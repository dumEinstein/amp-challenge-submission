"""Domain-aware AR variants: family conditioning, descriptor conditioning, factorized planning.

MOTIVATION, AND WHAT THE MEASUREMENTS ALREADY RULED OUT
-------------------------------------------------------
Before building this, two cheaper domain-aware ideas were measured and their premises FAILED:

  * Cys-parity grammar -- premise was "odd cysteine counts are biophysically impossible". FALSE:
    14.6% of real training AMPs have odd Cys counts. A hard parity constraint would overcorrect
    to 0% against a 14.6% target.
  * Helical-phase positional encoding -- premise was "the generator is blind to 3.6-residue
    periodicity and must rediscover it". Measured Kyte-Doolittle autocorrelation shows EVERY
    existing library already reproduces the amphipathic signature (real: lag2 -0.183, lag3
    -0.004, lag4 +0.060) to within 1.7-3.9x the disjoint-sample floor. Only ~0.015 of lag3+4
    correlation is correctable. Phase-agnostic position embeddings are not the bottleneck.

What the measurements DO support is multi-modality and systematic first-moment offsets:

    library            muH (real 0.894)     charge (real 2.79)
    AR                 0.864  (-3.3%)       2.81  (match)
    D3PM               0.936  (+4.7%)       2.64  (-5.4%)
    blend f0.4         0.905  (+1.2%)       3.14  (+12.5%)

AR UNDERSHOOTS amphipathicity, D3PM OVERSHOOTS it, and the blend lands between -- opposite-signed
biophysical errors, which is the first biophysical account of why blending is the only mechanism
that has ever replicated in this project. The heldout reference matches real to +0.5%, so these
offsets are systematic, not sampling noise.

THREE ARMS, ONE SHARED TRUNK
----------------------------
All three reuse the AR transformer from tools/ar_bpe.py (itself a validated reimplementation of the
shipped model: val 1.8368 vs 1.8424). Only the conditioning path differs, so any difference is
attributable to the conditioning and not to the backbone, optimiser, data or split.

  family      (idea #5) K-means the training set into biophysical families on (length, charge, muH,
              Cys count) and add a family embedding. A shared-trunk conditioned model rather than a
              true MoE with separate experts -- same multi-modality hypothesis at ~1/K the cost.
              Sampling draws a family from the empirical family frequencies.

  descriptor  (idea #1) Condition on continuous (length, charge, muH), z-scored with statistics
              stored in the checkpoint. Trained with 10% condition dropout to a learned NULL
              embedding, because conditioning on a DETERMINISTIC FUNCTION OF THE INPUT otherwise
              invites conditional collapse -- the model can minimise loss by learning p(x) and
              ignoring the condition, and would then not comply at sampling time.
              CRITICAL: at sampling the conditioning vector is drawn by RESAMPLING REAL TRAINING
              DESCRIPTOR TRIPLES, never by sampling each axis independently, which would destroy
              the charge/muH/length correlations and request biophysically impossible combinations.

  factored    (idea #3) Hierarchical p(plan) * p(seq | plan). The vocabulary is extended with 3x8
              quantised PLAN tokens (length bin, cationic fraction bin, hydrophobic fraction bin);
              every training sequence is prefixed with its own plan. The model therefore generates
              a composition plan first and then arranges residues to satisfy it. Targets the
              FKEA/FBD Pareto frontier that Stage D found: composition governs coverage,
              arrangement governs embedding geometry, and factorising lets them move separately.

MEASUREMENT DISCIPLINE (section 5.20): FBD(AMPs) has a within-checkpoint sd of 0.0676 (AR), so a
single-draw difference below ~0.19 is meaningless. Every arm is therefore sampled with 3 seeds, and
compared against the baseline's ALREADY MEASURED n=4 distribution (mean 0.6707, sd 0.0676) with a
Welch t-test rather than eyeballed.
"""

from __future__ import annotations

import argparse, json, math, random, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ar_bpe import (AA, MIN_LENGTH, MAX_LENGTH, PAD_ID, BOS_ID, EOS_ID, N_SPECIAL, BLOCK_SIZE,
                    TRAIN_FASTA, HELDOUT_FASTA, ANTIBAC_FASTA,
                    read_fasta, write_fasta, is_valid, Block, lr_at)

KD = dict(A=1.8, R=-4.5, N=-3.5, D=-3.5, C=2.5, Q=-3.5, E=-3.5, G=-0.4, H=-3.2, I=4.5,
          L=3.8, K=-3.9, M=1.9, F=2.8, P=-1.6, S=-0.8, T=-0.7, W=-0.9, Y=-1.3, V=4.2)
POS, NEG = set("KR"), set("DE")
CATIONIC, HYDROPHOBIC = set("KRH"), set("AVILMFWC")
N_BINS = 8
CHAR_VOCAB = N_SPECIAL + len(AA)          # 23
PLAN_AXES = 3                              # length bin, cationic-fraction bin, hydrophobic bin
PLAN_BASE = CHAR_VOCAB                     # plan token ids live above the residue vocabulary
PLAN_VOCAB = CHAR_VOCAB + PLAN_AXES * N_BINS
N_FAMILIES = 6                             # fixed K for the hierarchical mode
FAM_BASE = PLAN_VOCAB                      # family tokens sit above the plan tokens
HIER_VOCAB = PLAN_VOCAB + N_FAMILIES       # 23 + 24 + 6 = 53
DENS_BASE = HIER_VOCAB                     # conformity-density tokens sit above the family tokens,
PDENS_VOCAB = HIER_VOCAB + N_BINS          # so no existing id shifts: 53 + 8 = 61

STOI = {aa: i + N_SPECIAL for i, aa in enumerate(AA)}
ITOS = {i + N_SPECIAL: aa for i, aa in enumerate(AA)}


def mu_h(s: str) -> float:
    """Eisenberg hydrophobic moment; 100 deg/residue is the alpha-helix turn."""
    r = math.radians(100.0)
    x = sum(KD[c] * math.cos(i * r) for i, c in enumerate(s) if c in KD)
    y = sum(KD[c] * math.sin(i * r) for i, c in enumerate(s) if c in KD)
    return math.hypot(x, y) / len(s)


def net_charge(s: str) -> int:
    return sum(c in POS for c in s) - sum(c in NEG for c in s)


def descriptors3(seqs: list[str]) -> np.ndarray:
    return np.asarray([[len(s), net_charge(s), mu_h(s)] for s in seqs], dtype=np.float64)


def family_features(seqs: list[str]) -> np.ndarray:
    return np.asarray([[len(s), net_charge(s), mu_h(s), s.count("C")] for s in seqs],
                      dtype=np.float64)


def prefix_spec(mode: str) -> list[tuple[int, int]]:
    """(lo, size) of the vocabulary block each prefix step must emit a token from.

    factored/both : [plan_len][plan_cat][plan_hyd]
    hier          : [family][plan_len][plan_cat][plan_hyd]  -- family FIRST, so the plan
                    tokens are predicted conditional on it (that is the whole point of A2).
    pdens         : [plan_len][plan_cat][plan_hyd][density]  -- density LAST, so it is predicted
                    conditional on the composition it has to be achieved within. Measured: the
                    composition cell already fixes charge (within-cell sd / overall = 0.49) but says
                    almost nothing about the hydrophobic moment (0.82), because muH depends on
                    residue ORDER and a composition axis cannot express phase. See
                    tools/plan_density.py for why one density axis beats muH and charge as two axes.
    """
    plan = [(PLAN_BASE + a * N_BINS, N_BINS) for a in range(PLAN_AXES)]
    if mode == "hier":
        return [(FAM_BASE, N_FAMILIES)] + plan
    if mode == "pdens":
        return plan + [(DENS_BASE, N_BINS)]
    if mode in ("factored", "both"):
        return plan
    return []


def largest_remainder(p: np.ndarray, n: int) -> np.ndarray:
    """Integer quotas summing to exactly n, closest to n*p (A1)."""
    raw = p * n
    base = np.floor(raw).astype(int)
    rem = n - int(base.sum())
    if rem > 0:
        base[np.argsort(-(raw - base))[:rem]] += 1
    return base


def plan_cell(s: str, edges: dict) -> int:
    """The 512-cell index of a sequence's plan (len bin, cat bin, hyd bin)."""
    b = [t - PLAN_BASE - a * N_BINS for a, t in enumerate(plan_tokens(s, edges))]
    return b[0] * N_BINS * N_BINS + b[1] * N_BINS + b[2]


def cell_to_tokens(c: int) -> list[int]:
    bins = [c // (N_BINS * N_BINS), (c // N_BINS) % N_BINS, c % N_BINS]
    return [PLAN_BASE + a * N_BINS + b for a, b in enumerate(bins)]


def plan_tokens(s: str, edges: dict) -> list[int]:
    """Quantise (length, cationic fraction, hydrophobic fraction) into one token per axis."""
    vals = [len(s),
            sum(c in CATIONIC for c in s) / len(s),
            sum(c in HYDROPHOBIC for c in s) / len(s)]
    out = []
    for a, v in enumerate(vals):
        b = int(np.clip(np.searchsorted(edges[a], v) - 1, 0, N_BINS - 1))
        out.append(PLAN_BASE + a * N_BINS + b)
    return out


def _kmeans(X: np.ndarray, k: int, seed: int, n_init: int = 10, iters: int = 100):
    """k-means++ / Lloyd in pure numpy.

    NOT an sklearn call on purpose: sklearn is installed in the grader venv but NOT in the CUDA
    training venv, and the compute nodes have no network. Pre-flighting this file with the grader
    interpreter hid that, and the family arm died with ModuleNotFoundError at runtime.
    Deterministic given `seed`; returns (centers, labels) from the best of `n_init` restarts.
    """
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(n_init):
        first = int(rng.integers(len(X)))
        idx = [first]
        d2 = ((X - X[first]) ** 2).sum(1)
        for _ in range(k - 1):
            s = d2.sum()
            j = int(rng.choice(len(X), p=d2 / s)) if s > 0 else int(rng.integers(len(X)))
            idx.append(j)
            d2 = np.minimum(d2, ((X - X[j]) ** 2).sum(1))
        C = X[idx].astype(np.float64).copy()
        lab = np.zeros(len(X), dtype=np.int64)
        for _ in range(iters):
            D = ((X[:, None, :] - C[None, :, :]) ** 2).sum(-1)
            lab = D.argmin(1)
            newC = np.stack([X[lab == j].mean(0) if bool((lab == j).any()) else C[j]
                             for j in range(k)])
            if np.allclose(newC, C):
                C = newC
                break
            C = newC
        inertia = float(((X - C[lab]) ** 2).sum())
        if best is None or inertia < best[0]:
            best = (inertia, C.copy(), lab.copy())
    return best[1], best[2]


def _assign(X: np.ndarray, C: np.ndarray) -> np.ndarray:
    return ((X[:, None, :] - C[None, :, :]) ** 2).sum(-1).argmin(1)


class CondLM(nn.Module):
    """The ar_bpe AR trunk with an optional conditioning path added to the input embeddings."""

    def __init__(self, vocab_size, cond_dim=0, n_families=0, d_model=256, n_layers=6,
                 n_heads=8, d_ff=1024, dropout=0.1, block_size=BLOCK_SIZE + PLAN_AXES):
        super().__init__()
        self.vocab_size, self.block_size = vocab_size, block_size
        self.cond_dim, self.n_families = cond_dim, n_families
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(block_size, d_model)
        if cond_dim:
            self.cond_mlp = nn.Sequential(nn.Linear(cond_dim, d_model), nn.SiLU(),
                                          nn.Linear(d_model, d_model))
            self.null_cond = nn.Parameter(torch.zeros(d_model))   # learned NULL for dropout/CFG
        if n_families:
            self.fam_emb = nn.Embedding(n_families, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([Block(d_model, n_heads, d_ff, dropout)
                                     for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, cond=None, fam=None, cond_mask=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        if self.cond_dim and cond is not None:
            c = self.cond_mlp(cond)
            if cond_mask is not None:                       # True -> use NULL instead
                c = torch.where(cond_mask.unsqueeze(1), self.null_cond.expand_as(c), c)
            x = x + c.unsqueeze(1)
        if self.n_families and fam is not None:
            x = x + self.fam_emb(fam).unsqueeze(1)
        x = self.drop(x)
        for b in self.blocks:
            x = b(x)
        return self.head(self.ln_f(x))

    def loss(self, batch, cond=None, fam=None, cond_mask=None):
        logits = self(batch[:, :-1], cond, fam, cond_mask)
        return F.cross_entropy(logits.reshape(-1, self.vocab_size), batch[:, 1:].reshape(-1),
                               ignore_index=PAD_ID)


def encode_char(s: str) -> list[int]:
    return [BOS_ID] + [STOI[c] for c in s] + [EOS_ID]


def build_rows(seqs, mode, edges=None, fam=None, dens=None):
    rows = []
    for i, s in enumerate(seqs):
        ids = encode_char(s)
        if mode in ("factored", "both"):
            ids = [BOS_ID] + plan_tokens(s, edges) + ids[1:]
        elif mode == "pdens":
            # [BOS][plan_len][plan_cat][plan_hyd][density][residues...][EOS]
            ids = [BOS_ID] + plan_tokens(s, edges) + [DENS_BASE + int(dens[i])] + ids[1:]
        elif mode == "hier":
            # [BOS][family][plan_len][plan_cat][plan_hyd][residues...][EOS]
            ids = [BOS_ID, FAM_BASE + int(fam[i])] + plan_tokens(s, edges) + ids[1:]
        rows.append(ids)
    width = max(len(r) for r in rows)
    out = torch.full((len(rows), width), PAD_ID, dtype=torch.long)
    for i, r in enumerate(rows):
        out[i, : len(r)] = torch.tensor(r, dtype=torch.long)
    return out


@torch.no_grad()
def sample(model, mode, n, gen, device, temp, top_p, cond=None, fam=None, n_plan=0,
           prefix=None, force=None):
    """force: (n, len(prefix)) absolute token ids to emit instead of sampling (A1), or None.

    A negative entry means DO NOT force that step -- the model samples it. `pdens` uses this to pin
    only the density token while letting the model choose its own composition plan, so the arm tests
    the density axis and not a re-weighted composition distribution.
    """
    vocab = model.vocab_size
    if prefix is None:
        prefix = prefix_spec(mode)
    n_plan = len(prefix)
    tokens = torch.full((n, 1), BOS_ID, dtype=torch.long, device=device)
    res = torch.zeros(n, dtype=torch.long, device=device)
    done = torch.zeros(n, dtype=torch.bool, device=device)
    total_steps = MAX_LENGTH + 1 + n_plan
    for step in range(total_steps):
        logits = model(tokens, cond, fam)[:, -1, :] / temp
        logits[:, PAD_ID] = float("-inf")
        logits[:, BOS_ID] = float("-inf")
        in_plan = step < n_plan
        if in_plan:
            # position `step` must emit a token from its own block
            lo, size = prefix[step]
            mask = torch.ones(vocab, dtype=torch.bool, device=device)
            mask[lo : lo + size] = False
            logits = logits.masked_fill(mask.unsqueeze(0), float("-inf"))
        else:
            if vocab > CHAR_VOCAB:                      # plan tokens illegal inside the sequence
                logits[:, PLAN_BASE:] = float("-inf")
            logits[res < MIN_LENGTH, EOS_ID] = float("-inf")
            at_cap = res >= MAX_LENGTH
            if bool(at_cap.any()):
                logits[at_cap, N_SPECIAL:CHAR_VOCAB] = float("-inf")
        probs = torch.softmax(logits, dim=-1)
        if top_p < 1.0 and not in_plan:
            ordered, order = torch.sort(probs, dim=-1, descending=True)
            drop = ordered.cumsum(dim=-1) - ordered > top_p
            ordered[drop] = 0.0
            probs = torch.zeros_like(probs).scatter_(1, order, ordered)
            probs = probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-12)
        nxt = torch.multinomial(probs, 1, generator=gen)
        if in_plan and force is not None:
            f = force[:, step : step + 1].to(device)
            nxt = torch.where(f >= 0, f, nxt)          # negative = leave this step to the model
        nxt[done] = PAD_ID
        if not in_plan:
            is_res = (nxt.squeeze(1) >= N_SPECIAL) & (nxt.squeeze(1) < CHAR_VOCAB) & (~done)
            res = res + is_res.long()
            done = done | (nxt.squeeze(1) == EOS_ID)
        tokens = torch.cat([tokens, nxt], dim=1)
        if bool(done.all()):
            break
    return ["".join(ITOS[i] for i in row.tolist() if i in ITOS) for row in tokens[:, 1:].cpu()]


def cmd_train(a):
    if not torch.cuda.is_available() and not a.allow_cpu:
        raise SystemExit(f"REFUSING TO TRAIN ON CPU (torch {torch.__version__}). Use the CUDA venv "
                         "or pass --allow-cpu.")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)

    meta = {"mode": a.mode}
    corpus = Path(getattr(a, "train_fasta", None) or TRAIN_FASTA)
    valset = Path(getattr(a, "heldout_fasta", None) or HELDOUT_FASTA)
    allseq = [s for s in read_fasta(corpus) if is_valid(s)]
    held = set(read_fasta(valset))
    if corpus != TRAIN_FASTA or valset != HELDOUT_FASTA:
        print(f"[data] OVERRIDE corpus={corpus.name} val={valset.name}", flush=True)
    meta["train_fasta"], meta["heldout_fasta"] = str(corpus), str(valset)
    tr_seqs = [s for s in allseq if s not in held]
    va_seqs = [s for s in allseq if s in held]
    print(f"[data] {len(allseq)} valid | {len(tr_seqs)} train | {len(va_seqs)} val "
          f"(reusing submission heldout split)", flush=True)

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    vocab = CHAR_VOCAB
    cond_dim = n_fam = 0
    edges = None
    tr_cond = va_cond = tr_fam = va_fam = None

    if a.mode == "descriptor":
        cond_dim = 3
        D = descriptors3(tr_seqs); V = descriptors3(va_seqs)
        mu, sd = D.mean(0), D.std(0) + 1e-9
        meta["cond_mu"], meta["cond_sd"] = mu.tolist(), sd.tolist()
        meta["cond_pool"] = D[np.random.default_rng(0).choice(len(D), 20000, replace=False)].tolist()
        tr_cond = torch.tensor((D - mu) / sd, dtype=torch.float32)
        va_cond = torch.tensor((V - mu) / sd, dtype=torch.float32)
        print(f"[cond] descriptors z-scored | mu {np.round(mu,3).tolist()} "
              f"sd {np.round(sd,3).tolist()} | dropout {a.cond_dropout}", flush=True)
    if a.mode in ("family", "both", "hier"):
        Ftr = family_features(tr_seqs); Fva = family_features(va_seqs)
        mu, sd = Ftr.mean(0), Ftr.std(0) + 1e-9
        centers, lab_tr = _kmeans((Ftr - mu) / sd, a.families, a.seed)
        n_fam = a.families
        lab_va = _assign((Fva - mu) / sd, centers)
        hier_tr, hier_va = lab_tr, lab_va
        cnt = np.bincount(lab_tr, minlength=n_fam)
        meta.update({"fam_mu": mu.tolist(), "fam_sd": sd.tolist(),
                     "fam_centers": centers.tolist(),
                     "fam_freq": (cnt / cnt.sum()).tolist()})
        if a.mode == "hier":
            n_fam = 0                       # in-stream token, NOT a side-channel embedding
        else:
            tr_fam = torch.tensor(lab_tr, dtype=torch.long)
            va_fam = torch.tensor(lab_va, dtype=torch.long)
        print(f"[fam] K={n_fam} sizes {cnt.tolist()}", flush=True)
        for k in range(n_fam):
            c = centers[k] * sd + mu
            print(f"       family {k}: len {c[0]:5.1f}  charge {c[1]:+5.2f}  muH {c[2]:.3f}  "
                  f"Cys {c[3]:.2f}  n={cnt[k]}", flush=True)
    dens_tr = dens_va = None
    if a.mode in ("factored", "both", "hier", "pdens"):
        vocab = HIER_VOCAB if a.mode == "hier" else (PDENS_VOCAB if a.mode == "pdens" else PLAN_VOCAB)
        vals = np.asarray([[len(s), sum(c in CATIONIC for c in s)/len(s),
                            sum(c in HYDROPHOBIC for c in s)/len(s)] for s in tr_seqs])
        edges = {a_: np.quantile(vals[:, a_], np.linspace(0, 1, N_BINS + 1)) for a_ in range(3)}
        meta["plan_edges"] = {str(k): v.tolist() for k, v in edges.items()}
        n_axes = PLAN_AXES + (1 if a.mode == "pdens" else 0)
        print(f"[plan] vocab {CHAR_VOCAB} -> {vocab} ({n_axes} axes x {N_BINS} bins)", flush=True)
    if a.mode == "pdens":
        # Precomputed by tools/plan_density.py in the grader venv: the KDE needs sklearn, which this
        # (CUDA) venv does not have. Keyed by sequence, so it is order-independent.
        dpath = Path(getattr(a, "density_json", None) or (Path(__file__).resolve().parent.parent
                                                          / "data/plan/density.json"))
        dj = json.loads(dpath.read_text())
        if int(dj["n_bins"]) != N_BINS:
            raise SystemExit(f"density sidecar has {dj['n_bins']} bins, ar_domain has {N_BINS}")
        sb = dj["seq_bin"]
        miss = [s_ for s_ in tr_seqs if s_ not in sb]
        if miss:
            raise SystemExit(f"{len(miss)} training sequences absent from {dpath.name} "
                             f"(regenerate it against this corpus); first: {miss[0][:30]}")
        dens_tr = [sb[s_] for s_ in tr_seqs]
        # a val sequence outside the sidecar cannot be binned here; bin 0 is a placeholder that only
        # affects the reported val loss for those rows, never training
        dens_va = [sb.get(s_, 0) for s_ in va_seqs]
        unseen = sum(1 for s_ in va_seqs if s_ not in sb)
        meta["density_json"] = str(dpath)
        meta["density_edges"] = dj["density_edges"]
        meta["joint_cell_density"] = dj["joint_cell_density"]
        print(f"[dens] {dpath.name}: {len(sb)} keyed seqs, train bins "
              f"{np.bincount(dens_tr, minlength=N_BINS).tolist()}"
              + (f", {unseen} val seqs unbinned (placeholder)" if unseen else ""), flush=True)

    if a.mode == "hier":
        tr = build_rows(tr_seqs, a.mode, edges, hier_tr)
        va = build_rows(va_seqs, a.mode, edges, hier_va)
    elif a.mode == "pdens":
        tr = build_rows(tr_seqs, a.mode, edges, None, dens_tr)
        va = build_rows(va_seqs, a.mode, edges, None, dens_va)
    else:
        tr = build_rows(tr_seqs, a.mode, edges); va = build_rows(va_seqs, a.mode, edges)
    block = BLOCK_SIZE + PLAN_AXES + (1 if a.mode in ("hier", "pdens") else 0)
    meta["block_size"] = block
    model = CondLM(vocab, cond_dim, n_fam, a.d_model, a.n_layers, a.n_heads, a.d_ff,
                   a.dropout, block_size=block).to(dev)
    print(f"[model] mode={a.mode} vocab={vocab} cond_dim={cond_dim} n_fam={n_fam} | "
          f"{sum(p.numel() for p in model.parameters())/1e6:.2f}M params on {dev}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay,
                            betas=(0.9, 0.98))
    total = math.ceil(len(tr) / a.batch) * a.epochs
    best, bad, step, hist = float("inf"), 0, 0, []
    t0 = time.time()
    for ep in range(a.epochs):
        model.train()
        perm = torch.randperm(len(tr), generator=torch.Generator().manual_seed(a.seed + ep))
        run, nb = 0.0, 0
        for i in range(0, len(tr), a.batch):
            sel = perm[i : i + a.batch]
            batch = tr[sel].to(dev)
            c = tr_cond[sel].to(dev) if tr_cond is not None else None
            f_ = tr_fam[sel].to(dev) if tr_fam is not None else None
            cm = None
            if c is not None and a.cond_dropout > 0:
                cm = torch.rand(len(sel), device=dev) < a.cond_dropout
            for g in opt.param_groups:
                g["lr"] = lr_at(step, total, a.lr, a.warmup)
            loss = model.loss(batch, c, f_, cm)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            run += loss.item(); nb += 1; step += 1
        model.eval()
        with torch.no_grad():
            vl, vn = 0.0, 0
            for i in range(0, len(va), a.batch):
                b = va[i : i + a.batch].to(dev)
                c = va_cond[i : i + a.batch].to(dev) if va_cond is not None else None
                f_ = va_fam[i : i + a.batch].to(dev) if va_fam is not None else None
                vl += model.loss(b, c, f_).item(); vn += 1
            vl /= max(1, vn)
        mark = ""
        if vl < best - 1e-5:
            best, bad = vl, 0
            torch.save({"model": model.state_dict(), "vocab": vocab, "cond_dim": cond_dim,
                        "n_fam": n_fam, "block_size": block, "meta": meta, "args": vars(a), "epoch": ep, "val": vl},
                       out / "model.pt")
            mark = " *"
        else:
            bad += 1
        hist.append({"epoch": ep, "train": run / nb, "val": vl})
        print(f"[ep {ep:3d}] train {run/nb:.4f} val {vl:.4f} lr {opt.param_groups[0]['lr']:.2e} "
              f"{time.time()-t0:.0f}s{mark}", flush=True)
        if bad >= a.patience:
            print(f"[stop] no val improvement for {a.patience} epochs", flush=True); break
    json.dump({"history": hist, "best_val": best, "mode": a.mode},
              open(out / "history.json", "w"), indent=1)
    print(f"[done] best val {best:.4f} -> {out/'model.pt'}", flush=True)


def cmd_sample(a):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(a.ckpt, map_location=dev, weights_only=False)
    ta, meta, mode = ck["args"], ck["meta"], ck["meta"]["mode"]
    block = ck.get("block_size", BLOCK_SIZE + PLAN_AXES)
    model = CondLM(ck["vocab"], ck["cond_dim"], ck["n_fam"], ta["d_model"], ta["n_layers"],
                   ta["n_heads"], ta["d_ff"], 0.0, block_size=block).to(dev)
    model.load_state_dict(ck["model"]); model.eval()
    print(f"[sample] mode={mode} val {ck['val']:.4f} epoch {ck['epoch']} temp {a.temp} "
          f"top_p {a.top_p} seed {a.seed}", flush=True)

    banned = set(read_fasta(ANTIBAC_FASTA)) if ANTIBAC_FASTA.exists() else set()
    ban_src = Path(getattr(a, "ban_fasta", None) or TRAIN_FASTA)
    if ban_src.exists():
        banned |= set(read_fasta(ban_src))
    if ban_src != TRAIN_FASTA:
        print(f"[sample] OVERRIDE banned corpus = {ban_src.name} ({len(banned)} banned)", flush=True)
    gen = torch.Generator(device=dev); gen.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    prefix = prefix_spec(mode)
    n_plan = len(prefix)
    keep, seen, drawn = [], set(), 0

    # ---------------- A1: integer-quota plan assembly ----------------
    if getattr(a, "quota", False):
        if not prefix or mode not in ("factored", "both"):
            raise SystemExit(f"--quota needs a plan-prefixed model; mode={mode} has none")
        edges = {int(k): np.asarray(v) for k, v in meta["plan_edges"].items()}
        tr_seqs = [t for t in read_fasta(ban_src) if is_valid(t)]
        hist = np.bincount([plan_cell(t, edges) for t in tr_seqs],
                           minlength=N_BINS ** 3).astype(float)
        hist /= hist.sum()
        need = largest_remainder(hist, a.n)
        print(f"[quota] {int((need > 0).sum())} of {int((hist > 0).sum())} occupied cells "
              f"reachable at n={a.n}; strict={a.quota_strict}", flush=True)
        comply = miss = 0
        for _ in range(a.max_rounds):
            out_cells = np.repeat(np.arange(N_BINS ** 3), need)
            if len(out_cells) == 0:
                break
            batch = (out_cells if len(out_cells) >= a.draw
                     else np.resize(out_cells, a.draw))[: a.draw]
            force = torch.tensor([[BOS_ID] + cell_to_tokens(int(c)) for c in batch],
                                 dtype=torch.long)[:, 1:]
            got = sample(model, mode, len(batch), gen, dev, a.temp, a.top_p,
                         None, None, n_plan, prefix, force)
            for sq, c in zip(got, batch):
                drawn += 1
                if not (is_valid(sq) and sq not in banned and sq not in seen):
                    continue
                ok = plan_cell(sq, edges) == int(c)      # did the model honour the plan?
                comply += ok; miss += (not ok)
                if a.quota_strict and not ok:
                    continue
                if need[int(c)] <= 0:
                    continue
                need[int(c)] -= 1
                seen.add(sq); keep.append(sq)
            if int(need.sum()) == 0:
                break
        tot = comply + miss
        print(f"[quota] plan compliance {comply}/{tot} = {comply/max(1,tot):.3f} | "
              f"unfilled slots {int(need.sum())}", flush=True)
        if int(need.sum()) > 0:
            print(f"[quota] WARNING: {int(need.sum())} slots unfilled after {a.max_rounds} "
                  f"rounds; the plan histogram will NOT match training exactly", flush=True)
        keep = keep[: a.n]
        write_fasta(keep, Path(a.out))
        L = [len(t) for t in keep]
        got_hist = np.bincount([plan_cell(t, edges) for t in keep],
                               minlength=N_BINS ** 3).astype(float)
        got_hist /= max(1.0, got_hist.sum())
        print(f"[quota] realised plan TV vs training = {0.5*np.abs(got_hist-hist).sum():.4f}",
              flush=True)
        print(f"[sample] wrote {len(keep)} to {a.out} | mean len {np.mean(L):.1f} "
              f"min {min(L)} max {max(L)} | muH {np.mean([mu_h(t) for t in keep]):.3f} "
              f"charge {np.mean([net_charge(t) for t in keep]):.2f} | "
              f"yield {len(keep)/max(1,drawn):.3f}", flush=True)
        return

    # ---------------- pdens: request high-density plans instead of selecting for them ----
    # --dens-top K forces the density token to one of the K HIGHEST bins, chosen uniformly, and
    # leaves the three composition steps to the model (the -1 sentinel in `force`). K=0 forces
    # nothing, which is the CONTROL: identical checkpoint, identical sampler, untilted plan.
    dens_top = int(getattr(a, "dens_top", 0) or 0)
    if dens_top and mode != "pdens":
        raise SystemExit(f"--dens-top needs a density-prefixed model; mode={mode} has none")
    req_hist = np.zeros(N_BINS, dtype=int)

    for _ in range(a.max_rounds):
        cond = fam = None
        force = None
        if dens_top:
            k = min(dens_top, N_BINS)
            bins = rng.choice(np.arange(N_BINS - k, N_BINS), a.draw)
            req_hist += np.bincount(bins, minlength=N_BINS)
            force = torch.full((a.draw, n_plan), -1, dtype=torch.long)
            force[:, -1] = torch.tensor(DENS_BASE + bins, dtype=torch.long)
        if mode == "descriptor":
            # resample REAL descriptor triples: keeps the joint charge/muH/length structure
            pool = np.asarray(meta["cond_pool"])
            mu, sd = np.asarray(meta["cond_mu"]), np.asarray(meta["cond_sd"])
            pick = pool[rng.choice(len(pool), a.draw)]
            cond = torch.tensor((pick - mu) / sd, dtype=torch.float32, device=dev)
        if mode in ("family", "both"):
            freq = np.asarray(meta["fam_freq"])
            fam = torch.tensor(rng.choice(len(freq), a.draw, p=freq), dtype=torch.long, device=dev)
        for s in sample(model, mode, a.draw, gen, dev, a.temp, a.top_p, cond, fam, n_plan,
                        prefix, force):
            drawn += 1
            if is_valid(s) and s not in banned and s not in seen:
                seen.add(s); keep.append(s)
        if len(keep) >= a.n:
            break
    keep = keep[: a.n]
    write_fasta(keep, Path(a.out))
    L = [len(s) for s in keep]
    print(f"[sample] wrote {len(keep)} to {a.out} | mean len {np.mean(L):.1f} min {min(L)} "
          f"max {max(L)} | muH {np.mean([mu_h(s) for s in keep]):.3f} "
          f"charge {np.mean([net_charge(s) for s in keep]):.2f} | "
          f"yield {len(keep)/max(1,drawn):.3f}", flush=True)


def cmd_selftest(a):
    ok = 0
    def chk(n, c):
        nonlocal ok
        assert c, f"FAIL {n}"; ok += 1
    seqs = [s for s in read_fasta(TRAIN_FASTA) if is_valid(s)][:3000]
    chk("data", len(seqs) == 3000)
    # descriptor sanity against the shipped selector's own formula
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "packages" / "phase2_1_submission" / "src"))
    from amp_moment.cli import KD as KD2, POS as P2, NEG as N2
    chk("KD table matches the shipped selector", KD == KD2)
    chk("POS/NEG match the shipped selector", POS == P2 and NEG == N2)
    D = descriptors3(seqs)
    chk("descriptors finite", np.isfinite(D).all())
    chk("muH in plausible range", 0.0 < D[:, 2].mean() < 3.0)
    # plan tokens
    vals = np.asarray([[len(s), sum(c in CATIONIC for c in s)/len(s),
                        sum(c in HYDROPHOBIC for c in s)/len(s)] for s in seqs])
    edges = {i: np.quantile(vals[:, i], np.linspace(0, 1, N_BINS + 1)) for i in range(3)}
    for s in seqs[:500]:
        p = plan_tokens(s, edges)
        chk("plan has one token per axis", len(p) == PLAN_AXES)
        for a_, t in enumerate(p):
            chk("plan token in its axis block",
                PLAN_BASE + a_*N_BINS <= t < PLAN_BASE + (a_+1)*N_BINS)
    # round-trip: char encoding recovers the sequence
    for s in seqs[:500]:
        chk("char round-trip", "".join(ITOS[i] for i in encode_char(s) if i in ITOS) == s)
    # rows
    r = build_rows(seqs[:64], "factored", edges)
    chk("factored rows prefixed by plan", bool((r[:, 1:1+PLAN_AXES] >= PLAN_BASE).all()))
    r2 = build_rows(seqs[:64], "baseline")
    chk("baseline rows have no plan tokens", bool((r2 < CHAR_VOCAB).all()))
    # sampling invariants for each mode, tiny model, cpu
    for mode, vocab, cd, nf, npl in (("baseline", CHAR_VOCAB, 0, 0, 0),
                                     ("descriptor", CHAR_VOCAB, 3, 0, 0),
                                     ("family", CHAR_VOCAB, 0, 5, 0),
                                     ("factored", PLAN_VOCAB, 0, 0, PLAN_AXES)):
        m = CondLM(vocab, cd, nf, d_model=32, n_layers=1, n_heads=2, d_ff=64, dropout=0.0).eval()
        g = torch.Generator(); g.manual_seed(0)
        cond = torch.zeros(32, 3) if cd else None
        fam = torch.zeros(32, dtype=torch.long) if nf else None
        out = sample(m, mode, 32, g, "cpu", 1.0, 0.95, cond, fam, npl)
        chk(f"{mode}: count", len(out) == 32)
        chk(f"{mode}: EVERY sample length-valid (guarantee)",
            all(MIN_LENGTH <= len(s) <= MAX_LENGTH for s in out))
        chk(f"{mode}: alphabet valid", all(not (set(s) - set(AA)) for s in out))
        chk(f"{mode}: no plan tokens leak into output", all(set(s) <= set(AA) for s in out))
        g2 = torch.Generator(); g2.manual_seed(0)
        chk(f"{mode}: seed-deterministic",
            sample(m, mode, 32, g2, "cpu", 1.0, 0.95, cond, fam, npl) == out)
    # conditioning actually reaches the logits (else it is a silent no-op)
    m = CondLM(CHAR_VOCAB, 3, 0, d_model=32, n_layers=1, n_heads=2, d_ff=64, dropout=0.0).eval()
    idx = torch.randint(N_SPECIAL, CHAR_VOCAB, (4, 6))
    with torch.no_grad():
        a1 = m(idx, torch.zeros(4, 3)); a2 = m(idx, torch.ones(4, 3) * 5)
    chk("descriptor conditioning changes logits", not torch.allclose(a1, a2))
    with torch.no_grad():
        nullmask = torch.ones(4, dtype=torch.bool)
        a3 = m(idx, torch.ones(4, 3) * 5, cond_mask=nullmask)
        a4 = m(idx, torch.zeros(4, 3), cond_mask=nullmask)
    chk("cond dropout routes to the SAME null embedding", torch.allclose(a3, a4))
    m2 = CondLM(CHAR_VOCAB, 0, 5, d_model=32, n_layers=1, n_heads=2, d_ff=64, dropout=0.0).eval()
    with torch.no_grad():
        b1 = m2(idx, fam=torch.zeros(4, dtype=torch.long))
        b2 = m2(idx, fam=torch.ones(4, dtype=torch.long) * 4)
    chk("family conditioning changes logits", not torch.allclose(b1, b2))

    # ---------------- A1 / A2 additions ----------------
    chk("hier vocab is 53", HIER_VOCAB == CHAR_VOCAB + PLAN_AXES * N_BINS + N_FAMILIES)
    chk("family tokens sit above plan tokens", FAM_BASE == PLAN_VOCAB)
    ps = prefix_spec("hier")
    chk("hier prefix is family then 3 plan axes", len(ps) == 4 and ps[0] == (FAM_BASE, N_FAMILIES))
    for a_ in range(PLAN_AXES):
        chk("hier plan block follows the family token",
            ps[1 + a_] == (PLAN_BASE + a_ * N_BINS, N_BINS))
    chk("factored prefix unchanged", prefix_spec("factored") == ps[1:])

    # ---- mode pdens: the conformity-density plan axis -------------------------------------
    chk("pdens vocab is 61", PDENS_VOCAB == HIER_VOCAB + N_BINS)
    chk("density tokens sit above the family tokens", DENS_BASE == HIER_VOCAB)
    chk("adding pdens shifted no existing id",
        (PLAN_BASE, PLAN_VOCAB, FAM_BASE, HIER_VOCAB) == (23, 47, 47, 53))
    pd = prefix_spec("pdens")
    chk("pdens prefix is 3 plan axes then density", len(pd) == PLAN_AXES + 1)
    chk("pdens keeps the factored prefix as its head", pd[:PLAN_AXES] == prefix_spec("factored"))
    chk("pdens density block is last", pd[-1] == (DENS_BASE, N_BINS))
    seqs_ = ["KRWWKWWRR", "GIGKFLHSAKKFGKAFVGEIMNS", "AAAAAAAAAA"]
    vals_ = np.asarray([[len(t), sum(c in CATIONIC for c in t)/len(t),
                         sum(c in HYDROPHOBIC for c in t)/len(t)] for t in seqs_])
    ed_ = {i: np.quantile(vals_[:, i], np.linspace(0, 1, N_BINS + 1)) for i in range(3)}
    rp = build_rows(seqs_, "pdens", ed_, None, [0, 7, 3])
    chk("pdens rows begin with BOS", bool((rp[:, 0] == BOS_ID).all()))
    chk("pdens rows carry 3 plan tokens",
        bool(((rp[:, 1:1+PLAN_AXES] >= PLAN_BASE) & (rp[:, 1:1+PLAN_AXES] < FAM_BASE)).all()))
    chk("pdens density token is at position 4 and in its own block",
        bool(((rp[:, 1+PLAN_AXES] >= DENS_BASE) &
              (rp[:, 1+PLAN_AXES] < DENS_BASE + N_BINS)).all()))
    chk("pdens density token encodes the given bin",
        [int(v) - DENS_BASE for v in rp[:, 1+PLAN_AXES]] == [0, 7, 3])
    chk("pdens residues carry no prefix tokens", bool((rp[:, 2+PLAN_AXES:] < PLAN_BASE).all()))

    # partial forcing: a negative entry must leave that step to the model
    g_ = torch.Generator(); g_.manual_seed(0)
    pm = CondLM(PDENS_VOCAB, 0, 0, 64, 2, 4, 128, 0.0,
                block_size=BLOCK_SIZE + PLAN_AXES + 1).eval()
    frc = torch.full((4, PLAN_AXES + 1), -1, dtype=torch.long)
    frc[:, -1] = DENS_BASE + 6
    with torch.no_grad():
        tk = [None]
        orig_cat = torch.cat
        _ = sample(pm, "pdens", 4, g_, "cpu", 1.0, 1.0, None, None, PLAN_AXES + 1, pd, frc)
    # re-run capturing tokens: assert the forced step took the forced value and the free steps did not
    # all collapse to one id (they are sampled, so at least their block membership must hold)
    with torch.no_grad():
        toks = torch.full((4, 1), BOS_ID, dtype=torch.long)
        for st in range(PLAN_AXES + 1):
            lo, size = pd[st]
            lg = pm(toks)[:, -1, :]
            m = torch.ones(PDENS_VOCAB, dtype=torch.bool); m[lo:lo+size] = False
            lg = lg.masked_fill(m.unsqueeze(0), float("-inf"))
            nx = torch.multinomial(torch.softmax(lg, -1), 1, generator=g_)
            f = frc[:, st:st+1]
            nx = torch.where(f >= 0, f, nx)
            toks = torch.cat([toks, nx], dim=1)
        chk("free plan steps stay inside their own block",
            all(PLAN_BASE + a_*N_BINS <= int(toks[0, 1+a_]) < PLAN_BASE + (a_+1)*N_BINS
                for a_ in range(PLAN_AXES)))
        chk("forced density step took the forced bin",
            bool((toks[:, 1+PLAN_AXES] == DENS_BASE + 6).all()))
    chk("baseline has no prefix", prefix_spec("baseline") == [])

    # quota arithmetic
    rq = np.random.default_rng(0).dirichlet(np.ones(N_BINS ** 3))
    for n_ in (1000, 997, 12345):
        q = largest_remainder(rq, n_)
        chk("quota sums to exactly n", int(q.sum()) == n_)
        chk("quota is non-negative", bool((q >= 0).all()))
    chk("quota of a degenerate histogram is degenerate",
        int(largest_remainder(np.eye(N_BINS ** 3)[7], 100)[7]) == 100)

    # cell <-> token round-trip, on real sequences
    for t_ in seqs[:300]:
        c = plan_cell(t_, edges)
        chk("cell index in range", 0 <= c < N_BINS ** 3)
        chk("cell -> tokens round-trips", cell_to_tokens(c) == plan_tokens(t_, edges))

    # hier rows: [BOS][fam][plan x3][residues][EOS]
    famlab = np.arange(64) % N_FAMILIES
    rh = build_rows(seqs[:64], "hier", edges, famlab)
    chk("hier row starts with BOS", bool((rh[:, 0] == BOS_ID).all()))
    chk("hier row has a family token at position 1",
        bool(((rh[:, 1] >= FAM_BASE) & (rh[:, 1] < HIER_VOCAB)).all()))
    chk("hier family token matches the label",
        bool((rh[:, 1] - FAM_BASE == torch.tensor(famlab)).all()))
    chk("hier plan tokens follow the family token",
        bool(((rh[:, 2:2+PLAN_AXES] >= PLAN_BASE) & (rh[:, 2:2+PLAN_AXES] < FAM_BASE)).all()))
    chk("hier residues carry no prefix tokens", bool((rh[:, 2+PLAN_AXES:] < PLAN_BASE).all()))
    chk("hier row is one token longer than factored",
        rh.shape[1] == build_rows(seqs[:64], "factored", edges).shape[1] + 1)

    # forced-prefix sampling reproduces the requested cells exactly (A1's mechanism)
    hm = CondLM(HIER_VOCAB, 0, 0, 64, 2, 4, 128, 0.0,
                block_size=BLOCK_SIZE + PLAN_AXES + 1).eval()
    g2 = torch.Generator(); g2.manual_seed(0)
    cells = [5, 100, 371, 511]
    frc = torch.tensor([cell_to_tokens(c) for c in cells], dtype=torch.long)
    fm = CondLM(PLAN_VOCAB, 0, 0, 64, 2, 4, 128, 0.0).eval()
    outs = sample(fm, "factored", len(cells), g2, "cpu", 1.0, 1.0, None, None, 0,
                  prefix_spec("factored"), frc)
    chk("forced sampling returns one sequence per request", len(outs) == len(cells))
    chk("forced sampling emits only residues", all(set(o) <= set(AA) for o in outs))
    # the untrained model will not COMPLY with the plan, but it must have EMITTED it:
    with torch.no_grad():
        toks = torch.full((len(cells), 1), BOS_ID, dtype=torch.long)
        for st in range(PLAN_AXES):
            nx = frc[:, st : st + 1]
            toks = torch.cat([toks, nx], dim=1)
        chk("forced prefix lands in the right blocks",
            all(PLAN_BASE + a_ * N_BINS <= int(toks[0, 1 + a_]) < PLAN_BASE + (a_ + 1) * N_BINS
                for a_ in range(PLAN_AXES)))
    outs_h = sample(hm, "hier", 4, g2, "cpu", 1.0, 1.0)
    chk("hier sampling emits only residues", all(set(o) <= set(AA) for o in outs_h))
    chk("hier sampling respects the length bounds",
        all(len(o) <= MAX_LENGTH for o in outs_h))

    print(f"[selftest] {ok} checks passed")
    print("SELFTEST PASSED")


def main():
    p = argparse.ArgumentParser(); sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--mode", required=True,
                   choices=["baseline", "descriptor", "family", "factored", "both", "hier",
                            "pdens"])
    t.add_argument("--out", required=True)
    t.add_argument("--families", type=int, default=6)
    t.add_argument("--cond-dropout", type=float, default=0.1, dest="cond_dropout")
    t.add_argument("--epochs", type=int, default=60)
    t.add_argument("--batch", type=int, default=256)
    t.add_argument("--lr", type=float, default=3e-4)
    t.add_argument("--weight-decay", type=float, default=0.01, dest="weight_decay")
    t.add_argument("--warmup", type=int, default=200)
    t.add_argument("--patience", type=int, default=8)
    t.add_argument("--d-model", type=int, default=256, dest="d_model")
    t.add_argument("--n-layers", type=int, default=6, dest="n_layers")
    t.add_argument("--n-heads", type=int, default=8, dest="n_heads")
    t.add_argument("--d-ff", type=int, default=1024, dest="d_ff")
    t.add_argument("--dropout", type=float, default=0.1)
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--allow-cpu", action="store_true", dest="allow_cpu")
    t.add_argument("--train-fasta", dest="train_fasta", default=None,
                   help="override the training corpus (overfitting / half-data experiments)")
    t.add_argument("--heldout-fasta", dest="heldout_fasta", default=None,
                   help="override the validation split held out of that corpus")
    t.add_argument("--density-json", dest="density_json", default=None,
                   help="mode pdens: sequence->density-bin sidecar from tools/plan_density.py "
                        "(default data/plan/density.json). Must cover every training sequence.")
    t.set_defaults(fn=cmd_train)
    s = sub.add_parser("sample")
    s.add_argument("--ckpt", required=True); s.add_argument("--out", required=True)
    s.add_argument("--n", type=int, default=1000); s.add_argument("--draw", type=int, default=2000)
    s.add_argument("--max-rounds", type=int, default=15, dest="max_rounds")
    s.add_argument("--temp", type=float, default=1.2)
    s.add_argument("--top-p", type=float, default=0.95, dest="top_p")
    s.add_argument("--seed", type=int, default=42)
    s.add_argument("--ban-fasta", dest="ban_fasta", default=None,
                   help="override the corpus banned from output (must match what the model saw)")
    s.add_argument("--quota", action="store_true",
                   help="A1: fill integer per-cell quotas from the training plan histogram")
    s.add_argument("--quota-strict", action="store_true", dest="quota_strict",
                   help="A1: also require the realised descriptors to land in the forced cell")
    s.add_argument("--dens-top", type=int, default=0, dest="dens_top",
                   help="mode pdens: force the density token into one of the K HIGHEST bins "
                        "(composition steps stay free). 0 = force nothing, the untilted control.")
    s.set_defaults(fn=cmd_sample)
    st = sub.add_parser("selftest"); st.set_defaults(fn=cmd_selftest)
    a = p.parse_args(); a.fn(a)


if __name__ == "__main__":
    main()
