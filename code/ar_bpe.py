"""BPE vocabulary-size sweep for the autoregressive peptide transformer.

Standalone on purpose: `submission/` is the shipped deliverable and its `data.py` freezes
VOCAB_SIZE at module scope, so a sweep cannot parameterise it without editing the artifact.
The transformer here is a faithful copy of `submission/src/amp_lm/model.py` (same blocks, same
init, same weight tying) with the vocabulary made a constructor argument, so the ONLY thing that
varies across arms is the tokenizer.

THREE THINGS THAT NEEDED CARE
-----------------------------
1. LENGTH CONTROL MOVES FROM TOKEN SPACE TO RESIDUE SPACE. The original bans EOS while
   `position < MIN_LENGTH` and forces EOS at `position >= MAX_LENGTH`, which is exact only
   because one token == one residue. With BPE a token carries 1..k residues, so this tracks a
   per-row residue count and masks any token whose residue length exceeds the remaining budget.
   Because every one of the 20 amino acids is kept as a single-residue token (`initial_alphabet`),
   the legal set is never empty while budget >= 1, so 8 <= len <= 50 is GUARANTEED, not filtered.
2. BLOCK SIZE IS HELD AT 52 FOR EVERY ARM. BPE shortens sequences, so a data-derived block size
   would shrink with vocab -- and then a sample that happened to draw many single-residue tokens
   would run off the end of the position embedding. Fixing it at MAX_LENGTH+2 keeps the
   architecture byte-identical across arms and keeps the worst case in range.
3. THE HELD-OUT SPLIT IS REUSED, NOT REDERIVED. `submission/checkpoint/heldout.fasta` (2195 seqs,
   a subset of training.fasta) is the reference set every prior 650M score in this project was
   measured against. Re-splitting would make FBD(AMPs) incomparable to the existing AR reference
   (absolute 0.6331), so those exact sequences are excluded from training here.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

AA = "ACDEFGHIKLMNPQRSTVWY"
MIN_LENGTH, MAX_LENGTH = 8, 50
PAD_ID, BOS_ID, EOS_ID, N_SPECIAL = 0, 1, 2, 3
BLOCK_SIZE = MAX_LENGTH + 2
SPECIALS = ["<pad>", "<bos>", "<eos>"]

ROOT = Path(__file__).resolve().parent.parent

# The corpora are CC0 (public domain) and are bundled in this repository under data/, so the
# sampler's novelty filter and the identity screen work from a clean checkout with no external
# downloads. The second path in each pair is the original development layout, kept as a fallback
# so the scripts still run inside the project tree they were written in.
def _data(name: str, *fallbacks: Path) -> Path:
    bundled = ROOT / "data" / name
    if bundled.exists():
        return bundled
    for fb in fallbacks:
        if fb.exists():
            return fb
    return bundled


TRAIN_FASTA = _data("training.fasta",
                    ROOT / "amp-grader-demo" / "data" / "training" / "training.fasta")
HELDOUT_FASTA = _data("heldout.fasta",
                      ROOT / "submission" / "checkpoint" / "heldout.fasta")
ANTIBAC_FASTA = _data("antibacterial.fasta",
                      ROOT / "packages" / "phase2_1_submission" / "data" / "antibacterial.fasta")


def read_fasta(path: Path) -> list[str]:
    seqs, cur = [], []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if cur:
                seqs.append("".join(cur))
                cur = []
        else:
            cur.append(line.upper())
    if cur:
        seqs.append("".join(cur))
    return seqs


def write_fasta(seqs: list[str], path: Path) -> None:
    with open(path, "w") as f:
        for i, s in enumerate(seqs, 1):
            f.write(f">seq{i}\n{s}\n")


def is_valid(s: str) -> bool:
    return MIN_LENGTH <= len(s) <= MAX_LENGTH and not (set(s) - set(AA))


# --------------------------------------------------------------------------- tokenizer


class Tok:
    """BPE over the residue alphabet. vocab_size counts the 3 specials."""

    def __init__(self, vocab: list[str], merges: list[tuple[str, str]] | None = None):
        self.itos = SPECIALS + vocab
        self.stoi = {t: i for i, t in enumerate(self.itos)}
        self.merges = merges or []
        self.rank = {tuple(m): i for i, m in enumerate(self.merges)}
        # residue length of every token id; specials contribute 0 residues
        self.tok_len = np.zeros(len(self.itos), dtype=np.int64)
        for i, t in enumerate(self.itos):
            self.tok_len[i] = 0 if i < N_SPECIAL else len(t)

    @property
    def size(self) -> int:
        return len(self.itos)

    def _bpe(self, seq: str) -> list[str]:
        if not self.rank:
            return list(seq)
        parts = list(seq)
        while len(parts) > 1:
            best, best_rank = None, None
            for i in range(len(parts) - 1):
                r = self.rank.get((parts[i], parts[i + 1]))
                if r is not None and (best_rank is None or r < best_rank):
                    best, best_rank = i, r
            if best is None:
                break
            parts[best : best + 2] = [parts[best] + parts[best + 1]]
        return parts

    def encode(self, seq: str) -> list[int]:
        return [BOS_ID] + [self.stoi[p] for p in self._bpe(seq)] + [EOS_ID]

    def decode(self, ids) -> str:
        return "".join(self.itos[i] for i in ids if i >= N_SPECIAL)

    def save(self, path: Path) -> None:
        path.write_text(json.dumps({"vocab": self.itos[N_SPECIAL:], "merges": self.merges}))

    @staticmethod
    def load(path: Path) -> "Tok":
        d = json.loads(Path(path).read_text())
        return Tok(d["vocab"], [tuple(m) for m in d["merges"]])

    @staticmethod
    def train(seqs: list[str], vocab_size: int) -> "Tok":
        """Standard BPE: greedily merge the most frequent adjacent pair. The 20 residues are
        always present, so `vocab_size == 23` yields exactly the character-level baseline."""
        target_merges = max(0, vocab_size - N_SPECIAL - len(AA))
        words: dict[tuple[str, ...], int] = {}
        for s in seqs:
            k = tuple(s)
            words[k] = words.get(k, 0) + 1
        vocab = list(AA)
        merges: list[tuple[str, str]] = []
        for _ in range(target_merges):
            pairs: dict[tuple[str, str], int] = {}
            for w, c in words.items():
                for i in range(len(w) - 1):
                    p = (w[i], w[i + 1])
                    pairs[p] = pairs.get(p, 0) + c
            if not pairs:
                break
            best = max(pairs.items(), key=lambda kv: (kv[1], kv[0]))[0]
            if pairs[best] < 2:
                break
            merged = best[0] + best[1]
            merges.append(best)
            vocab.append(merged)
            new_words: dict[tuple[str, ...], int] = {}
            for w, c in words.items():
                parts, i = [], 0
                while i < len(w):
                    if i < len(w) - 1 and (w[i], w[i + 1]) == best:
                        parts.append(merged)
                        i += 2
                    else:
                        parts.append(w[i])
                        i += 1
                k = tuple(parts)
                new_words[k] = new_words.get(k, 0) + c
            words = new_words
        return Tok(vocab, merges)


# --------------------------------------------------------------------------- model
# Faithful copy of submission/src/amp_lm/model.py with vocab_size parameterised.


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads, self.d_head = n_heads, d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = dropout
        self.resid_drop = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0, is_causal=True
        )
        return self.resid_drop(self.proj(y.transpose(1, 2).contiguous().view(B, T, C)))


class Block(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model), nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class PeptideLM(nn.Module):
    def __init__(self, vocab_size, d_model=256, n_layers=6, n_heads=8, d_ff=1024,
                 dropout=0.1, block_size=BLOCK_SIZE):
        super().__init__()
        self.vocab_size, self.block_size = vocab_size, block_size
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(block_size, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([Block(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight          # weight tying, as in the original
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx):
        pos = torch.arange(idx.shape[1], device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        for b in self.blocks:
            x = b(x)
        return self.head(self.ln_f(x))

    def loss(self, batch):
        logits = self(batch[:, :-1])
        return F.cross_entropy(
            logits.reshape(-1, self.vocab_size), batch[:, 1:].reshape(-1), ignore_index=PAD_ID
        )


# --------------------------------------------------------------------------- data


def build_tensor(seqs: list[str], tok: Tok) -> torch.Tensor:
    rows = [tok.encode(s) for s in seqs]
    width = max(len(r) for r in rows)
    out = torch.full((len(rows), width), PAD_ID, dtype=torch.long)
    for i, r in enumerate(rows):
        out[i, : len(r)] = torch.tensor(r, dtype=torch.long)
    return out


def lr_at(step, total, base, warmup):
    if step < warmup:
        return base * (step + 1) / warmup
    prog = (step - warmup) / max(1, total - warmup)
    return 0.05 * base + 0.95 * base * 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))


# --------------------------------------------------------------------------- sampling


@torch.no_grad()
def sample_batch(model, tok, n, gen, device, temperature, top_p):
    tok_len = torch.from_numpy(tok.tok_len).to(device)
    tokens = torch.full((n, 1), BOS_ID, dtype=torch.long, device=device)
    res = torch.zeros(n, dtype=torch.long, device=device)
    done = torch.zeros(n, dtype=torch.bool, device=device)
    for _ in range(MAX_LENGTH + 1):
        logits = model(tokens)[:, -1, :] / temperature
        logits[:, PAD_ID] = float("-inf")
        logits[:, BOS_ID] = float("-inf")
        remaining = (MAX_LENGTH - res).clamp(min=0)
        # a token is illegal if its residue length exceeds the remaining budget
        too_long = tok_len.unsqueeze(0) > remaining.unsqueeze(1)
        too_long[:, :N_SPECIAL] = False
        logits = logits.masked_fill(too_long, float("-inf"))
        logits[res < MIN_LENGTH, EOS_ID] = float("-inf")
        probs = torch.softmax(logits, dim=-1)
        if top_p < 1.0:
            ordered, order = torch.sort(probs, dim=-1, descending=True)
            drop = ordered.cumsum(dim=-1) - ordered > top_p
            ordered[drop] = 0.0
            probs = torch.zeros_like(probs).scatter_(1, order, ordered)
            probs = probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-12)
        nxt = torch.multinomial(probs, 1, generator=gen)
        nxt[done] = PAD_ID
        res = res + torch.where(done, torch.zeros_like(res), tok_len[nxt.squeeze(1)])
        tokens = torch.cat([tokens, nxt], dim=1)
        done |= nxt.squeeze(1) == EOS_ID
        if bool(done.all()):
            break
    return [tok.decode(r.tolist()) for r in tokens[:, 1:].cpu()]


# --------------------------------------------------------------------------- commands


def cmd_train(a):
    if not torch.cuda.is_available() and not a.allow_cpu:
        raise SystemExit(
            f"REFUSING TO TRAIN ON CPU: torch reports cuda unavailable (torch {torch.__version__}). "
            "Use ~/.venvs/perceptgrid-qwen3/bin/python, or pass --allow-cpu.")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)

    allseq = [s for s in read_fasta(TRAIN_FASTA) if is_valid(s)]
    held = set(read_fasta(HELDOUT_FASTA))
    train_seqs = [s for s in allseq if s not in held]
    val_seqs = [s for s in allseq if s in held]
    print(f"[data] {len(allseq)} valid | {len(train_seqs)} train | {len(val_seqs)} val "
          f"(reusing submission heldout split)", flush=True)

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    tp = out / "tokenizer.json"
    if tp.exists():
        tok = Tok.load(tp)
    else:
        t0 = time.time()
        tok = Tok.train(train_seqs, a.vocab_size)
        tok.save(tp)
        print(f"[bpe] target {a.vocab_size} -> actual {tok.size} "
              f"({len(tok.merges)} merges, {time.time()-t0:.0f}s)", flush=True)
    enc = [tok.encode(s) for s in train_seqs[:2000]]
    print(f"[bpe] vocab {tok.size} | mean tokens/seq {np.mean([len(e)-2 for e in enc]):.2f} "
          f"| max {max(len(e)-2 for e in enc)} | compression "
          f"{np.mean([len(s) for s in train_seqs[:2000]])/np.mean([len(e)-2 for e in enc]):.3f}x",
          flush=True)

    tr = build_tensor(train_seqs, tok); va = build_tensor(val_seqs, tok)
    model = PeptideLM(tok.size, a.d_model, a.n_layers, a.n_heads, a.d_ff, a.dropout).to(dev)
    print(f"[model] vocab {tok.size} | {sum(p.numel() for p in model.parameters())/1e6:.2f}M params "
          f"on {dev} ({torch.cuda.get_device_name(0) if dev=='cuda' else 'cpu'})", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay, betas=(0.9, 0.98))
    spe = math.ceil(len(tr) / a.batch)
    total = spe * a.epochs
    best, bad, step, hist = float("inf"), 0, 0, []
    t0 = time.time()
    for ep in range(a.epochs):
        model.train()
        perm = torch.randperm(len(tr), generator=torch.Generator().manual_seed(a.seed + ep))
        run, nb = 0.0, 0
        for i in range(0, len(tr), a.batch):
            batch = tr[perm[i : i + a.batch]].to(dev)
            for g in opt.param_groups:
                g["lr"] = lr_at(step, total, a.lr, a.warmup)
            loss = model.loss(batch)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            run += loss.item(); nb += 1; step += 1
        model.eval()
        with torch.no_grad():
            vl = float(np.mean([model.loss(va[i:i+a.batch].to(dev)).item()
                                for i in range(0, len(va), a.batch)]))
        # per-TOKEN val loss is not comparable across vocab sizes (different sequence lengths);
        # the per-RESIDUE bound is. bits/residue = val_loss * tokens_per_seq / residues_per_seq
        tps = np.mean([len(e) - 1 for e in [tok.encode(s) for s in val_seqs[:500]]])
        rps = np.mean([len(s) for s in val_seqs[:500]])
        bpr = vl * tps / rps / math.log(2)
        mark = ""
        if vl < best - 1e-5:
            best, bad = vl, 0
            torch.save({"model": model.state_dict(), "vocab_size": tok.size,
                        "args": vars(a), "epoch": ep, "val": vl, "bits_per_residue": bpr},
                       out / "model.pt")
            mark = " *"
        else:
            bad += 1
        hist.append({"epoch": ep, "train": run / nb, "val": vl, "bits_per_residue": bpr})
        print(f"[ep {ep:3d}] train {run/nb:.4f} val {vl:.4f} bits/res {bpr:.4f} "
              f"lr {opt.param_groups[0]['lr']:.2e} {time.time()-t0:.0f}s{mark}", flush=True)
        if bad >= a.patience:
            print(f"[stop] no val improvement for {a.patience} epochs", flush=True)
            break
    json.dump({"history": hist, "best_val": best, "vocab_size": tok.size},
              open(out / "history.json", "w"), indent=1)
    print(f"[done] best val {best:.4f} -> {out/'model.pt'}", flush=True)


def cmd_sample(a):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(a.ckpt, map_location=dev, weights_only=False)
    tok = Tok.load(Path(a.ckpt).parent / "tokenizer.json")
    ta = ck["args"]
    model = PeptideLM(ck["vocab_size"], ta["d_model"], ta["n_layers"], ta["n_heads"],
                      ta["d_ff"], 0.0).to(dev)
    model.load_state_dict(ck["model"]); model.eval()
    print(f"[sample] vocab {ck['vocab_size']} val {ck['val']:.4f} epoch {ck['epoch']} "
          f"temp {a.temp} top_p {a.top_p}", flush=True)

    banned = set(read_fasta(ANTIBAC_FASTA)) if ANTIBAC_FASTA.exists() else set()
    if TRAIN_FASTA.exists():
        banned |= set(read_fasta(TRAIN_FASTA))
    print(f"[sample] {len(banned)} banned sequences", flush=True)
    gen = torch.Generator(device=dev); gen.manual_seed(a.seed)
    keep, seen, drawn = [], set(), 0
    for _ in range(a.max_rounds):
        for s in sample_batch(model, tok, a.draw, gen, dev, a.temp, a.top_p):
            drawn += 1
            if is_valid(s) and s not in banned and s not in seen:
                seen.add(s); keep.append(s)
        if len(keep) >= a.n:
            break
    keep = keep[: a.n]
    write_fasta(keep, Path(a.out))
    L = [len(s) for s in keep]
    print(f"[sample] wrote {len(keep)} to {a.out} | mean len {np.mean(L):.1f} "
          f"min {min(L)} max {max(L)} | yield {len(keep)/max(1,drawn):.3f}", flush=True)


def cmd_selftest(a):
    ok = 0

    def chk(name, cond):
        nonlocal ok
        assert cond, f"FAIL {name}"
        ok += 1

    seqs = [s for s in read_fasta(TRAIN_FASTA) if is_valid(s)]
    chk("training data loaded", len(seqs) > 40000)
    sub = seqs[:4000]
    for V in (23, 50, 100, 200):
        tok = Tok.train(sub, V)
        chk(f"V={V} size exact", tok.size == V)
        chk(f"V={V} specials at 0,1,2", tok.itos[:3] == SPECIALS)
        chk(f"V={V} all 20 residues single-token",
            all(aa in tok.stoi and tok.tok_len[tok.stoi[aa]] == 1 for aa in AA))
        chk(f"V={V} merges count", len(tok.merges) == max(0, V - N_SPECIAL - len(AA)))
        # round-trip on held-out-of-tokenizer sequences
        bad = [s for s in seqs[4000:6000] if tok.decode(tok.encode(s)) != s]
        chk(f"V={V} round-trip exact on 2000 unseen", not bad)
        chk(f"V={V} tok_len matches token strings",
            all(tok.tok_len[i] == len(t) for i, t in enumerate(tok.itos) if i >= N_SPECIAL))
        chk(f"V={V} encode wraps with BOS/EOS",
            tok.encode(sub[0])[0] == BOS_ID and tok.encode(sub[0])[-1] == EOS_ID)
        enc = [len(tok.encode(s)) - 2 for s in sub]
        chk(f"V={V} encoded fits block size", max(enc) <= MAX_LENGTH)
        if V == 23:
            chk("V=23 is exactly character-level",
                all(len(tok.encode(s)) - 2 == len(s) for s in sub[:500]))
        else:
            chk(f"V={V} compresses vs char level", np.mean(enc) < np.mean([len(s) for s in sub]))
    # sampling invariants with an untrained model (cheap, cpu)
    tok = Tok.train(sub, 100)
    m = PeptideLM(tok.size, d_model=32, n_layers=1, n_heads=2, d_ff=64, dropout=0.0)
    m.eval()
    g = torch.Generator(); g.manual_seed(0)
    out = sample_batch(m, tok, 64, g, "cpu", 1.0, 0.95)
    chk("sampler returns requested count", len(out) == 64)
    chk("EVERY sample is length-valid (guarantee, not filter)",
        all(MIN_LENGTH <= len(s) <= MAX_LENGTH for s in out))
    chk("every sample is alphabet-valid", all(not (set(s) - set(AA)) for s in out))
    g2 = torch.Generator(); g2.manual_seed(0)
    chk("sampling is seed-deterministic", sample_batch(m, tok, 64, g2, "cpu", 1.0, 0.95) == out)
    # tensor build
    t = build_tensor(sub[:100], tok)
    chk("batch is padded rectangular", t.shape[0] == 100 and t.shape[1] <= BLOCK_SIZE)
    chk("pad only at the right", all((row == PAD_ID).nonzero().numel() == 0 or
        bool((row[(row == PAD_ID).nonzero()[0].item():] == PAD_ID).all()) for row in t))
    print(f"[selftest] {ok} checks passed")
    print("SELFTEST PASSED")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train")
    t.add_argument("--vocab-size", type=int, required=True, dest="vocab_size")
    t.add_argument("--out", required=True)
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
    t.set_defaults(fn=cmd_train)

    s = sub.add_parser("sample")
    s.add_argument("--ckpt", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--n", type=int, default=1000)
    s.add_argument("--draw", type=int, default=2000)
    s.add_argument("--max-rounds", type=int, default=15, dest="max_rounds")
    s.add_argument("--temp", type=float, default=1.2)
    s.add_argument("--top-p", type=float, default=0.95, dest="top_p")
    s.add_argument("--seed", type=int, default=42)
    s.set_defaults(fn=cmd_sample)

    st = sub.add_parser("selftest"); st.set_defaults(fn=cmd_selftest)
    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
