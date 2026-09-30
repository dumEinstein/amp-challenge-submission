# Selection and ranking procedure for the top-100 list

This documents exactly how the 100 submitted candidates were chosen from the 50,000-sequence library.
Every step is deterministic code with a fixed seed; no sequence was hand-picked or hand-edited.

**Submitted list:** `top100.fasta` / `top100.csv` (synthesizability-gated).
**Also included for comparison:** `top100_unconstrained.fasta` / `.csv` (activity only, no gate).

## Step 0 — the library (no selection)

The 50,000-member library is the raw output of the generator. Nothing is ranked, filtered or reordered
to produce it beyond four hard constraints applied **inside the sampler at draw time**:

1. **Alphabet** — the 20 standard proteinogenic amino acids only.
2. **Length** — 8 to 50 residues, enforced in *logit space*: the EOS token is masked before position 8
   and forced at position 50, so the window is structurally unviolable rather than learned.
3. **Uniqueness** — exact duplicates rejected as they are drawn.
4. **Novelty** — any sequence appearing in the training corpus or the known-antibacterial
   corpus (both aggregated from public AMP databases) is rejected as it is drawn.

```
uv run generate.py
```

which is equivalent to the underlying sampler call:

```
python code/ar_domain.py sample --ckpt code/factored_model.pt \
    --out library_50k.fasta \
    --n 50000 --draw 8000 --max-rounds 80 --temp 1.2 --top-p 0.95 --seed 20260930
```

The seed is fixed and the sampler deterministic given it, so re-running reproduces the library.

**The library matches the real corpus closely on the properties we did not optimise:**

| | mean length | mean Cys | ≥4 Cys | >35 residues |
|---|---|---|---|---|
| library (50,000) | 20.0 | 0.86 | 8.5% | 9.3% |
| real training AMPs (43,911) | 19.9 | 0.90 | 9.2% | 9.3% |

## Step 1 — synthesizability gate

Candidates are restricted to **at most 2 cysteines** and **at most 35 residues** *before* ranking, so
the shortlist is drawn from feasible peptides rather than filtered afterwards. 41,322 of 50,000
(82.6%) are feasible.

**Why.** The aggregation score scores synthesizability explicitly, 25 of the 100 go to the wet lab at
random so the whole list must be viable, and the activity oracle has a strong bias that works against
both. Ranking by amPEPpy alone returns long, cysteine-rich, defensin-like sequences:

| | mean length | mean Cys | ≥4 Cys | >35 residues |
|---|---|---|---|---|
| top-100, activity only | 30.7 | 2.91 | 33.0% | 39.0% |
| top-100, gated (submitted) | 22.3 | 0.93 | 0.0% | 0.0% |
| real training AMPs | 19.9 | 0.90 | 9.2% | 9.3% |

The unconstrained list runs ~3.2x the real mean cysteine count and ~3.6x the real rate of ≥4
cysteines. Multiple disulfides raise synthesis cost, invite scrambling, and must fold correctly for
the peptide to be active at all.

*Correction of record:* an earlier draft of this file justified the gate with "no real training AMP
has ≥4 cysteines, mean 0.46". That was computed on a truncated `[:5000]` slice of an evidently ordered
corpus and is wrong; the full corpus reads mean 0.90 with 9.2% above 4. The gate is still justified,
by a smaller margin than first stated.

The gate slightly *over*-corrects on one axis — the submitted list has 0% at ≥4 cysteines where real
AMPs have 9.2% — which is a deliberate trade of a little realism for synthesis viability.

## Step 2 — rank by predicted activity

Every feasible sequence is scored by **amPEPpy**, a published descriptor-based antimicrobial activity
oracle, via `seqme`'s `ThirdPartyModel` entry point. The score is per-sequence, so the ranking is
exact and order-independent; there is no set-level objective and no cohort effect. Over all 50,000 the
distribution is mean 0.5729, sd 0.2009.

amPEPpy is used **only here**. It plays no part in generating the library, so the 50,000-sequence
submission is not conditioned on it in any way.

## Step 3 — the 80% identity screen

Walking down the ranking, a candidate is accepted only if its identity to every peptide in the
known-AMP reference set is at most **80%**. A candidate above the threshold is dropped and replaced by
the next one down, which is the substitution rule the competition specifies. 52 candidates were
rejected this way; the submitted list has maximum identity 0.800 and mean 0.639.

**Identity definition.** Longest common contiguous subsequence between candidate and reference,
divided by the length of the shorter. This is *ungapped* and therefore conservative: it may reject
candidates a gapped aligner would accept, but will not accept one a gapped aligner would reject at the
same threshold. For tractability against ~44k references, each candidate is compared only to
references sharing at least one 4-mer and within a factor of two in length — pairs outside that band
cannot reach 80% identity under this definition.

**Reference set.** We hold no standalone copy of DBAASP, dbAMP or APD. The screen runs
against the corpus we do have: the training set unioned with the known-antibacterial set. Per its own
FASTA annotations that corpus is **drawn from DBAASP, dbAMP and APD among others** — the same
databases the competition screens against — so this is a closely aligned reference set rather than a
loose proxy. The organisers' screen remains authoritative and may still reject candidates that pass
ours. Each candidate's measured identity is in `top100.csv` so the margin is
visible per row rather than merely asserted.

```
python code/kaggle_rank.py --library library_50k.fasta \
    --known data/training.fasta data/antibacterial.fasta \
    --out-top top100.fasta --out-csv top100.csv \
    --thirdparty-dir <seqme-thirdparty-dir> --jobs 8 \
    --max-cys 2 --max-length 35 --shortlist 800
```

## What the gate costs, and what the ranker choice costs

The gate costs almost nothing in predicted activity: **0.9910 → 0.9860 mean**, a 0.5% relative drop,
in exchange for a profile far closer to the real corpus. That asymmetry is why it is applied.

Separately, the choice of ranker determines the entire wet-lab list. A from-scratch alternative —
density rank under a 2-D kernel density estimate over (hydrophobic moment, net charge) fitted on the
training corpus — selects an almost disjoint set: its top-100 **overlaps the activity top-100 by 1 of
100**, preferring short peptides (mean 16.5) where amPEPpy prefers long ones. We submit the activity
ranking because the stated objective for this list is potency and amPEPpy is a published potency
oracle, whereas the density ranker measures only typicality in two descriptors. The orthogonality is
recorded because it means our list reflects one declared notion of "best", not a consensus.

Finally, note that the reported 0.9860 is the mean of the top 0.2% of scores under the same oracle
family that scores the challenge. It is a selection statistic, not independent evidence of potency;
the MIC and HC50 measurements are the real test and nothing here predicts them.

## Manual intervention

None. Every filter, the gate, the ranking and the identity screen are deterministic code. No candidate
was added, removed or reordered by inspection at any point.
