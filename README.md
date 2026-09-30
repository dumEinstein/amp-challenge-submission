# AMP Challenge submission

A 4.76M-parameter causal transformer trained from scratch, with an in-stream factorised plan prefix
(length / cationic fraction / hydrophobic fraction, 8 quantile bins each). No pretrained protein
language model is used in generation.

## Training data — CC0 public domain

The training corpus (43,911 sequences) is derived from the **MarLys AMP database**, a
**CC0 / public-domain** aggregation of 103,000 peptides from thirteen public AMP databases, filtered
to the 8–50 residue window with a held-out slice reserved by the assignment authors.

> B. Marczak, M. Jaromin, A. Bocian, A. Łyskowski. *MarLys AMP: An integrated bioinformatics platform
> for antimicrobial peptide analysis using DIAMOND and Biopython tools with optimized database
> repeatability indices DAIRI & IDAIRI.* SSRN 6418316, 2026.

The corpus is **not redistributed here**; it is referenced rather than copied.
`TRAINING_DATA_MANIFEST.md` gives the source, SHA-256 checksums, the measured structure and the
reproduction recipe (MarLys → filter to 8–50 residues → remove the held-out slice), and
`training_ids.txt` lists all 43,911 `MLAMP` accessions, so the exact corpus can be reconstructed
from the upstream database and verified byte-for-byte against the checksums. We can supply the
files directly to the organisers on request.

No proprietary or non-public data contributes to this submission, so the Full Requirements' clause
on releasing non-public data imposes nothing further.

Obtained via the `amp-grader-demo` bundle distributed for COL870 at IIT Delhi; that assignment is
built on Phase 1 of AMP Challenge 2027 and states that students are free to enter the competition.

| file | contents |
|---|---|
| `library_50k.fasta` / `.csv` | the 50,000-sequence library |
| `top100.fasta` / `.csv` | **the submitted ranked top 100** (synthesizability-gated) |
| `top100_unconstrained.fasta` / `.csv` | activity-only ranking, for comparison — not submitted |
| `TRAINING_DATA_MANIFEST.md`, `training_ids.txt` | source, checksums, reproduction recipe, accessions |
| `ABSTRACT.md` | method abstract |
| `DATA_AND_FILTERS.md` | data provenance, external databases, filters, manual intervention |
| `RANKING.md` | the selection and ranking procedure, with its limitations |
| `generate.py` | **entry point** — regenerates the library with the submitted settings |
| `code/ar_domain.py`, `code/ar_bpe.py` | model, training and sampling |
| `code/kaggle_rank.py` | ranking, synthesizability gate, 80%-identity screen |
| `code/factored_model.pt` | trained weights (4.76M parameters) |
| `data/` | *not in this repository* — where the referenced corpora must be placed to re-run |
| `pyproject.toml`, `LICENSE` | pinned environment; MIT licence |

## Reproduce the library

```
uv sync
uv run generate.py
```

`generate.py` takes no required arguments: the seed (20260930) and every sampling parameter are
hardcoded defaults matching the submitted run, so two runs produce identical output. It writes
`library_50k.fasta` and `library_50k.csv`.

The underlying sampler can also be called directly, which is exactly what `generate.py` does:

```
uv run python code/ar_domain.py sample --ckpt code/factored_model.pt \
    --out library_50k.fasta --n 50000 --draw 8000 --max-rounds 80 \
    --temp 1.2 --top-p 0.95 --seed 20260930
```

`pyproject.toml` pins torch 2.14.0 (CPU) and numpy 2.4.6, the versions the submitted library was
drawn under. Generation needs no GPU.

The sampler's novelty filter rejects any draw already present in the reference corpora, so it reads
`data/training.fasta` and `data/antibacterial.fasta`. Those files are not redistributed here, and
`generate.py` **refuses to run without them** rather than emit a library that would silently differ
from the submitted one. Obtain them per `TRAINING_DATA_MANIFEST.md` and place them under `data/`.

## Reproduce the top-100 ranking

```
uv run python code/kaggle_rank.py \
    --library library_50k.fasta \
    --known data/training.fasta data/antibacterial.fasta \
    --out-top top100.fasta --out-csv top100.csv \
    --thirdparty-dir <seqme-thirdparty-dir> --jobs 8 \
    --max-cys 2 --max-length 35 --shortlist 800
```

This step additionally requires `seqme` with the amPEPpy third-party oracle installed; it is not
part of the pinned dependency set, because the library itself is generated without it. See
`RANKING.md`.

## Verified constraints

Library: 50,000 sequences, all unique, 8–50 residues, 20 standard amino acids only, no exact match to
any peptide in the training corpus. Top 100: all unique, all a subset of the library, all ≤2 cysteines
and ≤35 residues, all ≤80% identity to the training corpus — which is drawn from DBAASP, dbAMP and APD
among others, so it is closely aligned with the organisers' reference database, though theirs remains
authoritative. All peptides are linear with free termini; no terminal modification is intended.
