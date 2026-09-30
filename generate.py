#!/usr/bin/env python
"""Entry point for AMP Challenge reproducibility verification.

Running this script with no arguments regenerates the submitted 50,000-sequence library:

    uv run generate.py

Every sampling parameter below is the one used for the submitted library, hardcoded as the
default, so two runs produce byte-identical output. Writes both `library_50k.fasta` and
`library_50k.csv` next to this script (override with --out-dir).
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "code"))

# The exact configuration the submitted library was drawn with.
SEED = 20260930
N = 50000
DRAW = 8000
MAX_ROUNDS = 80
TEMP = 1.2
TOP_P = 0.95
CKPT = ROOT / "code" / "factored_model.pt"

# Mirrors ar_bpe._data: prefer a corpus placed under data/, else the original development layout.
_FALLBACKS = {
    "training.fasta": ROOT / "amp-grader-demo" / "data" / "training" / "training.fasta",
    "antibacterial.fasta": ROOT / "packages" / "phase2_1_submission" / "data" / "antibacterial.fasta",
}


def _corpus(name: str) -> Path:
    bundled = ROOT / "data" / name
    if bundled.exists():
        return bundled
    fb = _FALLBACKS.get(name)
    return fb if fb is not None and fb.exists() else bundled


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", type=Path, default=ROOT,
                   help="where to write library_50k.fasta/.csv (default: repository root)")
    p.add_argument("--seed", type=int, default=SEED,
                   help=f"sampling seed (default: {SEED}, the submitted one)")
    p.add_argument("--n", type=int, default=N, help=f"sequences to generate (default: {N})")
    a = p.parse_args()

    a.out_dir.mkdir(parents=True, exist_ok=True)
    fasta = a.out_dir / "library_50k.fasta"
    csv_path = a.out_dir / "library_50k.csv"

    if not CKPT.exists():
        sys.exit(f"missing model weights: {CKPT}")

    # The sampler rejects any draw that already exists in the training or known-antibacterial
    # corpora. Those corpora are NOT redistributed in this repository (see
    # TRAINING_DATA_MANIFEST.md), and without them that novelty filter would silently pass
    # everything and the library would not reproduce. So refuse to run rather than emit a
    # library that looks right and is not.
    #
    # Resolved here rather than imported from ar_bpe, which pulls in torch: this way the
    # explanation appears even on a checkout where dependencies are not installed yet.
    missing = [p for p in (_corpus("training.fasta"), _corpus("antibacterial.fasta"))
               if not p.exists()]
    if missing:
        sys.exit(
            "Cannot reproduce the library: the novelty-filter corpora are missing:\n"
            + "".join(f"  {p}\n" for p in missing)
            + "\nThese are not redistributed here. Obtain them as described in\n"
            "TRAINING_DATA_MANIFEST.md (source, reproduction recipe and SHA-256 checksums),\n"
            "place them at the paths above, and re-run. Running without them would produce a\n"
            "library that does not match the submitted one."
        )

    import ar_domain

    argv = [
        "ar_domain.py", "sample",
        "--ckpt", str(CKPT),
        "--out", str(fasta),
        "--n", str(a.n),
        "--draw", str(DRAW),
        "--max-rounds", str(MAX_ROUNDS),
        "--temp", str(TEMP),
        "--top-p", str(TOP_P),
        "--seed", str(a.seed),
    ]
    old_argv, sys.argv = sys.argv, argv
    try:
        ar_domain.main()
    finally:
        sys.argv = old_argv

    # Mirror the FASTA into the submitted CSV shape (id,sequence with 1-based ids).
    from ar_bpe import read_fasta

    seqs = read_fasta(fasta)
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "sequence"])
        for i, s in enumerate(seqs, 1):
            w.writerow([i, s])
    print(f"[generate] wrote {fasta} and {csv_path} ({len(seqs)} sequences)")


if __name__ == "__main__":
    main()
