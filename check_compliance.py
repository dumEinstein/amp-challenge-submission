#!/usr/bin/env python
"""Self-check of the submitted files against the competition's peptide design constraints.

    python check_compliance.py

Needs no dependencies beyond the standard library, so it runs before `uv sync`. Exits non-zero
if any check fails.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
AA = set("ACDEFGHIKLMNPQRSTVWY")
MIN_LEN, MAX_LEN = 8, 50
MAX_CYS, MAX_TOP_LEN = 2, 35  # the synthesizability gate, top-100 only


def read_fasta(path: Path) -> list[str]:
    seqs: list[str] = []
    cur: list[str] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if cur:
                seqs.append("".join(cur))
                cur = []
        else:
            cur.append(line)
    if cur:
        seqs.append("".join(cur))
    return seqs


def main() -> int:
    lib = read_fasta(ROOT / "library_50k.fasta")
    top = read_fasta(ROOT / "top100.fasta")

    checks: list[tuple[str, bool]] = [
        ("library has exactly 50,000 sequences", len(lib) == 50000),
        ("library sequences all unique", len(set(lib)) == 50000),
        ("library lengths within 8-50", all(MIN_LEN <= len(s) <= MAX_LEN for s in lib)),
        ("library uses only the 20 standard amino acids", all(set(s) <= AA for s in lib)),
        ("top-100 has exactly 100 sequences", len(top) == 100),
        ("top-100 sequences all unique", len(set(top)) == 100),
        ("top-100 is a subset of the library", set(top) <= set(lib)),
        ("top-100 lengths within 8-50", all(MIN_LEN <= len(s) <= MAX_LEN for s in top)),
        ("top-100 uses only the 20 standard amino acids", all(set(s) <= AA for s in top)),
        (f"top-100 all <= {MAX_CYS} cysteines", all(s.count("C") <= MAX_CYS for s in top)),
        (f"top-100 all <= {MAX_TOP_LEN} residues", all(len(s) <= MAX_TOP_LEN for s in top)),
    ]

    # The novelty checks need the reference corpora, which are not redistributed here (see
    # TRAINING_DATA_MANIFEST.md). Skip them rather than fail when the corpora are absent.
    train_f = ROOT / "data" / "training.fasta"
    antibac_f = ROOT / "data" / "antibacterial.fasta"
    if train_f.exists() and antibac_f.exists():
        known = set(read_fasta(train_f)) | set(read_fasta(antibac_f))
        checks += [
            ("library has no exact match in the known-AMP corpus", not set(lib) & known),
            ("top-100 has no exact match in the known-AMP corpus", not set(top) & known),
        ]
        novelty_note = None
    else:
        novelty_note = ("SKIP  novelty checks: reference corpora not present under data/ "
                        "(see TRAINING_DATA_MANIFEST.md)")

    for name, path in (("library_50k", "library_50k.csv"), ("top100", "top100.csv")):
        with open(ROOT / path, newline="") as fh:
            rows = [r["sequence"] for r in csv.DictReader(fh)]
        ref = lib if name == "library_50k" else top
        checks.append((f"{path} matches the FASTA in order and content", rows == ref))

    failed = 0
    for name, ok in checks:
        print(f"{'PASS' if ok else 'FAIL'}  {name}")
        failed += not ok
    if novelty_note:
        print(novelty_note)
    print(f"\n{len(checks) - failed}/{len(checks)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
