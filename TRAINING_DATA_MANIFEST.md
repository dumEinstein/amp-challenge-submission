# Training data — provenance and reproduction

## Source (from the COL870 assignment handout, IIT Delhi, §1.2)

> "Teams are provided with a training dataset derived from the **MarLys AMP database [1], a CC0
> (public domain) aggregation of 103,000 peptide sequences from thirteen public AMP databases**. We
> filtered MarLys to the assignment's window length (8-50 residues - the whole corpus is already
> restricted to the 20 standard amino acids) and reserved a held-out slice."

**[1]** B. Marczak, M. Jaromin, A. Bocian, A. Lyskowski. *MarLys AMP: An integrated bioinformatics
platform for antimicrobial peptide analysis using DIAMOND and Biopython tools with optimized database
repeatability indices DAIRI & IDAIRI.* SSRN 6418316, 2026.

**Licence: CC0 / public domain.** There is no restriction on redistribution, so `training.fasta` is
included in this package rather than merely referenced.

The `MLAMP` identifiers carried by every sequence are MarLys AMP accessions; they run to
`MLAMP0103200`, consistent with the 103,000-sequence upstream corpus.

## Reproduction recipe

1. Obtain the MarLys AMP database (CC0), 103,000 peptides aggregated from thirteen public AMP
   databases.
2. Restrict to sequences of length 8-50 residues. (The corpus is already limited to the 20 standard
   proteinogenic amino acids.)
3. Remove the held-out slice reserved by the assignment authors, which is not distributed to teams.

Step 3 is the one part we cannot reproduce independently: the held-out split was made by the
assignment authors and we received only its complement. The checksums below pin the exact file we
trained on, so the result of the recipe can be verified against it.

## Checksums (SHA-256)

```
training.fasta       81b7afe25ae975eb52598a4356b6b7bc9761a60140649ef9d92a1e9b678fa20e   43911 records
antibacterial.fasta  cbbeac64ba95746d87961e8ad9dd0849ae8058d15a300b2e7f6990730ca521e9   39448 records
```

## Structure, as measured

- `training.fasta`: 43,911 records, all distinct by identifier and by sequence. This is the full
  training corpus; a validation split for early stopping was taken from it by our code, not supplied.
- `antibacterial.fasta` is a **strict subset**: all 39,448 of its sequences occur in `training.fasta`
  and none outside it, so the union is exactly 43,911.
- 39,448 of 43,911 (89.8%) training records therefore also carry the `dbs=` annotations present in
  `antibacterial.fasta` headers, which name the upstream databases per peptide. The remaining 4,463
  (10.2%) carry no annotation in the distributed files; per the handout all 43,911 come from the same
  CC0 MarLys aggregation, so the missing annotations reflect what the assignment shipped, not a
  different data source.

## Databases named in the annotations

| database | annotated sequences |
|---|---|
| dbAMP | 20,933 |
| DRAMP | 18,395 |
| DBAASP | 14,496 |
| CAMP | 11,898 |
| SATPdb | 9,740 |
| APD | 2,041 |
| AMPDB | 1,270 |
| DADP | 601 |
| InverPep | 424 |
| CancerPPD | 396 |
| BaAMPs | 182 |
| CyBase | 169 |

The handout states thirteen upstream databases; twelve appear in the annotations of the subset we
received.

## How we obtained the file

Bundled inside `amp-grader-demo`, a runnable copy of the grading pipeline distributed for COL870 at
IIT Delhi (assignment document on Moodle). The assignment is explicitly built on Phase 1 of AMP
Challenge 2027 and states that "students are free to participate in the NeurIPS competition".
