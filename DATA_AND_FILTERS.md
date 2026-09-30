# Training data, external databases, and interventions

## Training data — provenance

Per the COL870 assignment handout (IIT Delhi, §1.2), `training.fasta` is **derived from the MarLys AMP
database — a CC0 (public domain) aggregation of 103,000 peptide sequences from thirteen public AMP
databases** — filtered to the 8–50 residue window, with a held-out slice reserved by the assignment
authors and not distributed.

> B. Marczak, M. Jaromin, A. Bocian, A. Łyskowski. *MarLys AMP: An integrated bioinformatics platform
> for antimicrobial peptide analysis using DIAMOND and Biopython tools with optimized database
> repeatability indices DAIRI & IDAIRI.* SSRN 6418316, 2026.

- **43,911 sequences**, each carrying a MarLys `MLAMP` accession. This is the sole training corpus; a
  validation split for early stopping was taken from it by our own code.
- **`antibacterial.fasta`** — a strict subset (39,448 of the 43,911; none outside) used **only** as a
  rejection list at sampling time and as the reference for the identity screen. It never contributes
  gradients. Its headers annotate each peptide with the upstream databases it occurs in: dbAMP,
  DRAMP, DBAASP, CAMP, SATPdb, APD, AMPDB, DADP, InverPep, CancerPPD, BaAMPs, CyBase.

**Licence.** The handout states the upstream MarLys aggregation is CC0 / public domain; we have not
independently verified that at the source and report it as the handout's claim. No proprietary or
non-public data is involved either way, so the Full Requirements' data-release clause imposes
nothing further.

The corpus is **referenced, not redistributed** in this repository.
`TRAINING_DATA_MANIFEST.md` gives SHA-256 checksums, the measured structure, and the reproduction
recipe (MarLys → filter to 8–50 → remove the held-out slice); `training_ids.txt` lists all 43,911
accessions, so the exact file can be reconstructed and verified. We can supply it directly to the
organisers on request.

No other corpus contributes gradients.

## External databases
- **None used for training** beyond the aggregated corpus described above.
- We do not hold standalone local copies of DBAASP, dbAMP or APD. The 80%-identity screen therefore
  runs against the corpus above — which, per its own annotations, is **drawn from DBAASP, dbAMP and
  APD among others**, i.e. the same databases the competition screens against. The organisers' screen
  remains authoritative, but this is a closely aligned reference set rather than a loose proxy.

## Pretrained models
- **None in generation.** The generator is trained from scratch; no protein language model, no
  pretrained embedding, and no external weights influence which sequences are produced.
- **In ranking only:** amPEPpy, a published descriptor-based activity oracle, ranks candidates for the
  top-100 list. It plays no part in producing the 50,000-sequence library.
- ESM-2 650M was used **offline, as a measurement instrument** during model selection, because the
  `seqme` metrics are computed in its embedding space. It is not part of the submitted generation or
  ranking pipeline.

## Computational filters applied
Applied at draw time, inside the sampler:
1. **alphabet** — the 20 standard proteinogenic amino acids only;
2. **length** — 8 to 50 residues, enforced in logit space;
3. **uniqueness** — exact duplicates rejected;
4. **novelty** — any sequence present in the training or known-antibacterial corpora rejected, so the
   library contains no exact match to any peptide in the aggregated public databases above.

Applied to the top-100 list only:
5. **synthesizability gate** — at most 2 cysteines and at most 35 residues, applied before ranking;
6. **ranking** by amPEPpy predicted activity;
7. **identity screen** — rejection of any candidate above 80% identity to the reference set, replaced
   by the next valid candidate.

## Manual intervention
**None.** No sequence was hand-picked, hand-edited, or removed by inspection. Every filter above is
deterministic code with a fixed seed, and the library is reproducible by re-running the entry point.
All peptides are linear with free termini; no amidation or other terminal modification is implied or
intended.
