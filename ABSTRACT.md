# AMP Challenge submission — abstract

**Method.** A 4.76 M-parameter causal transformer trained from scratch on a
43,911-sequence corpus derived from the MarLys AMP database (CC0 public domain; 103,000 peptides
aggregated from thirteen public AMP databases), filtered to 8–50 residues, with a **factorised two-stage generative distribution**: each sequence is prefixed
in-stream with a three-token "plan" describing its length bin, cationic fraction (K/R/H) and
hydrophobic fraction (A/V/I/L/M/F/W/C), each quantised into 8 training quantiles. The model therefore
emits a composition plan and then arranges residues to satisfy it, so composition and arrangement can
move independently. No pretrained protein language model is used anywhere in generation.

**Architecture.** 6-layer pre-norm decoder, d_model 256, 8 heads, d_ff 1024, context 52, learned
positional embeddings, output head weight-tied to the token embedding. Vocabulary 47 = 20 residues +
3 special + 24 plan tokens. Next-token cross-entropy; AdamW, lr 3e-4, cosine schedule with warmup,
batch 256, early stopping on a held-out split.

**Generation.** Nucleus sampling, temperature 1.2, top-p 0.95, fixed seed. The 8–50 residue window is
enforced **in logit space** — EOS is masked before position 8 and forced at 50 — so the length
constraint is structurally unviolable rather than learned. Sampling refills across rounds until the
target count of unique, in-alphabet, novel sequences is reached; duplicates and any sequence appearing
in the training or known-antibacterial corpora are rejected at draw time.

**Why this model.** It was selected by a cohort-independent aggregate (FKEA normalised against a fixed
scale rather than the scoring cohort, so libraries scored months apart remain comparable) computed
under ESM-2 650M across 34 arms, including discrete diffusion (D3PM with PAD as the absorbing state,
a complete 2x2x2 factorial over width/steps/schedule plus four structured transition matrices),
non-autoregressive and AR-decoder VAEs, continuous latent diffusion, adversarial training, and AR x VAE
library blending. The factorised transformer and its variants occupied the top of that ranking. Every
reported difference is averaged over >= 3 sampling seeds, because the Frechet metric has a
within-checkpoint draw standard deviation of 0.004-0.068 depending on the generator, and four earlier
single-draw findings in this project did not survive replication.

**Top-100 ranking.** Candidates are ranked by amPEPpy predicted antimicrobial activity, a per-sequence
descriptor-based oracle. The ranked shortlist is then screened so that no submitted candidate exceeds
80% sequence identity to any peptide in our local known-AMP reference set; candidates over the
threshold are dropped and replaced by the next valid candidate, as the competition specifies.

**Reproducibility.** Generation is a single command with a fixed default seed; re-running reproduces
the library exactly. Model weights, the training script, the sampling script and the ranking script
are all in the repository.
