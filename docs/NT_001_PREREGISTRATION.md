# NT-001: Perturbation-Response Foundation Model Pre-Registration

**Date**: 2026-10-31
**Status**: pre-registered (not yet run)
**Repo**: yav3/NeuroTunesAI

## Question

Can a conditional VAE trained on in-regime GEO perturbation data
(≥30 samples per series, human only) predict held-out post-perturbation
gene expression profiles better than a mean-baseline and a linear-shift
baseline?

## Data

- **Source**: GEO DataSets, in-regime series (≥30 independent samples)
  from Merlin's `geo_dataset_scout` (782 total across human and mouse)
- **Species**: human only (species firewall: declare or refuse, never
  default — per `merlinaineuralplus2/backend/methods/species.py`)
- **Regime**: data-driven (≥30 samples) per Merlin's `method_router.py`;
  series below this threshold are excluded, not down-weighted
- **Split**: 80% train / 10% validation / 10% held-out test,
  `seed=2026`, stratified by program
- **Programs**: autophagy (30 series), reprogramming (118),
  senescence (235), t_cell_exhaustion (134), plasma_rejuvenation (20,
  human subset only), microglia_apoe4 (65), hsc_aging (24, human subset
  only), erythropoiesis (102), aging_time_course (27)

### Data curation steps (to be done before training)

1. For each candidate series, read the GEO design to confirm ≥30
   **independent biological** samples (not technical replicates, single
   cells, or plate wells).
2. Confirm species is human from `!Sample_organism_ch1`, not from gene
   symbol casing (upper-case symbols do not decide — species.py rule).
3. Confirm a two-condition (perturbation vs control) design exists, or
   a time-course with ≥2 timepoints.
4. Record the curated series list with accession, sample count, design,
   and species confirmation method in
   `docs/records/NT-001_curated_series.json` before training begins.

## Model

- **Architecture**: Conditional VAE (`PerturbationResponseModel`)
  - Expression encoder: `[gene_dim → 512 → 256 → 2×latent_dim]` (μ, log σ²)
  - Perturbation encoder: `[perturbation_dim → 128 → 64]`
  - Decoder: `[latent_dim + 64 → 256 → 512 → gene_dim]`
- **Latent dim**: 128
- **Perturbation dim**: 64 (one-hot perturbation type + continuous dose)
- **Training**: AdamW, lr=1e-4, weight_decay=0.01, cosine warmup
  1000 steps, KL weight β=1.0 (standard VAE, no annealing)
- **Epochs**: early-stopped on validation MSE, patience=10
- **Batch size**: 64

## Criteria (declared before run)

All criteria are evaluated on the held-out test split only. The model
arm and each baseline arm use the same test samples.

| # | Metric | Arm | Op | Threshold/Versus | Description |
|---|--------|-----|----|------------------|-------------|
| C1 | mse | model | < | mean_baseline | Model MSE < mean-baseline MSE on held-out test |
| C2 | mse | model | < | linear_shift | Model MSE < linear-shift baseline MSE on held-out test |
| C3 | pearson_r | model | > | 0.5 | Mean per-gene Pearson r > 0.5 on held-out test |
| C4 | pearson_r | model | > | mean_baseline | Model Pearson r > mean-baseline Pearson r on held-out test |

A criterion that is not met is a finding, recorded as-is. The criteria
are never tuned to the result.

## Baselines

- **mean_baseline**: for each condition in the test set, predict
  `target = mean(training-set expression for that condition)`. This is
  the strongest trivial baseline: it captures condition-level effects
  without learning gene-gene structure.
- **linear_shift**: predict `target = control_expression +
  mean(target − control)` from the training set. This captures the
  average perturbation shift without learning nonlinear interactions.

## Controls

- **Permutation control**: shuffle perturbation labels (which condition
  each sample belongs to), retrain on the shuffled data with the same
  hyperparameters and seed, measure the same four metrics. This
  establishes the null distribution — the performance a model achieves
  when the perturbation–expression relationship is broken.
- **Same-run requirement**: the model, both baselines, and the
  permutation control are all evaluated in the same script invocation
  on the same held-out split. No cherry-picking of runs.

## Record format

Content-addressed JSON following Merlin's `app/inventions/harness.py`
pattern:

```
record_hash = SHA-256(experiment_id || json(parameters) || json(per_arm_metrics))
```

The record file is written to `docs/records/NT-001_result.json` with
fields: `experiment_id`, `date`, `parameters`, `arms` (dict of arm →
metric dict), `stats` (permutation p-values), `criteria` (list of
{criterion, met, detail}), `record_hash`.

## What this is NOT

- This is not a foundation model in the LLM sense. It is a narrow
  conditional generative model trained on curated perturbation
  expression data, operating within the governed evidence layer that
  Merlin provides.
- This does not replace Merlin's GRN ranking pipeline. It is a
  complementary module: GRN ranks gene targets by network topology;
  this model predicts what happens to expression after a perturbation.
- No number from this pre-registration may appear on any page, deck, or
  partner document until the run completes and the record is hashed.

## Cross-repo dependencies

| Dependency | Repo | File |
|---|---|---|
| Species firewall | merlinaineuralplus2 | `backend/methods/species.py` |
| Regime selection | merlinaineuralplus2 | `backend/methods/method_router.py` |
| GEO series candidates | merlinaineuralplus2 | `docs/data/geo_dataset_scout.json` |
| Data registry convention | merlinaineuralplus2 | `backend/data/registry.json` |
| Evidence harness pattern | merlinaineuralplus2 | `backend/app/inventions/harness.py` |
| Study record convention | merlinaineuralplus2 | `docs/partnerships/records/*.json` |
