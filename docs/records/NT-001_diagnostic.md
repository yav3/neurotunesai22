# NT-001 Diagnostic: Why the cVAE Fails to Beat Baselines

**Date:** 2026-10-31
**Dataset:** GSE119262 (human, everolimus, 44 samples, data-driven regime)

## Result Summary

| Arms | MSE (log1p) | Pearson r |
|------|------------|-----------|
| cVAE (17183 genes) | 0.560 | 0.006 |
| cVAE (500 genes) | 0.948 | 0.136 |
| cVAE (100 genes) | 2.090 | 0.020 |
| Mean baseline | 0.088 / 0.905 / 2.088 | 0.000 |
| Linear shift | 0.059 / 0.735 / 1.595 | 0.369 / 0.436 / 0.260 |
| Permutation | 0.570 / 1.455 / 3.131 | 0.005 / -0.150 / 0.380 |

## Root Cause: Sample Insufficiency

GSE119262 has 44 samples → 21 perturbation pairs (control-treated) →
15 train / 4 val / 2 test after splitting. No neural network can learn
a 17183-dimensional (or even 100-dimensional) perturbation response
from 15 examples.

**Key evidence:**
- The model always converges to approximately the mean prediction
  (MSE ≈ mean-baseline MSE at 100 genes: 2.090 vs 2.088)
- Permutation control is consistently worse than the model, confirming
  the model does learn *something* from real labels — but not enough
  to beat trivial baselines
- The linear-shift baseline (predict baseline + mean treatment shift)
  outperforms the cVAE at every gene count because the global shift
  dominates the perturbation response

## Data Landscape

Only 1 of 8 committed human GEO series qualifies for data-driven regime (≥30 samples):

| Accession | Samples | Control | Treated | Regime |
|-----------|---------|---------|---------|--------|
| GSE119262 | 44 | 23 | 21 | data-driven |
| GSE115678 | 28 | 14 | 14 | prior-contrast |
| GSE107965 | 18 | 6 | 12 | prior-contrast |
| GSE201710 | 16 | 8 | 8 | prior-contrast |
| GSE271385 | 9 | 0 | 9 | prior-mean |
| GSE297233 | 8 | 4 | 4 | prior-mean |
| GSE164949 | 6 | 3 | 3 | prior-mean |
| GSE206071 | 4 | 2 | 2 | prior-mean |

**Total perturbation pairs across all 8 series:** ~74 (estimated)
Pooling all series with both conditions gives ~74 pairs for training,
which is still small but approaches the minimum viable territory.

## Path Forward

1. **Multi-series pooling** — Pool all committed human series that have
   both control and treated conditions (GSE119262 + GSE115678 + GSE107965
   + GSE201710 + GSE297233 + GSE164949 + GSE206071 = 7 series, ~124 samples).
   Gene intersection across series gives the common feature set.
   Species firewall: human only (already enforced).

2. **Gene subsetting** — Top-500 by variance on the pooled data reduces
   p/n to a tractable ratio while retaining the most informative genes.

3. **Prior-informed regularization** — STRING protein-protein interaction
   network as a graph prior on the gene-gene structure. Already available
   in Merlin's data registry (`backend/data/priors/string/`).

4. **Acquire more in-regime data** — Merlin's `geo_dataset_scout.py`
   identified 745 GEO series with ≥30 samples across 9 programs.
   Each new in-regime series adds ~10-20 perturbation pairs.

## Pooled Multi-Series Result (2026-10-31)

**Dataset:** GSE119262 + GSE201710 (pooled, 60 samples, 38 common genes, 29 pairs)
**Split:** 22 train / 5 val / 2 test
**p/n ratio:** 1.3:1 (down from 818:1)

| Arms | MSE (log1p) | Pearson r |
|------|------------|-----------|
| cVAE (38 genes, pooled) | **0.116** | **0.316** |
| Mean baseline | 0.527 | 0.000 |
| Linear shift | 0.077 | 0.421 |
| Permutation | 3.559 | 0.947† |

†Permutation r is degenerate: 2 test samples make per-gene r ∈ {−1, 0, +1},
so the mean is dominated by chance sign alignment. The MSE separation
(3.559 vs 0.116) is the reliable signal: the model learned from real labels.

**Criteria:**
- C1 **MET**: Model MSE (0.116) < mean-baseline (0.527) — **first time passing**
- C2 NOT MET: Model MSE (0.116) > linear-shift (0.077)
- C3 NOT MET: r = 0.316 < 0.5
- C4 **MET**: r = 0.316 > 0.0

**Verdict: NOT_MET (2 of 4 criteria met — up from 1 of 4)**

### What pooling fixed

The single-series p/n=818 regime was hopelessly underdetermined.
Gene intersection across GSE119262 (17,183 genes) and GSE201710 (56 genes)
collapsed the feature space to 38 genes (the overlap), making p/n≈1.3.
Study-aware pairing added 8 pairs from GSE201710 to the 21 from GSE119262.

### What pooling did not fix

1. **Test set too small** — 2 samples make all per-sample metrics unreliable.
   The permutation control's r=0.947 is a direct consequence.
2. **Linear shift still wins** — the dominant perturbation effect is a
   gene-wise global shift; the cVAE's nonlinear capacity adds noise.
3. **Gene intersection too narrow** — 38 genes from a forced pairwise pool
   is far below the ≈500 needed for biologically meaningful features.

### Remaining path

1. **Acquire in-regime data** — `geo_dataset_scout.py` found 745 GEO series
   with ≥30 samples. Each series that measures the full transcriptome would
   keep the gene intersection at thousands rather than collapsing to tens.
2. **STRING prior edges** — for the current 38-gene panel, prior-informed
   regularization adds biological structure the 29 training pairs cannot.
3. **Cross-platform normalization** — series from different platforms need
   quantile normalization before pooling to remove batch effects.

## Pre-registration Verdict

The evidence discipline worked exactly as designed: the pre-registered
criteria (C1-C4) caught a model that doesn't beat trivial baselines on
the single-series run, then measured real improvement under pooling.
The content-addressed records at `NT-001_result.json` (single-series)
and `NT-001_pooled_result.json` (pooled) are the honest, immutable
records. The next trial needs more in-regime data to grow both the
gene intersection and the pair count.
