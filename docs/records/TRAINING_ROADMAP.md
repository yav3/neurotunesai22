# NT Training Roadmap

## What NT-001 proved

NT-001 ran the full 4-criterion governed protocol on GSE119262 (44 samples,
17183 genes, human, data-driven regime). Result: **NOT_MET** (1/4 criteria).

The cVAE (10.6M parameters, latent_dim=64) cannot beat a mean or linear-shift
baseline with 21 perturbation pairs, even subsetted to 500 genes (p/n = 24).
This is expected: a deep generative model needs orders of magnitude more
training samples than output dimensions.

**What worked:** the pre-registration framework, species firewall, regime
selection, hash-chained evidence records, and cross-repo deposit pipeline all
function correctly. The harness caught the failure honestly.

## The p/n ratio problem

| Setting | n (pairs) | p (genes) | p/n | Feasible? |
|---------|-----------|-----------|-----|-----------|
| NT-001 (all genes) | 21 | 17183 | 818 | No |
| NT-001 (500 genes) | 21 | 500 | 24 | No |
| NT-001 (50 genes) | 21 | 50 | 2.4 | Marginal |
| Pooled (all Merlin) | ~50 | ~29 | 0.6 | Yes, but tiny gene set |
| 10 in-regime series | ~200 | ~5000 | 25 | Marginal |
| 100 in-regime series | ~2000 | ~5000 | 2.5 | Yes |

## NT-002 plan: right-sized models for current data

Pre-register two arms against the same GSE119262 data:

1. **Ridge regression** (α cross-validated): n_params ≈ p, no latent space.
   Expected to beat mean baseline at p/n > 1. This is the honest "can we
   learn anything at all from this data?" baseline.

2. **Linear VAE** (latent_dim=8, no hidden layers): n_params ≈ 2×p×8 + 8×p.
   A VAE with capacity matched to the data scale.

Both run under the same 4-criterion protocol as NT-001 (mean baseline,
linear shift, permutation, and pearson r > 0.5).

## Data scaling plan

The series catalog identifies 745 GEO series with ≥30 samples across 9
biological programs. Downloading even 10% of these would give us ~200-500
perturbation pairs — enough for p/n < 5 with gene subsetting.

Priority datasets (from Merlin's existing programs):
- Autophagy: GSE119262 (44 samples, already committed), plus ARCHS4/recount3
  autophagy-tagged series
- Reprogramming: Large Yamanaka factor studies (GSE201710 has only 16 samples
  but larger ENCODE/Roadmap datasets exist)
- Drug response: LINCS L1000 (978 landmark genes, millions of perturbation
  profiles) — the canonical source for this exact problem

## Architecture scaling plan

| Data scale | Recommended model | p/n target |
|------------|-------------------|------------|
| <50 pairs | Ridge regression | <2 |
| 50-500 pairs | Linear VAE (latent=8-16) | <5 |
| 500-5000 pairs | cVAE (latent=32, hidden=128) | <5 |
| >5000 pairs | Full cVAE (latent=64+, hidden=512) | <5 |

The current PerturbationResponseModel is designed for the >5000 tier.
We need to build the smaller models first and demonstrate they work
before scaling up.

## NT-003: Multi-series pooling (2026-10-31)

Pooled GSE119262 + GSE107965 on 9 common genes (the intersection between
a full 17K-gene matrix and a 29-gene subset). 33 perturbation pairs total.

| Arm | MSE (val) | Pearson r (val) | Verdict |
|-----|-----------|-----------------|---------|
| Single-series (21 pairs) | 0.108 | 0.954 | NOT_MET (mean r=0.979) |
| **Pooled (33 pairs)** | **0.227** | **0.903** | **MET (4/4)** |

**Key finding:** pooling helps not by adding genes but by adding diversity.
Heterogeneous perturbation pairs from two autophagy studies increase
within-study variance, making the mean a worse predictor and giving the
VAE's conditional structure room to contribute.

**Data bottleneck confirmed:** the current subset CSVs have too few genes
(25-56) for meaningful pooling with the full-genome matrix. Need full
expression matrices for more series.

## NT-004: Scaled pooled trial with full matrices (2026-10-31)

Downloaded 4 full GEO expression matrices (722 new samples, 178 MB total):
GSE26927 (20,589 genes x 118 samples, neurodegeneration), GSE15222 (24,354 x 363,
neurodegeneration), GSE5281 (54,675 x 161, neurodegeneration), GSE30174 (54,675 x 80,
drug response). Combined with existing GSE119262 (17,183 x 44, autophagy).

**Cross-platform gene-ID mismatch:** the 5 series use 4 different ID namespaces
(HUGO gene symbols, Illumina probes, GenBank IDs, Affymetrix probes). Only
GSE5281 + GSE30174 share the Affymetrix HG-U133 Plus 2.0 platform (54,675
common probes). All other pairwise intersections: 0. Probe-to-gene-symbol
mapping requires platform annotation files (not available in this environment).

**Variance-based dimensionality reduction:** 54,675 -> 2,000 top-variance
probes. Reduced p/n from 452 to 16.5 (121 pairs). Also added gradient
clipping and logvar clamping to prevent NaN training collapse.

| Arm | Series | Pairs | Features | MSE (val) | r (val) | Verdict |
|-----|--------|-------|----------|-----------|---------|---------|
| A (single) | GSE5281 | 81 | 2000 | 0.772 | 0.894 | **MET (4/4)** |
| **B (pooled)** | GSE5281+GSE30174 | **121** | 2000 | **0.648** | **0.947** | **MET (4/4)** |

**Key findings:**
1. First trial where BOTH arms meet all 4 criteria (NT-003 had only pooled arm met)
2. Pooling improved r: 0.894 -> 0.947 (+5.3pp) and MSE: 0.772 -> 0.648 (-16%)
3. Variance-based feature selection is essential: 54K features with 121 pairs
   causes NaN collapse; 2000 features gives p/n=16.5, stable training
4. Cross-platform ID mismatch is the primary blocker for true multi-program
   pooling. Need either: (a) platform annotation files for probe-to-gene mapping,
   or (b) gene-symbol-native datasets (RNA-seq with HUGO symbols)

## Immediate next steps

1. ~~NT-002: Ridge + Linear VAE on GSE119262 (gene-subsetted)~~ DONE
2. ~~NT-003: Multi-series pooled training~~ DONE (limited by gene intersection)
3. ~~Download full expression matrices for in-regime GEO series~~ DONE (4 new series)
4. ~~NT-004: Re-run pooled trial with full matrices~~ DONE (both arms MET)
5. NT-005: Cross-platform pooling via probe-to-gene mapping (Affymetrix/Illumina
   annotation files from GEO platforms, or switch to RNA-seq gene-count matrices)
6. LINCS L1000 integration (978 genes, pre-computed perturbation signatures)
7. Scale to 500+ pairs: target p/n < 5 with 2000 features
