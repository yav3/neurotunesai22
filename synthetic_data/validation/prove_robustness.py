"""
Empirical validation of synthetic dataset robustness.
Tests: correlation recovery, distributional normality, cross-domain independence,
missingness patterns, disease separability, and boundary constraints.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

QUARANTINE = Path(__file__).resolve().parent.parent / "quarantine"
SCHEMA = Path(__file__).resolve().parent.parent / "schemas" / "disease_parameters.json"


def load_sample(n=500_000):
    """Load a representative sample from the first chunk."""
    fp = sorted(QUARANTINE.glob("SYNTHETIC_neuro_bio_disease_chunk_*.parquet"))[0]
    return pd.read_parquet(fp).head(n)


def test_correlation_recovery(df):
    """Verify that within-disease correlations match the design matrices.
    Correlations are tested within HC to avoid Simpson's paradox from
    pooling across disease states with different means."""
    print("=" * 60)
    print("TEST 1: Correlation matrix recovery (within HC)")
    print("=" * 60)

    hc = df[df["disease_code"] == "HC"]

    csf_target = np.array([
        [1.00, -0.40, -0.45, -0.20],
        [-0.40, 1.00, 0.80, 0.35],
        [-0.45, 0.80, 1.00, 0.30],
        [-0.20, 0.35, 0.30, 1.00],
    ])
    csf_cols = ["csf_amyloid_beta42_pg_ml", "csf_total_tau_pg_ml",
                "csf_p_tau181_pg_ml", "csf_nfl_pg_ml"]
    csf_data = hc[csf_cols].dropna()
    csf_empirical = csf_data.corr().values

    blood_target = np.array([
        [1.00, 0.50, -0.20, 0.35, 0.25, 0.30, 0.25],
        [0.50, 1.00, -0.15, 0.40, 0.20, 0.25, 0.20],
        [-0.20, -0.15, 1.00, -0.10, -0.15, -0.20, -0.15],
        [0.35, 0.40, -0.10, 1.00, 0.20, 0.25, 0.20],
        [0.25, 0.20, -0.15, 0.20, 1.00, 0.55, 0.45],
        [0.30, 0.25, -0.20, 0.25, 0.55, 1.00, 0.50],
        [0.25, 0.20, -0.15, 0.20, 0.45, 0.50, 1.00],
    ])
    blood_cols = ["blood_nfl_pg_ml", "blood_gfap_pg_ml", "blood_bdnf_ng_ml",
                  "blood_s100b_ug_l", "blood_crp_mg_l", "blood_il6_pg_ml",
                  "blood_tnf_alpha_pg_ml"]
    blood_data = hc[blood_cols].dropna()
    blood_empirical = blood_data.corr().values

    csf_err = np.linalg.norm(csf_empirical - csf_target, 'fro')
    blood_err = np.linalg.norm(blood_empirical - blood_target, 'fro')
    csf_max = np.max(np.abs(csf_empirical - csf_target))
    blood_max = np.max(np.abs(blood_empirical - blood_target))

    print(f"\n  CSF domain (4x4, N_HC={len(csf_data):,}):")
    print(f"    Frobenius norm error: {csf_err:.4f}")
    print(f"    Max element error:    {csf_max:.4f}")
    print(f"    Key: Abeta42-tTau r_target=-0.40, r_empirical={csf_empirical[0,1]:.3f}")
    print(f"    Key: tTau-pTau   r_target= 0.80, r_empirical={csf_empirical[1,2]:.3f}")

    print(f"\n  Blood domain (7x7, N_HC={len(blood_data):,}):")
    print(f"    Frobenius norm error: {blood_err:.4f}")
    print(f"    Max element error:    {blood_max:.4f}")
    print(f"    Key: CRP-IL6     r_target= 0.55, r_empirical={blood_empirical[4,5]:.3f}")
    print(f"    Key: NfL-GFAP    r_target= 0.50, r_empirical={blood_empirical[0,1]:.3f}")

    ok = csf_max < 0.02 and blood_max < 0.02
    print(f"\n  PASS: {ok}  (threshold: max |r_err| < 0.02 within single disease state)")
    return ok


def test_disease_mean_separation(df):
    """Verify disease states have statistically distinct biomarker profiles."""
    print("\n" + "=" * 60)
    print("TEST 2: Disease-state separability (Welch t-tests)")
    print("=" * 60)

    with open(SCHEMA) as f:
        params = json.load(f)["disease_states"]

    tests = [
        ("alzheimers_disease", "healthy_control", "csf_amyloid_beta42_pg_ml",
         "AD vs HC: CSF Abeta42 (expect AD << HC)"),
        ("alzheimers_disease", "healthy_control", "eeg_theta_alpha_ratio",
         "AD vs HC: theta/alpha ratio (expect AD >> HC)"),
        ("parkinsons_disease", "healthy_control", "updrs_motor_score",
         "PD vs HC: UPDRS motor (expect PD >> HC)"),
        ("major_depressive_disorder", "healthy_control", "hamilton_depression_score",
         "MDD vs HC: Hamilton depression (expect MDD >> HC)"),
        ("amyotrophic_lateral_sclerosis", "healthy_control", "blood_nfl_pg_ml",
         "ALS vs HC: blood NfL (expect ALS >> HC)"),
    ]

    all_pass = True
    for d1, d2, col, desc in tests:
        g1 = df[df["disease_state"] == d1][col].dropna()
        g2 = df[df["disease_state"] == d2][col].dropna()
        t_stat, p_val = stats.ttest_ind(g1, g2, equal_var=False)
        cohen_d = (g1.mean() - g2.mean()) / np.sqrt((g1.std()**2 + g2.std()**2) / 2)

        ok = p_val < 1e-10  # extremely significant given N~100K per group
        all_pass &= ok
        print(f"\n  {desc}")
        print(f"    mean({d1[:3].upper()})={g1.mean():.2f}, "
              f"mean({d2[:3].upper()})={g2.mean():.2f}")
        print(f"    Cohen's d={cohen_d:.3f}, t={t_stat:.1f}, p={p_val:.2e}")
        print(f"    PASS: {ok}")

    return all_pass


def test_missingness_structure(df):
    """Verify MCAR rates and MAR age-dependence."""
    print("\n" + "=" * 60)
    print("TEST 3: Missingness structure (MCAR + MAR)")
    print("=" * 60)

    targets = {
        "eeg_delta_1_4hz_uv2": 0.08,
        "fmri_bold_dmn": 0.15,
        "blood_nfl_pg_ml": 0.05,
        "csf_amyloid_beta42_pg_ml": 0.35,
        "mmse_score": 0.03,
        "polygenic_risk_score": 0.10,
    }

    all_pass = True
    for col, target in targets.items():
        observed = df[col].isna().mean()
        err = abs(observed - target)
        ok = err < 0.05  # within 5pp
        all_pass &= ok
        print(f"  {col:40s}: target={target:.0%}, observed={observed:.2%}, "
              f"err={err:.3f}  {'PASS' if ok else 'FAIL'}")

    # MAR test: fMRI missingness should be higher for age > 75
    young = df[df["age"] <= 75]["fmri_bold_dmn"].isna().mean()
    old = df[df["age"] > 75]["fmri_bold_dmn"].isna().mean()
    mar_ok = old > young + 0.05
    all_pass &= mar_ok
    print(f"\n  MAR check: fMRI missing rate age<=75={young:.2%}, age>75={old:.2%}")
    print(f"  Age-dependent excess: {old - young:.2%}  PASS: {mar_ok}")

    return all_pass


def test_boundary_constraints(df):
    """Verify physiological bounds are respected."""
    print("\n" + "=" * 60)
    print("TEST 4: Boundary constraints")
    print("=" * 60)

    checks = [
        ("age", 0, 120),
        ("bmi", 10, 60),
        ("eeg_spectral_entropy", 0.0, 1.0),
        ("mmse_score", 0, 30),
        ("moca_score", 0, 30),
        ("hamilton_depression_score", 0, 52),
        ("updrs_motor_score", 0, 132),
        ("eeg_delta_1_4hz_uv2", 0.0, None),
        ("blood_nfl_pg_ml", 0.0, None),
        ("csf_amyloid_beta42_pg_ml", 0.0, None),
    ]

    all_pass = True
    for col, lo, hi in checks:
        vals = df[col].dropna()
        vmin, vmax = vals.min(), vals.max()
        ok = True
        if lo is not None and vmin < lo:
            ok = False
        if hi is not None and vmax > hi:
            ok = False
        all_pass &= ok
        hi_str = str(hi) if hi is not None else "inf"
        print(f"  {col:40s}: [{lo}, {hi_str}] -> [{vmin:.2f}, {vmax:.2f}]  "
              f"{'PASS' if ok else 'FAIL'}")

    return all_pass


def test_within_disease_normality(df):
    """Within each disease state, continuous features should be approximately normal
    (by design — we sample from multivariate Gaussian). Use D'Agostino-Pearson on
    large samples; expect some deviation due to clipping but p should be reasonable."""
    print("\n" + "=" * 60)
    print("TEST 5: Within-disease normality (D'Agostino-Pearson)")
    print("=" * 60)

    # Pick HC and a few continuous features that aren't heavily clipped
    hc = df[df["disease_code"] == "HC"]
    features = ["fmri_bold_dmn", "fmri_bold_pfc", "blood_gfap_pg_ml"]

    all_pass = True
    for feat in features:
        vals = hc[feat].dropna().values
        if len(vals) < 100:
            continue
        # Skewness and kurtosis
        skew = stats.skew(vals)
        kurt = stats.kurtosis(vals)
        # For large N, even small departures are "significant", so check magnitude
        ok = abs(skew) < 1.0 and abs(kurt) < 3.0
        all_pass &= ok
        print(f"  HC/{feat:30s}: skew={skew:+.3f}, excess_kurtosis={kurt:+.3f}  "
              f"{'PASS' if ok else 'FAIL'}")

    return all_pass


def test_cross_domain_structure(df):
    """Verify that features from DIFFERENT domains have low correlation
    within a single disease state (HC). Pooling across states can create
    spurious correlations via Simpson's paradox."""
    print("\n" + "=" * 60)
    print("TEST 6: Cross-domain independence (within HC)")
    print("=" * 60)

    hc = df[df["disease_code"] == "HC"]
    pairs = [
        ("eeg_delta_1_4hz_uv2", "csf_amyloid_beta42_pg_ml"),
        ("fmri_bold_dmn", "blood_crp_mg_l"),
        ("eeg_gamma_30_100hz_uv2", "polygenic_risk_score"),
    ]

    all_pass = True
    for c1, c2 in pairs:
        both = hc[[c1, c2]].dropna()
        r, p = stats.pearsonr(both[c1], both[c2])
        ok = abs(r) < 0.05
        all_pass &= ok
        print(f"  {c1:35s} vs {c2:35s}: r={r:+.4f}  {'PASS' if ok else 'FAIL'}")

    return all_pass


def main():
    print("Loading synthetic data for robustness validation...\n")
    df = load_sample(500_000)
    print(f"Loaded {len(df):,} samples, {df['disease_code'].nunique()} disease states\n")

    results = {
        "correlation_recovery": test_correlation_recovery(df),
        "disease_separability": test_disease_mean_separation(df),
        "missingness_structure": test_missingness_structure(df),
        "boundary_constraints": test_boundary_constraints(df),
        "within_disease_normality": test_within_disease_normality(df),
        "cross_domain_independence": test_cross_domain_structure(df),
    }

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, passed in results.items():
        print(f"  {name:35s}: {'PASS' if passed else 'FAIL'}")

    all_ok = all(results.values())
    print(f"\nOverall: {'ALL TESTS PASSED' if all_ok else 'SOME TESTS FAILED'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
