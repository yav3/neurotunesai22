"""
SYNTHETIC DATA GENERATOR — Neuro/Bio Disease-State Dataset
===========================================================
Generates ≥10M synthetic samples across 13 human disease states with
biologically plausible multivariate distributions, realistic missingness
patterns, and inter-feature correlations.

ALL OUTPUT IS SYNTHETIC. No real patient data is used or produced.

Output: Parquet files in synthetic_data/quarantine/
Each file tagged with synthetic=true in Parquet metadata.

Disease states: HC, AD, PD, MDD, EP, MS, ALS, HD, TBI, IS, SZ, BD, ASD, MCI
Feature domains: EEG spectral, fMRI BOLD/connectivity, blood/CSF biomarkers,
                 genomics, clinical scores, demographics

Usage:
    python3 -I generate_neuro_bio_dataset.py [--total-samples N] [--chunk-size M]
"""

import argparse
import json
import os
import sys
import time
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


SCRIPT_DIR = Path(__file__).resolve().parent
QUARANTINE_DIR = SCRIPT_DIR.parent / "quarantine"
SCHEMA_PATH = SCRIPT_DIR.parent / "schemas" / "disease_parameters.json"

SEED = 20261031
PARQUET_METADATA = {
    b"synthetic": b"true",
    b"generator": b"generate_neuro_bio_dataset.py",
    b"version": b"1.0.0",
    b"warning": b"SYNTHETIC DATA - NOT FOR CLINICAL USE",
}


# ── Correlation structures by domain ──────────────────────────────────────
# Approximate within-domain Pearson r values observed in real datasets.
# These define a positive-definite correlation skeleton; actual magnitudes
# are scaled per disease state via the parameterized std.

EEG_CORR = np.array([
    # delta  theta  alpha  beta   gamma
    [1.00,  0.55, -0.40, -0.30, -0.20],  # delta
    [0.55,  1.00, -0.30, -0.25, -0.15],  # theta
    [-0.40, -0.30,  1.00,  0.35,  0.20],  # alpha
    [-0.30, -0.25,  0.35,  1.00,  0.40],  # beta
    [-0.20, -0.15,  0.20,  0.40,  1.00],  # gamma
])

FMRI_BOLD_CORR = np.array([
    # DMN   PFC   Hipp  Amyg  Motor Cereb Thal  BG
    [1.00, 0.45, 0.35, 0.20, 0.10, 0.10, 0.25, 0.15],
    [0.45, 1.00, 0.30, 0.25, 0.15, 0.10, 0.30, 0.20],
    [0.35, 0.30, 1.00, 0.45, 0.10, 0.10, 0.25, 0.15],
    [0.20, 0.25, 0.45, 1.00, 0.10, 0.05, 0.20, 0.10],
    [0.10, 0.15, 0.10, 0.10, 1.00, 0.30, 0.25, 0.45],
    [0.10, 0.10, 0.10, 0.05, 0.30, 1.00, 0.20, 0.25],
    [0.25, 0.30, 0.25, 0.20, 0.25, 0.20, 1.00, 0.35],
    [0.15, 0.20, 0.15, 0.10, 0.45, 0.25, 0.35, 1.00],
])

BLOOD_CORR = np.array([
    # NfL   GFAP  BDNF  S100B CRP   IL6   TNFa
    [1.00, 0.50, -0.20, 0.35, 0.25, 0.30, 0.25],
    [0.50, 1.00, -0.15, 0.40, 0.20, 0.25, 0.20],
    [-0.20, -0.15, 1.00, -0.10, -0.15, -0.20, -0.15],
    [0.35, 0.40, -0.10, 1.00, 0.20, 0.25, 0.20],
    [0.25, 0.20, -0.15, 0.20, 1.00, 0.55, 0.45],
    [0.30, 0.25, -0.20, 0.25, 0.55, 1.00, 0.50],
    [0.25, 0.20, -0.15, 0.20, 0.45, 0.50, 1.00],
])

CSF_CORR = np.array([
    # Ab42  tTau  pTau  NfL
    [1.00, -0.40, -0.45, -0.20],
    [-0.40, 1.00, 0.80, 0.35],
    [-0.45, 0.80, 1.00, 0.30],
    [-0.20, 0.35, 0.30, 1.00],
])

# Missingness rates by domain (MCAR + MAR components)
MISSINGNESS = {
    "eeg": 0.08,
    "fmri_bold": 0.15,
    "fmri_conn": 0.15,
    "blood": 0.05,
    "csf": 0.35,  # CSF is invasive → high missingness
    "clinical": 0.03,
    "genomic": 0.10,
}


def nearest_pd(A):
    """Project a matrix to the nearest positive-definite matrix."""
    B = (A + A.T) / 2
    eigvals, eigvecs = np.linalg.eigh(B)
    eigvals = np.maximum(eigvals, 1e-6)
    return eigvecs @ np.diag(eigvals) @ eigvecs.T


def correlated_normals(rng, means, stds, corr, n):
    """Draw n samples from N(means, Sigma) where Sigma = diag(stds) @ corr @ diag(stds)."""
    D = np.diag(stds)
    cov = D @ corr @ D
    cov = nearest_pd(cov)
    return rng.multivariate_normal(means, cov, size=n)


def lognormal_from_params(rng, mean, std, n):
    """Generate lognormal samples with the specified arithmetic mean and std."""
    var = std ** 2
    mu_ln = np.log(mean ** 2 / np.sqrt(var + mean ** 2))
    sigma_ln = np.sqrt(np.log(1 + var / mean ** 2))
    return rng.lognormal(mu_ln, sigma_ln, size=n)


def apply_missingness(rng, df_block, rate, mechanism="mcar"):
    """Inject missing values. MCAR baseline with optional MAR on age."""
    if rate <= 0:
        return df_block
    mask = rng.random(df_block.shape) < rate
    return df_block.mask(pd.DataFrame(mask, index=df_block.index, columns=df_block.columns))


def generate_chunk(disease_params, chunk_size, rng, chunk_id):
    """Generate one chunk of synthetic data across all disease states."""
    all_records = []
    states = list(disease_params.items())
    weights = np.array([v["prevalence_weight"] for _, v in states])
    weights /= weights.sum()

    counts = rng.multinomial(chunk_size, weights)

    for (disease_name, params), n_samples in zip(states, counts):
        if n_samples == 0:
            continue

        code = params["code"]
        demo = params["demographics"]

        # ── Subject IDs ──
        subject_ids = [f"SYNTH-{code}-{chunk_id:04d}-{i:07d}" for i in range(n_samples)]

        # ── Demographics ──
        age = rng.normal(demo["age"]["mean"], demo["age"]["std"], n_samples)
        age = np.clip(age, demo["age"]["min"], demo["age"]["max"]).astype(int)

        sex = rng.binomial(1, demo["sex_female_prob"], n_samples)
        sex_labels = np.where(sex == 1, "F", "M")

        bmi = rng.normal(demo["bmi"]["mean"], demo["bmi"]["std"], n_samples)
        bmi = np.clip(bmi, demo["bmi"]["min"], demo["bmi"]["max"])

        # ── EEG spectral power ──
        eeg_p = params["eeg_power_spectral"]
        eeg_means = np.array([eeg_p[k]["mean"] for k in
                              ["delta_1_4hz", "theta_4_8hz", "alpha_8_13hz",
                               "beta_13_30hz", "gamma_30_100hz"]])
        eeg_stds = np.array([eeg_p[k]["std"] for k in
                             ["delta_1_4hz", "theta_4_8hz", "alpha_8_13hz",
                              "beta_13_30hz", "gamma_30_100hz"]])
        eeg_vals = correlated_normals(rng, eeg_means, eeg_stds, EEG_CORR, n_samples)
        eeg_vals = np.maximum(eeg_vals, 0.1)  # power is non-negative

        # ── EEG derived ──
        eeg_d = params["eeg_derived"]
        tar = rng.normal(eeg_d["theta_alpha_ratio"]["mean"],
                         eeg_d["theta_alpha_ratio"]["std"], n_samples)
        tar = np.maximum(tar, 0.05)
        se = rng.normal(eeg_d["spectral_entropy"]["mean"],
                        eeg_d["spectral_entropy"]["std"], n_samples)
        se = np.clip(se, 0.0, 1.0)
        apf = rng.normal(eeg_d["alpha_peak_freq_hz"]["mean"],
                         eeg_d["alpha_peak_freq_hz"]["std"], n_samples)
        apf = np.clip(apf, 4.0, 14.0)

        # ── fMRI BOLD ──
        fmri_p = params["fmri_bold"]
        bold_keys = ["default_mode_network", "prefrontal_cortex", "hippocampus",
                     "amygdala", "motor_cortex", "cerebellum", "thalamus", "basal_ganglia"]
        bold_means = np.array([fmri_p[k]["mean"] for k in bold_keys])
        bold_stds = np.array([fmri_p[k]["std"] for k in bold_keys])
        bold_vals = correlated_normals(rng, bold_means, bold_stds, FMRI_BOLD_CORR, n_samples)

        # ── fMRI connectivity ──
        conn_p = params["fmri_connectivity"]
        conn_keys = ["dmn_pfcx", "hipp_amyg", "motor_bg", "thal_cortical"]
        conn_vals = np.column_stack([
            np.clip(rng.normal(conn_p[k]["mean"], conn_p[k]["std"], n_samples), -1, 1)
            for k in conn_keys
        ])

        # ── Blood biomarkers ──
        bld_p = params["blood_biomarkers"]
        bld_keys = ["nfl_pg_ml", "gfap_pg_ml", "bdnf_ng_ml", "s100b_ug_l",
                     "crp_mg_l", "il6_pg_ml", "tnf_alpha_pg_ml"]
        bld_means = np.array([bld_p[k]["mean"] for k in bld_keys])
        bld_stds = np.array([bld_p[k]["std"] for k in bld_keys])

        # Use lognormal for skewed markers, normal for others
        bld_vals = correlated_normals(rng, bld_means, bld_stds, BLOOD_CORR, n_samples)
        for j, k in enumerate(bld_keys):
            mn = bld_p[k].get("min", 0)
            bld_vals[:, j] = np.maximum(bld_vals[:, j], mn)

        # ── CSF biomarkers ──
        csf_p = params["csf_biomarkers"]
        csf_keys = ["amyloid_beta42_pg_ml", "total_tau_pg_ml",
                     "p_tau181_pg_ml", "nfl_csf_pg_ml"]
        csf_means = np.array([csf_p[k]["mean"] for k in csf_keys])
        csf_stds = np.array([csf_p[k]["std"] for k in csf_keys])
        csf_vals = correlated_normals(rng, csf_means, csf_stds, CSF_CORR, n_samples)
        for j, k in enumerate(csf_keys):
            mn = csf_p[k].get("min", 0)
            csf_vals[:, j] = np.maximum(csf_vals[:, j], mn)

        # ── Genomic ──
        gen_p = params["genomic"]
        apoe_e4 = rng.binomial(1, gen_p["apoe_e4_carrier_prob"], n_samples)
        prs = rng.normal(gen_p["polygenic_risk_score"]["mean"],
                         gen_p["polygenic_risk_score"]["std"], n_samples)

        # ── Clinical scores ──
        clin_p = params["clinical_scores"]
        mmse = rng.normal(clin_p["mmse"]["mean"], clin_p["mmse"]["std"], n_samples)
        mmse = np.clip(np.round(mmse), clin_p["mmse"]["min"], clin_p["mmse"]["max"]).astype(int)

        moca = rng.normal(clin_p["moca"]["mean"], clin_p["moca"]["std"], n_samples)
        moca = np.clip(np.round(moca), clin_p["moca"]["min"], clin_p["moca"]["max"]).astype(int)

        hamd = rng.normal(clin_p["hamilton_depression"]["mean"],
                          clin_p["hamilton_depression"]["std"], n_samples)
        hamd = np.clip(np.round(hamd),
                       clin_p["hamilton_depression"]["min"],
                       clin_p["hamilton_depression"]["max"]).astype(int)

        updrs = rng.normal(clin_p["updrs_motor"]["mean"],
                           clin_p["updrs_motor"]["std"], n_samples)
        updrs = np.clip(np.round(updrs),
                        clin_p["updrs_motor"]["min"],
                        clin_p["updrs_motor"]["max"]).astype(int)

        # ── Assemble DataFrame ──
        df = pd.DataFrame({
            "subject_id": subject_ids,
            "disease_state": disease_name,
            "disease_code": code,
            "is_synthetic": True,
            "age": age,
            "sex": sex_labels,
            "bmi": np.round(bmi, 1),
            # EEG
            "eeg_delta_1_4hz_uv2": np.round(eeg_vals[:, 0], 2),
            "eeg_theta_4_8hz_uv2": np.round(eeg_vals[:, 1], 2),
            "eeg_alpha_8_13hz_uv2": np.round(eeg_vals[:, 2], 2),
            "eeg_beta_13_30hz_uv2": np.round(eeg_vals[:, 3], 2),
            "eeg_gamma_30_100hz_uv2": np.round(eeg_vals[:, 4], 2),
            "eeg_theta_alpha_ratio": np.round(tar, 3),
            "eeg_spectral_entropy": np.round(se, 4),
            "eeg_alpha_peak_freq_hz": np.round(apf, 2),
            # fMRI BOLD
            "fmri_bold_dmn": np.round(bold_vals[:, 0], 3),
            "fmri_bold_pfc": np.round(bold_vals[:, 1], 3),
            "fmri_bold_hippocampus": np.round(bold_vals[:, 2], 3),
            "fmri_bold_amygdala": np.round(bold_vals[:, 3], 3),
            "fmri_bold_motor_ctx": np.round(bold_vals[:, 4], 3),
            "fmri_bold_cerebellum": np.round(bold_vals[:, 5], 3),
            "fmri_bold_thalamus": np.round(bold_vals[:, 6], 3),
            "fmri_bold_basal_ganglia": np.round(bold_vals[:, 7], 3),
            # fMRI connectivity
            "fmri_conn_dmn_pfc": np.round(conn_vals[:, 0], 4),
            "fmri_conn_hipp_amyg": np.round(conn_vals[:, 1], 4),
            "fmri_conn_motor_bg": np.round(conn_vals[:, 2], 4),
            "fmri_conn_thal_cortical": np.round(conn_vals[:, 3], 4),
            # Blood biomarkers
            "blood_nfl_pg_ml": np.round(bld_vals[:, 0], 2),
            "blood_gfap_pg_ml": np.round(bld_vals[:, 1], 1),
            "blood_bdnf_ng_ml": np.round(bld_vals[:, 2], 2),
            "blood_s100b_ug_l": np.round(bld_vals[:, 3], 4),
            "blood_crp_mg_l": np.round(bld_vals[:, 4], 2),
            "blood_il6_pg_ml": np.round(bld_vals[:, 5], 2),
            "blood_tnf_alpha_pg_ml": np.round(bld_vals[:, 6], 2),
            # CSF biomarkers
            "csf_amyloid_beta42_pg_ml": np.round(csf_vals[:, 0], 1),
            "csf_total_tau_pg_ml": np.round(csf_vals[:, 1], 1),
            "csf_p_tau181_pg_ml": np.round(csf_vals[:, 2], 2),
            "csf_nfl_pg_ml": np.round(csf_vals[:, 3], 1),
            # Genomic
            "apoe_e4_carrier": apoe_e4.astype(bool),
            "polygenic_risk_score": np.round(prs, 4),
            # Clinical
            "mmse_score": mmse,
            "moca_score": moca,
            "hamilton_depression_score": hamd,
            "updrs_motor_score": updrs,
        })

        # ── Apply missingness per domain ──
        eeg_cols = [c for c in df.columns if c.startswith("eeg_")]
        bold_cols = [c for c in df.columns if c.startswith("fmri_bold_")]
        conn_cols = [c for c in df.columns if c.startswith("fmri_conn_")]
        blood_cols = [c for c in df.columns if c.startswith("blood_")]
        csf_cols = [c for c in df.columns if c.startswith("csf_")]
        clin_cols = ["mmse_score", "moca_score", "hamilton_depression_score", "updrs_motor_score"]
        gen_cols = ["apoe_e4_carrier", "polygenic_risk_score"]

        df[eeg_cols] = apply_missingness(rng, df[eeg_cols], MISSINGNESS["eeg"])
        df[bold_cols] = apply_missingness(rng, df[bold_cols], MISSINGNESS["fmri_bold"])
        df[conn_cols] = apply_missingness(rng, df[conn_cols], MISSINGNESS["fmri_conn"])
        df[blood_cols] = apply_missingness(rng, df[blood_cols], MISSINGNESS["blood"])
        df[csf_cols] = apply_missingness(rng, df[csf_cols], MISSINGNESS["csf"])
        df[clin_cols] = apply_missingness(rng, df[clin_cols], MISSINGNESS["clinical"])
        df[gen_cols] = apply_missingness(rng, df[gen_cols], MISSINGNESS["genomic"])

        # MAR: older subjects more likely to have missing fMRI
        age_mask = (age > 75) & (rng.random(n_samples) < 0.25)
        for col in bold_cols + conn_cols:
            df.loc[age_mask, col] = np.nan

        all_records.append(df)

    return pd.concat(all_records, ignore_index=True)


def write_parquet_chunk(df, output_dir, chunk_id):
    """Write a chunk to parquet with synthetic metadata."""
    fname = f"SYNTHETIC_neuro_bio_disease_chunk_{chunk_id:04d}.parquet"
    path = output_dir / fname

    table = pa.Table.from_pandas(df)
    existing_meta = table.schema.metadata or {}
    merged_meta = {**existing_meta, **PARQUET_METADATA}
    table = table.replace_schema_metadata(merged_meta)

    pq.write_table(table, str(path), compression="snappy")
    return path


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic neuro/bio disease-state dataset (≥10M samples)")
    parser.add_argument("--total-samples", type=int, default=10_000_000,
                        help="Total number of synthetic samples to generate (default: 10M)")
    parser.add_argument("--chunk-size", type=int, default=500_000,
                        help="Samples per Parquet file (default: 500K)")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else QUARANTINE_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(SCHEMA_PATH) as f:
        schema = json.load(f)
    disease_params = schema["disease_states"]

    rng = np.random.default_rng(args.seed)
    n_chunks = (args.total_samples + args.chunk_size - 1) // args.chunk_size

    print(f"{'=' * 60}")
    print(f"SYNTHETIC DATA GENERATOR v1.0.0")
    print(f"{'=' * 60}")
    print(f"Total samples:  {args.total_samples:,}")
    print(f"Chunk size:     {args.chunk_size:,}")
    print(f"Num chunks:     {n_chunks}")
    print(f"Disease states: {len(disease_params)}")
    print(f"Output dir:     {output_dir}")
    print(f"Seed:           {args.seed}")
    print(f"{'=' * 60}")

    total_written = 0
    t0 = time.time()

    for i in range(n_chunks):
        remaining = args.total_samples - total_written
        this_chunk = min(args.chunk_size, remaining)
        if this_chunk <= 0:
            break

        t1 = time.time()
        df = generate_chunk(disease_params, this_chunk, rng, i)
        path = write_parquet_chunk(df, output_dir, i)
        elapsed = time.time() - t1
        total_written += len(df)

        print(f"  Chunk {i + 1:3d}/{n_chunks}: {len(df):>9,} samples → {path.name}  "
              f"({elapsed:.1f}s, {total_written:,} total)")

    elapsed_total = time.time() - t0
    print(f"{'=' * 60}")
    print(f"Done. {total_written:,} samples in {elapsed_total:.1f}s")
    print(f"Output: {output_dir}")

    # Write manifest
    manifest = {
        "synthetic": True,
        "generator_version": "1.0.0",
        "total_samples": total_written,
        "num_chunks": n_chunks,
        "chunk_size": args.chunk_size,
        "seed": args.seed,
        "disease_states": list(disease_params.keys()),
        "feature_count": len(df.columns),
        "columns": list(df.columns),
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    manifest_path = output_dir / "SYNTHETIC_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
