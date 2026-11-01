"""
SYNTHETIC LONGITUDINAL DATA GENERATOR — PK/PD Drug Modeling
=============================================================
Generates multi-visit patient time series with disease progression and
pharmacological interventions for neurological/psychiatric conditions.

ALL OUTPUT IS SYNTHETIC. No real patient data is used or produced.

Mathematical framework:
  Baseline:       X_i(0) ~ N(mu_d, Sigma_d)     (from disease_parameters.json)
  Progression:    X_i(t) = X_i(0) + beta_d * t + eps_i(t)
  PK (oral):      C(t) = (F*D*ka)/(Vd*(ka-ke)) * (exp(-ke*t) - exp(-ka*t))
  PK (IV):        C(t) = (D/Vd) * exp(-ke*t)
  PK (SS avg):    Css = (F*D) / (Vd*ke*tau)
  PD (Emax):      E(C) = Emax * C^gamma / (EC50^gamma + C^gamma)
  Combined:       Y_i(t) = X_i(t) + E(C_i(t))
  IIV:            theta_i = theta_pop * exp(eta_i), eta ~ N(0, omega^2)
  Within-subj:    eps_i(t) = rho * eps_i(t-1) + sqrt(1-rho^2) * z,  AR(1)

Usage:
    python3 -I generate_longitudinal_pkpd.py [--total-patients N] [--seed S]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

SCRIPT_DIR = Path(__file__).resolve().parent
SCHEMA_DIR = SCRIPT_DIR.parent / "schemas"
QUARANTINE_DIR = SCRIPT_DIR.parent / "quarantine"

DISEASE_SCHEMA = SCHEMA_DIR / "disease_parameters.json"
DRUG_SCHEMA = SCHEMA_DIR / "drug_parameters.json"
PROGRESSION_SCHEMA = SCHEMA_DIR / "progression_parameters.json"

SEED = 20261031
AR1_RHO = 0.7

PARQUET_METADATA = {
    b"synthetic": b"true",
    b"generator": b"generate_longitudinal_pkpd.py",
    b"version": b"1.0.0",
    b"warning": b"SYNTHETIC DATA - NOT FOR CLINICAL USE",
}

# Reuse correlation structures from cross-sectional generator
EEG_CORR = np.array([
    [1.00, 0.55, -0.40, -0.30, -0.20],
    [0.55, 1.00, -0.30, -0.25, -0.15],
    [-0.40, -0.30, 1.00, 0.35, 0.20],
    [-0.30, -0.25, 0.35, 1.00, 0.40],
    [-0.20, -0.15, 0.20, 0.40, 1.00],
])

FMRI_BOLD_CORR = np.array([
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
    [1.00, 0.50, -0.20, 0.35, 0.25, 0.30, 0.25],
    [0.50, 1.00, -0.15, 0.40, 0.20, 0.25, 0.20],
    [-0.20, -0.15, 1.00, -0.10, -0.15, -0.20, -0.15],
    [0.35, 0.40, -0.10, 1.00, 0.20, 0.25, 0.20],
    [0.25, 0.20, -0.15, 0.20, 1.00, 0.55, 0.45],
    [0.30, 0.25, -0.20, 0.25, 0.55, 1.00, 0.50],
    [0.25, 0.20, -0.15, 0.20, 0.45, 0.50, 1.00],
])

CSF_CORR = np.array([
    [1.00, -0.40, -0.45, -0.20],
    [-0.40, 1.00, 0.80, 0.35],
    [-0.45, 0.80, 1.00, 0.30],
    [-0.20, 0.35, 0.30, 1.00],
])

MISSINGNESS = {
    "eeg": 0.08, "fmri_bold": 0.15, "fmri_conn": 0.15,
    "blood": 0.05, "csf": 0.35, "clinical": 0.03, "genomic": 0.10,
}

# ── Hardcoded fallback drug parameters ──────────────────────────────────
FALLBACK_DRUGS = {
    "alzheimers_disease": [
        {
            "name": "donepezil", "drug_class": "AChE_inhibitor", "route": "oral",
            "dose_mg": 10.0, "doses_per_day": 1,
            "pk": {"ka_per_hr": 0.5, "ke_per_hr": 0.0099, "vd_l_per_kg": 12.0, "bioavailability": 1.0},
            "pd": {"emax": 2.5, "ec50_ng_ml": 15.0, "hill": 1.5},
            "target": "mmse_score", "assignment_prob": 0.50,
            "iiv": {"omega_ka": 0.3, "omega_ke": 0.25, "omega_vd": 0.2},
        },
        {
            "name": "memantine", "drug_class": "NMDA_antagonist", "route": "oral",
            "dose_mg": 20.0, "doses_per_day": 1,
            "pk": {"ka_per_hr": 0.3, "ke_per_hr": 0.0087, "vd_l_per_kg": 9.0, "bioavailability": 1.0},
            "pd": {"emax": 1.5, "ec50_ng_ml": 100.0, "hill": 1.0},
            "target": "mmse_score", "assignment_prob": 0.20,
            "iiv": {"omega_ka": 0.3, "omega_ke": 0.2, "omega_vd": 0.2},
        },
        {
            "name": "lecanemab", "drug_class": "anti_amyloid_mAb", "route": "iv",
            "dose_mg_per_kg": 10.0, "interval_days": 14,
            "pk": {"ke_per_hr": 0.003, "vd_l_per_kg": 0.07},
            "pd": {"emax": 150.0, "ec50_ng_ml": 50000.0, "hill": 1.2},
            "target": "csf_amyloid_beta42_pg_ml", "assignment_prob": 0.15,
            "iiv": {"omega_ke": 0.3, "omega_vd": 0.15},
        },
    ],
    "parkinsons_disease": [
        {
            "name": "levodopa_carbidopa", "drug_class": "dopamine_precursor", "route": "oral",
            "dose_mg": 150.0, "doses_per_day": 3,
            "pk": {"ka_per_hr": 1.5, "ke_per_hr": 0.69, "vd_l_per_kg": 0.9, "bioavailability": 0.3},
            "pd": {"emax": -15.0, "ec50_ng_ml": 300.0, "hill": 1.5},
            "target": "updrs_motor_score", "assignment_prob": 0.55,
            "iiv": {"omega_ka": 0.35, "omega_ke": 0.3, "omega_vd": 0.25},
        },
        {
            "name": "pramipexole", "drug_class": "D2_D3_agonist", "route": "oral",
            "dose_mg": 1.0, "doses_per_day": 3,
            "pk": {"ka_per_hr": 0.8, "ke_per_hr": 0.077, "vd_l_per_kg": 7.0, "bioavailability": 0.9},
            "pd": {"emax": -7.0, "ec50_ng_ml": 0.3, "hill": 1.5},
            "target": "updrs_motor_score", "assignment_prob": 0.25,
            "iiv": {"omega_ka": 0.3, "omega_ke": 0.25, "omega_vd": 0.2},
        },
    ],
    "major_depressive_disorder": [
        {
            "name": "fluoxetine", "drug_class": "SSRI", "route": "oral",
            "dose_mg": 40.0, "doses_per_day": 1,
            "pk": {"ka_per_hr": 0.3, "ke_per_hr": 0.014, "vd_l_per_kg": 35.0, "bioavailability": 0.72},
            "pd": {"emax": -8.0, "ec50_ng_ml": 200.0, "hill": 1.5},
            "target": "hamilton_depression_score", "assignment_prob": 0.45,
            "iiv": {"omega_ka": 0.35, "omega_ke": 0.4, "omega_vd": 0.3},
        },
        {
            "name": "venlafaxine", "drug_class": "SNRI", "route": "oral",
            "dose_mg": 150.0, "doses_per_day": 1,
            "pk": {"ka_per_hr": 0.7, "ke_per_hr": 0.14, "vd_l_per_kg": 7.5, "bioavailability": 0.45},
            "pd": {"emax": -10.0, "ec50_ng_ml": 100.0, "hill": 1.5},
            "target": "hamilton_depression_score", "assignment_prob": 0.30,
            "iiv": {"omega_ka": 0.3, "omega_ke": 0.35, "omega_vd": 0.25},
        },
    ],
    "amyotrophic_lateral_sclerosis": [
        {
            "name": "riluzole", "drug_class": "glutamate_modulator", "route": "oral",
            "dose_mg": 50.0, "doses_per_day": 2,
            "pk": {"ka_per_hr": 1.0, "ke_per_hr": 0.058, "vd_l_per_kg": 3.4, "bioavailability": 0.6},
            "pd": {"emax": -0.35, "ec50_ng_ml": 200.0, "hill": 1.0},
            "target": "blood_nfl_pg_ml", "pd_mode": "rate_modifier",
            "assignment_prob": 0.60,
            "iiv": {"omega_ka": 0.3, "omega_ke": 0.3, "omega_vd": 0.2},
        },
        {
            "name": "edaravone", "drug_class": "free_radical_scavenger", "route": "iv",
            "dose_mg": 60.0, "interval_days": 1,
            "pk": {"ke_per_hr": 0.28, "vd_l_per_kg": 0.15},
            "pd": {"emax": -0.25, "ec50_ng_ml": 500.0, "hill": 1.0},
            "target": "blood_nfl_pg_ml", "pd_mode": "rate_modifier",
            "assignment_prob": 0.20,
            "iiv": {"omega_ke": 0.25, "omega_vd": 0.2},
        },
    ],
    "epilepsy": [
        {
            "name": "levetiracetam", "drug_class": "SV2A_modulator", "route": "oral",
            "dose_mg": 1000.0, "doses_per_day": 2,
            "pk": {"ka_per_hr": 1.2, "ke_per_hr": 0.099, "vd_l_per_kg": 0.6, "bioavailability": 1.0},
            "pd": {"emax": -5.0, "ec50_ng_ml": 20000.0, "hill": 1.5},
            "target": "eeg_delta_1_4hz_uv2",
            "assignment_prob": 0.65,
            "iiv": {"omega_ka": 0.25, "omega_ke": 0.2, "omega_vd": 0.15},
        },
    ],
    "multiple_sclerosis": [
        {
            "name": "ocrelizumab", "drug_class": "anti_CD20_mAb", "route": "iv",
            "dose_mg": 600.0, "interval_days": 182,
            "pk": {"ke_per_hr": 0.001, "vd_l_per_kg": 0.07},
            "pd": {"emax": -0.46, "ec50_ng_ml": 10000.0, "hill": 1.0},
            "target": "blood_nfl_pg_ml", "pd_mode": "rate_modifier",
            "assignment_prob": 0.55,
            "iiv": {"omega_ke": 0.3, "omega_vd": 0.15},
        },
    ],
}

# ── Hardcoded fallback progression parameters ───────────────────────────
FALLBACK_PROGRESSION = {
    "healthy_control": {
        "trajectories": {
            "mmse_score": {"annual_rate": -0.1, "std": 0.5},
            "blood_nfl_pg_ml": {"annual_rate": 0.5, "std": 0.3},
            "eeg_theta_alpha_ratio": {"annual_rate": 0.005, "std": 0.01},
        },
        "visit_interval_months": 12, "total_years": 5, "dropout_rate_per_year": 0.03,
    },
    "alzheimers_disease": {
        "trajectories": {
            "mmse_score": {"annual_rate": -2.5, "std": 1.0},
            "csf_amyloid_beta42_pg_ml": {"annual_rate": -50.0, "std": 20.0},
            "csf_total_tau_pg_ml": {"annual_rate": 30.0, "std": 15.0},
            "eeg_theta_alpha_ratio": {"annual_rate": 0.08, "std": 0.03},
            "blood_nfl_pg_ml": {"annual_rate": 3.0, "std": 1.5},
            "fmri_bold_dmn": {"annual_rate": -0.05, "std": 0.02},
            "fmri_conn_dmn_pfc": {"annual_rate": -0.03, "std": 0.01},
        },
        "visit_interval_months": 6, "total_years": 5, "dropout_rate_per_year": 0.08,
    },
    "parkinsons_disease": {
        "trajectories": {
            "updrs_motor_score": {"annual_rate": 4.0, "std": 2.0},
            "blood_nfl_pg_ml": {"annual_rate": 2.0, "std": 1.0},
            "fmri_bold_basal_ganglia": {"annual_rate": -0.03, "std": 0.015},
            "eeg_beta_13_30hz_uv2": {"annual_rate": -0.5, "std": 0.3},
        },
        "visit_interval_months": 6, "total_years": 5, "dropout_rate_per_year": 0.06,
    },
    "major_depressive_disorder": {
        "trajectories": {
            "hamilton_depression_score": {"annual_rate": 0.0, "std": 3.0},
            "blood_bdnf_ng_ml": {"annual_rate": -0.5, "std": 0.3},
        },
        "episodic": True,
        "relapse_prob_per_year": 0.5,
        "remission_months_mean": 6.0,
        "remission_hamilton_mean": 5.0,
        "relapse_hamilton_mean": 22.0,
        "visit_interval_months": 3, "total_years": 4, "dropout_rate_per_year": 0.10,
    },
    "amyotrophic_lateral_sclerosis": {
        "trajectories": {
            "blood_nfl_pg_ml": {"annual_rate": 15.0, "std": 5.0},
            "blood_gfap_pg_ml": {"annual_rate": 20.0, "std": 8.0},
        },
        "visit_interval_months": 3, "total_years": 3, "dropout_rate_per_year": 0.20,
    },
    "epilepsy": {
        "trajectories": {
            "eeg_delta_1_4hz_uv2": {"annual_rate": 0.5, "std": 0.3},
            "eeg_theta_4_8hz_uv2": {"annual_rate": 0.3, "std": 0.2},
        },
        "visit_interval_months": 6, "total_years": 4, "dropout_rate_per_year": 0.05,
    },
    "multiple_sclerosis": {
        "trajectories": {
            "blood_nfl_pg_ml": {"annual_rate": 2.0, "std": 1.5},
            "fmri_bold_dmn": {"annual_rate": -0.03, "std": 0.02},
        },
        "visit_interval_months": 6, "total_years": 5, "dropout_rate_per_year": 0.05,
    },
    "mild_cognitive_impairment": {
        "trajectories": {
            "mmse_score": {"annual_rate": -1.0, "std": 0.8},
            "csf_amyloid_beta42_pg_ml": {"annual_rate": -30.0, "std": 15.0},
            "blood_nfl_pg_ml": {"annual_rate": 1.5, "std": 0.8},
        },
        "visit_interval_months": 6, "total_years": 5, "dropout_rate_per_year": 0.06,
    },
    "huntingtons_disease": {
        "trajectories": {
            "blood_nfl_pg_ml": {"annual_rate": 5.0, "std": 2.5},
            "mmse_score": {"annual_rate": -1.5, "std": 0.8},
        },
        "visit_interval_months": 6, "total_years": 5, "dropout_rate_per_year": 0.07,
    },
    "traumatic_brain_injury": {
        "trajectories": {
            "blood_nfl_pg_ml": {"annual_rate": -3.0, "std": 2.0},
            "blood_gfap_pg_ml": {"annual_rate": -5.0, "std": 3.0},
            "mmse_score": {"annual_rate": 1.0, "std": 0.5},
        },
        "visit_interval_months": 3, "total_years": 2, "dropout_rate_per_year": 0.12,
    },
    "ischemic_stroke": {
        "trajectories": {
            "blood_nfl_pg_ml": {"annual_rate": -2.0, "std": 1.5},
            "mmse_score": {"annual_rate": 0.5, "std": 0.4},
        },
        "visit_interval_months": 3, "total_years": 2, "dropout_rate_per_year": 0.10,
    },
    "schizophrenia": {
        "trajectories": {
            "fmri_bold_pfc": {"annual_rate": -0.02, "std": 0.01},
            "fmri_conn_dmn_pfc": {"annual_rate": -0.01, "std": 0.008},
        },
        "visit_interval_months": 6, "total_years": 4, "dropout_rate_per_year": 0.12,
    },
    "bipolar_disorder": {
        "trajectories": {
            "hamilton_depression_score": {"annual_rate": 0.0, "std": 3.0},
        },
        "episodic": True,
        "relapse_prob_per_year": 0.6,
        "remission_months_mean": 5.0,
        "remission_hamilton_mean": 4.0,
        "relapse_hamilton_mean": 18.0,
        "visit_interval_months": 3, "total_years": 4, "dropout_rate_per_year": 0.08,
    },
    "autism_spectrum_disorder": {
        "trajectories": {
            "eeg_gamma_30_100hz_uv2": {"annual_rate": 0.1, "std": 0.05},
        },
        "visit_interval_months": 12, "total_years": 5, "dropout_rate_per_year": 0.04,
    },
}

BIOMARKER_BOUNDS = {
    "mmse_score": (0, 30), "moca_score": (0, 30),
    "hamilton_depression_score": (0, 52), "updrs_motor_score": (0, 132),
    "eeg_spectral_entropy": (0.0, 1.0),
    "eeg_delta_1_4hz_uv2": (0.1, None), "eeg_theta_4_8hz_uv2": (0.1, None),
    "eeg_alpha_8_13hz_uv2": (0.1, None), "eeg_beta_13_30hz_uv2": (0.1, None),
    "eeg_gamma_30_100hz_uv2": (0.1, None), "eeg_theta_alpha_ratio": (0.05, None),
    "eeg_alpha_peak_freq_hz": (4.0, 14.0),
    "blood_nfl_pg_ml": (1.0, None), "blood_gfap_pg_ml": (5.0, None),
    "blood_bdnf_ng_ml": (1.0, None), "blood_s100b_ug_l": (0.01, None),
    "blood_crp_mg_l": (0.1, None), "blood_il6_pg_ml": (0.1, None),
    "blood_tnf_alpha_pg_ml": (0.1, None),
    "csf_amyloid_beta42_pg_ml": (100.0, None), "csf_total_tau_pg_ml": (50.0, None),
    "csf_p_tau181_pg_ml": (5.0, None), "csf_nfl_pg_ml": (100.0, None),
    "fmri_conn_dmn_pfc": (-1.0, 1.0), "fmri_conn_hipp_amyg": (-1.0, 1.0),
    "fmri_conn_motor_bg": (-1.0, 1.0), "fmri_conn_thal_cortical": (-1.0, 1.0),
    "bmi": (10.0, 60.0), "age": (0, 120),
}


def nearest_pd(A):
    B = (A + A.T) / 2
    eigvals, eigvecs = np.linalg.eigh(B)
    eigvals = np.maximum(eigvals, 1e-6)
    return eigvecs @ np.diag(eigvals) @ eigvecs.T


def correlated_normals(rng, means, stds, corr, n):
    D = np.diag(stds)
    cov = D @ corr @ D
    cov = nearest_pd(cov)
    return rng.multivariate_normal(means, cov, size=n)


def clip_biomarker(values, col_name):
    lo, hi = BIOMARKER_BOUNDS.get(col_name, (None, None))
    if lo is not None:
        values = np.maximum(values, lo)
    if hi is not None:
        values = np.minimum(values, hi)
    return values


def load_schemas():
    with open(DISEASE_SCHEMA) as f:
        disease_params = json.load(f)["disease_states"]

    drug_params = FALLBACK_DRUGS.copy()
    if DRUG_SCHEMA.exists():
        try:
            with open(DRUG_SCHEMA) as f:
                raw = json.load(f)
            if "drugs" in raw and "disease_drug_mapping" in raw:
                drug_params = {}
                for disease, drug_names in raw["disease_drug_mapping"].items():
                    drugs_list = []
                    for dname in drug_names:
                        d = raw["drugs"].get(dname)
                        if not d:
                            continue
                        dosing = d.get("dosing", {})
                        pk_raw = d.get("pk_parameters", {})
                        pd_raw = d.get("pd_parameters", {})
                        iiv = d.get("interindividual_variability", {})
                        emax_val = pd_raw.get("emax", 0)
                        if "emax_fractional_reduction" in pd_raw:
                            emax_val = -pd_raw["emax_fractional_reduction"]
                        pd_direction = d.get("pd_direction", "")
                        pd_mode = "additive"
                        if pd_direction == "decrease_rate" or \
                                pd_raw.get("effect_type", "").endswith("rate_reduction"):
                            pd_mode = "rate_modifier"
                            emax_val = -abs(pd_raw.get("emax_fractional_reduction",
                                                       abs(emax_val)))
                        normalized = {
                            "name": dname,
                            "drug_class": d.get("drug_class", ""),
                            "route": d.get("route", "oral"),
                            "dose_mg": dosing.get("amount_mg", 0),
                            "doses_per_day": dosing.get("doses_per_day", 1),
                            "pk": {
                                "ka_per_hr": pk_raw.get("ka_per_hr", 1.0),
                                "ke_per_hr": pk_raw.get("ke_per_hr", 0.01),
                                "vd_l_per_kg": pk_raw.get("vd_l_per_kg", 1.0),
                                "bioavailability": pk_raw.get("bioavailability", 1.0),
                            },
                            "pd": {
                                "emax": emax_val,
                                "ec50_ng_ml": pd_raw.get("ec50_ng_ml", 100),
                                "hill": pd_raw.get("hill_coefficient",
                                                   pd_raw.get("hill", 1.0)),
                            },
                            "target": d.get("target_biomarker", ""),
                            "assignment_prob": d.get("treatment_probability", 0.3),
                            "iiv": iiv,
                            "pd_mode": pd_mode,
                        }
                        if d.get("route") == "iv":
                            normalized.pop("doses_per_day", None)
                            iv_dosing = dosing
                            if "amount_mg_per_kg" in iv_dosing:
                                normalized["dose_mg_per_kg"] = iv_dosing["amount_mg_per_kg"]
                                normalized.pop("dose_mg", None)
                            if "interval_days" in iv_dosing:
                                normalized["interval_days"] = iv_dosing["interval_days"]
                        drugs_list.append(normalized)
                    drug_params[disease] = drugs_list
            elif "drugs_by_disease" in raw:
                drug_params = raw["drugs_by_disease"]
        except (json.JSONDecodeError, KeyError):
            pass

    prog_params = FALLBACK_PROGRESSION.copy()
    if PROGRESSION_SCHEMA.exists():
        try:
            with open(PROGRESSION_SCHEMA) as f:
                raw = json.load(f)
            if "progression_models" in raw:
                for disease, info in raw["progression_models"].items():
                    normalized = {}
                    if "biomarker_trajectories" in info:
                        normalized["trajectories"] = info["biomarker_trajectories"]
                    elif "trajectories" in info:
                        normalized["trajectories"] = info["trajectories"]
                    else:
                        normalized["trajectories"] = {}
                    vs = info.get("visit_schedule", {})
                    normalized["visit_interval_months"] = vs.get("interval_months", 6)
                    normalized["total_years"] = vs.get("total_years", 5)
                    normalized["dropout_rate_per_year"] = vs.get("dropout_rate_per_year", 0.05)
                    if info.get("model") == "episodic":
                        normalized["episodic"] = True
                        ep = info.get("episode_parameters", {})
                        normalized["relapse_prob_per_year"] = ep.get("relapse_prob_per_year", 0.5)
                        normalized["remission_months_mean"] = ep.get("remission_duration_months_mean", 6.0)
                        normalized["remission_hamilton_mean"] = ep.get("remission_hamilton_mean", 5.0)
                        normalized["relapse_hamilton_mean"] = ep.get("relapse_hamilton_mean", 22.0)
                    prog_params[disease] = normalized
        except (json.JSONDecodeError, KeyError):
            pass

    return disease_params, drug_params, prog_params


# ── PK/PD engine ────────────────────────────────────────────────────────

def compute_pk_concentration(drug, patient_pk, time_on_drug_months, body_weight_kg):
    """Compute steady-state average concentration in ng/mL."""
    pk = patient_pk
    route = drug.get("route", "oral")

    if route == "oral":
        dose_mg = drug["dose_mg"]
        tau_hr = 24.0 / drug.get("doses_per_day", 1)
        F = pk.get("bioavailability", 1.0)
        ke = pk["ke_per_hr"]
        ka = pk.get("ka_per_hr", 1.0)
        vd_ml = pk["vd_l_per_kg"] * body_weight_kg * 1000.0
        if vd_ml <= 0 or ke <= 0 or tau_hr <= 0:
            return 0.0
        t_half_hr = 0.693 / ke
        if t_half_hr < tau_hr:
            # Short t½: use Cmax approximation (pulsatile dosing)
            tmax = np.log(ka / ke) / (ka - ke) if ka != ke else 1.0 / ka
            cmax = (F * dose_mg * 1e6 * ka) / (vd_ml * (ka - ke)) * (
                np.exp(-ke * tmax) - np.exp(-ka * tmax)) if ka != ke else (
                F * dose_mg * 1e6) / (vd_ml * np.e)
            accumulation = 1.0 / (1.0 - np.exp(-ke * tau_hr))
            return cmax * accumulation
        else:
            # Long t½: Css_avg with accumulation
            css_avg = (F * dose_mg * 1e6) / (vd_ml * ke * tau_hr)
            accumulation = 1.0 / (1.0 - np.exp(-ke * tau_hr))
            return css_avg * min(accumulation, 20.0)
    else:
        # IV: Css_avg = D_ng / (Vd_mL * ke * tau_hr)
        dose_mg = drug.get("dose_mg", 0)
        if dose_mg == 0:
            dose_mg = drug.get("dose_mg_per_kg", 10.0) * body_weight_kg
        interval_days = drug.get("interval_days", 14)
        ke = pk["ke_per_hr"]
        vd_ml = pk["vd_l_per_kg"] * body_weight_kg * 1000.0
        if vd_ml <= 0 or ke <= 0:
            return 0.0
        tau_hr = interval_days * 24.0
        css_avg = (dose_mg * 1e6) / (vd_ml * ke * tau_hr)
        return css_avg


def compute_pd_effect(drug, concentration):
    """Emax model: E = Emax * C^gamma / (EC50^gamma + C^gamma)."""
    pd = drug["pd"]
    emax = pd["emax"]
    ec50 = pd["ec50_ng_ml"]
    gamma = pd.get("hill", 1.0)
    if concentration <= 0 or ec50 <= 0:
        return 0.0
    return emax * (concentration ** gamma) / (ec50 ** gamma + concentration ** gamma)


def sample_patient_pk(rng, drug):
    """Apply inter-individual variability to PK parameters."""
    pk_pop = drug["pk"].copy()
    iiv = drug.get("iiv", {})
    patient_pk = {}
    for param, value in pk_pop.items():
        if isinstance(value, (int, float)):
            omega = iiv.get(f"omega_{param.split('_')[0]}", 0.0)
            if omega > 0:
                patient_pk[param] = value * np.exp(rng.normal(0, omega))
            else:
                patient_pk[param] = value
        else:
            patient_pk[param] = value
    return patient_pk


# ── Baseline biomarker generation ───────────────────────────────────────

def generate_baseline(rng, params, n):
    """Generate baseline biomarkers for n patients of one disease state."""
    demo = params["demographics"]
    age = rng.normal(demo["age"]["mean"], demo["age"]["std"], n)
    age = np.clip(age, demo["age"]["min"], demo["age"]["max"]).astype(int)
    sex = np.where(rng.random(n) < demo["sex_female_prob"], "F", "M")
    bmi = np.clip(rng.normal(demo["bmi"]["mean"], demo["bmi"]["std"], n),
                  demo["bmi"]["min"], demo["bmi"]["max"])

    eeg_p = params["eeg_power_spectral"]
    eeg_keys = ["delta_1_4hz", "theta_4_8hz", "alpha_8_13hz", "beta_13_30hz", "gamma_30_100hz"]
    eeg_means = np.array([eeg_p[k]["mean"] for k in eeg_keys])
    eeg_stds = np.array([eeg_p[k]["std"] for k in eeg_keys])
    eeg_vals = correlated_normals(rng, eeg_means, eeg_stds, EEG_CORR, n)
    eeg_vals = np.maximum(eeg_vals, 0.1)

    eeg_d = params["eeg_derived"]
    tar = np.maximum(rng.normal(eeg_d["theta_alpha_ratio"]["mean"],
                                eeg_d["theta_alpha_ratio"]["std"], n), 0.05)
    se = np.clip(rng.normal(eeg_d["spectral_entropy"]["mean"],
                            eeg_d["spectral_entropy"]["std"], n), 0.0, 1.0)
    apf = np.clip(rng.normal(eeg_d["alpha_peak_freq_hz"]["mean"],
                             eeg_d["alpha_peak_freq_hz"]["std"], n), 4.0, 14.0)

    fmri_p = params["fmri_bold"]
    bold_keys = ["default_mode_network", "prefrontal_cortex", "hippocampus",
                 "amygdala", "motor_cortex", "cerebellum", "thalamus", "basal_ganglia"]
    bold_means = np.array([fmri_p[k]["mean"] for k in bold_keys])
    bold_stds = np.array([fmri_p[k]["std"] for k in bold_keys])
    bold_vals = correlated_normals(rng, bold_means, bold_stds, FMRI_BOLD_CORR, n)

    conn_p = params["fmri_connectivity"]
    conn_keys = ["dmn_pfcx", "hipp_amyg", "motor_bg", "thal_cortical"]
    conn_vals = np.column_stack([
        np.clip(rng.normal(conn_p[k]["mean"], conn_p[k]["std"], n), -1, 1)
        for k in conn_keys
    ])

    bld_p = params["blood_biomarkers"]
    bld_keys = ["nfl_pg_ml", "gfap_pg_ml", "bdnf_ng_ml", "s100b_ug_l",
                "crp_mg_l", "il6_pg_ml", "tnf_alpha_pg_ml"]
    bld_means = np.array([bld_p[k]["mean"] for k in bld_keys])
    bld_stds = np.array([bld_p[k]["std"] for k in bld_keys])
    bld_vals = correlated_normals(rng, bld_means, bld_stds, BLOOD_CORR, n)
    for j, k in enumerate(bld_keys):
        bld_vals[:, j] = np.maximum(bld_vals[:, j], bld_p[k].get("min", 0))

    csf_p = params["csf_biomarkers"]
    csf_keys = ["amyloid_beta42_pg_ml", "total_tau_pg_ml", "p_tau181_pg_ml", "nfl_csf_pg_ml"]
    csf_means = np.array([csf_p[k]["mean"] for k in csf_keys])
    csf_stds = np.array([csf_p[k]["std"] for k in csf_keys])
    csf_vals = correlated_normals(rng, csf_means, csf_stds, CSF_CORR, n)
    for j, k in enumerate(csf_keys):
        csf_vals[:, j] = np.maximum(csf_vals[:, j], csf_p[k].get("min", 0))

    gen_p = params["genomic"]
    apoe = rng.binomial(1, gen_p["apoe_e4_carrier_prob"], n)
    prs = rng.normal(gen_p["polygenic_risk_score"]["mean"],
                     gen_p["polygenic_risk_score"]["std"], n)

    clin_p = params["clinical_scores"]
    mmse = np.clip(np.round(rng.normal(clin_p["mmse"]["mean"], clin_p["mmse"]["std"], n)),
                   clin_p["mmse"]["min"], clin_p["mmse"]["max"]).astype(int)
    moca = np.clip(np.round(rng.normal(clin_p["moca"]["mean"], clin_p["moca"]["std"], n)),
                   clin_p["moca"]["min"], clin_p["moca"]["max"]).astype(int)
    hamd = np.clip(np.round(rng.normal(clin_p["hamilton_depression"]["mean"],
                                       clin_p["hamilton_depression"]["std"], n)),
                   clin_p["hamilton_depression"]["min"],
                   clin_p["hamilton_depression"]["max"]).astype(int)
    updrs = np.clip(np.round(rng.normal(clin_p["updrs_motor"]["mean"],
                                        clin_p["updrs_motor"]["std"], n)),
                    clin_p["updrs_motor"]["min"],
                    clin_p["updrs_motor"]["max"]).astype(int)

    baseline = {
        "age": age, "sex": sex, "bmi": np.round(bmi, 1),
        "eeg_delta_1_4hz_uv2": eeg_vals[:, 0], "eeg_theta_4_8hz_uv2": eeg_vals[:, 1],
        "eeg_alpha_8_13hz_uv2": eeg_vals[:, 2], "eeg_beta_13_30hz_uv2": eeg_vals[:, 3],
        "eeg_gamma_30_100hz_uv2": eeg_vals[:, 4],
        "eeg_theta_alpha_ratio": tar, "eeg_spectral_entropy": se,
        "eeg_alpha_peak_freq_hz": apf,
        "fmri_bold_dmn": bold_vals[:, 0], "fmri_bold_pfc": bold_vals[:, 1],
        "fmri_bold_hippocampus": bold_vals[:, 2], "fmri_bold_amygdala": bold_vals[:, 3],
        "fmri_bold_motor_ctx": bold_vals[:, 4], "fmri_bold_cerebellum": bold_vals[:, 5],
        "fmri_bold_thalamus": bold_vals[:, 6], "fmri_bold_basal_ganglia": bold_vals[:, 7],
        "fmri_conn_dmn_pfc": conn_vals[:, 0], "fmri_conn_hipp_amyg": conn_vals[:, 1],
        "fmri_conn_motor_bg": conn_vals[:, 2], "fmri_conn_thal_cortical": conn_vals[:, 3],
        "blood_nfl_pg_ml": bld_vals[:, 0], "blood_gfap_pg_ml": bld_vals[:, 1],
        "blood_bdnf_ng_ml": bld_vals[:, 2], "blood_s100b_ug_l": bld_vals[:, 3],
        "blood_crp_mg_l": bld_vals[:, 4], "blood_il6_pg_ml": bld_vals[:, 5],
        "blood_tnf_alpha_pg_ml": bld_vals[:, 6],
        "csf_amyloid_beta42_pg_ml": csf_vals[:, 0], "csf_total_tau_pg_ml": csf_vals[:, 1],
        "csf_p_tau181_pg_ml": csf_vals[:, 2], "csf_nfl_pg_ml": csf_vals[:, 3],
        "apoe_e4_carrier": apoe.astype(bool), "polygenic_risk_score": prs,
        "mmse_score": mmse, "moca_score": moca,
        "hamilton_depression_score": hamd, "updrs_motor_score": updrs,
    }
    return baseline


# ── Longitudinal engine ─────────────────────────────────────────────────

def assign_drug(rng, disease_name, drug_list):
    """Assign a drug to a patient. Returns drug dict or None."""
    if not drug_list:
        return None
    probs = [d.get("assignment_prob", 0) for d in drug_list]
    no_treatment_prob = max(0, 1.0 - sum(probs))
    probs.append(no_treatment_prob)
    choice = rng.choice(len(probs), p=np.array(probs) / sum(probs))
    if choice >= len(drug_list):
        return None
    return drug_list[choice]


def generate_patient_timeseries(rng, patient_idx, disease_name, disease_code,
                                 baseline, prog_info, drug, chunk_id):
    """Generate full visit-level time series for one patient."""
    patient_id = f"SYN_P{chunk_id:03d}{patient_idx:07d}"

    interval_m = prog_info.get("visit_interval_months", 6)
    total_years = prog_info.get("total_years", 5)
    dropout_rate = prog_info.get("dropout_rate_per_year", 0.05)
    n_visits = int(total_years * 12 / interval_m) + 1
    trajectories = prog_info.get("trajectories", {})
    is_episodic = prog_info.get("episodic", False)

    body_weight = baseline["bmi"] * (1.75 ** 2) / 10000.0 * 100.0
    body_weight = max(40.0, min(150.0, body_weight * 0.7 + 30.0))

    patient_pk = None
    drug_name = ""
    drug_dose = 0.0
    if drug is not None:
        patient_pk = sample_patient_pk(rng, drug)
        drug_name = drug["name"]
        drug_dose = drug.get("dose_mg", drug.get("dose_mg_per_kg", 0) * body_weight)

    ar1_state = {}
    episodic_state = "remission" if is_episodic else None
    months_in_state = 0

    rows = []
    for v in range(n_visits):
        month = v * interval_m
        jitter_days = int(rng.normal(0, 10))
        visit_day = month * 30.44 + jitter_days
        years = month / 12.0

        if v > 0 and rng.random() < (dropout_rate * interval_m / 12.0):
            break

        if v > 0:
            severity = 0.0
            mmse_val = baseline.get("mmse_score", 28)
            if isinstance(mmse_val, np.ndarray):
                mmse_val = mmse_val.item()
            severity = max(0, (28 - mmse_val) / 28.0)
            updrs_val = baseline.get("updrs_motor_score", 0)
            if isinstance(updrs_val, np.ndarray):
                updrs_val = updrs_val.item()
            severity = max(severity, updrs_val / 132.0)
            if rng.random() < severity * dropout_rate * interval_m / 12.0:
                break

        row = {
            "patient_id": patient_id,
            "visit_number": v,
            "visit_month": month,
            "visit_day": int(round(visit_day)),
            "disease_state": disease_name,
            "disease_code": disease_code,
            "drug_name": drug_name if drug else "",
            "drug_dose_mg": round(drug_dose, 1) if drug else 0.0,
        }

        concentration = 0.0
        drug_effect = 0.0
        if drug is not None and patient_pk is not None and month > 0:
            concentration = compute_pk_concentration(drug, patient_pk, month, body_weight)
            drug_effect = compute_pd_effect(drug, concentration)

        row["drug_concentration_ng_ml"] = round(concentration, 3)
        row["drug_effect"] = round(drug_effect, 4)

        if is_episodic:
            months_in_state += interval_m
            if episodic_state == "remission":
                trans_prob = prog_info.get("relapse_prob_per_year", 0.5) * interval_m / 12.0
                if drug is not None and concentration > 0:
                    frac_effect = abs(drug_effect) / max(abs(drug["pd"]["emax"]), 1e-9)
                    trans_prob *= max(0.2, 1.0 - frac_effect)
                if rng.random() < trans_prob:
                    episodic_state = "relapse"
                    months_in_state = 0
            else:
                remission_mean = prog_info.get("remission_months_mean", 6.0)
                if months_in_state > rng.exponential(remission_mean):
                    episodic_state = "remission"
                    months_in_state = 0

        for col, bl_val in baseline.items():
            if col in ("sex", "apoe_e4_carrier"):
                row[col] = bl_val
                continue

            val = float(bl_val)

            if col in trajectories:
                traj = trajectories[col]
                annual_rate = traj.get("annual_rate",
                              traj.get("baseline_annual_rate", 0.0))
                noise_std = traj.get("std", 0)

                if drug is not None and drug.get("target") == col:
                    pd_mode = drug.get("pd_mode", "additive")
                    if pd_mode == "rate_modifier":
                        frac = abs(compute_pd_effect(drug, concentration))
                        annual_rate *= max(0, 1.0 - frac)
                    else:
                        val += drug_effect

                val += annual_rate * years

                ar_key = col
                if ar_key not in ar1_state:
                    ar1_state[ar_key] = 0.0
                innovation = rng.normal(0, noise_std * np.sqrt(interval_m / 12.0))
                ar1_state[ar_key] = AR1_RHO * ar1_state[ar_key] + np.sqrt(1 - AR1_RHO ** 2) * innovation
                val += ar1_state[ar_key]

            if is_episodic and col == "hamilton_depression_score":
                if episodic_state == "relapse":
                    val = prog_info.get("relapse_hamilton_mean", 22.0) + rng.normal(0, 3)
                else:
                    val = prog_info.get("remission_hamilton_mean", 5.0) + rng.normal(0, 2)

            val = clip_biomarker(np.array([val]), col)[0]

            if col in ("mmse_score", "moca_score", "hamilton_depression_score",
                       "updrs_motor_score", "age"):
                val = int(round(val))

            row[col] = val

        row["age"] = int(baseline["age"]) + int(round(years))
        row["is_synthetic"] = True
        rows.append(row)

    return rows


def generate_patient_chunk(rng, disease_params, drug_params, prog_params,
                           patient_indices, chunk_id):
    """Generate longitudinal data for a chunk of patients."""
    states = list(disease_params.items())
    weights = np.array([v["prevalence_weight"] for _, v in states])
    weights /= weights.sum()

    n_patients = len(patient_indices)
    disease_assignments = rng.choice(len(states), size=n_patients, p=weights)

    all_rows = []
    for local_idx, (patient_offset, disease_idx) in enumerate(
            zip(patient_indices, disease_assignments)):
        disease_name, params = states[disease_idx]
        code = params["code"]

        baseline = generate_baseline(rng, params, 1)
        baseline_scalar = {k: (v[0] if hasattr(v, '__len__') and len(v) == 1 else v)
                          for k, v in baseline.items()}

        drugs_for_disease = drug_params.get(disease_name, [])
        if isinstance(drugs_for_disease, dict):
            drugs_for_disease = drugs_for_disease.get("drugs", [])
        drug = assign_drug(rng, disease_name, drugs_for_disease)

        prog = prog_params.get(disease_name, {
            "trajectories": {}, "visit_interval_months": 6,
            "total_years": 5, "dropout_rate_per_year": 0.05,
        })

        rows = generate_patient_timeseries(
            rng, patient_offset, disease_name, code,
            baseline_scalar, prog, drug, chunk_id)
        all_rows.extend(rows)

    return all_rows


def apply_visit_missingness(rng, df):
    """Apply domain-level MCAR missingness to a visit-level dataframe."""
    for c in df.columns:
        if df[c].dtype == bool:
            df[c] = df[c].astype("boolean")
        elif df[c].dtype == np.int64:
            df[c] = df[c].astype("Int64")
    eeg_cols = [c for c in df.columns if c.startswith("eeg_")]
    bold_cols = [c for c in df.columns if c.startswith("fmri_bold_")]
    conn_cols = [c for c in df.columns if c.startswith("fmri_conn_")]
    blood_cols = [c for c in df.columns if c.startswith("blood_")]
    csf_cols = [c for c in df.columns if c.startswith("csf_")]
    clin_cols = ["mmse_score", "moca_score", "hamilton_depression_score", "updrs_motor_score"]
    gen_cols = ["apoe_e4_carrier", "polygenic_risk_score"]

    for cols, rate in [(eeg_cols, MISSINGNESS["eeg"]),
                       (bold_cols, MISSINGNESS["fmri_bold"]),
                       (conn_cols, MISSINGNESS["fmri_conn"]),
                       (blood_cols, MISSINGNESS["blood"]),
                       (csf_cols, MISSINGNESS["csf"]),
                       (clin_cols, MISSINGNESS["clinical"]),
                       (gen_cols, MISSINGNESS["genomic"])]:
        cols_present = [c for c in cols if c in df.columns]
        if cols_present and rate > 0:
            mask = rng.random((len(df), len(cols_present))) < rate
            for j, c in enumerate(cols_present):
                df.loc[mask[:, j], c] = np.nan

    if "age" in df.columns:
        age_vals = df["age"].values
        old_mask = (age_vals > 75) & (rng.random(len(df)) < 0.25)
        fmri_cols = bold_cols + conn_cols
        fmri_present = [c for c in fmri_cols if c in df.columns]
        for c in fmri_present:
            df.loc[old_mask, c] = np.nan

    return df


def write_parquet(df, output_dir, chunk_id):
    fname = f"SYNTHETIC_longitudinal_pkpd_chunk_{chunk_id:03d}.parquet"
    path = output_dir / fname
    table = pa.Table.from_pandas(df, preserve_index=False)
    existing_meta = table.schema.metadata or {}
    merged_meta = {**existing_meta, **PARQUET_METADATA}
    table = table.replace_schema_metadata(merged_meta)
    pq.write_table(table, str(path), compression="snappy")
    return path


def main():
    parser = argparse.ArgumentParser(
        description="Generate longitudinal synthetic neuro/bio dataset with PK/PD modeling")
    parser.add_argument("--total-patients", type=int, default=50_000)
    parser.add_argument("--chunk-size", type=int, default=10_000,
                        help="Patients per chunk (not rows)")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else QUARANTINE_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    disease_params, drug_params, prog_params = load_schemas()

    n_chunks = (args.total_patients + args.chunk_size - 1) // args.chunk_size

    print(f"{'=' * 60}")
    print(f"LONGITUDINAL PK/PD SYNTHETIC DATA GENERATOR v1.0.0")
    print(f"{'=' * 60}")
    print(f"Total patients: {args.total_patients:,}")
    print(f"Chunk size:     {args.chunk_size:,} patients")
    print(f"Num chunks:     {n_chunks}")
    print(f"Disease states: {len(disease_params)}")
    print(f"Drug configs:   {sum(len(v) if isinstance(v, list) else len(v.get('drugs', [])) for v in drug_params.values())}")
    print(f"Output dir:     {output_dir}")
    print(f"Seed:           {args.seed}")
    print(f"{'=' * 60}")

    rng = np.random.default_rng(args.seed)
    total_patients = 0
    total_rows = 0
    t0 = time.time()

    for i in range(n_chunks):
        remaining = args.total_patients - total_patients
        this_chunk = min(args.chunk_size, remaining)
        if this_chunk <= 0:
            break

        patient_indices = list(range(total_patients, total_patients + this_chunk))

        t1 = time.time()
        rows = generate_patient_chunk(rng, disease_params, drug_params,
                                      prog_params, patient_indices, i)

        df = pd.DataFrame(rows)

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col in BIOMARKER_BOUNDS:
                lo, hi = BIOMARKER_BOUNDS[col]
                if lo is not None:
                    df[col] = df[col].clip(lower=lo)
                if hi is not None:
                    df[col] = df[col].clip(upper=hi)

        df = apply_visit_missingness(rng, df)

        path = write_parquet(df, output_dir, i)
        elapsed = time.time() - t1
        total_patients += this_chunk
        total_rows += len(df)

        print(f"  Chunk {i + 1:3d}/{n_chunks}: {this_chunk:>7,} patients, "
              f"{len(df):>9,} rows -> {path.name}  ({elapsed:.1f}s)")

    elapsed_total = time.time() - t0
    print(f"{'=' * 60}")
    print(f"Done. {total_patients:,} patients, {total_rows:,} total rows in {elapsed_total:.1f}s")

    manifest = {
        "synthetic": True,
        "generator": "generate_longitudinal_pkpd.py",
        "generator_version": "1.0.0",
        "total_patients": total_patients,
        "total_rows": total_rows,
        "num_chunks": n_chunks,
        "patients_per_chunk": args.chunk_size,
        "seed": args.seed,
        "disease_states": list(disease_params.keys()),
        "drugs_modeled": [d["name"] for drugs in drug_params.values()
                          for d in (drugs if isinstance(drugs, list) else drugs.get("drugs", []))],
        "pk_model": "1-compartment (oral: first-order absorption; IV: bolus)",
        "pd_model": "Emax with Hill coefficient",
        "progression_model": "linear + AR(1) residuals (episodic for MDD/BD)",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    manifest_path = output_dir / "SYNTHETIC_longitudinal_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
