"""
De-identified archetypal patient prototype generator.

Produces curated composite patient profiles for each of 14 disease states,
representing early/mild, typical/moderate, advanced/severe, atypical, and
treatment-responsive presentations. All profiles are entirely synthetic.

Output: SYNTHETIC_prototypes.json + SYNTHETIC_prototypes.parquet
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def build_prototypes():
    """Return the full prototype dictionary, hardcoded for reproducibility."""
    return {
        "healthy_control": [
            {
                "prototype_id": "HC_YOUNG_001",
                "archetype": "young_healthy",
                "label": "Young healthy adult, no neurological risk factors",
                "demographics": {
                    "age": 32, "sex": "M", "bmi": 23.5,
                    "apoe_e4_carrier": False, "education_years": 16
                },
                "baseline_biomarkers": {
                    "mmse_score": 30, "moca_score": 29,
                    "hamilton_depression_score": 1, "updrs_motor_score": 0,
                    "csf_amyloid_beta42_pg_ml": 1200,
                    "csf_total_tau_pg_ml": 180,
                    "csf_p_tau181_pg_ml": 18,
                    "csf_nfl_pg_ml": 400,
                    "blood_nfl_pg_ml": 8.0,
                    "blood_gfap_pg_ml": 55.0,
                    "blood_bdnf_ng_ml": 28.0,
                    "blood_crp_mg_l": 0.8,
                    "blood_il6_pg_ml": 0.9,
                    "eeg_theta_alpha_ratio": 0.35,
                    "eeg_spectral_entropy": 0.78,
                    "eeg_alpha_peak_freq_hz": 10.5,
                    "fmri_bold_dmn": 0.05,
                    "fmri_connectivity_dmn_pfcx": 0.50,
                    "polygenic_risk_score": -0.3
                },
                "treatment": None,
                "trajectory_summary": [
                    {"month": 0, "mmse": 30, "blood_nfl": 8.0, "status": "baseline"},
                    {"month": 24, "mmse": 30, "blood_nfl": 8.2, "status": "stable"},
                    {"month": 48, "mmse": 30, "blood_nfl": 8.5, "status": "stable"}
                ],
                "clinical_narrative": (
                    "32-year-old man with no family history of neurodegenerative disease. "
                    "APOE e3/e3 genotype. All biomarkers within normal ranges. Cognitive "
                    "testing unremarkable. Enrolled as a healthy control in a longitudinal "
                    "neuroimaging study. No change over 4-year follow-up."
                )
            },
            {
                "prototype_id": "HC_AGING_002",
                "archetype": "aging_healthy",
                "label": "Aging healthy adult with normal age-related changes",
                "demographics": {
                    "age": 72, "sex": "F", "bmi": 27.0,
                    "apoe_e4_carrier": False, "education_years": 14
                },
                "baseline_biomarkers": {
                    "mmse_score": 28, "moca_score": 26,
                    "hamilton_depression_score": 3, "updrs_motor_score": 3,
                    "csf_amyloid_beta42_pg_ml": 980,
                    "csf_total_tau_pg_ml": 280,
                    "csf_p_tau181_pg_ml": 26,
                    "csf_nfl_pg_ml": 700,
                    "blood_nfl_pg_ml": 18.0,
                    "blood_gfap_pg_ml": 110.0,
                    "blood_bdnf_ng_ml": 20.0,
                    "blood_crp_mg_l": 2.0,
                    "blood_il6_pg_ml": 2.0,
                    "eeg_theta_alpha_ratio": 0.45,
                    "eeg_spectral_entropy": 0.72,
                    "eeg_alpha_peak_freq_hz": 9.8,
                    "fmri_bold_dmn": -0.05,
                    "fmri_connectivity_dmn_pfcx": 0.42,
                    "polygenic_risk_score": 0.1
                },
                "treatment": None,
                "trajectory_summary": [
                    {"month": 0, "mmse": 28, "blood_nfl": 18.0, "status": "baseline"},
                    {"month": 24, "mmse": 28, "blood_nfl": 19.0, "status": "stable"},
                    {"month": 48, "mmse": 27, "blood_nfl": 20.0, "status": "normal aging"}
                ],
                "clinical_narrative": (
                    "72-year-old woman with no cognitive complaints. Mild age-related "
                    "increases in CSF tau and blood NfL, within expected norms for age. "
                    "Slightly elevated inflammatory markers consistent with normal aging. "
                    "MMSE stable at 27-28 over 4-year follow-up."
                )
            },
        ],

        "alzheimers_disease": [
            {
                "prototype_id": "AD_EARLY_001",
                "archetype": "early_mild",
                "label": "Early-stage Alzheimer's, amyloid-positive, preserved function",
                "demographics": {
                    "age": 68, "sex": "F", "bmi": 24.0,
                    "apoe_e4_carrier": True, "education_years": 16
                },
                "baseline_biomarkers": {
                    "mmse_score": 26, "moca_score": 23,
                    "hamilton_depression_score": 5, "updrs_motor_score": 1,
                    "csf_amyloid_beta42_pg_ml": 450,
                    "csf_total_tau_pg_ml": 350,
                    "csf_p_tau181_pg_ml": 45,
                    "csf_nfl_pg_ml": 900,
                    "blood_nfl_pg_ml": 25.0,
                    "blood_gfap_pg_ml": 150.0,
                    "blood_bdnf_ng_ml": 18.0,
                    "blood_crp_mg_l": 2.5,
                    "blood_il6_pg_ml": 2.2,
                    "eeg_theta_alpha_ratio": 0.65,
                    "eeg_spectral_entropy": 0.65,
                    "eeg_alpha_peak_freq_hz": 9.0,
                    "fmri_bold_dmn": -0.30,
                    "fmri_connectivity_dmn_pfcx": 0.32,
                    "polygenic_risk_score": 1.2
                },
                "treatment": {
                    "drug": "donepezil", "dose_mg": 10, "response": "partial"
                },
                "trajectory_summary": [
                    {"month": 0, "mmse": 26, "csf_ab42": 450, "status": "baseline"},
                    {"month": 12, "mmse": 24, "csf_ab42": 420, "status": "mild decline"},
                    {"month": 24, "mmse": 22, "csf_ab42": 390, "status": "moderate decline"},
                    {"month": 36, "mmse": 19, "csf_ab42": 360, "status": "significant decline"}
                ],
                "clinical_narrative": (
                    "68-year-old woman, APOE e4 heterozygous, presenting with subtle "
                    "episodic memory deficits. CSF profile consistent with early AD "
                    "pathology (low Abeta42, mildly elevated tau). Started on donepezil "
                    "10mg with initial stabilization, followed by gradual decline over "
                    "3 years. EEG shows increased theta/alpha ratio. fMRI reveals early "
                    "default mode network hypoactivation."
                )
            },
            {
                "prototype_id": "AD_TYPICAL_002",
                "archetype": "typical_moderate",
                "label": "Moderate Alzheimer's with classic biomarker profile",
                "demographics": {
                    "age": 76, "sex": "F", "bmi": 23.0,
                    "apoe_e4_carrier": True, "education_years": 12
                },
                "baseline_biomarkers": {
                    "mmse_score": 18, "moca_score": 14,
                    "hamilton_depression_score": 8, "updrs_motor_score": 4,
                    "csf_amyloid_beta42_pg_ml": 320,
                    "csf_total_tau_pg_ml": 550,
                    "csf_p_tau181_pg_ml": 72,
                    "csf_nfl_pg_ml": 1400,
                    "blood_nfl_pg_ml": 42.0,
                    "blood_gfap_pg_ml": 220.0,
                    "blood_bdnf_ng_ml": 14.0,
                    "blood_crp_mg_l": 3.5,
                    "blood_il6_pg_ml": 3.0,
                    "eeg_theta_alpha_ratio": 1.10,
                    "eeg_spectral_entropy": 0.55,
                    "eeg_alpha_peak_freq_hz": 8.2,
                    "fmri_bold_dmn": -0.60,
                    "fmri_connectivity_dmn_pfcx": 0.22,
                    "polygenic_risk_score": 1.5
                },
                "treatment": {
                    "drug": "donepezil+memantine", "dose_mg": "10+20",
                    "response": "modest"
                },
                "trajectory_summary": [
                    {"month": 0, "mmse": 18, "csf_ab42": 320, "status": "baseline"},
                    {"month": 12, "mmse": 15, "csf_ab42": 280, "status": "progressive decline"},
                    {"month": 24, "mmse": 11, "csf_ab42": 250, "status": "severe impairment"},
                    {"month": 36, "mmse": 8, "csf_ab42": 230, "status": "advanced disease"}
                ],
                "clinical_narrative": (
                    "76-year-old woman, APOE e4 homozygous, diagnosed 3 years prior. "
                    "MMSE 18 at current evaluation. Pronounced CSF signature: very low "
                    "Abeta42 (320 pg/mL), elevated total tau (550 pg/mL) and p-tau181 "
                    "(72 pg/mL). On combined donepezil 10mg and memantine 20mg with "
                    "modest benefit. Marked EEG slowing and DMN disconnection on fMRI."
                )
            },
            {
                "prototype_id": "AD_SEVERE_003",
                "archetype": "advanced_severe",
                "label": "Advanced Alzheimer's with severe cognitive and functional impairment",
                "demographics": {
                    "age": 82, "sex": "M", "bmi": 21.5,
                    "apoe_e4_carrier": True, "education_years": 10
                },
                "baseline_biomarkers": {
                    "mmse_score": 6, "moca_score": 3,
                    "hamilton_depression_score": 4, "updrs_motor_score": 12,
                    "csf_amyloid_beta42_pg_ml": 220,
                    "csf_total_tau_pg_ml": 780,
                    "csf_p_tau181_pg_ml": 105,
                    "csf_nfl_pg_ml": 2200,
                    "blood_nfl_pg_ml": 65.0,
                    "blood_gfap_pg_ml": 320.0,
                    "blood_bdnf_ng_ml": 10.0,
                    "blood_crp_mg_l": 5.0,
                    "blood_il6_pg_ml": 4.5,
                    "eeg_theta_alpha_ratio": 1.80,
                    "eeg_spectral_entropy": 0.40,
                    "eeg_alpha_peak_freq_hz": 7.0,
                    "fmri_bold_dmn": -0.90,
                    "fmri_connectivity_dmn_pfcx": 0.12,
                    "polygenic_risk_score": 1.8
                },
                "treatment": {
                    "drug": "memantine", "dose_mg": 20, "response": "minimal"
                },
                "trajectory_summary": [
                    {"month": 0, "mmse": 6, "csf_ab42": 220, "status": "baseline severe"},
                    {"month": 12, "mmse": 3, "csf_ab42": 200, "status": "near floor"},
                    {"month": 24, "mmse": 0, "csf_ab42": 190, "status": "end-stage"}
                ],
                "clinical_narrative": (
                    "82-year-old man with 8-year disease course. MMSE 6, fully dependent "
                    "for ADLs. CSF shows maximally abnormal AD signature. Severe EEG "
                    "slowing with theta/alpha ratio of 1.8. Continued on memantine 20mg "
                    "for behavioral symptoms. Low BMI reflects disease-related cachexia."
                )
            },
            {
                "prototype_id": "AD_EARLYONSET_004",
                "archetype": "atypical_early_onset",
                "label": "Early-onset Alzheimer's (<60 years), rapid progression",
                "demographics": {
                    "age": 54, "sex": "M", "bmi": 26.0,
                    "apoe_e4_carrier": True, "education_years": 18
                },
                "baseline_biomarkers": {
                    "mmse_score": 22, "moca_score": 18,
                    "hamilton_depression_score": 12, "updrs_motor_score": 2,
                    "csf_amyloid_beta42_pg_ml": 280,
                    "csf_total_tau_pg_ml": 620,
                    "csf_p_tau181_pg_ml": 85,
                    "csf_nfl_pg_ml": 1600,
                    "blood_nfl_pg_ml": 48.0,
                    "blood_gfap_pg_ml": 200.0,
                    "blood_bdnf_ng_ml": 15.0,
                    "blood_crp_mg_l": 2.0,
                    "blood_il6_pg_ml": 2.5,
                    "eeg_theta_alpha_ratio": 0.90,
                    "eeg_spectral_entropy": 0.58,
                    "eeg_alpha_peak_freq_hz": 8.5,
                    "fmri_bold_dmn": -0.50,
                    "fmri_connectivity_dmn_pfcx": 0.25,
                    "polygenic_risk_score": 2.1
                },
                "treatment": {
                    "drug": "donepezil", "dose_mg": 10, "response": "poor"
                },
                "trajectory_summary": [
                    {"month": 0, "mmse": 22, "csf_ab42": 280, "status": "baseline"},
                    {"month": 12, "mmse": 16, "csf_ab42": 240, "status": "rapid decline"},
                    {"month": 24, "mmse": 10, "csf_ab42": 210, "status": "severe"},
                    {"month": 36, "mmse": 5, "csf_ab42": 195, "status": "end-stage"}
                ],
                "clinical_narrative": (
                    "54-year-old man with early-onset AD, APOE e4 homozygous. Presented "
                    "with executive dysfunction and visuospatial deficits at age 52. "
                    "Unusually aggressive CSF profile for age (very high tau, very low "
                    "Abeta42). High polygenic risk score (2.1 SD). Rapid decline of "
                    "~6 MMSE points/year despite donepezil. Comorbid depression."
                )
            },
            {
                "prototype_id": "AD_TREATMENT_005",
                "archetype": "treatment_responsive",
                "label": "Alzheimer's responding to lecanemab + donepezil combination",
                "demographics": {
                    "age": 70, "sex": "F", "bmi": 25.5,
                    "apoe_e4_carrier": True, "education_years": 14
                },
                "baseline_biomarkers": {
                    "mmse_score": 24, "moca_score": 21,
                    "hamilton_depression_score": 4, "updrs_motor_score": 1,
                    "csf_amyloid_beta42_pg_ml": 400,
                    "csf_total_tau_pg_ml": 380,
                    "csf_p_tau181_pg_ml": 50,
                    "csf_nfl_pg_ml": 1000,
                    "blood_nfl_pg_ml": 30.0,
                    "blood_gfap_pg_ml": 160.0,
                    "blood_bdnf_ng_ml": 17.0,
                    "blood_crp_mg_l": 2.2,
                    "blood_il6_pg_ml": 2.0,
                    "eeg_theta_alpha_ratio": 0.70,
                    "eeg_spectral_entropy": 0.63,
                    "eeg_alpha_peak_freq_hz": 8.8,
                    "fmri_bold_dmn": -0.25,
                    "fmri_connectivity_dmn_pfcx": 0.30,
                    "polygenic_risk_score": 1.0
                },
                "treatment": {
                    "drug": "lecanemab+donepezil",
                    "dose_mg": "10mg/kg biweekly + 10mg daily",
                    "response": "good"
                },
                "trajectory_summary": [
                    {"month": 0, "mmse": 24, "csf_ab42": 400, "status": "baseline"},
                    {"month": 6, "mmse": 24, "csf_ab42": 480, "status": "amyloid clearance"},
                    {"month": 12, "mmse": 23, "csf_ab42": 540, "status": "stabilized"},
                    {"month": 24, "mmse": 22, "csf_ab42": 560, "status": "slowed decline"},
                    {"month": 36, "mmse": 21, "csf_ab42": 570, "status": "sustained benefit"}
                ],
                "clinical_narrative": (
                    "70-year-old woman, APOE e4 heterozygous, enrolled in anti-amyloid "
                    "therapy. Lecanemab 10mg/kg biweekly plus donepezil 10mg daily. "
                    "CSF Abeta42 rose from 400 to 570 pg/mL over 36 months, indicating "
                    "amyloid plaque clearance. MMSE decline slowed to ~1 point/year "
                    "(vs expected 2.5/year). ARIA monitoring negative."
                )
            },
        ],

        "parkinsons_disease": [
            {
                "prototype_id": "PD_TREMOR_001",
                "archetype": "early_tremor_dominant",
                "label": "Early Parkinson's, tremor-dominant, good levodopa response",
                "demographics": {
                    "age": 62, "sex": "M", "bmi": 26.0,
                    "apoe_e4_carrier": False, "education_years": 14
                },
                "baseline_biomarkers": {
                    "mmse_score": 28, "moca_score": 26,
                    "hamilton_depression_score": 6, "updrs_motor_score": 22,
                    "csf_amyloid_beta42_pg_ml": 950,
                    "csf_total_tau_pg_ml": 240,
                    "csf_p_tau181_pg_ml": 24,
                    "csf_nfl_pg_ml": 800,
                    "blood_nfl_pg_ml": 20.0,
                    "blood_gfap_pg_ml": 100.0,
                    "blood_bdnf_ng_ml": 18.0,
                    "blood_crp_mg_l": 2.0,
                    "blood_il6_pg_ml": 1.8,
                    "eeg_theta_alpha_ratio": 0.50,
                    "eeg_spectral_entropy": 0.68,
                    "eeg_alpha_peak_freq_hz": 9.5,
                    "fmri_bold_dmn": -0.10,
                    "fmri_bold_basal_ganglia": -0.40,
                    "fmri_connectivity_motor_bg": 0.28,
                    "polygenic_risk_score": 0.8
                },
                "treatment": {
                    "drug": "levodopa_carbidopa", "dose_mg": "100/25 TID",
                    "response": "excellent"
                },
                "trajectory_summary": [
                    {"month": 0, "updrs": 22, "blood_nfl": 20.0, "status": "baseline"},
                    {"month": 12, "updrs": 18, "blood_nfl": 22.0, "status": "on treatment, improved"},
                    {"month": 24, "updrs": 22, "blood_nfl": 24.0, "status": "wearing off emerging"},
                    {"month": 48, "updrs": 30, "blood_nfl": 28.0, "status": "motor fluctuations"}
                ],
                "clinical_narrative": (
                    "62-year-old man presenting with right-hand resting tremor for 18 "
                    "months. UPDRS motor score 22. Excellent initial response to "
                    "levodopa/carbidopa 100/25mg TID with UPDRS improving to 18. "
                    "Cognitive function preserved. Mild basal ganglia hypoactivation on "
                    "fMRI. After 2 years, beginning to develop end-of-dose wearing off."
                )
            },
            {
                "prototype_id": "PD_RIGID_002",
                "archetype": "akinetic_rigid",
                "label": "Akinetic-rigid Parkinson's with gait instability",
                "demographics": {
                    "age": 71, "sex": "M", "bmi": 24.5,
                    "apoe_e4_carrier": False, "education_years": 12
                },
                "baseline_biomarkers": {
                    "mmse_score": 26, "moca_score": 23,
                    "hamilton_depression_score": 10, "updrs_motor_score": 38,
                    "csf_amyloid_beta42_pg_ml": 880,
                    "csf_total_tau_pg_ml": 300,
                    "csf_p_tau181_pg_ml": 28,
                    "csf_nfl_pg_ml": 1100,
                    "blood_nfl_pg_ml": 32.0,
                    "blood_gfap_pg_ml": 130.0,
                    "blood_bdnf_ng_ml": 14.0,
                    "blood_crp_mg_l": 2.8,
                    "blood_il6_pg_ml": 2.5,
                    "eeg_theta_alpha_ratio": 0.60,
                    "eeg_spectral_entropy": 0.62,
                    "eeg_alpha_peak_freq_hz": 9.0,
                    "fmri_bold_dmn": -0.20,
                    "fmri_bold_basal_ganglia": -0.55,
                    "fmri_connectivity_motor_bg": 0.20,
                    "polygenic_risk_score": 0.9
                },
                "treatment": {
                    "drug": "levodopa_carbidopa+pramipexole",
                    "dose_mg": "200/50 TID + 1.0mg TID",
                    "response": "moderate"
                },
                "trajectory_summary": [
                    {"month": 0, "updrs": 38, "blood_nfl": 32.0, "status": "baseline"},
                    {"month": 12, "updrs": 32, "blood_nfl": 34.0, "status": "on treatment"},
                    {"month": 24, "updrs": 38, "blood_nfl": 37.0, "status": "declining"},
                    {"month": 48, "updrs": 50, "blood_nfl": 42.0, "status": "advanced"}
                ],
                "clinical_narrative": (
                    "71-year-old man with 5-year disease course, predominantly "
                    "akinetic-rigid phenotype with postural instability. UPDRS motor "
                    "38. On combined levodopa and pramipexole with moderate benefit. "
                    "Mild cognitive impairment (MoCA 23). Elevated NfL suggesting "
                    "neurodegeneration. Comorbid depression (Hamilton 10)."
                )
            },
            {
                "prototype_id": "PD_ADVANCED_003",
                "archetype": "advanced_motor_fluctuations",
                "label": "Advanced Parkinson's with motor fluctuations and dyskinesia",
                "demographics": {
                    "age": 74, "sex": "F", "bmi": 22.0,
                    "apoe_e4_carrier": False, "education_years": 12
                },
                "baseline_biomarkers": {
                    "mmse_score": 23, "moca_score": 19,
                    "hamilton_depression_score": 14, "updrs_motor_score": 55,
                    "csf_amyloid_beta42_pg_ml": 800,
                    "csf_total_tau_pg_ml": 350,
                    "csf_p_tau181_pg_ml": 32,
                    "csf_nfl_pg_ml": 1500,
                    "blood_nfl_pg_ml": 45.0,
                    "blood_gfap_pg_ml": 170.0,
                    "blood_bdnf_ng_ml": 12.0,
                    "blood_crp_mg_l": 3.5,
                    "blood_il6_pg_ml": 3.2,
                    "eeg_theta_alpha_ratio": 0.75,
                    "eeg_spectral_entropy": 0.58,
                    "eeg_alpha_peak_freq_hz": 8.5,
                    "fmri_bold_dmn": -0.35,
                    "fmri_bold_basal_ganglia": -0.70,
                    "fmri_connectivity_motor_bg": 0.15,
                    "polygenic_risk_score": 0.7
                },
                "treatment": {
                    "drug": "levodopa_carbidopa",
                    "dose_mg": "200/50 QID + entacapone",
                    "response": "complicated"
                },
                "trajectory_summary": [
                    {"month": 0, "updrs": 55, "blood_nfl": 45.0, "status": "baseline"},
                    {"month": 12, "updrs": 60, "blood_nfl": 48.0, "status": "worsening"},
                    {"month": 24, "updrs": 68, "blood_nfl": 52.0, "status": "advanced"}
                ],
                "clinical_narrative": (
                    "74-year-old woman with 12-year disease course. UPDRS motor 55. "
                    "Prominent motor fluctuations with 3-hour wearing-off periods and "
                    "peak-dose dyskinesias. On high-dose levodopa (800mg/day) with "
                    "entacapone. Emerging dementia (MMSE 23, MoCA 19). Significant "
                    "depression. Being considered for deep brain stimulation."
                )
            },
            {
                "prototype_id": "PD_YOUNG_004",
                "archetype": "atypical_young_onset",
                "label": "Young-onset Parkinson's (<50 years), slow progression",
                "demographics": {
                    "age": 44, "sex": "M", "bmi": 25.0,
                    "apoe_e4_carrier": False, "education_years": 16
                },
                "baseline_biomarkers": {
                    "mmse_score": 29, "moca_score": 28,
                    "hamilton_depression_score": 8, "updrs_motor_score": 15,
                    "csf_amyloid_beta42_pg_ml": 1050,
                    "csf_total_tau_pg_ml": 200,
                    "csf_p_tau181_pg_ml": 20,
                    "csf_nfl_pg_ml": 650,
                    "blood_nfl_pg_ml": 15.0,
                    "blood_gfap_pg_ml": 75.0,
                    "blood_bdnf_ng_ml": 20.0,
                    "blood_crp_mg_l": 1.5,
                    "blood_il6_pg_ml": 1.2,
                    "eeg_theta_alpha_ratio": 0.42,
                    "eeg_spectral_entropy": 0.72,
                    "eeg_alpha_peak_freq_hz": 10.0,
                    "fmri_bold_dmn": -0.05,
                    "fmri_bold_basal_ganglia": -0.30,
                    "fmri_connectivity_motor_bg": 0.32,
                    "polygenic_risk_score": 1.3
                },
                "treatment": {
                    "drug": "pramipexole", "dose_mg": "1.5mg TID",
                    "response": "good"
                },
                "trajectory_summary": [
                    {"month": 0, "updrs": 15, "blood_nfl": 15.0, "status": "baseline"},
                    {"month": 12, "updrs": 12, "blood_nfl": 16.0, "status": "on agonist"},
                    {"month": 24, "updrs": 16, "blood_nfl": 17.0, "status": "slow progression"},
                    {"month": 48, "updrs": 20, "blood_nfl": 19.0, "status": "stable"}
                ],
                "clinical_narrative": (
                    "44-year-old man with young-onset PD diagnosed at age 42. Started "
                    "on pramipexole to delay levodopa initiation. UPDRS motor 15, "
                    "with good response to dopamine agonist. Cognition fully preserved. "
                    "Slower progression than typical late-onset PD. Higher genetic risk "
                    "score (PRS 1.3 SD). Anxiety and mild depression managed with SSRI."
                )
            },
        ],

        "major_depressive_disorder": [
            {
                "prototype_id": "MDD_FIRST_001",
                "archetype": "first_episode_moderate",
                "label": "First depressive episode, moderate severity",
                "demographics": {
                    "age": 34, "sex": "F", "bmi": 27.0,
                    "apoe_e4_carrier": False, "education_years": 16
                },
                "baseline_biomarkers": {
                    "mmse_score": 28, "moca_score": 26,
                    "hamilton_depression_score": 20, "updrs_motor_score": 0,
                    "csf_amyloid_beta42_pg_ml": 1080,
                    "csf_total_tau_pg_ml": 200,
                    "csf_p_tau181_pg_ml": 20,
                    "csf_nfl_pg_ml": 500,
                    "blood_nfl_pg_ml": 10.0,
                    "blood_gfap_pg_ml": 70.0,
                    "blood_bdnf_ng_ml": 15.0,
                    "blood_crp_mg_l": 2.5,
                    "blood_il6_pg_ml": 2.8,
                    "eeg_theta_alpha_ratio": 0.50,
                    "eeg_spectral_entropy": 0.68,
                    "eeg_alpha_peak_freq_hz": 9.5,
                    "fmri_bold_dmn": 0.30,
                    "fmri_bold_amygdala": 0.45,
                    "fmri_connectivity_dmn_pfcx": 0.50,
                    "polygenic_risk_score": 0.5
                },
                "treatment": {
                    "drug": "fluoxetine", "dose_mg": 20, "response": "good"
                },
                "trajectory_summary": [
                    {"month": 0, "hamilton": 20, "bdnf": 15.0, "status": "acute episode"},
                    {"month": 2, "hamilton": 14, "bdnf": 17.0, "status": "partial response"},
                    {"month": 6, "hamilton": 6, "bdnf": 22.0, "status": "remission"},
                    {"month": 12, "hamilton": 4, "bdnf": 24.0, "status": "sustained remission"}
                ],
                "clinical_narrative": (
                    "34-year-old woman with first major depressive episode following job "
                    "loss. Hamilton score 20 (moderate). Elevated inflammatory markers "
                    "(CRP 2.5, IL-6 2.8). Reduced BDNF (15 ng/mL). fMRI shows DMN "
                    "hyperactivity and amygdala hyperreactivity. Started on fluoxetine "
                    "20mg, achieving remission by month 6 with BDNF normalization."
                )
            },
            {
                "prototype_id": "MDD_RECURRENT_002",
                "archetype": "recurrent_severe",
                "label": "Recurrent severe depression, multiple episodes",
                "demographics": {
                    "age": 52, "sex": "F", "bmi": 30.0,
                    "apoe_e4_carrier": False, "education_years": 14
                },
                "baseline_biomarkers": {
                    "mmse_score": 26, "moca_score": 24,
                    "hamilton_depression_score": 28, "updrs_motor_score": 1,
                    "csf_amyloid_beta42_pg_ml": 1020,
                    "csf_total_tau_pg_ml": 230,
                    "csf_p_tau181_pg_ml": 22,
                    "csf_nfl_pg_ml": 600,
                    "blood_nfl_pg_ml": 14.0,
                    "blood_gfap_pg_ml": 95.0,
                    "blood_bdnf_ng_ml": 10.0,
                    "blood_crp_mg_l": 4.5,
                    "blood_il6_pg_ml": 4.0,
                    "eeg_theta_alpha_ratio": 0.55,
                    "eeg_spectral_entropy": 0.62,
                    "eeg_alpha_peak_freq_hz": 9.2,
                    "fmri_bold_dmn": 0.50,
                    "fmri_bold_amygdala": 0.60,
                    "fmri_connectivity_dmn_pfcx": 0.55,
                    "polygenic_risk_score": 1.0
                },
                "treatment": {
                    "drug": "venlafaxine", "dose_mg": 225, "response": "partial"
                },
                "trajectory_summary": [
                    {"month": 0, "hamilton": 28, "crp": 4.5, "status": "severe episode"},
                    {"month": 3, "hamilton": 18, "crp": 3.5, "status": "partial response"},
                    {"month": 9, "hamilton": 10, "crp": 2.5, "status": "improved"},
                    {"month": 18, "hamilton": 22, "crp": 4.0, "status": "relapse"},
                    {"month": 24, "hamilton": 12, "crp": 3.0, "status": "re-stabilized"}
                ],
                "clinical_narrative": (
                    "52-year-old woman with fourth depressive episode over 20 years. "
                    "Hamilton score 28 (severe). Elevated inflammatory burden (CRP 4.5, "
                    "IL-6 4.0). Very low BDNF (10 ng/mL). Suboptimal response to "
                    "venlafaxine 225mg, requiring augmentation. Relapse at 18 months "
                    "despite maintenance therapy. Mild cognitive complaints (MoCA 24)."
                )
            },
            {
                "prototype_id": "MDD_RESISTANT_003",
                "archetype": "treatment_resistant",
                "label": "Treatment-resistant depression, failed multiple trials",
                "demographics": {
                    "age": 45, "sex": "M", "bmi": 32.0,
                    "apoe_e4_carrier": False, "education_years": 12
                },
                "baseline_biomarkers": {
                    "mmse_score": 27, "moca_score": 25,
                    "hamilton_depression_score": 32, "updrs_motor_score": 0,
                    "csf_amyloid_beta42_pg_ml": 1050,
                    "csf_total_tau_pg_ml": 210,
                    "csf_p_tau181_pg_ml": 21,
                    "csf_nfl_pg_ml": 550,
                    "blood_nfl_pg_ml": 12.0,
                    "blood_gfap_pg_ml": 85.0,
                    "blood_bdnf_ng_ml": 8.0,
                    "blood_crp_mg_l": 6.0,
                    "blood_il6_pg_ml": 5.5,
                    "eeg_theta_alpha_ratio": 0.58,
                    "eeg_spectral_entropy": 0.58,
                    "eeg_alpha_peak_freq_hz": 9.0,
                    "fmri_bold_dmn": 0.65,
                    "fmri_bold_amygdala": 0.70,
                    "fmri_connectivity_dmn_pfcx": 0.58,
                    "polygenic_risk_score": 1.4
                },
                "treatment": {
                    "drug": "venlafaxine+lithium augmentation",
                    "dose_mg": "225+900",
                    "response": "poor"
                },
                "trajectory_summary": [
                    {"month": 0, "hamilton": 32, "bdnf": 8.0, "status": "treatment resistant"},
                    {"month": 6, "hamilton": 26, "bdnf": 9.0, "status": "minimal response"},
                    {"month": 12, "hamilton": 28, "bdnf": 8.5, "status": "persistent"},
                    {"month": 24, "hamilton": 24, "bdnf": 10.0, "status": "partial improvement"}
                ],
                "clinical_narrative": (
                    "45-year-old man with treatment-resistant depression, failed 3 "
                    "adequate SSRI/SNRI trials. Hamilton 32 (very severe). Markedly "
                    "elevated CRP (6.0) and IL-6 (5.5), very low BDNF (8 ng/mL). "
                    "High genetic loading (PRS 1.4 SD). On venlafaxine 225mg with "
                    "lithium augmentation, showing only modest improvement. Being "
                    "evaluated for neuromodulation (rTMS/ECT)."
                )
            },
            {
                "prototype_id": "MDD_RESPONSIVE_004",
                "archetype": "treatment_responsive",
                "label": "SSRI-responsive depression with full remission",
                "demographics": {
                    "age": 28, "sex": "F", "bmi": 23.0,
                    "apoe_e4_carrier": False, "education_years": 18
                },
                "baseline_biomarkers": {
                    "mmse_score": 29, "moca_score": 28,
                    "hamilton_depression_score": 22, "updrs_motor_score": 0,
                    "csf_amyloid_beta42_pg_ml": 1150,
                    "csf_total_tau_pg_ml": 180,
                    "csf_p_tau181_pg_ml": 17,
                    "csf_nfl_pg_ml": 420,
                    "blood_nfl_pg_ml": 8.0,
                    "blood_gfap_pg_ml": 60.0,
                    "blood_bdnf_ng_ml": 14.0,
                    "blood_crp_mg_l": 1.8,
                    "blood_il6_pg_ml": 2.0,
                    "eeg_theta_alpha_ratio": 0.48,
                    "eeg_spectral_entropy": 0.70,
                    "eeg_alpha_peak_freq_hz": 9.8,
                    "fmri_bold_dmn": 0.25,
                    "fmri_bold_amygdala": 0.40,
                    "fmri_connectivity_dmn_pfcx": 0.48,
                    "polygenic_risk_score": 0.3
                },
                "treatment": {
                    "drug": "fluoxetine", "dose_mg": 20, "response": "excellent"
                },
                "trajectory_summary": [
                    {"month": 0, "hamilton": 22, "bdnf": 14.0, "status": "acute episode"},
                    {"month": 1, "hamilton": 16, "bdnf": 16.0, "status": "early response"},
                    {"month": 3, "hamilton": 6, "bdnf": 22.0, "status": "remission"},
                    {"month": 12, "hamilton": 3, "bdnf": 26.0, "status": "sustained well-being"}
                ],
                "clinical_narrative": (
                    "28-year-old woman with moderate depression (Hamilton 22) following "
                    "relationship dissolution. Low inflammatory markers, mildly reduced "
                    "BDNF. Rapid response to fluoxetine 20mg with remission by week 8. "
                    "BDNF normalized to 26 ng/mL. fMRI DMN hyperactivity resolved. "
                    "Maintained on fluoxetine for 12 months, then successfully tapered."
                )
            },
        ],

        "epilepsy": [
            {
                "prototype_id": "EP_CONTROLLED_001",
                "archetype": "well_controlled_focal",
                "label": "Well-controlled focal epilepsy on levetiracetam",
                "demographics": {
                    "age": 35, "sex": "M", "bmi": 25.0,
                    "apoe_e4_carrier": False, "education_years": 16
                },
                "baseline_biomarkers": {
                    "mmse_score": 29, "moca_score": 28,
                    "hamilton_depression_score": 4, "updrs_motor_score": 0,
                    "csf_amyloid_beta42_pg_ml": 1100,
                    "csf_total_tau_pg_ml": 190,
                    "csf_p_tau181_pg_ml": 18,
                    "csf_nfl_pg_ml": 500,
                    "blood_nfl_pg_ml": 10.0,
                    "blood_gfap_pg_ml": 70.0,
                    "blood_bdnf_ng_ml": 22.0,
                    "blood_crp_mg_l": 1.2,
                    "blood_il6_pg_ml": 1.0,
                    "eeg_theta_alpha_ratio": 0.55,
                    "eeg_spectral_entropy": 0.60,
                    "eeg_alpha_peak_freq_hz": 9.8,
                    "fmri_bold_dmn": 0.00,
                    "polygenic_risk_score": 0.6
                },
                "treatment": {
                    "drug": "levetiracetam", "dose_mg": "1000mg BID",
                    "response": "seizure-free"
                },
                "trajectory_summary": [
                    {"month": 0, "seizures_per_month": 3, "status": "baseline"},
                    {"month": 3, "seizures_per_month": 1, "status": "improving"},
                    {"month": 6, "seizures_per_month": 0, "status": "seizure-free"},
                    {"month": 24, "seizures_per_month": 0, "status": "sustained control"}
                ],
                "clinical_narrative": (
                    "35-year-old man with focal epilepsy originating from left temporal "
                    "lobe, diagnosed at age 28. Seizure-free on levetiracetam 1000mg "
                    "BID for 18 months. EEG shows mildly elevated theta activity but "
                    "no epileptiform discharges. Cognition preserved. Low inflammatory "
                    "burden."
                )
            },
            {
                "prototype_id": "EP_RESISTANT_002",
                "archetype": "drug_resistant_temporal",
                "label": "Drug-resistant temporal lobe epilepsy, surgical candidate",
                "demographics": {
                    "age": 42, "sex": "F", "bmi": 28.0,
                    "apoe_e4_carrier": False, "education_years": 12
                },
                "baseline_biomarkers": {
                    "mmse_score": 26, "moca_score": 23,
                    "hamilton_depression_score": 14, "updrs_motor_score": 0,
                    "csf_amyloid_beta42_pg_ml": 1000,
                    "csf_total_tau_pg_ml": 260,
                    "csf_p_tau181_pg_ml": 25,
                    "csf_nfl_pg_ml": 750,
                    "blood_nfl_pg_ml": 18.0,
                    "blood_gfap_pg_ml": 110.0,
                    "blood_bdnf_ng_ml": 16.0,
                    "blood_crp_mg_l": 2.5,
                    "blood_il6_pg_ml": 2.2,
                    "eeg_theta_alpha_ratio": 0.80,
                    "eeg_spectral_entropy": 0.50,
                    "eeg_alpha_peak_freq_hz": 8.8,
                    "fmri_bold_dmn": -0.15,
                    "polygenic_risk_score": 1.2
                },
                "treatment": {
                    "drug": "levetiracetam+carbamazepine",
                    "dose_mg": "1500mg BID + 400mg BID",
                    "response": "poor"
                },
                "trajectory_summary": [
                    {"month": 0, "seizures_per_month": 8, "status": "uncontrolled"},
                    {"month": 6, "seizures_per_month": 6, "status": "modest reduction"},
                    {"month": 12, "seizures_per_month": 7, "status": "drug resistant"},
                    {"month": 24, "seizures_per_month": 5, "status": "pre-surgical eval"}
                ],
                "clinical_narrative": (
                    "42-year-old woman with temporal lobe epilepsy since age 15, "
                    "refractory to 4 AED trials. 6-8 seizures/month on dual therapy. "
                    "EEG shows bilateral temporal spikes. Mild cognitive decline (MoCA "
                    "23), comorbid depression (Hamilton 14). Elevated NfL (18 pg/mL) "
                    "suggesting seizure-related neuronal injury. Undergoing pre-surgical "
                    "evaluation for temporal lobectomy."
                )
            },
            {
                "prototype_id": "EP_GENERALIZED_003",
                "archetype": "generalized_epilepsy",
                "label": "Idiopathic generalized epilepsy, juvenile myoclonic type",
                "demographics": {
                    "age": 22, "sex": "F", "bmi": 22.0,
                    "apoe_e4_carrier": False, "education_years": 14
                },
                "baseline_biomarkers": {
                    "mmse_score": 30, "moca_score": 29,
                    "hamilton_depression_score": 3, "updrs_motor_score": 0,
                    "csf_amyloid_beta42_pg_ml": 1150,
                    "csf_total_tau_pg_ml": 170,
                    "csf_p_tau181_pg_ml": 16,
                    "csf_nfl_pg_ml": 420,
                    "blood_nfl_pg_ml": 8.0,
                    "blood_gfap_pg_ml": 55.0,
                    "blood_bdnf_ng_ml": 24.0,
                    "blood_crp_mg_l": 0.8,
                    "blood_il6_pg_ml": 0.7,
                    "eeg_theta_alpha_ratio": 0.45,
                    "eeg_spectral_entropy": 0.55,
                    "eeg_alpha_peak_freq_hz": 10.0,
                    "fmri_bold_dmn": 0.05,
                    "polygenic_risk_score": 0.8
                },
                "treatment": {
                    "drug": "levetiracetam", "dose_mg": "750mg BID",
                    "response": "good"
                },
                "trajectory_summary": [
                    {"month": 0, "seizures_per_month": 4, "status": "baseline"},
                    {"month": 3, "seizures_per_month": 1, "status": "controlled"},
                    {"month": 12, "seizures_per_month": 0, "status": "seizure-free"},
                    {"month": 24, "seizures_per_month": 0, "status": "maintained"}
                ],
                "clinical_narrative": (
                    "22-year-old woman with juvenile myoclonic epilepsy diagnosed at "
                    "age 14. Morning myoclonic jerks and occasional generalized "
                    "tonic-clonic seizures triggered by sleep deprivation. EEG shows "
                    "generalized 4-6 Hz polyspike-wave. Well-controlled on "
                    "levetiracetam 750mg BID. Normal cognition and biomarkers."
                )
            },
        ],

        "multiple_sclerosis": [
            {
                "prototype_id": "MS_RRMS_001",
                "archetype": "relapsing_remitting_early",
                "label": "Early relapsing-remitting MS, first demyelinating event",
                "demographics": {
                    "age": 30, "sex": "F", "bmi": 24.0,
                    "apoe_e4_carrier": False, "education_years": 16
                },
                "baseline_biomarkers": {
                    "mmse_score": 29, "moca_score": 27,
                    "hamilton_depression_score": 8, "updrs_motor_score": 2,
                    "csf_amyloid_beta42_pg_ml": 1080,
                    "csf_total_tau_pg_ml": 210,
                    "csf_p_tau181_pg_ml": 20,
                    "csf_nfl_pg_ml": 900,
                    "blood_nfl_pg_ml": 22.0,
                    "blood_gfap_pg_ml": 120.0,
                    "blood_bdnf_ng_ml": 18.0,
                    "blood_crp_mg_l": 3.0,
                    "blood_il6_pg_ml": 3.5,
                    "eeg_theta_alpha_ratio": 0.48,
                    "eeg_spectral_entropy": 0.68,
                    "eeg_alpha_peak_freq_hz": 9.5,
                    "fmri_bold_dmn": -0.10,
                    "polygenic_risk_score": 1.0
                },
                "treatment": {
                    "drug": "ocrelizumab", "dose_mg": "600mg IV q6months",
                    "response": "good"
                },
                "trajectory_summary": [
                    {"month": 0, "relapses_ytd": 2, "blood_nfl": 22.0, "status": "active RRMS"},
                    {"month": 6, "relapses_ytd": 0, "blood_nfl": 16.0, "status": "on treatment"},
                    {"month": 12, "relapses_ytd": 0, "blood_nfl": 12.0, "status": "NEDA"},
                    {"month": 24, "relapses_ytd": 0, "blood_nfl": 11.0, "status": "sustained NEDA"}
                ],
                "clinical_narrative": (
                    "30-year-old woman presenting with optic neuritis and subsequent "
                    "brain MRI showing 12 T2 lesions. CSF oligoclonal bands positive. "
                    "2 relapses in first year. NfL elevated at 22 pg/mL. Started on "
                    "ocrelizumab 600mg IV q6months, achieving no evidence of disease "
                    "activity (NEDA) by month 12 with NfL normalization."
                )
            },
            {
                "prototype_id": "MS_SPMS_002",
                "archetype": "secondary_progressive",
                "label": "Secondary progressive MS with accumulating disability",
                "demographics": {
                    "age": 52, "sex": "F", "bmi": 26.0,
                    "apoe_e4_carrier": False, "education_years": 14
                },
                "baseline_biomarkers": {
                    "mmse_score": 26, "moca_score": 23,
                    "hamilton_depression_score": 12, "updrs_motor_score": 5,
                    "csf_amyloid_beta42_pg_ml": 1000,
                    "csf_total_tau_pg_ml": 280,
                    "csf_p_tau181_pg_ml": 26,
                    "csf_nfl_pg_ml": 1200,
                    "blood_nfl_pg_ml": 35.0,
                    "blood_gfap_pg_ml": 200.0,
                    "blood_bdnf_ng_ml": 14.0,
                    "blood_crp_mg_l": 3.5,
                    "blood_il6_pg_ml": 3.8,
                    "eeg_theta_alpha_ratio": 0.55,
                    "eeg_spectral_entropy": 0.62,
                    "eeg_alpha_peak_freq_hz": 9.0,
                    "fmri_bold_dmn": -0.25,
                    "polygenic_risk_score": 1.2
                },
                "treatment": {
                    "drug": "ocrelizumab",
                    "dose_mg": "600mg IV q6months",
                    "response": "partial"
                },
                "trajectory_summary": [
                    {"month": 0, "edss": 4.5, "blood_nfl": 35.0, "status": "SPMS baseline"},
                    {"month": 12, "edss": 5.0, "blood_nfl": 30.0, "status": "slow progression"},
                    {"month": 24, "edss": 5.5, "blood_nfl": 28.0, "status": "progressive"},
                    {"month": 48, "edss": 6.0, "blood_nfl": 26.0, "status": "requires cane"}
                ],
                "clinical_narrative": (
                    "52-year-old woman with RRMS diagnosed at age 28, transitioned to "
                    "SPMS 5 years ago. EDSS 4.5 with progressive gait difficulty. "
                    "Elevated GFAP (200 pg/mL) indicating astrocytic damage. NfL 35 "
                    "pg/mL. On ocrelizumab with reduced relapse frequency but "
                    "accumulating disability. Cognitive slowing (MoCA 23). Depression "
                    "managed with SSRI."
                )
            },
            {
                "prototype_id": "MS_RESPONSIVE_003",
                "archetype": "treatment_responsive",
                "label": "Highly active RRMS with excellent treatment response",
                "demographics": {
                    "age": 26, "sex": "F", "bmi": 22.5,
                    "apoe_e4_carrier": False, "education_years": 18
                },
                "baseline_biomarkers": {
                    "mmse_score": 30, "moca_score": 28,
                    "hamilton_depression_score": 6, "updrs_motor_score": 1,
                    "csf_amyloid_beta42_pg_ml": 1100,
                    "csf_total_tau_pg_ml": 200,
                    "csf_p_tau181_pg_ml": 19,
                    "csf_nfl_pg_ml": 1100,
                    "blood_nfl_pg_ml": 28.0,
                    "blood_gfap_pg_ml": 140.0,
                    "blood_bdnf_ng_ml": 20.0,
                    "blood_crp_mg_l": 2.8,
                    "blood_il6_pg_ml": 3.2,
                    "eeg_theta_alpha_ratio": 0.42,
                    "eeg_spectral_entropy": 0.72,
                    "eeg_alpha_peak_freq_hz": 10.0,
                    "fmri_bold_dmn": 0.00,
                    "polygenic_risk_score": 0.9
                },
                "treatment": {
                    "drug": "ocrelizumab", "dose_mg": "600mg IV q6months",
                    "response": "excellent"
                },
                "trajectory_summary": [
                    {"month": 0, "relapses_ytd": 3, "blood_nfl": 28.0, "status": "highly active"},
                    {"month": 6, "relapses_ytd": 0, "blood_nfl": 14.0, "status": "rapid response"},
                    {"month": 12, "relapses_ytd": 0, "blood_nfl": 10.0, "status": "NEDA"},
                    {"month": 36, "relapses_ytd": 0, "blood_nfl": 9.0, "status": "sustained NEDA"}
                ],
                "clinical_narrative": (
                    "26-year-old woman with highly active RRMS: 3 relapses and 8 new "
                    "T2 lesions in first year. Rapidly escalated to ocrelizumab. NfL "
                    "dropped from 28 to 10 pg/mL within 12 months. Achieved sustained "
                    "NEDA over 3 years. Full cognitive recovery. No disability "
                    "accumulation."
                )
            },
        ],

        "amyotrophic_lateral_sclerosis": [
            {
                "prototype_id": "ALS_LIMB_001",
                "archetype": "limb_onset_typical",
                "label": "Limb-onset ALS, typical progression",
                "demographics": {
                    "age": 58, "sex": "M", "bmi": 25.0,
                    "apoe_e4_carrier": False, "education_years": 14
                },
                "baseline_biomarkers": {
                    "mmse_score": 28, "moca_score": 26,
                    "hamilton_depression_score": 10, "updrs_motor_score": 8,
                    "csf_amyloid_beta42_pg_ml": 1050,
                    "csf_total_tau_pg_ml": 250,
                    "csf_p_tau181_pg_ml": 24,
                    "csf_nfl_pg_ml": 3500,
                    "blood_nfl_pg_ml": 80.0,
                    "blood_gfap_pg_ml": 180.0,
                    "blood_bdnf_ng_ml": 12.0,
                    "blood_crp_mg_l": 3.0,
                    "blood_il6_pg_ml": 3.5,
                    "eeg_theta_alpha_ratio": 0.45,
                    "eeg_spectral_entropy": 0.68,
                    "eeg_alpha_peak_freq_hz": 9.5,
                    "fmri_bold_dmn": -0.05,
                    "fmri_bold_motor_cortex": -0.40,
                    "polygenic_risk_score": 0.8
                },
                "treatment": {
                    "drug": "riluzole", "dose_mg": "50mg BID",
                    "response": "modest"
                },
                "trajectory_summary": [
                    {"month": 0, "blood_nfl": 80.0, "alsfrs_r_equiv": 40, "status": "baseline"},
                    {"month": 6, "blood_nfl": 90.0, "alsfrs_r_equiv": 36, "status": "declining"},
                    {"month": 12, "blood_nfl": 100.0, "alsfrs_r_equiv": 32, "status": "progressive"},
                    {"month": 24, "blood_nfl": 120.0, "alsfrs_r_equiv": 22, "status": "advanced"}
                ],
                "clinical_narrative": (
                    "58-year-old man with limb-onset ALS presenting with right hand "
                    "weakness and fasciculations. EMG confirms widespread denervation. "
                    "Blood NfL markedly elevated at 80 pg/mL, consistent with rapid "
                    "motor neuron loss. CSF NfL extremely high (3500 pg/mL). On riluzole "
                    "50mg BID. Progressive decline over 2 years with loss of hand "
                    "function and emerging respiratory compromise."
                )
            },
            {
                "prototype_id": "ALS_BULBAR_002",
                "archetype": "bulbar_onset_rapid",
                "label": "Bulbar-onset ALS, rapid progression",
                "demographics": {
                    "age": 65, "sex": "F", "bmi": 22.0,
                    "apoe_e4_carrier": False, "education_years": 12
                },
                "baseline_biomarkers": {
                    "mmse_score": 27, "moca_score": 24,
                    "hamilton_depression_score": 15, "updrs_motor_score": 10,
                    "csf_amyloid_beta42_pg_ml": 1020,
                    "csf_total_tau_pg_ml": 270,
                    "csf_p_tau181_pg_ml": 26,
                    "csf_nfl_pg_ml": 5000,
                    "blood_nfl_pg_ml": 120.0,
                    "blood_gfap_pg_ml": 250.0,
                    "blood_bdnf_ng_ml": 9.0,
                    "blood_crp_mg_l": 4.0,
                    "blood_il6_pg_ml": 4.5,
                    "eeg_theta_alpha_ratio": 0.50,
                    "eeg_spectral_entropy": 0.64,
                    "eeg_alpha_peak_freq_hz": 9.2,
                    "fmri_bold_dmn": -0.10,
                    "fmri_bold_motor_cortex": -0.55,
                    "polygenic_risk_score": 0.6
                },
                "treatment": {
                    "drug": "riluzole+edaravone",
                    "dose_mg": "50mg BID + 60mg IV cycles",
                    "response": "minimal"
                },
                "trajectory_summary": [
                    {"month": 0, "blood_nfl": 120.0, "alsfrs_r_equiv": 35, "status": "baseline"},
                    {"month": 6, "blood_nfl": 150.0, "alsfrs_r_equiv": 26, "status": "rapid decline"},
                    {"month": 12, "blood_nfl": 180.0, "alsfrs_r_equiv": 18, "status": "severe"}
                ],
                "clinical_narrative": (
                    "65-year-old woman with bulbar-onset ALS presenting with dysarthria "
                    "and dysphagia. Blood NfL 120 pg/mL — among the highest levels "
                    "seen, predicting rapid course. On riluzole and edaravone infusion "
                    "cycles with minimal benefit. PEG placed at 6 months. Progressive "
                    "respiratory failure. Cognitive screening shows mild frontal "
                    "dysfunction (MoCA 24)."
                )
            },
            {
                "prototype_id": "ALS_SLOW_003",
                "archetype": "slow_progressor",
                "label": "Slow-progressing ALS on riluzole, maintained function",
                "demographics": {
                    "age": 52, "sex": "M", "bmi": 27.0,
                    "apoe_e4_carrier": False, "education_years": 16
                },
                "baseline_biomarkers": {
                    "mmse_score": 29, "moca_score": 27,
                    "hamilton_depression_score": 6, "updrs_motor_score": 5,
                    "csf_amyloid_beta42_pg_ml": 1080,
                    "csf_total_tau_pg_ml": 220,
                    "csf_p_tau181_pg_ml": 21,
                    "csf_nfl_pg_ml": 1800,
                    "blood_nfl_pg_ml": 45.0,
                    "blood_gfap_pg_ml": 120.0,
                    "blood_bdnf_ng_ml": 16.0,
                    "blood_crp_mg_l": 2.0,
                    "blood_il6_pg_ml": 2.0,
                    "eeg_theta_alpha_ratio": 0.42,
                    "eeg_spectral_entropy": 0.70,
                    "eeg_alpha_peak_freq_hz": 9.8,
                    "fmri_bold_dmn": 0.00,
                    "fmri_bold_motor_cortex": -0.25,
                    "polygenic_risk_score": 0.4
                },
                "treatment": {
                    "drug": "riluzole", "dose_mg": "50mg BID",
                    "response": "beneficial"
                },
                "trajectory_summary": [
                    {"month": 0, "blood_nfl": 45.0, "alsfrs_r_equiv": 42, "status": "baseline"},
                    {"month": 12, "blood_nfl": 50.0, "alsfrs_r_equiv": 39, "status": "slow decline"},
                    {"month": 24, "blood_nfl": 55.0, "alsfrs_r_equiv": 36, "status": "stable"},
                    {"month": 48, "blood_nfl": 65.0, "alsfrs_r_equiv": 30, "status": "gradual"}
                ],
                "clinical_narrative": (
                    "52-year-old man with limb-onset ALS diagnosed 4 years ago, slow "
                    "progressor phenotype. Blood NfL 45 pg/mL — lower than typical ALS, "
                    "consistent with slower neuronal loss. ALSFRS-R declining ~3 "
                    "points/year (vs typical 9-12). Maintained ambulation and "
                    "respiratory function at 4 years. Cognition preserved. On riluzole "
                    "since diagnosis."
                )
            },
        ],

        "huntingtons_disease": [
            {
                "prototype_id": "HD_PREMANIFEST_001",
                "archetype": "pre_manifest",
                "label": "Pre-manifest Huntington's, gene-positive, no motor symptoms",
                "demographics": {
                    "age": 35, "sex": "M", "bmi": 25.5,
                    "apoe_e4_carrier": False, "education_years": 16
                },
                "baseline_biomarkers": {
                    "mmse_score": 29, "moca_score": 27,
                    "hamilton_depression_score": 8, "updrs_motor_score": 2,
                    "csf_amyloid_beta42_pg_ml": 1100,
                    "csf_total_tau_pg_ml": 220,
                    "csf_p_tau181_pg_ml": 20,
                    "csf_nfl_pg_ml": 800,
                    "blood_nfl_pg_ml": 18.0,
                    "blood_gfap_pg_ml": 90.0,
                    "blood_bdnf_ng_ml": 18.0,
                    "blood_crp_mg_l": 1.5,
                    "blood_il6_pg_ml": 1.2,
                    "eeg_theta_alpha_ratio": 0.42,
                    "eeg_spectral_entropy": 0.70,
                    "eeg_alpha_peak_freq_hz": 9.8,
                    "fmri_bold_dmn": -0.05,
                    "fmri_bold_basal_ganglia": -0.20,
                    "polygenic_risk_score": 2.5
                },
                "treatment": None,
                "trajectory_summary": [
                    {"month": 0, "mmse": 29, "blood_nfl": 18.0, "status": "pre-manifest"},
                    {"month": 24, "mmse": 28, "blood_nfl": 22.0, "status": "subtle changes"},
                    {"month": 48, "mmse": 27, "blood_nfl": 28.0, "status": "prodromal"}
                ],
                "clinical_narrative": (
                    "35-year-old man, CAG repeat 42 (HTT gene). No motor symptoms yet. "
                    "Subtle caudate volume loss on MRI, mildly elevated blood NfL "
                    "(18 pg/mL) suggesting early neurodegeneration. Psychiatric symptoms "
                    "emerging (irritability, mild depression). Enrolled in HD "
                    "observational cohort. Estimated 10-15 years from motor onset."
                )
            },
            {
                "prototype_id": "HD_MANIFEST_002",
                "archetype": "manifest_motor",
                "label": "Manifest Huntington's with chorea and cognitive decline",
                "demographics": {
                    "age": 48, "sex": "F", "bmi": 22.0,
                    "apoe_e4_carrier": False, "education_years": 14
                },
                "baseline_biomarkers": {
                    "mmse_score": 22, "moca_score": 18,
                    "hamilton_depression_score": 16, "updrs_motor_score": 15,
                    "csf_amyloid_beta42_pg_ml": 1000,
                    "csf_total_tau_pg_ml": 320,
                    "csf_p_tau181_pg_ml": 30,
                    "csf_nfl_pg_ml": 2500,
                    "blood_nfl_pg_ml": 55.0,
                    "blood_gfap_pg_ml": 180.0,
                    "blood_bdnf_ng_ml": 11.0,
                    "blood_crp_mg_l": 3.0,
                    "blood_il6_pg_ml": 3.5,
                    "eeg_theta_alpha_ratio": 0.65,
                    "eeg_spectral_entropy": 0.55,
                    "eeg_alpha_peak_freq_hz": 8.5,
                    "fmri_bold_dmn": -0.30,
                    "fmri_bold_basal_ganglia": -0.60,
                    "polygenic_risk_score": 2.8
                },
                "treatment": None,
                "trajectory_summary": [
                    {"month": 0, "mmse": 22, "blood_nfl": 55.0, "status": "manifest"},
                    {"month": 12, "mmse": 19, "blood_nfl": 62.0, "status": "progressive"},
                    {"month": 24, "mmse": 16, "blood_nfl": 70.0, "status": "moderate stage"},
                    {"month": 48, "mmse": 10, "blood_nfl": 85.0, "status": "advanced"}
                ],
                "clinical_narrative": (
                    "48-year-old woman, CAG repeat 45, motor onset at age 43. "
                    "Prominent chorea, executive dysfunction, and personality changes. "
                    "MMSE 22. NfL markedly elevated at 55 pg/mL. Severe caudate "
                    "atrophy on MRI. Depression and irritability managed with SSRIs. "
                    "Progressive functional decline over 4 years."
                )
            },
        ],

        "traumatic_brain_injury": [
            {
                "prototype_id": "TBI_MILD_001",
                "archetype": "mild_concussion",
                "label": "Mild TBI (concussion), typical recovery trajectory",
                "demographics": {
                    "age": 28, "sex": "M", "bmi": 26.0,
                    "apoe_e4_carrier": False, "education_years": 16
                },
                "baseline_biomarkers": {
                    "mmse_score": 27, "moca_score": 25,
                    "hamilton_depression_score": 8, "updrs_motor_score": 0,
                    "csf_amyloid_beta42_pg_ml": 1050,
                    "csf_total_tau_pg_ml": 280,
                    "csf_p_tau181_pg_ml": 28,
                    "csf_nfl_pg_ml": 900,
                    "blood_nfl_pg_ml": 25.0,
                    "blood_gfap_pg_ml": 150.0,
                    "blood_s100b_ug_l": 0.15,
                    "blood_crp_mg_l": 3.0,
                    "blood_il6_pg_ml": 4.0,
                    "eeg_theta_alpha_ratio": 0.55,
                    "eeg_spectral_entropy": 0.62,
                    "eeg_alpha_peak_freq_hz": 9.2,
                    "fmri_bold_dmn": -0.10,
                    "polygenic_risk_score": 0.0
                },
                "treatment": None,
                "trajectory_summary": [
                    {"month": 0, "mmse": 27, "blood_nfl": 25.0, "status": "acute post-TBI"},
                    {"month": 1, "mmse": 28, "blood_nfl": 18.0, "status": "recovering"},
                    {"month": 3, "mmse": 29, "blood_nfl": 12.0, "status": "near baseline"},
                    {"month": 6, "mmse": 30, "blood_nfl": 10.0, "status": "recovered"}
                ],
                "clinical_narrative": (
                    "28-year-old man, sports-related concussion. Brief loss of "
                    "consciousness (<30 seconds), GCS 15. Acute elevation of blood "
                    "NfL (25 pg/mL) and S100B (0.15 ug/L) reflecting transient "
                    "neuronal and glial injury. Mild cognitive symptoms and headaches "
                    "for 2 weeks. Full recovery by 3 months with normalization of "
                    "biomarkers."
                )
            },
            {
                "prototype_id": "TBI_MODERATE_002",
                "archetype": "moderate",
                "label": "Moderate TBI with persistent post-concussive symptoms",
                "demographics": {
                    "age": 42, "sex": "M", "bmi": 28.0,
                    "apoe_e4_carrier": True, "education_years": 12
                },
                "baseline_biomarkers": {
                    "mmse_score": 24, "moca_score": 21,
                    "hamilton_depression_score": 14, "updrs_motor_score": 3,
                    "csf_amyloid_beta42_pg_ml": 900,
                    "csf_total_tau_pg_ml": 400,
                    "csf_p_tau181_pg_ml": 40,
                    "csf_nfl_pg_ml": 1500,
                    "blood_nfl_pg_ml": 45.0,
                    "blood_gfap_pg_ml": 280.0,
                    "blood_s100b_ug_l": 0.35,
                    "blood_crp_mg_l": 5.0,
                    "blood_il6_pg_ml": 6.0,
                    "eeg_theta_alpha_ratio": 0.70,
                    "eeg_spectral_entropy": 0.55,
                    "eeg_alpha_peak_freq_hz": 8.5,
                    "fmri_bold_dmn": -0.30,
                    "fmri_connectivity_dmn_pfcx": 0.28,
                    "polygenic_risk_score": 0.5
                },
                "treatment": None,
                "trajectory_summary": [
                    {"month": 0, "mmse": 24, "blood_nfl": 45.0, "status": "acute moderate TBI"},
                    {"month": 3, "mmse": 26, "blood_nfl": 30.0, "status": "partial recovery"},
                    {"month": 12, "mmse": 27, "blood_nfl": 20.0, "status": "persistent symptoms"},
                    {"month": 24, "mmse": 27, "blood_nfl": 16.0, "status": "plateau"}
                ],
                "clinical_narrative": (
                    "42-year-old man, motor vehicle accident. GCS 10 at scene, brief "
                    "ICU admission. Contusion in right frontal lobe. APOE e4 carrier — "
                    "poorer prognosis. Markedly elevated GFAP (280 pg/mL) and NfL "
                    "(45 pg/mL). Persistent headaches, cognitive complaints, and "
                    "depression at 12 months. Incomplete recovery with plateau at 2 years."
                )
            },
            {
                "prototype_id": "TBI_SEVERE_003",
                "archetype": "severe_diffuse_axonal",
                "label": "Severe TBI with diffuse axonal injury",
                "demographics": {
                    "age": 25, "sex": "M", "bmi": 24.0,
                    "apoe_e4_carrier": False, "education_years": 14
                },
                "baseline_biomarkers": {
                    "mmse_score": 18, "moca_score": 14,
                    "hamilton_depression_score": 10, "updrs_motor_score": 8,
                    "csf_amyloid_beta42_pg_ml": 800,
                    "csf_total_tau_pg_ml": 600,
                    "csf_p_tau181_pg_ml": 55,
                    "csf_nfl_pg_ml": 5000,
                    "blood_nfl_pg_ml": 120.0,
                    "blood_gfap_pg_ml": 500.0,
                    "blood_s100b_ug_l": 0.80,
                    "blood_crp_mg_l": 8.0,
                    "blood_il6_pg_ml": 10.0,
                    "eeg_theta_alpha_ratio": 1.20,
                    "eeg_spectral_entropy": 0.40,
                    "eeg_alpha_peak_freq_hz": 7.5,
                    "fmri_bold_dmn": -0.50,
                    "fmri_connectivity_dmn_pfcx": 0.15,
                    "polygenic_risk_score": 0.2
                },
                "treatment": None,
                "trajectory_summary": [
                    {"month": 0, "mmse": 18, "blood_nfl": 120.0, "status": "post-acute severe TBI"},
                    {"month": 6, "mmse": 22, "blood_nfl": 60.0, "status": "recovering"},
                    {"month": 12, "mmse": 24, "blood_nfl": 35.0, "status": "improving"},
                    {"month": 24, "mmse": 25, "blood_nfl": 22.0, "status": "residual deficits"}
                ],
                "clinical_narrative": (
                    "25-year-old man, high-speed MVA with diffuse axonal injury. GCS 5 "
                    "at scene, 2-week coma. Extremely elevated NfL (120 pg/mL), GFAP "
                    "(500 pg/mL) and S100B (0.80 ug/L). Severe EEG slowing. Prolonged "
                    "inpatient rehabilitation. Gradual cognitive improvement over 2 "
                    "years but persistent executive dysfunction and processing speed "
                    "deficits."
                )
            },
        ],

        "ischemic_stroke": [
            {
                "prototype_id": "IS_ACUTE_001",
                "archetype": "acute_mca",
                "label": "Acute MCA territory ischemic stroke",
                "demographics": {
                    "age": 68, "sex": "M", "bmi": 29.0,
                    "apoe_e4_carrier": False, "education_years": 12
                },
                "baseline_biomarkers": {
                    "mmse_score": 20, "moca_score": 16,
                    "hamilton_depression_score": 6, "updrs_motor_score": 5,
                    "csf_amyloid_beta42_pg_ml": 950,
                    "csf_total_tau_pg_ml": 450,
                    "csf_p_tau181_pg_ml": 35,
                    "csf_nfl_pg_ml": 3000,
                    "blood_nfl_pg_ml": 80.0,
                    "blood_gfap_pg_ml": 350.0,
                    "blood_s100b_ug_l": 0.50,
                    "blood_crp_mg_l": 8.0,
                    "blood_il6_pg_ml": 12.0,
                    "eeg_theta_alpha_ratio": 0.90,
                    "eeg_spectral_entropy": 0.48,
                    "eeg_alpha_peak_freq_hz": 8.0,
                    "fmri_bold_dmn": -0.35,
                    "polygenic_risk_score": 0.3
                },
                "treatment": None,
                "trajectory_summary": [
                    {"month": 0, "mmse": 20, "blood_nfl": 80.0, "status": "acute stroke"},
                    {"month": 1, "mmse": 22, "blood_nfl": 50.0, "status": "early recovery"},
                    {"month": 6, "mmse": 25, "blood_nfl": 25.0, "status": "rehabilitation"},
                    {"month": 12, "mmse": 26, "blood_nfl": 18.0, "status": "chronic phase"}
                ],
                "clinical_narrative": (
                    "68-year-old man with acute left MCA ischemic stroke, treated with "
                    "IV tPA at 3 hours. NIHSS 12 at presentation. Massively elevated "
                    "inflammatory markers (CRP 8.0, IL-6 12.0) in acute phase. NfL "
                    "peaked at 80 pg/mL, GFAP at 350 pg/mL. Progressive recovery with "
                    "rehabilitation. Residual mild aphasia and right arm weakness at "
                    "12 months."
                )
            },
            {
                "prototype_id": "IS_CHRONIC_002",
                "archetype": "chronic_post_stroke",
                "label": "Chronic post-stroke with vascular cognitive impairment",
                "demographics": {
                    "age": 75, "sex": "F", "bmi": 28.0,
                    "apoe_e4_carrier": True, "education_years": 10
                },
                "baseline_biomarkers": {
                    "mmse_score": 23, "moca_score": 19,
                    "hamilton_depression_score": 12, "updrs_motor_score": 4,
                    "csf_amyloid_beta42_pg_ml": 850,
                    "csf_total_tau_pg_ml": 320,
                    "csf_p_tau181_pg_ml": 30,
                    "csf_nfl_pg_ml": 1200,
                    "blood_nfl_pg_ml": 28.0,
                    "blood_gfap_pg_ml": 180.0,
                    "blood_s100b_ug_l": 0.12,
                    "blood_crp_mg_l": 4.0,
                    "blood_il6_pg_ml": 3.5,
                    "eeg_theta_alpha_ratio": 0.65,
                    "eeg_spectral_entropy": 0.58,
                    "eeg_alpha_peak_freq_hz": 8.8,
                    "fmri_bold_dmn": -0.20,
                    "fmri_connectivity_dmn_pfcx": 0.28,
                    "polygenic_risk_score": 0.8
                },
                "treatment": None,
                "trajectory_summary": [
                    {"month": 0, "mmse": 23, "blood_nfl": 28.0, "status": "chronic post-stroke"},
                    {"month": 12, "mmse": 22, "blood_nfl": 26.0, "status": "stable"},
                    {"month": 24, "mmse": 21, "blood_nfl": 30.0, "status": "slow decline"},
                    {"month": 48, "mmse": 18, "blood_nfl": 35.0, "status": "vascular dementia"}
                ],
                "clinical_narrative": (
                    "75-year-old woman, 3 years post left MCA stroke with residual "
                    "aphasia. Progressive vascular cognitive impairment (MMSE 23, "
                    "declining). APOE e4 carrier raising concern for mixed vascular-AD "
                    "pathology. Persistently elevated NfL and GFAP. Post-stroke "
                    "depression (Hamilton 12). White matter hyperintensities on MRI."
                )
            },
        ],

        "schizophrenia": [
            {
                "prototype_id": "SZ_FIRST_001",
                "archetype": "first_episode_positive",
                "label": "First-episode schizophrenia, prominent positive symptoms",
                "demographics": {
                    "age": 22, "sex": "M", "bmi": 23.0,
                    "apoe_e4_carrier": False, "education_years": 12
                },
                "baseline_biomarkers": {
                    "mmse_score": 27, "moca_score": 24,
                    "hamilton_depression_score": 10, "updrs_motor_score": 1,
                    "csf_amyloid_beta42_pg_ml": 1080,
                    "csf_total_tau_pg_ml": 210,
                    "csf_p_tau181_pg_ml": 20,
                    "csf_nfl_pg_ml": 600,
                    "blood_nfl_pg_ml": 12.0,
                    "blood_gfap_pg_ml": 80.0,
                    "blood_bdnf_ng_ml": 15.0,
                    "blood_crp_mg_l": 2.5,
                    "blood_il6_pg_ml": 3.0,
                    "eeg_theta_alpha_ratio": 0.52,
                    "eeg_spectral_entropy": 0.62,
                    "eeg_alpha_peak_freq_hz": 9.2,
                    "fmri_bold_dmn": 0.20,
                    "fmri_bold_pfc": -0.25,
                    "fmri_connectivity_dmn_pfcx": 0.52,
                    "polygenic_risk_score": 1.5
                },
                "treatment": None,
                "trajectory_summary": [
                    {"month": 0, "mmse": 27, "status": "first psychotic episode"},
                    {"month": 6, "mmse": 27, "status": "stabilized on antipsychotic"},
                    {"month": 12, "mmse": 27, "status": "maintained remission"},
                    {"month": 24, "mmse": 26, "status": "relapse risk period"}
                ],
                "clinical_narrative": (
                    "22-year-old man with first psychotic episode: auditory "
                    "hallucinations, paranoid delusions, and social withdrawal over 6 "
                    "months. Mild cognitive deficits (MoCA 24). Elevated inflammatory "
                    "markers and reduced BDNF. fMRI shows PFC hypoactivation and DMN "
                    "dysregulation. High polygenic risk score (1.5 SD)."
                )
            },
            {
                "prototype_id": "SZ_CHRONIC_002",
                "archetype": "chronic_negative",
                "label": "Chronic schizophrenia, predominant negative symptoms",
                "demographics": {
                    "age": 45, "sex": "M", "bmi": 32.0,
                    "apoe_e4_carrier": False, "education_years": 10
                },
                "baseline_biomarkers": {
                    "mmse_score": 24, "moca_score": 20,
                    "hamilton_depression_score": 8, "updrs_motor_score": 4,
                    "csf_amyloid_beta42_pg_ml": 1020,
                    "csf_total_tau_pg_ml": 250,
                    "csf_p_tau181_pg_ml": 24,
                    "csf_nfl_pg_ml": 750,
                    "blood_nfl_pg_ml": 16.0,
                    "blood_gfap_pg_ml": 100.0,
                    "blood_bdnf_ng_ml": 10.0,
                    "blood_crp_mg_l": 4.0,
                    "blood_il6_pg_ml": 4.5,
                    "eeg_theta_alpha_ratio": 0.60,
                    "eeg_spectral_entropy": 0.55,
                    "eeg_alpha_peak_freq_hz": 8.8,
                    "fmri_bold_dmn": 0.15,
                    "fmri_bold_pfc": -0.40,
                    "fmri_connectivity_dmn_pfcx": 0.58,
                    "polygenic_risk_score": 1.8
                },
                "treatment": None,
                "trajectory_summary": [
                    {"month": 0, "mmse": 24, "status": "chronic negative symptoms"},
                    {"month": 12, "mmse": 23, "status": "stable, functional impairment"},
                    {"month": 24, "mmse": 23, "status": "no improvement"},
                    {"month": 48, "mmse": 22, "status": "slow cognitive decline"}
                ],
                "clinical_narrative": (
                    "45-year-old man with 22-year illness course. Predominant negative "
                    "symptoms: avolition, flat affect, social isolation. Cognitive "
                    "decline (MMSE 24, MoCA 20). Metabolic syndrome from long-term "
                    "antipsychotic use (BMI 32). Low BDNF (10 ng/mL), elevated "
                    "inflammatory markers. Marked PFC hypoactivation on fMRI."
                )
            },
            {
                "prototype_id": "SZ_RESPONSIVE_003",
                "archetype": "treatment_responsive",
                "label": "Schizophrenia with good treatment response, functional recovery",
                "demographics": {
                    "age": 30, "sex": "F", "bmi": 25.0,
                    "apoe_e4_carrier": False, "education_years": 14
                },
                "baseline_biomarkers": {
                    "mmse_score": 28, "moca_score": 26,
                    "hamilton_depression_score": 6, "updrs_motor_score": 0,
                    "csf_amyloid_beta42_pg_ml": 1060,
                    "csf_total_tau_pg_ml": 200,
                    "csf_p_tau181_pg_ml": 19,
                    "csf_nfl_pg_ml": 550,
                    "blood_nfl_pg_ml": 10.0,
                    "blood_gfap_pg_ml": 70.0,
                    "blood_bdnf_ng_ml": 18.0,
                    "blood_crp_mg_l": 1.8,
                    "blood_il6_pg_ml": 2.0,
                    "eeg_theta_alpha_ratio": 0.48,
                    "eeg_spectral_entropy": 0.68,
                    "eeg_alpha_peak_freq_hz": 9.5,
                    "fmri_bold_dmn": 0.10,
                    "fmri_bold_pfc": -0.15,
                    "fmri_connectivity_dmn_pfcx": 0.45,
                    "polygenic_risk_score": 0.8
                },
                "treatment": None,
                "trajectory_summary": [
                    {"month": 0, "mmse": 28, "status": "acute psychosis, treatment started"},
                    {"month": 3, "mmse": 29, "status": "symptom remission"},
                    {"month": 12, "mmse": 29, "status": "functional recovery"},
                    {"month": 24, "mmse": 29, "status": "sustained remission, employed"}
                ],
                "clinical_narrative": (
                    "30-year-old woman with second psychotic episode, rapid response "
                    "to second-generation antipsychotic. Minimal negative symptoms. "
                    "Low inflammatory burden, near-normal BDNF. Cognitive function "
                    "preserved. Returned to part-time employment within 6 months. "
                    "Lower genetic loading than typical (PRS 0.8 SD)."
                )
            },
        ],

        "bipolar_disorder": [
            {
                "prototype_id": "BD_MANIC_001",
                "archetype": "manic_episode",
                "label": "Bipolar I, acute manic episode",
                "demographics": {
                    "age": 32, "sex": "M", "bmi": 24.0,
                    "apoe_e4_carrier": False, "education_years": 16
                },
                "baseline_biomarkers": {
                    "mmse_score": 27, "moca_score": 25,
                    "hamilton_depression_score": 2, "updrs_motor_score": 0,
                    "csf_amyloid_beta42_pg_ml": 1080,
                    "csf_total_tau_pg_ml": 200,
                    "csf_p_tau181_pg_ml": 19,
                    "csf_nfl_pg_ml": 550,
                    "blood_nfl_pg_ml": 12.0,
                    "blood_gfap_pg_ml": 75.0,
                    "blood_bdnf_ng_ml": 30.0,
                    "blood_crp_mg_l": 3.0,
                    "blood_il6_pg_ml": 3.5,
                    "eeg_theta_alpha_ratio": 0.40,
                    "eeg_spectral_entropy": 0.75,
                    "eeg_alpha_peak_freq_hz": 10.5,
                    "fmri_bold_dmn": 0.10,
                    "fmri_bold_amygdala": 0.50,
                    "fmri_bold_pfc": -0.20,
                    "polygenic_risk_score": 1.2
                },
                "treatment": None,
                "trajectory_summary": [
                    {"month": 0, "hamilton": 2, "bdnf": 30.0, "status": "acute mania"},
                    {"month": 1, "hamilton": 5, "bdnf": 25.0, "status": "stabilizing"},
                    {"month": 3, "hamilton": 4, "bdnf": 22.0, "status": "euthymic"},
                    {"month": 9, "hamilton": 18, "bdnf": 12.0, "status": "depressive switch"}
                ],
                "clinical_narrative": (
                    "32-year-old man in acute manic episode: grandiosity, decreased "
                    "sleep, pressured speech, impulsive spending. Elevated BDNF "
                    "(30 ng/mL, paradoxically high in mania). Increased inflammatory "
                    "markers. fMRI shows amygdala hyperactivation and PFC hypofunction. "
                    "Stabilized with lithium and antipsychotic, then switched to "
                    "depressive phase at 9 months."
                )
            },
            {
                "prototype_id": "BD_DEPRESSIVE_002",
                "archetype": "depressive_episode",
                "label": "Bipolar II, depressive episode",
                "demographics": {
                    "age": 40, "sex": "F", "bmi": 28.0,
                    "apoe_e4_carrier": False, "education_years": 14
                },
                "baseline_biomarkers": {
                    "mmse_score": 27, "moca_score": 25,
                    "hamilton_depression_score": 24, "updrs_motor_score": 0,
                    "csf_amyloid_beta42_pg_ml": 1060,
                    "csf_total_tau_pg_ml": 210,
                    "csf_p_tau181_pg_ml": 20,
                    "csf_nfl_pg_ml": 580,
                    "blood_nfl_pg_ml": 12.0,
                    "blood_gfap_pg_ml": 80.0,
                    "blood_bdnf_ng_ml": 12.0,
                    "blood_crp_mg_l": 3.5,
                    "blood_il6_pg_ml": 3.8,
                    "eeg_theta_alpha_ratio": 0.55,
                    "eeg_spectral_entropy": 0.62,
                    "eeg_alpha_peak_freq_hz": 9.2,
                    "fmri_bold_dmn": 0.35,
                    "fmri_bold_amygdala": 0.45,
                    "fmri_bold_pfc": -0.20,
                    "polygenic_risk_score": 1.0
                },
                "treatment": None,
                "trajectory_summary": [
                    {"month": 0, "hamilton": 24, "bdnf": 12.0, "status": "bipolar depression"},
                    {"month": 2, "hamilton": 16, "bdnf": 16.0, "status": "improving"},
                    {"month": 6, "hamilton": 8, "bdnf": 20.0, "status": "remission"},
                    {"month": 12, "hamilton": 5, "bdnf": 22.0, "status": "euthymic"}
                ],
                "clinical_narrative": (
                    "40-year-old woman with bipolar II disorder in depressive phase "
                    "(Hamilton 24). Low BDNF (12 ng/mL), elevated CRP and IL-6. DMN "
                    "hyperactivity on fMRI. History of hypomania but no full mania. "
                    "Lamotrigine added to mood stabilizer with gradual improvement over "
                    "6 months."
                )
            },
            {
                "prototype_id": "BD_EUTHYMIC_003",
                "archetype": "euthymic_on_lithium",
                "label": "Bipolar I, euthymic on lithium maintenance",
                "demographics": {
                    "age": 50, "sex": "M", "bmi": 29.0,
                    "apoe_e4_carrier": False, "education_years": 14
                },
                "baseline_biomarkers": {
                    "mmse_score": 28, "moca_score": 26,
                    "hamilton_depression_score": 4, "updrs_motor_score": 0,
                    "csf_amyloid_beta42_pg_ml": 1050,
                    "csf_total_tau_pg_ml": 220,
                    "csf_p_tau181_pg_ml": 21,
                    "csf_nfl_pg_ml": 560,
                    "blood_nfl_pg_ml": 11.0,
                    "blood_gfap_pg_ml": 75.0,
                    "blood_bdnf_ng_ml": 20.0,
                    "blood_crp_mg_l": 2.0,
                    "blood_il6_pg_ml": 1.8,
                    "eeg_theta_alpha_ratio": 0.44,
                    "eeg_spectral_entropy": 0.70,
                    "eeg_alpha_peak_freq_hz": 9.8,
                    "fmri_bold_dmn": 0.05,
                    "fmri_bold_amygdala": 0.10,
                    "fmri_bold_pfc": -0.05,
                    "polygenic_risk_score": 0.9
                },
                "treatment": None,
                "trajectory_summary": [
                    {"month": 0, "hamilton": 4, "status": "euthymic baseline"},
                    {"month": 12, "hamilton": 3, "status": "stable"},
                    {"month": 24, "hamilton": 5, "status": "stable"},
                    {"month": 36, "hamilton": 4, "status": "maintained"}
                ],
                "clinical_narrative": (
                    "50-year-old man with bipolar I, euthymic on lithium 900mg for 8 "
                    "years. No episodes in past 5 years. Biomarkers near-normal: BDNF "
                    "20 ng/mL, low inflammatory markers. fMRI shows normalized "
                    "prefrontal and amygdala activation. Mild weight gain (BMI 29) and "
                    "thyroid monitoring on lithium."
                )
            },
        ],

        "autism_spectrum_disorder": [
            {
                "prototype_id": "ASD_HIGH_SUPPORT_001",
                "archetype": "high_support_needs",
                "label": "ASD with high support needs, intellectual disability",
                "demographics": {
                    "age": 12, "sex": "M", "bmi": 20.0,
                    "apoe_e4_carrier": False, "education_years": 6
                },
                "baseline_biomarkers": {
                    "mmse_score": 15, "moca_score": 10,
                    "hamilton_depression_score": 5, "updrs_motor_score": 2,
                    "csf_amyloid_beta42_pg_ml": 1150,
                    "csf_total_tau_pg_ml": 170,
                    "csf_p_tau181_pg_ml": 16,
                    "csf_nfl_pg_ml": 400,
                    "blood_nfl_pg_ml": 8.0,
                    "blood_gfap_pg_ml": 60.0,
                    "blood_bdnf_ng_ml": 22.0,
                    "blood_crp_mg_l": 1.5,
                    "blood_il6_pg_ml": 2.0,
                    "eeg_theta_alpha_ratio": 0.65,
                    "eeg_spectral_entropy": 0.58,
                    "eeg_alpha_peak_freq_hz": 9.0,
                    "fmri_bold_dmn": 0.15,
                    "fmri_connectivity_dmn_pfcx": 0.55,
                    "fmri_connectivity_hipp_amyg": 0.60,
                    "polygenic_risk_score": 1.5
                },
                "treatment": None,
                "trajectory_summary": [
                    {"month": 0, "status": "baseline assessment"},
                    {"month": 12, "status": "behavioral intervention"},
                    {"month": 24, "status": "slow developmental progress"},
                    {"month": 48, "status": "stable, supported living"}
                ],
                "clinical_narrative": (
                    "12-year-old boy with ASD Level 3 and co-occurring intellectual "
                    "disability. Limited verbal communication, repetitive behaviors, "
                    "sensory sensitivities. EEG shows elevated theta power. fMRI "
                    "reveals atypical DMN-PFC connectivity. Comorbid epileptiform "
                    "activity on EEG (no clinical seizures). In intensive behavioral "
                    "and speech therapy."
                )
            },
            {
                "prototype_id": "ASD_LOW_SUPPORT_002",
                "archetype": "low_support_needs_adult",
                "label": "ASD with low support needs, adult with employment",
                "demographics": {
                    "age": 30, "sex": "M", "bmi": 24.0,
                    "apoe_e4_carrier": False, "education_years": 18
                },
                "baseline_biomarkers": {
                    "mmse_score": 30, "moca_score": 28,
                    "hamilton_depression_score": 8, "updrs_motor_score": 0,
                    "csf_amyloid_beta42_pg_ml": 1100,
                    "csf_total_tau_pg_ml": 190,
                    "csf_p_tau181_pg_ml": 18,
                    "csf_nfl_pg_ml": 450,
                    "blood_nfl_pg_ml": 9.0,
                    "blood_gfap_pg_ml": 65.0,
                    "blood_bdnf_ng_ml": 20.0,
                    "blood_crp_mg_l": 1.2,
                    "blood_il6_pg_ml": 1.5,
                    "eeg_theta_alpha_ratio": 0.48,
                    "eeg_spectral_entropy": 0.65,
                    "eeg_alpha_peak_freq_hz": 9.5,
                    "fmri_bold_dmn": 0.10,
                    "fmri_connectivity_dmn_pfcx": 0.48,
                    "fmri_connectivity_hipp_amyg": 0.55,
                    "polygenic_risk_score": 0.8
                },
                "treatment": None,
                "trajectory_summary": [
                    {"month": 0, "status": "research assessment baseline"},
                    {"month": 12, "status": "stable"},
                    {"month": 24, "status": "stable, employed"}
                ],
                "clinical_narrative": (
                    "30-year-old man with ASD Level 1 diagnosed at age 25. Graduate "
                    "degree in engineering, employed in technical role. Social "
                    "communication difficulties and restricted interests. Comorbid "
                    "anxiety (Hamilton 8). Normal biomarkers. fMRI shows mildly atypical "
                    "social brain connectivity. Participates in autism research cohort."
                )
            },
        ],

        "mild_cognitive_impairment": [
            {
                "prototype_id": "MCI_AMNESTIC_001",
                "archetype": "amnestic_ad_trajectory",
                "label": "Amnestic MCI, likely progressing to AD",
                "demographics": {
                    "age": 70, "sex": "F", "bmi": 25.0,
                    "apoe_e4_carrier": True, "education_years": 14
                },
                "baseline_biomarkers": {
                    "mmse_score": 25, "moca_score": 22,
                    "hamilton_depression_score": 5, "updrs_motor_score": 1,
                    "csf_amyloid_beta42_pg_ml": 520,
                    "csf_total_tau_pg_ml": 320,
                    "csf_p_tau181_pg_ml": 42,
                    "csf_nfl_pg_ml": 850,
                    "blood_nfl_pg_ml": 22.0,
                    "blood_gfap_pg_ml": 130.0,
                    "blood_bdnf_ng_ml": 17.0,
                    "blood_crp_mg_l": 2.0,
                    "blood_il6_pg_ml": 2.0,
                    "eeg_theta_alpha_ratio": 0.55,
                    "eeg_spectral_entropy": 0.65,
                    "eeg_alpha_peak_freq_hz": 9.2,
                    "fmri_bold_dmn": -0.15,
                    "fmri_connectivity_dmn_pfcx": 0.35,
                    "polygenic_risk_score": 1.1
                },
                "treatment": None,
                "trajectory_summary": [
                    {"month": 0, "mmse": 25, "csf_ab42": 520, "status": "amnestic MCI"},
                    {"month": 12, "mmse": 24, "csf_ab42": 480, "status": "stable MCI"},
                    {"month": 24, "mmse": 22, "csf_ab42": 440, "status": "declining"},
                    {"month": 36, "mmse": 20, "csf_ab42": 400, "status": "converted to AD"}
                ],
                "clinical_narrative": (
                    "70-year-old woman with amnestic MCI, APOE e4 heterozygous. "
                    "Subjective memory complaints confirmed on neuropsychological "
                    "testing. CSF profile (low Abeta42, elevated p-tau181) indicates "
                    "underlying AD pathology. Converted to clinical AD diagnosis at "
                    "36 months. Pattern consistent with A+T+N+ biomarker profile."
                )
            },
            {
                "prototype_id": "MCI_NONAMNESTIC_002",
                "archetype": "non_amnestic_stable",
                "label": "Non-amnestic MCI, stable over follow-up",
                "demographics": {
                    "age": 68, "sex": "M", "bmi": 27.0,
                    "apoe_e4_carrier": False, "education_years": 16
                },
                "baseline_biomarkers": {
                    "mmse_score": 26, "moca_score": 23,
                    "hamilton_depression_score": 4, "updrs_motor_score": 2,
                    "csf_amyloid_beta42_pg_ml": 900,
                    "csf_total_tau_pg_ml": 260,
                    "csf_p_tau181_pg_ml": 25,
                    "csf_nfl_pg_ml": 700,
                    "blood_nfl_pg_ml": 16.0,
                    "blood_gfap_pg_ml": 100.0,
                    "blood_bdnf_ng_ml": 20.0,
                    "blood_crp_mg_l": 1.8,
                    "blood_il6_pg_ml": 1.5,
                    "eeg_theta_alpha_ratio": 0.48,
                    "eeg_spectral_entropy": 0.68,
                    "eeg_alpha_peak_freq_hz": 9.5,
                    "fmri_bold_dmn": -0.05,
                    "fmri_connectivity_dmn_pfcx": 0.40,
                    "polygenic_risk_score": 0.3
                },
                "treatment": None,
                "trajectory_summary": [
                    {"month": 0, "mmse": 26, "csf_ab42": 900, "status": "non-amnestic MCI"},
                    {"month": 12, "mmse": 26, "csf_ab42": 890, "status": "stable"},
                    {"month": 24, "mmse": 25, "csf_ab42": 880, "status": "stable"},
                    {"month": 48, "mmse": 25, "csf_ab42": 870, "status": "stable MCI"}
                ],
                "clinical_narrative": (
                    "68-year-old man with non-amnestic MCI (executive/attention "
                    "deficits). Normal CSF Abeta42 (900 pg/mL) — no underlying AD "
                    "pathology. APOE e3/e3. Stable over 4-year follow-up, no "
                    "conversion to dementia. Likely cerebrovascular contribution "
                    "(mild white matter disease). Low genetic risk."
                )
            },
            {
                "prototype_id": "MCI_CONVERTER_003",
                "archetype": "converter_to_ad",
                "label": "MCI converter with rapid biomarker-driven transition to AD",
                "demographics": {
                    "age": 75, "sex": "F", "bmi": 24.0,
                    "apoe_e4_carrier": True, "education_years": 12
                },
                "baseline_biomarkers": {
                    "mmse_score": 24, "moca_score": 20,
                    "hamilton_depression_score": 7, "updrs_motor_score": 2,
                    "csf_amyloid_beta42_pg_ml": 380,
                    "csf_total_tau_pg_ml": 450,
                    "csf_p_tau181_pg_ml": 60,
                    "csf_nfl_pg_ml": 1100,
                    "blood_nfl_pg_ml": 30.0,
                    "blood_gfap_pg_ml": 170.0,
                    "blood_bdnf_ng_ml": 14.0,
                    "blood_crp_mg_l": 2.5,
                    "blood_il6_pg_ml": 2.5,
                    "eeg_theta_alpha_ratio": 0.62,
                    "eeg_spectral_entropy": 0.60,
                    "eeg_alpha_peak_freq_hz": 8.8,
                    "fmri_bold_dmn": -0.25,
                    "fmri_connectivity_dmn_pfcx": 0.28,
                    "polygenic_risk_score": 1.4
                },
                "treatment": None,
                "trajectory_summary": [
                    {"month": 0, "mmse": 24, "csf_ab42": 380, "status": "MCI baseline"},
                    {"month": 12, "mmse": 21, "csf_ab42": 340, "status": "rapid decline"},
                    {"month": 18, "mmse": 19, "csf_ab42": 320, "status": "converted to AD"},
                    {"month": 24, "mmse": 17, "csf_ab42": 300, "status": "early AD"}
                ],
                "clinical_narrative": (
                    "75-year-old woman, APOE e4 homozygous, with MCI and markedly "
                    "abnormal CSF biomarkers (very low Abeta42 380, high p-tau181 60). "
                    "High NfL (30 pg/mL) and GFAP (170 pg/mL) predicting rapid "
                    "conversion. Converted to AD diagnosis at 18 months — faster than "
                    "typical. Biomarker profile identifies high-risk MCI population "
                    "suitable for anti-amyloid clinical trials."
                )
            },
        ],
    }


def generate_parquet(prototypes, output_dir):
    """Flatten prototypes into a Parquet file."""
    rows = []
    for disease_key, protos in prototypes.items():
        for p in protos:
            row = {
                "prototype_id": p["prototype_id"],
                "disease_state": disease_key,
                "archetype": p["archetype"],
                "label": p["label"],
                "clinical_narrative": p["clinical_narrative"],
                "is_synthetic": True,
            }
            row.update(p["demographics"])
            row.update(p["baseline_biomarkers"])

            if p.get("treatment"):
                row["treatment_drug"] = str(p["treatment"].get("drug", ""))
                row["treatment_dose_mg"] = str(p["treatment"].get("dose_mg", ""))
                row["treatment_response"] = str(p["treatment"].get("response", ""))
            else:
                row["treatment_drug"] = None
                row["treatment_dose_mg"] = None
                row["treatment_response"] = None

            rows.append(row)

    df = pd.DataFrame(rows)

    fp = Path(output_dir) / "SYNTHETIC_prototypes.parquet"
    table = pa.Table.from_pandas(df)
    table = table.replace_schema_metadata({
        **(table.schema.metadata or {}),
        b"synthetic": b"true",
        b"generator": b"generate_prototypes.py",
        b"description": b"SYNTHETIC DATA - NOT FOR CLINICAL DECISIONS",
    })
    pq.write_table(table, str(fp), compression="snappy")
    return fp, len(df)


def main():
    parser = argparse.ArgumentParser(
        description="Generate de-identified archetypal patient prototypes.")
    parser.add_argument("--output-dir", default=str(
        Path(__file__).resolve().parent.parent / "quarantine"))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    prototypes = build_prototypes()

    total = sum(len(v) for v in prototypes.values())
    print(f"Generated {total} prototypes across {len(prototypes)} disease states")

    output = {
        "_metadata": {
            "synthetic": True,
            "version": "1.0.0",
            "description": "De-identified archetypal patient prototypes. Entirely synthetic.",
            "generation_timestamp": now,
            "warning": "SYNTHETIC DATA - NOT FOR CLINICAL DECISIONS",
            "total_prototypes": total,
            "disease_states": len(prototypes),
        },
        "prototypes": prototypes,
    }

    json_fp = output_dir / "SYNTHETIC_prototypes.json"
    with open(json_fp, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  JSON: {json_fp} ({json_fp.stat().st_size / 1024:.1f} KB)")

    parquet_fp, n_rows = generate_parquet(prototypes, output_dir)
    print(f"  Parquet: {parquet_fp} ({n_rows} rows)")

    print("\nPrototypes per disease state:")
    for disease, protos in prototypes.items():
        ids = [p["prototype_id"] for p in protos]
        print(f"  {disease:35s}: {len(protos)}  [{', '.join(ids)}]")

    return 0


if __name__ == "__main__":
    sys.exit(main())
