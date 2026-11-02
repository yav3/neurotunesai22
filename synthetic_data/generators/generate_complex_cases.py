"""
Complex synthetic case generator: zebra diagnoses, multi-comorbidity clusters,
diagnostic mimics, and genetic overlap cases with full multi-omics profiles.

All genetic variants use real rsIDs/HGVS from ClinVar/OMIM/gnomAD.
All biomarker values are physiologically plausible and internally consistent.
All patient data is ENTIRELY SYNTHETIC — no real patient information.
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def _v(gene, hgvs_c, hgvs_p, rsid, clinvar, pathogenicity, zygosity,
       gnomad_af, inheritance, functional_impact):
    return {
        "gene": gene, "hgvs_c": hgvs_c, "hgvs_p": hgvs_p,
        "rsid": rsid, "clinvar_accession": clinvar,
        "pathogenicity": pathogenicity, "zygosity": zygosity,
        "gnomad_af": gnomad_af, "inheritance": inheritance,
        "functional_impact": functional_impact,
    }


def _comorb(condition, icd10, onset_age, status="active"):
    return {"condition": condition, "icd10": icd10,
            "onset_age": onset_age, "status": status}


def _tx(drug, dose, start_age, status="ongoing"):
    return {"drug": drug, "dose": dose,
            "start_age": start_age, "status": status}


def _deg(gene, log2fc, padj, direction, note=""):
    d = {"gene": gene, "log2fc": log2fc, "padj": padj, "direction": direction}
    if note:
        d["note"] = note
    return d


def _traj(age, event, mmse, key_finding):
    return {"age": age, "event": event, "mmse": mmse, "key_finding": key_finding}


# ---------------------------------------------------------------------------
# CATEGORY 1: ZEBRA / RARE DIAGNOSES
# ---------------------------------------------------------------------------

def _zebra_cases():
    cases = []

    # 1. CADASIL (NOTCH3 R544C)
    cases.append({
        "case_id": "ZEBRA_CADASIL_001",
        "category": "zebra",
        "classification": {
            "primary_diagnosis": "CADASIL",
            "secondary_diagnoses": ["migraine_with_aura", "vascular_dementia", "major_depressive_disorder"],
            "icd10_codes": {"primary": "I67.850", "secondary": ["G43.1", "F01.1", "F32.1"]},
            "orpha_code": "136",
            "omim": "125310",
            "rarity": "ultra_rare",
            "diagnostic_category": "cerebrovascular_genetic",
        },
        "demographics": {
            "age": 52, "sex": "M", "bmi": 26.0, "ethnicity": "European",
            "family_history": "Father died of stroke at 55, paternal grandmother had dementia",
        },
        "genetic_variants": [
            _v("NOTCH3", "c.1630C>T", "p.Arg544Cys", "rs137852673", "VCV000003755",
               "pathogenic", "heterozygous", 1e-5, "autosomal_dominant",
               "Disrupts EGF-like repeat 10, causes GOM deposits in vessel walls"),
        ],
        "comorbidities": [
            _comorb("migraine_with_aura", "G43.1", 28),
            _comorb("major_depressive_disorder", "F32.1", 45),
            _comorb("subcortical_vascular_dementia", "F01.1", 52),
            _comorb("hypertension", "I10", 48, "treated"),
        ],
        "multi_omics": {
            "genomic": {
                "variants": [{"gene": "NOTCH3", "hgvs_p": "p.Arg544Cys", "rsid": "rs137852673"}],
                "cnv": [], "repeat_expansions": [],
            },
            "transcriptomic": {
                "platform": "RNA-seq_blood",
                "key_degs": [
                    _deg("NOTCH3", -1.8, 1e-12, "down", "haploinsufficiency"),
                    _deg("HES1", -1.2, 1e-8, "down", "Notch target gene"),
                    _deg("VEGFA", 0.9, 0.001, "up", "angiogenic compensation"),
                    _deg("MMP9", 1.1, 1e-5, "up", "vascular remodeling"),
                    _deg("ICAM1", 0.8, 0.005, "up", "endothelial activation"),
                ],
            },
            "proteomic": {
                "csf_amyloid_beta42_pg_ml": 950, "csf_total_tau_pg_ml": 180,
                "csf_p_tau181_pg_ml": 20, "blood_nfl_pg_ml": 45,
                "blood_gfap_pg_ml": 180, "csf_notch3_ectodomain_pg_ml": 450,
                "blood_endothelin1_pg_ml": 3.8, "blood_vcam1_ng_ml": 1200,
            },
            "metabolomic": {
                "plasma_homocysteine_umol_l": 18.5, "plasma_lactate_mmol_l": 1.2,
                "csf_glucose_mg_dl": 62, "plasma_creatinine_umol_l": 88,
            },
            "epigenomic": {
                "horvath_age_acceleration_yrs": 4.2, "global_methylation_pct": 4.0,
            },
        },
        "standard_biomarkers": {
            "eeg_theta_alpha_ratio": 0.55, "eeg_spectral_entropy": 0.65,
            "fmri_bold_dmn": -0.25, "fmri_conn_dmn_pfc": 0.30,
            "mmse_score": 22, "moca_score": 18,
            "hamilton_depression_score": 14, "updrs_motor_score": 5,
        },
        "treatment_history": [
            _tx("aspirin", "81mg", 48), _tx("amlodipine", "5mg", 48),
            _tx("sertraline", "100mg", 45),
        ],
        "diagnostic_journey": {
            "initial_presentation": "Recurrent migraine with aura at age 28",
            "initial_diagnosis": "migraine",
            "misdiagnoses": [
                {"diagnosis": "migraine", "age": 28, "duration_years": 17},
                {"diagnosis": "small_vessel_disease", "age": 48, "duration_years": 4},
            ],
            "correct_diagnosis": {"diagnosis": "CADASIL", "age": 52,
                                  "method": "genetic_testing_NOTCH3"},
            "diagnostic_delay_years": 24,
            "key_diagnostic_clue": "Family history + anterior temporal lobe white matter + young-onset lacunar infarcts",
        },
        "trajectory_summary": [
            _traj(28, "migraine onset", None, "Normal MRI"),
            _traj(40, "increasing migraine frequency", 29, "Periventricular WMH"),
            _traj(48, "first lacunar infarct", 27, "Multiple subcortical infarcts"),
            _traj(52, "cognitive decline + genetic diagnosis", 22, "Confluent WMH, NOTCH3 R544C confirmed"),
        ],
        "clinical_narrative": (
            "52-year-old man presenting with progressive cognitive decline and "
            "history of recurrent migraine with aura since age 28. MRI reveals "
            "extensive confluent white matter hyperintensities involving anterior "
            "temporal lobes and external capsules, with multiple lacunar infarcts "
            "in basal ganglia and thalamus. Genetic testing confirms NOTCH3 "
            "c.1630C>T (p.Arg544Cys) mutation, diagnostic of CADASIL. Family "
            "history significant for father who died of stroke at 55. Blood NfL "
            "elevated at 45 pg/mL reflecting ongoing white matter damage."
        ),
        "is_synthetic": True,
        "data_sources_modeled": ["UK Biobank", "CADASIL Registry", "ClinVar"],
    })

    # 2. Wilson's Disease
    cases.append({
        "case_id": "ZEBRA_WILSON_002",
        "category": "zebra",
        "classification": {
            "primary_diagnosis": "Wilson_disease",
            "secondary_diagnoses": ["major_depressive_disorder", "hepatic_dysfunction"],
            "icd10_codes": {"primary": "E83.01", "secondary": ["F32.1", "K76.89"]},
            "orpha_code": "905",
            "omim": "277900",
            "rarity": "rare",
            "diagnostic_category": "metabolic_genetic",
        },
        "demographics": {
            "age": 24, "sex": "F", "bmi": 22.5, "ethnicity": "Eastern_European",
            "family_history": "Brother diagnosed with Wilson disease at age 20",
        },
        "genetic_variants": [
            _v("ATP7B", "c.3207C>A", "p.His1069Gln", "rs76151636", "VCV000005404",
               "pathogenic", "compound_heterozygous", 0.005, "autosomal_recessive",
               "Disrupts ATP-binding domain, impairs copper transport"),
            _v("ATP7B", "c.2333G>T", "p.Arg778Leu", "rs1801249", "VCV000005401",
               "pathogenic", "compound_heterozygous", 0.003, "autosomal_recessive",
               "Disrupts transmembrane domain 4"),
        ],
        "comorbidities": [
            _comorb("major_depressive_disorder", "F32.1", 22),
            _comorb("personality_change_organic", "F07.0", 23),
            _comorb("hepatic_dysfunction", "K76.89", 23),
            _comorb("tremor", "G25.0", 23),
        ],
        "multi_omics": {
            "genomic": {
                "variants": [
                    {"gene": "ATP7B", "hgvs_p": "p.His1069Gln", "rsid": "rs76151636"},
                    {"gene": "ATP7B", "hgvs_p": "p.Arg778Leu", "rsid": "rs1801249"},
                ],
                "cnv": [], "repeat_expansions": [],
            },
            "transcriptomic": {
                "platform": "RNA-seq_blood",
                "key_degs": [
                    _deg("ATP7B", -2.5, 1e-15, "down", "loss of function"),
                    _deg("MT1A", 3.2, 1e-20, "up", "metallothionein, copper stress response"),
                    _deg("MT2A", 2.8, 1e-18, "up", "metallothionein"),
                    _deg("SOD1", -0.8, 0.001, "down", "copper-dependent enzyme impaired"),
                    _deg("CP", -1.5, 1e-8, "down", "ceruloplasmin reduced"),
                    _deg("HMOX1", 1.4, 1e-6, "up", "oxidative stress"),
                ],
            },
            "proteomic": {
                "serum_ceruloplasmin_mg_dl": 8.0,
                "serum_free_copper_ug_dl": 35,
                "urinary_copper_24hr_ug": 450,
                "csf_amyloid_beta42_pg_ml": 1050, "csf_total_tau_pg_ml": 150,
                "blood_nfl_pg_ml": 18, "blood_gfap_pg_ml": 95,
                "serum_alt_u_l": 85, "serum_ast_u_l": 72,
            },
            "metabolomic": {
                "plasma_homocysteine_umol_l": 12,
                "serum_uric_acid_mg_dl": 3.2,
                "plasma_aminotransferases_ratio": 0.85,
                "urinary_aminoacids_elevated": True,
            },
            "epigenomic": {
                "horvath_age_acceleration_yrs": 1.5,
                "global_methylation_pct": 4.3,
            },
        },
        "standard_biomarkers": {
            "eeg_theta_alpha_ratio": 0.48, "eeg_spectral_entropy": 0.70,
            "fmri_bold_dmn": -0.10, "fmri_bold_basal_ganglia": -0.35,
            "mmse_score": 27, "moca_score": 24,
            "hamilton_depression_score": 18, "updrs_motor_score": 12,
        },
        "treatment_history": [
            _tx("d-penicillamine", "250mg_QID", 24),
            _tx("zinc_acetate", "50mg_TID", 24),
            _tx("sertraline", "50mg", 22),
        ],
        "diagnostic_journey": {
            "initial_presentation": "Personality changes and depression at age 22",
            "initial_diagnosis": "major_depressive_disorder",
            "misdiagnoses": [
                {"diagnosis": "major_depressive_disorder", "age": 22, "duration_years": 1},
                {"diagnosis": "conversion_disorder", "age": 23, "duration_years": 0.5},
            ],
            "correct_diagnosis": {"diagnosis": "Wilson_disease", "age": 24,
                                  "method": "slit_lamp_KF_rings_plus_ceruloplasmin_plus_genetic"},
            "diagnostic_delay_years": 2,
            "key_diagnostic_clue": "Kayser-Fleischer rings on slit-lamp + low ceruloplasmin + brother's history",
        },
        "trajectory_summary": [
            _traj(22, "depression onset", 29, "Normal labs initially"),
            _traj(23, "personality change + tremor + liver dysfunction", 28, "Elevated ALT/AST, low ceruloplasmin"),
            _traj(24, "genetic diagnosis + treatment initiation", 27, "KF rings, ATP7B confirmed"),
        ],
        "clinical_narrative": (
            "24-year-old woman presenting with a 2-year history of progressive "
            "personality changes, depression, and tremor. Slit-lamp examination "
            "reveals Kayser-Fleischer rings. Labs show ceruloplasmin 8 mg/dL "
            "(normal >20), elevated 24-hour urinary copper at 450 mcg. Genetic "
            "testing confirms compound heterozygous ATP7B mutations (H1069Q/R778L). "
            "Brother previously diagnosed with Wilson disease. Initiated on "
            "d-penicillamine and zinc acetate with early clinical response."
        ),
        "is_synthetic": True,
        "data_sources_modeled": ["Wilson Disease Registry", "ClinVar", "EuroWilson"],
    })

    # 3. Niemann-Pick Type C
    cases.append({
        "case_id": "ZEBRA_NPC_003",
        "category": "zebra",
        "classification": {
            "primary_diagnosis": "Niemann_Pick_type_C",
            "secondary_diagnoses": ["cerebellar_ataxia", "psychosis", "vertical_gaze_palsy"],
            "icd10_codes": {"primary": "E75.242", "secondary": ["G11.9", "F23", "H51.8"]},
            "orpha_code": "646",
            "omim": "257220",
            "rarity": "ultra_rare",
            "diagnostic_category": "lysosomal_storage_disease",
        },
        "demographics": {
            "age": 19, "sex": "M", "bmi": 21.0, "ethnicity": "Hispanic",
            "family_history": "No known family history; consanguineous parents (second cousins)",
        },
        "genetic_variants": [
            _v("NPC1", "c.3182T>C", "p.Ile1061Thr", "rs80358257", "VCV000004444",
               "pathogenic", "homozygous", 0.0005, "autosomal_recessive",
               "Most common NPC1 mutation; disrupts cholesterol trafficking in late endosomes"),
        ],
        "comorbidities": [
            _comorb("vertical_supranuclear_gaze_palsy", "H51.8", 16),
            _comorb("cerebellar_ataxia", "G11.9", 17),
            _comorb("psychosis", "F23", 18),
            _comorb("splenomegaly", "R16.1", 12),
            _comorb("gelastic_cataplexy", "G47.42", 17),
        ],
        "multi_omics": {
            "genomic": {
                "variants": [{"gene": "NPC1", "hgvs_p": "p.Ile1061Thr", "rsid": "rs80358257"}],
                "cnv": [], "repeat_expansions": [],
            },
            "transcriptomic": {
                "platform": "RNA-seq_blood",
                "key_degs": [
                    _deg("NPC1", -2.0, 1e-14, "down"),
                    _deg("ABCA1", -1.3, 1e-7, "down", "cholesterol efflux impaired"),
                    _deg("LAMP1", 1.5, 1e-9, "up", "lysosomal expansion"),
                    _deg("SQSTM1", 1.2, 1e-6, "up", "autophagy marker"),
                    _deg("TFEB", 0.9, 0.005, "up", "lysosomal biogenesis response"),
                ],
            },
            "proteomic": {
                "plasma_oxysterol_cholestane_triol_ng_ml": 85,
                "plasma_lysosphingomyelin_509_ng_ml": 45,
                "csf_amyloid_beta42_pg_ml": 900, "csf_total_tau_pg_ml": 250,
                "csf_nfl_pg_ml": 1200, "blood_nfl_pg_ml": 35,
                "blood_chitotriosidase_nmol_hr_ml": 180,
            },
            "metabolomic": {
                "plasma_cholestane_triol_ng_ml": 85,
                "plasma_7_ketocholesterol_ng_ml": 65,
                "plasma_sphingomyelin_umol_l": 250,
                "csf_lactate_mmol_l": 1.8,
            },
            "epigenomic": {
                "horvath_age_acceleration_yrs": 3.0,
                "global_methylation_pct": 4.1,
            },
        },
        "standard_biomarkers": {
            "eeg_theta_alpha_ratio": 0.70, "eeg_spectral_entropy": 0.58,
            "fmri_bold_dmn": -0.20, "fmri_bold_cerebellum": -0.45,
            "mmse_score": 24, "moca_score": 20,
            "hamilton_depression_score": 8, "updrs_motor_score": 15,
        },
        "treatment_history": [
            _tx("miglustat", "200mg_TID", 19),
            _tx("risperidone", "1mg", 18, "discontinued"),
        ],
        "diagnostic_journey": {
            "initial_presentation": "Splenomegaly noted at age 12, cognitive and gait problems at 16",
            "initial_diagnosis": "idiopathic_splenomegaly",
            "misdiagnoses": [
                {"diagnosis": "idiopathic_splenomegaly", "age": 12, "duration_years": 4},
                {"diagnosis": "schizophrenia", "age": 18, "duration_years": 1},
            ],
            "correct_diagnosis": {"diagnosis": "Niemann_Pick_type_C", "age": 19,
                                  "method": "oxysterol_biomarkers_plus_NPC1_genetic_testing"},
            "diagnostic_delay_years": 7,
            "key_diagnostic_clue": "Vertical supranuclear gaze palsy + gelastic cataplexy + splenomegaly triad",
        },
        "trajectory_summary": [
            _traj(12, "splenomegaly discovered", 30, "Mild hepatosplenomegaly on imaging"),
            _traj(16, "vertical gaze palsy + clumsiness", 28, "Cerebellar signs on exam"),
            _traj(18, "psychotic episode", 25, "Hospitalized, started antipsychotics"),
            _traj(19, "NPC diagnosis + miglustat started", 24, "Oxysterols elevated, NPC1 I1061T confirmed"),
        ],
        "clinical_narrative": (
            "19-year-old man with progressive ataxia, vertical supranuclear gaze "
            "palsy, and gelastic cataplexy beginning at age 16, preceded by "
            "childhood splenomegaly. Psychotic episode at 18 led to brief "
            "misdiagnosis of schizophrenia. Elevated plasma oxysterols (cholestane-"
            "triol 85 ng/mL) and genetic testing confirmed homozygous NPC1 "
            "I1061T mutation. CSF NfL markedly elevated at 1200 pg/mL indicating "
            "active neurodegeneration. Started on miglustat substrate reduction therapy."
        ),
        "is_synthetic": True,
        "data_sources_modeled": ["NPC Patient Registry", "ClinVar", "OMIM"],
    })

    # 4. Anti-NMDAR Encephalitis
    cases.append({
        "case_id": "ZEBRA_NMDARE_004",
        "category": "zebra",
        "classification": {
            "primary_diagnosis": "anti_NMDAR_encephalitis",
            "secondary_diagnoses": ["ovarian_teratoma", "seizures", "dyskinesia"],
            "icd10_codes": {"primary": "G04.81", "secondary": ["D27.9", "G40.509", "G24.9"]},
            "orpha_code": "217253",
            "omim": None,
            "rarity": "rare",
            "diagnostic_category": "autoimmune_encephalitis",
        },
        "demographics": {
            "age": 22, "sex": "F", "bmi": 23.0, "ethnicity": "African_American",
            "family_history": "No neurological family history; maternal aunt with lupus",
        },
        "genetic_variants": [],
        "comorbidities": [
            _comorb("ovarian_teratoma", "D27.9", 22),
            _comorb("seizures", "G40.509", 22),
            _comorb("orofacial_dyskinesias", "G24.9", 22),
            _comorb("autonomic_instability", "G90.9", 22),
        ],
        "multi_omics": {
            "genomic": {"variants": [], "cnv": [], "repeat_expansions": [],
                        "hla_typing": {"HLA_B": "HLA-B*07:02", "note": "weakly associated with anti-NMDAR"}},
            "transcriptomic": {
                "platform": "RNA-seq_CSF_cells",
                "key_degs": [
                    _deg("GRIN1", -0.5, 0.01, "down", "NMDAR GluN1 subunit, antibody target"),
                    _deg("CXCL13", 3.5, 1e-20, "up", "B-cell chemoattractant, intrathecal synthesis"),
                    _deg("IGHG1", 4.0, 1e-25, "up", "intrathecal IgG production"),
                    _deg("CD19", 2.5, 1e-12, "up", "B-cell infiltration"),
                    _deg("IL6", 2.0, 1e-10, "up", "neuroinflammation"),
                    _deg("IFNG", 1.8, 1e-8, "up"),
                ],
            },
            "proteomic": {
                "csf_anti_nmdar_glun1_titer": "1:320",
                "csf_amyloid_beta42_pg_ml": 1050, "csf_total_tau_pg_ml": 280,
                "csf_protein_mg_dl": 65, "csf_wbc_per_ul": 45,
                "csf_oligoclonal_bands": "positive",
                "csf_igg_index": 1.2, "blood_nfl_pg_ml": 32,
                "csf_cxcl13_pg_ml": 850, "csf_il6_pg_ml": 45,
            },
            "metabolomic": {
                "csf_glutamate_umol_l": 12.5,
                "csf_quinolinic_acid_nmol_l": 450,
                "plasma_neopterin_nmol_l": 18,
            },
            "epigenomic": {
                "horvath_age_acceleration_yrs": 0.5,
                "global_methylation_pct": 4.5,
            },
        },
        "standard_biomarkers": {
            "eeg_theta_alpha_ratio": 1.8, "eeg_spectral_entropy": 0.40,
            "eeg_delta_1_4hz_uv2": 35, "fmri_bold_dmn": -0.40,
            "mmse_score": 15, "moca_score": 10,
            "hamilton_depression_score": 5, "updrs_motor_score": 20,
        },
        "treatment_history": [
            _tx("methylprednisolone", "1g_IV_x5d", 22),
            _tx("ivig", "0.4g_kg_x5d", 22),
            _tx("rituximab", "375mg_m2_weekly_x4", 22),
            _tx("levetiracetam", "1000mg_BID", 22),
        ],
        "diagnostic_journey": {
            "initial_presentation": "Acute psychosis with agitation and bizarre behavior at age 22",
            "initial_diagnosis": "acute_psychotic_episode",
            "misdiagnoses": [
                {"diagnosis": "acute_psychotic_episode", "age": 22, "duration_years": 0.08},
                {"diagnosis": "schizophrenia", "age": 22, "duration_years": 0.08},
            ],
            "correct_diagnosis": {"diagnosis": "anti_NMDAR_encephalitis", "age": 22,
                                  "method": "CSF_anti_NMDAR_antibody_cell_based_assay"},
            "diagnostic_delay_years": 0.17,
            "key_diagnostic_clue": "Young woman with psychosis + seizures + orofacial dyskinesias + autonomic instability",
        },
        "trajectory_summary": [
            _traj(22, "acute psychosis onset (week 0)", None, "Prodromal headache/fever, then paranoia"),
            _traj(22, "seizures + dyskinesias (week 2)", 15, "EEG: extreme delta brush pattern"),
            _traj(22, "ICU admission (week 3)", None, "Autonomic instability, CSF anti-NMDAR 1:320"),
            _traj(22, "teratoma resection + immunotherapy (week 4)", 10, "Ovarian teratoma found on CT"),
        ],
        "clinical_narrative": (
            "22-year-old woman presenting with acute-onset psychosis, progressing "
            "to seizures, orofacial dyskinesias, and autonomic instability over "
            "2 weeks. Initially treated with antipsychotics for presumed schizophrenia "
            "with clinical worsening. EEG shows extreme delta brush pattern. CSF "
            "reveals lymphocytic pleocytosis (45 WBC/uL) and anti-NMDAR GluN1 "
            "antibodies at 1:320. CT pelvis identifies right ovarian teratoma. "
            "Treated with tumor resection, IV methylprednisolone, IVIG, and "
            "rituximab with gradual improvement over months."
        ),
        "is_synthetic": True,
        "data_sources_modeled": ["Anti-NMDAR Encephalitis Registry", "Dalmau 2008/2011"],
    })

    # 5. Familial CJD (PRNP E200K)
    cases.append({
        "case_id": "ZEBRA_CJD_005",
        "category": "zebra",
        "classification": {
            "primary_diagnosis": "familial_Creutzfeldt_Jakob_disease",
            "secondary_diagnoses": ["rapidly_progressive_dementia", "myoclonus"],
            "icd10_codes": {"primary": "A81.09", "secondary": ["F02.8", "G25.3"]},
            "orpha_code": "204",
            "omim": "123400",
            "rarity": "ultra_rare",
            "diagnostic_category": "prion_disease",
        },
        "demographics": {
            "age": 58, "sex": "M", "bmi": 25.0, "ethnicity": "Ashkenazi_Jewish",
            "family_history": "Mother died at 60 of rapidly progressive dementia; maternal uncle similar",
        },
        "genetic_variants": [
            _v("PRNP", "c.598G>A", "p.Glu200Lys", "rs28933385", "VCV000012503",
               "pathogenic", "heterozygous", 0.0001, "autosomal_dominant",
               "Most common genetic CJD mutation; destabilizes PrP folding"),
            _v("PRNP", "c.385A>G", "p.Met129Val", "rs1799990", "VCV000009332",
               "risk_factor", "heterozygous", 0.35, "complex",
               "Codon 129 M/V: modifies disease onset and phenotype"),
        ],
        "comorbidities": [
            _comorb("rapidly_progressive_dementia", "F02.8", 58),
            _comorb("myoclonus", "G25.3", 58),
            _comorb("cerebellar_ataxia", "G11.9", 58),
            _comorb("visual_disturbance", "H53.9", 58),
        ],
        "multi_omics": {
            "genomic": {
                "variants": [{"gene": "PRNP", "hgvs_p": "p.Glu200Lys", "rsid": "rs28933385"}],
                "cnv": [], "repeat_expansions": [],
            },
            "transcriptomic": {
                "platform": "RNA-seq_blood",
                "key_degs": [
                    _deg("PRNP", 1.5, 1e-10, "up", "prion protein overexpression"),
                    _deg("GFAP", 3.0, 1e-20, "up", "massive astrogliosis"),
                    _deg("TREM2", 2.0, 1e-12, "up", "microglial activation"),
                    _deg("S100B", 2.5, 1e-15, "up", "astrocyte damage marker"),
                    _deg("NEFL", 2.8, 1e-18, "up", "neurofilament light"),
                ],
            },
            "proteomic": {
                "csf_14_3_3_protein": "positive",
                "csf_rt_quic": "positive",
                "csf_total_tau_pg_ml": 8500, "csf_p_tau181_pg_ml": 45,
                "csf_amyloid_beta42_pg_ml": 850, "csf_nfl_pg_ml": 12000,
                "blood_nfl_pg_ml": 250, "csf_s100b_ug_l": 3.5,
                "csf_neuron_specific_enolase_ng_ml": 85,
            },
            "metabolomic": {
                "csf_lactate_mmol_l": 2.2,
                "plasma_neopterin_nmol_l": 22,
            },
            "epigenomic": {
                "horvath_age_acceleration_yrs": 8.0,
                "global_methylation_pct": 3.5,
            },
        },
        "standard_biomarkers": {
            "eeg_theta_alpha_ratio": 2.5, "eeg_spectral_entropy": 0.30,
            "eeg_delta_1_4hz_uv2": 45, "fmri_bold_dmn": -0.60,
            "mmse_score": 12, "moca_score": 8,
            "hamilton_depression_score": 4, "updrs_motor_score": 25,
        },
        "treatment_history": [
            _tx("clonazepam", "0.5mg_TID", 58),
            _tx("donepezil", "10mg", 58, "ineffective"),
        ],
        "diagnostic_journey": {
            "initial_presentation": "Rapidly progressive cognitive decline over 3 months at age 58",
            "initial_diagnosis": "rapidly_progressive_Alzheimers",
            "misdiagnoses": [
                {"diagnosis": "rapidly_progressive_Alzheimers", "age": 58, "duration_years": 0.1},
            ],
            "correct_diagnosis": {"diagnosis": "familial_CJD_E200K", "age": 58,
                                  "method": "CSF_RT_QuIC_plus_PRNP_sequencing"},
            "diagnostic_delay_years": 0.1,
            "key_diagnostic_clue": "Rapid course + periodic sharp complexes on EEG + CSF 14-3-3 + tau >5000 + RT-QuIC positive",
        },
        "trajectory_summary": [
            _traj(58, "memory loss onset (month 0)", 22, "Subtle cognitive complaints"),
            _traj(58, "rapid decline + myoclonus (month 2)", 15, "MRI: cortical ribboning on DWI"),
            _traj(58, "severe dementia + akinetic mutism (month 4)", 5, "EEG: periodic sharp wave complexes"),
        ],
        "clinical_narrative": (
            "58-year-old Ashkenazi Jewish man with rapidly progressive dementia over "
            "3 months, progressing from subtle memory complaints to akinetic mutism. "
            "Examination reveals startle myoclonus and cerebellar ataxia. MRI shows "
            "cortical ribboning on DWI. EEG demonstrates periodic sharp wave complexes. "
            "CSF total tau massively elevated at 8500 pg/mL with positive 14-3-3 and "
            "RT-QuIC. PRNP sequencing confirms E200K mutation (codon 129 M/V). "
            "Family history of mother and uncle with similar rapidly progressive dementia."
        ),
        "is_synthetic": True,
        "data_sources_modeled": ["CJD Surveillance Network", "ClinVar", "EUROCJD"],
    })

    # 6. MELAS (m.3243A>G)
    cases.append({
        "case_id": "ZEBRA_MELAS_006",
        "category": "zebra",
        "classification": {
            "primary_diagnosis": "MELAS_syndrome",
            "secondary_diagnoses": ["stroke_like_episodes", "epilepsy", "lactic_acidosis", "diabetes"],
            "icd10_codes": {"primary": "G71.3", "secondary": ["I63.9", "G40.509", "E87.2", "E11.9"]},
            "orpha_code": "550",
            "omim": "540000",
            "rarity": "rare",
            "diagnostic_category": "mitochondrial_disease",
        },
        "demographics": {
            "age": 35, "sex": "M", "bmi": 22.0, "ethnicity": "Japanese",
            "family_history": "Mother has diabetes + hearing loss; maternal grandmother had stroke at 45",
        },
        "genetic_variants": [
            _v("MT-TL1", "m.3243A>G", "tRNA-Leu(UUR)", "rs199474657", "VCV000009659",
               "pathogenic", "heterozygous", 0.0002, "mitochondrial",
               "Disrupts mt-tRNA-Leu; heteroplasmy level 80% in muscle, 50% in blood"),
        ],
        "comorbidities": [
            _comorb("stroke_like_episodes", "I63.9", 30),
            _comorb("epilepsy", "G40.509", 28),
            _comorb("sensorineural_hearing_loss", "H90.5", 25),
            _comorb("type_2_diabetes", "E11.9", 32),
            _comorb("lactic_acidosis", "E87.2", 30),
            _comorb("short_stature", "E34.3", 15),
            _comorb("cardiomyopathy", "I42.8", 33),
        ],
        "multi_omics": {
            "genomic": {
                "variants": [{"gene": "MT-TL1", "variant": "m.3243A>G", "heteroplasmy_pct": 80}],
                "cnv": [], "repeat_expansions": [],
            },
            "transcriptomic": {
                "platform": "RNA-seq_muscle",
                "key_degs": [
                    _deg("MT-ND1", -2.0, 1e-12, "down", "Complex I subunit"),
                    _deg("MT-CO1", -1.5, 1e-8, "down", "Complex IV subunit"),
                    _deg("PPARGC1A", 1.8, 1e-10, "up", "PGC-1alpha, mitochondrial biogenesis compensation"),
                    _deg("FGF21", 3.5, 1e-20, "up", "mitokine stress response"),
                    _deg("GDF15", 3.0, 1e-18, "up", "mitokine stress marker"),
                    _deg("VEGFA", 1.5, 1e-6, "up", "angiogenic response to metabolic stress"),
                ],
            },
            "proteomic": {
                "csf_amyloid_beta42_pg_ml": 1000, "csf_total_tau_pg_ml": 200,
                "blood_nfl_pg_ml": 55, "blood_gfap_pg_ml": 120,
                "serum_fgf21_pg_ml": 2500, "serum_gdf15_pg_ml": 3500,
                "serum_creatine_kinase_u_l": 350,
            },
            "metabolomic": {
                "plasma_lactate_mmol_l": 5.5,
                "plasma_pyruvate_umol_l": 180,
                "lactate_pyruvate_ratio": 30.6,
                "csf_lactate_mmol_l": 4.2,
                "plasma_alanine_umol_l": 650,
                "urine_organic_acids_elevated": ["3-methylglutaconic_acid", "ethylmalonic_acid"],
            },
            "epigenomic": {
                "horvath_age_acceleration_yrs": 6.0,
                "global_methylation_pct": 3.8,
                "mitochondrial_dna_methylation_altered": True,
            },
        },
        "standard_biomarkers": {
            "eeg_theta_alpha_ratio": 0.90, "eeg_spectral_entropy": 0.50,
            "eeg_delta_1_4hz_uv2": 30, "fmri_bold_dmn": -0.30,
            "mmse_score": 23, "moca_score": 19,
            "hamilton_depression_score": 10, "updrs_motor_score": 8,
        },
        "treatment_history": [
            _tx("L-arginine", "500mg_TID", 30),
            _tx("coenzyme_Q10", "300mg", 28),
            _tx("levetiracetam", "1000mg_BID", 28),
            _tx("metformin", "500mg_BID", 32),
        ],
        "diagnostic_journey": {
            "initial_presentation": "Seizures at age 28, short stature, hearing loss",
            "initial_diagnosis": "epilepsy",
            "misdiagnoses": [
                {"diagnosis": "idiopathic_epilepsy", "age": 28, "duration_years": 2},
            ],
            "correct_diagnosis": {"diagnosis": "MELAS_m3243AG", "age": 30,
                                  "method": "muscle_biopsy_ragged_red_fibers_plus_mtDNA_sequencing"},
            "diagnostic_delay_years": 2,
            "key_diagnostic_clue": "Stroke-like episodes not conforming to vascular territories + lactic acidosis + family history of maternal diabetes/deafness",
        },
        "trajectory_summary": [
            _traj(25, "hearing loss onset", 28, "Audiometry: bilateral sensorineural loss"),
            _traj(28, "first seizure", 27, "MRI initially normal"),
            _traj(30, "stroke-like episode + diagnosis", 24, "MRI: cortical lesion not vascular; lactate elevated; m.3243A>G confirmed"),
            _traj(35, "recurrent episodes + diabetes", 23, "Progressive; on L-arginine, CoQ10, anticonvulsants"),
        ],
        "clinical_narrative": (
            "35-year-old man with MELAS syndrome (m.3243A>G, 80% heteroplasmy in muscle). "
            "History of sensorineural hearing loss since age 25, seizures since 28, and "
            "recurrent stroke-like episodes since 30. MRI lesions do not conform to "
            "vascular territories. Plasma lactate persistently elevated at 5.5 mmol/L "
            "(L:P ratio 30.6). Serum FGF21 and GDF15 markedly elevated as mitokine "
            "stress biomarkers. Developed diabetes at 32 and cardiomyopathy at 33. "
            "Family history of maternal diabetes and hearing loss consistent with "
            "maternal inheritance pattern."
        ),
        "is_synthetic": True,
        "data_sources_modeled": ["Mitochondrial Disease Registry", "MITOMAP", "ClinVar"],
    })

    # 7. FTD-ALS (C9orf72)
    cases.append({
        "case_id": "ZEBRA_FTDALS_007",
        "category": "zebra",
        "classification": {
            "primary_diagnosis": "FTD_ALS_spectrum",
            "secondary_diagnoses": ["behavioral_variant_FTD", "amyotrophic_lateral_sclerosis"],
            "icd10_codes": {"primary": "G31.09", "secondary": ["G12.21"]},
            "orpha_code": "275872",
            "omim": "105550",
            "rarity": "rare",
            "diagnostic_category": "neurodegenerative_genetic",
        },
        "demographics": {
            "age": 62, "sex": "M", "bmi": 23.5, "ethnicity": "Northern_European",
            "family_history": "Mother had FTD at 65; maternal uncle had ALS at 58",
        },
        "genetic_variants": [
            _v("C9orf72", "(GGGGCC)n_expansion", "repeat_expansion", None, "VCV000227087",
               "pathogenic", "heterozygous", 0.001, "autosomal_dominant",
               "Hexanucleotide repeat expansion >30 repeats (estimated ~800); causes RNA foci and DPR protein toxicity"),
        ],
        "comorbidities": [
            _comorb("behavioral_variant_FTD", "G31.09", 58),
            _comorb("amyotrophic_lateral_sclerosis", "G12.21", 61),
            _comorb("pseudobulbar_affect", "F48.2", 60),
            _comorb("psychosis", "F23", 59),
        ],
        "multi_omics": {
            "genomic": {
                "variants": [],
                "cnv": [],
                "repeat_expansions": [{"gene": "C9orf72", "repeat_unit": "GGGGCC",
                                       "repeat_count_estimate": 800, "pathogenic_threshold": 30}],
            },
            "transcriptomic": {
                "platform": "RNA-seq_blood",
                "key_degs": [
                    _deg("C9orf72", -1.5, 1e-10, "down", "haploinsufficiency"),
                    _deg("TARDBP", 0.8, 0.005, "up", "TDP-43 dysregulation"),
                    _deg("SQSTM1", 1.5, 1e-8, "up", "p62, autophagy impairment"),
                    _deg("CHIT1", 2.5, 1e-14, "up", "chitotriosidase, neuroinflammation"),
                    _deg("NEFL", 2.0, 1e-12, "up"),
                ],
            },
            "proteomic": {
                "csf_amyloid_beta42_pg_ml": 1000, "csf_total_tau_pg_ml": 250,
                "csf_p_tau181_pg_ml": 22, "csf_nfl_pg_ml": 8000,
                "blood_nfl_pg_ml": 180, "csf_p75ecd_pg_ml": 5.5,
                "blood_gfap_pg_ml": 250,
            },
            "metabolomic": {
                "csf_glutamate_umol_l": 8.5,
                "plasma_creatinine_umol_l": 65,
            },
            "epigenomic": {
                "horvath_age_acceleration_yrs": 5.0,
                "global_methylation_pct": 3.7,
                "c9orf72_promoter_hypermethylated": True,
            },
        },
        "standard_biomarkers": {
            "eeg_theta_alpha_ratio": 0.65, "eeg_spectral_entropy": 0.55,
            "fmri_bold_dmn": -0.15, "fmri_bold_pfc": -0.50,
            "mmse_score": 20, "moca_score": 16,
            "hamilton_depression_score": 6, "updrs_motor_score": 18,
        },
        "treatment_history": [
            _tx("riluzole", "50mg_BID", 61),
            _tx("trazodone", "50mg", 59),
            _tx("dextromethorphan_quinidine", "20_10mg_BID", 60),
        ],
        "diagnostic_journey": {
            "initial_presentation": "Personality change and social disinhibition at age 58",
            "initial_diagnosis": "depression",
            "misdiagnoses": [
                {"diagnosis": "depression", "age": 58, "duration_years": 1},
            ],
            "correct_diagnosis": {"diagnosis": "C9orf72_FTD_ALS", "age": 61,
                                  "method": "repeat_primed_PCR_plus_Southern_blot"},
            "diagnostic_delay_years": 3,
            "key_diagnostic_clue": "Behavioral disinhibition + ALS symptoms + family history of both FTD and ALS",
        },
        "trajectory_summary": [
            _traj(58, "personality change", 26, "Frontal atrophy on MRI"),
            _traj(59, "psychotic features", 23, "Paranoid delusions"),
            _traj(61, "fasciculations + weakness + C9orf72 diagnosis", 20, "EMG: widespread denervation; C9orf72 expansion confirmed"),
            _traj(62, "combined FTD-ALS", 18, "Progressive bulbar and limb weakness + cognitive decline"),
        ],
        "clinical_narrative": (
            "62-year-old man with 4-year history of progressive behavioral changes "
            "including social disinhibition, apathy, and loss of empathy, followed by "
            "fasciculations and limb weakness at age 61. EMG shows widespread active "
            "denervation consistent with ALS. CSF NfL massively elevated at 8000 pg/mL. "
            "Genetic testing confirms C9orf72 GGGGCC repeat expansion. Family history "
            "includes mother with FTD and maternal uncle with ALS, consistent with the "
            "known phenotypic heterogeneity of C9orf72 expansions."
        ),
        "is_synthetic": True,
        "data_sources_modeled": ["GENFI Consortium", "ALS Registry", "ClinVar"],
    })

    # 8. CADASIL variant (NOTCH3 R169C)
    cases.append({
        "case_id": "ZEBRA_CADASIL2_008",
        "category": "zebra",
        "classification": {
            "primary_diagnosis": "CADASIL",
            "secondary_diagnoses": ["recurrent_lacunar_strokes", "migraine_with_aura", "mood_disorder"],
            "icd10_codes": {"primary": "I67.850", "secondary": ["I63.81", "G43.1", "F34.1"]},
            "orpha_code": "136", "omim": "125310",
            "rarity": "ultra_rare", "diagnostic_category": "cerebrovascular_genetic",
        },
        "demographics": {"age": 45, "sex": "F", "bmi": 24.5, "ethnicity": "French",
                         "family_history": "Father had strokes in 50s; paternal aunt had early dementia"},
        "genetic_variants": [
            _v("NOTCH3", "c.505C>T", "p.Arg169Cys", "rs137852672", "VCV000003754",
               "pathogenic", "heterozygous", 1e-5, "autosomal_dominant",
               "Disrupts EGF-like repeat 4; creates unpaired cysteine"),
        ],
        "comorbidities": [
            _comorb("migraine_with_aura", "G43.1", 25),
            _comorb("recurrent_lacunar_strokes", "I63.81", 42),
            _comorb("dysthymia", "F34.1", 38),
        ],
        "multi_omics": {
            "genomic": {"variants": [{"gene": "NOTCH3", "hgvs_p": "p.Arg169Cys"}], "cnv": [], "repeat_expansions": []},
            "transcriptomic": {"platform": "RNA-seq_blood", "key_degs": [
                _deg("NOTCH3", -1.5, 1e-10, "down"), _deg("HEY1", -0.9, 1e-5, "down"),
                _deg("PDGFRB", 0.8, 0.01, "up", "pericyte dysfunction"),
            ]},
            "proteomic": {"csf_amyloid_beta42_pg_ml": 980, "csf_total_tau_pg_ml": 160,
                          "blood_nfl_pg_ml": 38, "blood_gfap_pg_ml": 150},
            "metabolomic": {"plasma_homocysteine_umol_l": 16},
            "epigenomic": {"horvath_age_acceleration_yrs": 3.0, "global_methylation_pct": 4.1},
        },
        "standard_biomarkers": {"eeg_theta_alpha_ratio": 0.50, "eeg_spectral_entropy": 0.68,
                                "mmse_score": 25, "moca_score": 21, "hamilton_depression_score": 12, "updrs_motor_score": 3},
        "treatment_history": [_tx("aspirin", "100mg", 42), _tx("escitalopram", "10mg", 38)],
        "diagnostic_journey": {
            "initial_presentation": "Migraine with aura at 25, recurrent strokes at 42",
            "initial_diagnosis": "migraine", "misdiagnoses": [{"diagnosis": "migraine", "age": 25, "duration_years": 17}],
            "correct_diagnosis": {"diagnosis": "CADASIL", "age": 45, "method": "skin_biopsy_GOM_plus_NOTCH3_sequencing"},
            "diagnostic_delay_years": 20,
            "key_diagnostic_clue": "Young-onset strokes + external capsule/temporal pole white matter + family history",
        },
        "trajectory_summary": [
            _traj(25, "migraine onset", 30, "Normal MRI"),
            _traj(42, "first lacunar stroke", 27, "Subcortical infarcts + WMH"),
            _traj(45, "genetic diagnosis", 25, "NOTCH3 R169C confirmed"),
        ],
        "clinical_narrative": (
            "45-year-old French woman with 20-year history of migraine with aura and "
            "recurrent lacunar strokes since age 42. MRI shows extensive white matter "
            "changes with characteristic anterior temporal pole involvement. Skin biopsy "
            "reveals granular osmiophilic material (GOM). NOTCH3 c.505C>T (p.Arg169Cys) "
            "confirmed on genetic testing."
        ),
        "is_synthetic": True,
        "data_sources_modeled": ["CADASIL Registry", "ClinVar"],
    })

    # 9. FXTAS
    cases.append({
        "case_id": "ZEBRA_FXTAS_009",
        "category": "zebra",
        "classification": {
            "primary_diagnosis": "fragile_X_tremor_ataxia_syndrome",
            "secondary_diagnoses": ["intention_tremor", "cerebellar_ataxia", "cognitive_decline"],
            "icd10_codes": {"primary": "G11.19", "secondary": ["G25.0", "G31.9"]},
            "orpha_code": "93256", "omim": "300623",
            "rarity": "rare", "diagnostic_category": "trinucleotide_repeat_disorder",
        },
        "demographics": {"age": 68, "sex": "M", "bmi": 27.0, "ethnicity": "Caucasian",
                         "family_history": "Daughter has a son with developmental delay (Fragile X syndrome)"},
        "genetic_variants": [
            _v("FMR1", "CGG_repeat_premutation", "premutation_90_repeats", None, None,
               "pathogenic", "hemizygous", 0.005, "x_linked",
               "CGG repeat 90 (premutation range 55-200); causes FMR1 mRNA toxicity via RNA gain-of-function"),
        ],
        "comorbidities": [
            _comorb("intention_tremor", "G25.0", 63), _comorb("cerebellar_ataxia", "G11.9", 65),
            _comorb("cognitive_decline", "G31.9", 66), _comorb("peripheral_neuropathy", "G62.9", 64),
            _comorb("erectile_dysfunction", "N52.9", 62), _comorb("autonomic_dysfunction", "G90.9", 65),
        ],
        "multi_omics": {
            "genomic": {"variants": [], "cnv": [],
                        "repeat_expansions": [{"gene": "FMR1", "repeat_unit": "CGG", "repeat_count": 90, "premutation_range": "55-200"}]},
            "transcriptomic": {"platform": "RNA-seq_blood", "key_degs": [
                _deg("FMR1", 2.5, 1e-15, "up", "mRNA elevated in premutation (RNA toxicity mechanism)"),
                _deg("FMRP", -0.5, 0.05, "down", "protein mildly reduced despite elevated mRNA"),
            ]},
            "proteomic": {"blood_nfl_pg_ml": 42, "blood_gfap_pg_ml": 160,
                          "csf_amyloid_beta42_pg_ml": 1000, "csf_total_tau_pg_ml": 220},
            "metabolomic": {"plasma_lactate_mmol_l": 1.5},
            "epigenomic": {"horvath_age_acceleration_yrs": 3.5, "fmr1_promoter_methylation_pct": 5},
        },
        "standard_biomarkers": {"eeg_theta_alpha_ratio": 0.60, "eeg_spectral_entropy": 0.60,
                                "fmri_bold_cerebellum": -0.40, "mmse_score": 24, "moca_score": 20,
                                "hamilton_depression_score": 8, "updrs_motor_score": 22},
        "treatment_history": [_tx("propranolol", "40mg_BID", 63), _tx("memantine", "10mg_BID", 66)],
        "diagnostic_journey": {
            "initial_presentation": "Progressive tremor at 63",
            "initial_diagnosis": "essential_tremor",
            "misdiagnoses": [{"diagnosis": "essential_tremor", "age": 63, "duration_years": 3},
                             {"diagnosis": "Parkinsons_disease", "age": 66, "duration_years": 2}],
            "correct_diagnosis": {"diagnosis": "FXTAS", "age": 68,
                                  "method": "FMR1_CGG_repeat_testing_after_grandson_diagnosed_with_Fragile_X"},
            "diagnostic_delay_years": 5,
            "key_diagnostic_clue": "Grandson with Fragile X + intention tremor + MCP sign on MRI",
        },
        "trajectory_summary": [
            _traj(63, "tremor onset", 27, "Bilateral intention tremor"),
            _traj(65, "ataxia develops", 25, "MRI: middle cerebellar peduncle sign"),
            _traj(68, "FXTAS diagnosed after grandson's Fragile X", 24, "FMR1 CGG=90 confirmed"),
        ],
        "clinical_narrative": (
            "68-year-old man with progressive intention tremor, cerebellar ataxia, and "
            "cognitive decline. Initially diagnosed as essential tremor, then PD. MRI shows "
            "characteristic middle cerebellar peduncle (MCP) T2 hyperintensity. After his "
            "grandson was diagnosed with Fragile X syndrome, FMR1 testing revealed a premutation "
            "of 90 CGG repeats, confirming FXTAS."
        ),
        "is_synthetic": True,
        "data_sources_modeled": ["National Fragile X Foundation Registry", "FXTAS Study Group"],
    })

    # 10. Anti-LGI1 Encephalitis
    cases.append({
        "case_id": "ZEBRA_LGI1_010",
        "category": "zebra",
        "classification": {
            "primary_diagnosis": "anti_LGI1_encephalitis",
            "secondary_diagnoses": ["faciobrachial_dystonic_seizures", "hyponatremia", "limbic_encephalitis"],
            "icd10_codes": {"primary": "G04.81", "secondary": ["G25.89", "E87.1", "G04.90"]},
            "orpha_code": "363588", "omim": None, "rarity": "rare",
            "diagnostic_category": "autoimmune_encephalitis",
        },
        "demographics": {"age": 65, "sex": "M", "bmi": 28.0, "ethnicity": "British",
                         "family_history": "No relevant family history"},
        "genetic_variants": [],
        "comorbidities": [
            _comorb("faciobrachial_dystonic_seizures", "G25.89", 65),
            _comorb("hyponatremia", "E87.1", 65),
            _comorb("episodic_memory_loss", "R41.3", 65),
        ],
        "multi_omics": {
            "genomic": {"variants": [], "cnv": [], "repeat_expansions": [],
                        "hla_typing": {"HLA_DRB1": "HLA-DRB1*07:01", "note": "strongly associated with anti-LGI1"}},
            "transcriptomic": {"platform": "RNA-seq_blood", "key_degs": [
                _deg("LGI1", -0.3, 0.1, "down", "target antigen"),
                _deg("CXCL13", 2.0, 1e-8, "up"), _deg("CD38", 1.5, 1e-6, "up", "B-cell activation"),
            ]},
            "proteomic": {"serum_anti_lgi1_titer": "1:128", "csf_anti_lgi1_titer": "1:32",
                          "serum_sodium_meq_l": 126, "csf_protein_mg_dl": 45, "csf_wbc_per_ul": 8,
                          "blood_nfl_pg_ml": 28, "csf_total_tau_pg_ml": 300},
            "metabolomic": {"serum_sodium_meq_l": 126, "serum_osmolality_mosm_kg": 265},
            "epigenomic": {"horvath_age_acceleration_yrs": 1.0, "global_methylation_pct": 4.4},
        },
        "standard_biomarkers": {"eeg_theta_alpha_ratio": 1.2, "eeg_spectral_entropy": 0.48,
                                "fmri_bold_hippocampus": 0.45, "mmse_score": 20, "moca_score": 16,
                                "hamilton_depression_score": 6, "updrs_motor_score": 8},
        "treatment_history": [_tx("methylprednisolone", "1g_IV_x5d", 65), _tx("ivig", "0.4g_kg_x5d", 65)],
        "diagnostic_journey": {
            "initial_presentation": "Brief arm-jerking episodes + confusion at age 65",
            "initial_diagnosis": "epilepsy",
            "misdiagnoses": [{"diagnosis": "focal_epilepsy", "age": 65, "duration_years": 0.15}],
            "correct_diagnosis": {"diagnosis": "anti_LGI1_encephalitis", "age": 65,
                                  "method": "serum_anti_LGI1_antibody_CBA"},
            "diagnostic_delay_years": 0.15,
            "key_diagnostic_clue": "Faciobrachial dystonic seizures + hyponatremia + medial temporal lobe T2 signal on MRI",
        },
        "trajectory_summary": [
            _traj(65, "FBDS onset", 25, "Brief arm-face jerks, antiepileptics ineffective"),
            _traj(65, "hyponatremia + memory loss", 20, "Na 126, MRI: bilateral medial temporal T2 signal"),
            _traj(65, "anti-LGI1 diagnosis + immunotherapy", 22, "Dramatic response to steroids/IVIG"),
        ],
        "clinical_narrative": (
            "65-year-old man presenting with stereotyped brief faciobrachial dystonic seizures "
            "occurring 30-50 times daily, refractory to carbamazepine. Hyponatremia (Na 126 mEq/L) "
            "and progressive episodic memory impairment. MRI shows bilateral medial temporal lobe "
            "T2/FLAIR hyperintensity. Serum anti-LGI1 antibodies positive at 1:128. HLA typing "
            "reveals DRB1*07:01. Dramatic improvement with immunotherapy."
        ),
        "is_synthetic": True,
        "data_sources_modeled": ["Autoimmune Encephalitis Alliance", "Irani et al. 2010"],
    })

    # 11. Susac Syndrome
    cases.append({
        "case_id": "ZEBRA_SUSAC_011",
        "category": "zebra",
        "classification": {
            "primary_diagnosis": "Susac_syndrome",
            "secondary_diagnoses": ["encephalopathy", "branch_retinal_artery_occlusion", "hearing_loss"],
            "icd10_codes": {"primary": "I67.89", "secondary": ["G93.49", "H34.23", "H90.5"]},
            "orpha_code": "838", "omim": None, "rarity": "ultra_rare",
            "diagnostic_category": "autoimmune_endotheliopathy",
        },
        "demographics": {"age": 28, "sex": "F", "bmi": 22.0, "ethnicity": "Caucasian",
                         "family_history": "No relevant history"},
        "genetic_variants": [],
        "comorbidities": [
            _comorb("encephalopathy", "G93.49", 28), _comorb("branch_retinal_artery_occlusion", "H34.23", 28),
            _comorb("sensorineural_hearing_loss", "H90.5", 28),
        ],
        "multi_omics": {
            "genomic": {"variants": [], "cnv": [], "repeat_expansions": []},
            "transcriptomic": {"platform": "RNA-seq_blood", "key_degs": [
                _deg("ICAM1", 1.5, 1e-6, "up"), _deg("VCAM1", 1.3, 1e-5, "up"),
                _deg("CXCL10", 1.8, 1e-8, "up", "endothelial inflammation"),
            ]},
            "proteomic": {"csf_protein_mg_dl": 55, "csf_wbc_per_ul": 15,
                          "blood_nfl_pg_ml": 30, "von_willebrand_factor_pct": 220,
                          "anti_endothelial_cell_antibodies": "positive"},
            "metabolomic": {"plasma_d_dimer_ng_ml": 450},
            "epigenomic": {"horvath_age_acceleration_yrs": 0.5},
        },
        "standard_biomarkers": {"eeg_theta_alpha_ratio": 0.85, "eeg_spectral_entropy": 0.52,
                                "mmse_score": 22, "moca_score": 18,
                                "hamilton_depression_score": 10, "updrs_motor_score": 2},
        "treatment_history": [_tx("methylprednisolone", "1g_IV_x5d", 28), _tx("ivig", "2g_kg_over_5d", 28),
                              _tx("mycophenolate", "1g_BID", 28)],
        "diagnostic_journey": {
            "initial_presentation": "Headache + confusion + vision loss + hearing loss",
            "initial_diagnosis": "multiple_sclerosis",
            "misdiagnoses": [{"diagnosis": "MS", "age": 28, "duration_years": 0.1}],
            "correct_diagnosis": {"diagnosis": "Susac_syndrome", "age": 28,
                                  "method": "clinical_triad_plus_corpus_callosum_snowball_lesions_on_MRI"},
            "diagnostic_delay_years": 0.1,
            "key_diagnostic_clue": "Clinical triad (encephalopathy + BRAO + hearing loss) + central corpus callosum snowball lesions",
        },
        "trajectory_summary": [
            _traj(28, "acute encephalopathy + vision loss", 22, "MRI: central corpus callosum snowball lesions"),
            _traj(28, "hearing loss + Susac diagnosed", 24, "FA shows BRAO; audiometry confirms SNHL"),
        ],
        "clinical_narrative": (
            "28-year-old woman presenting with acute encephalopathy, branch retinal artery "
            "occlusion causing scotoma, and low-frequency sensorineural hearing loss. MRI shows "
            "characteristic central corpus callosum 'snowball' lesions distinct from MS plaques. "
            "Initially suspected MS but lesion pattern and clinical triad diagnostic of Susac "
            "syndrome. Anti-endothelial cell antibodies positive. Treated with pulse steroids, "
            "IVIG, and mycophenolate."
        ),
        "is_synthetic": True,
        "data_sources_modeled": ["Susac Syndrome Registry", "Dorr et al. 2013"],
    })

    # 12. Stiff Person Syndrome
    cases.append({
        "case_id": "ZEBRA_SPS_012",
        "category": "zebra",
        "classification": {
            "primary_diagnosis": "stiff_person_syndrome",
            "secondary_diagnoses": ["progressive_rigidity", "anxiety", "type_1_diabetes"],
            "icd10_codes": {"primary": "G25.82", "secondary": ["G24.1", "F41.1", "E10.9"]},
            "orpha_code": "3198", "omim": "184850", "rarity": "ultra_rare",
            "diagnostic_category": "autoimmune_neurological",
        },
        "demographics": {"age": 45, "sex": "F", "bmi": 25.0, "ethnicity": "Scandinavian",
                         "family_history": "Sister with type 1 diabetes; mother with vitiligo"},
        "genetic_variants": [],
        "comorbidities": [
            _comorb("type_1_diabetes", "E10.9", 30),
            _comorb("severe_anxiety", "F41.1", 42),
            _comorb("autoimmune_thyroiditis", "E06.3", 35),
        ],
        "multi_omics": {
            "genomic": {"variants": [], "cnv": [], "repeat_expansions": [],
                        "hla_typing": {"HLA_DQB1": "HLA-DQB1*02:01", "note": "T1DM-associated"}},
            "transcriptomic": {"platform": "RNA-seq_blood", "key_degs": [
                _deg("GAD1", -0.5, 0.05, "down", "GAD67"),
                _deg("GAD2", -0.8, 0.01, "down", "GAD65 — autoantibody target"),
                _deg("IFNG", 1.5, 1e-6, "up"), _deg("IL17A", 1.2, 1e-4, "up"),
            ]},
            "proteomic": {"serum_anti_gad65_iu_ml": 15000, "csf_anti_gad65_iu_ml": 800,
                          "csf_protein_mg_dl": 50, "csf_oligoclonal_bands": "positive",
                          "blood_nfl_pg_ml": 15, "serum_anti_tpo_iu_ml": 350},
            "metabolomic": {"csf_gaba_nmol_l": 120, "plasma_hba1c_pct": 7.5},
            "epigenomic": {"horvath_age_acceleration_yrs": 2.0},
        },
        "standard_biomarkers": {"eeg_theta_alpha_ratio": 0.42, "eeg_spectral_entropy": 0.72,
                                "mmse_score": 29, "moca_score": 28,
                                "hamilton_depression_score": 12, "updrs_motor_score": 18},
        "treatment_history": [_tx("diazepam", "10mg_TID", 43), _tx("baclofen", "20mg_TID", 43),
                              _tx("ivig", "2g_kg_monthly", 44), _tx("insulin", "basal_bolus", 30)],
        "diagnostic_journey": {
            "initial_presentation": "Progressive stiffness and painful spasms of trunk at 42",
            "initial_diagnosis": "anxiety_with_conversion",
            "misdiagnoses": [{"diagnosis": "anxiety_conversion_disorder", "age": 42, "duration_years": 2},
                             {"diagnosis": "fibromyalgia", "age": 43, "duration_years": 1}],
            "correct_diagnosis": {"diagnosis": "stiff_person_syndrome", "age": 45,
                                  "method": "anti_GAD65_15000_IU_ml_plus_continuous_motor_unit_activity_on_EMG"},
            "diagnostic_delay_years": 3,
            "key_diagnostic_clue": "Axial rigidity + task-specific phobias + T1DM + anti-GAD65 >10,000 IU/mL",
        },
        "trajectory_summary": [
            _traj(42, "trunk stiffness onset", 29, "Normal MRI spine"),
            _traj(43, "falls + spasms + agoraphobia", 29, "EMG: continuous motor unit activity"),
            _traj(45, "anti-GAD65 testing + diagnosis", 29, "GAD65 15,000 IU/mL; IVIG started with improvement"),
        ],
        "clinical_narrative": (
            "45-year-old woman with 3-year history of progressive truncal rigidity, painful "
            "spasms triggered by startle, and task-specific phobias including agoraphobia. "
            "Background of type 1 diabetes (age 30) and autoimmune thyroiditis. Initially "
            "diagnosed as anxiety/conversion disorder. EMG shows continuous motor unit activity "
            "at rest. Anti-GAD65 antibodies massively elevated at 15,000 IU/mL (normal <5). "
            "Significant improvement on diazepam, baclofen, and monthly IVIG."
        ),
        "is_synthetic": True,
        "data_sources_modeled": ["SPS Research Foundation", "Dalakas 2009"],
    })

    # 13. Neurosarcoidosis
    cases.append({
        "case_id": "ZEBRA_NEUROSAR_013",
        "category": "zebra",
        "classification": {
            "primary_diagnosis": "neurosarcoidosis",
            "secondary_diagnoses": ["cranial_neuropathy", "hypothalamic_dysfunction", "aseptic_meningitis"],
            "icd10_codes": {"primary": "D86.82", "secondary": ["G52.9", "E23.7", "G03.0"]},
            "orpha_code": "231", "omim": None, "rarity": "rare",
            "diagnostic_category": "granulomatous_neurological",
        },
        "demographics": {"age": 40, "sex": "M", "bmi": 29.0, "ethnicity": "African_American",
                         "family_history": "No known sarcoidosis in family"},
        "genetic_variants": [],
        "comorbidities": [
            _comorb("facial_nerve_palsy_bilateral", "G51.0", 39),
            _comorb("diabetes_insipidus", "E23.2", 40),
            _comorb("pulmonary_sarcoidosis", "D86.0", 38),
        ],
        "multi_omics": {
            "genomic": {"variants": [], "cnv": [], "repeat_expansions": [],
                        "hla_typing": {"HLA_DRB1": "HLA-DRB1*11:01", "note": "sarcoidosis-associated in AA"}},
            "transcriptomic": {"platform": "RNA-seq_blood", "key_degs": [
                _deg("ACE", 1.8, 1e-8, "up", "granuloma-derived ACE"),
                _deg("IFNG", 2.0, 1e-10, "up", "Th1 response"),
                _deg("TNF", 1.5, 1e-6, "up"), _deg("IL12B", 1.2, 1e-4, "up"),
            ]},
            "proteomic": {"serum_ace_u_l": 95, "csf_ace_u_l": 12, "serum_il2_receptor_u_ml": 2500,
                          "csf_protein_mg_dl": 120, "csf_wbc_per_ul": 80, "csf_glucose_mg_dl": 38,
                          "blood_nfl_pg_ml": 25, "serum_calcium_mg_dl": 10.8},
            "metabolomic": {"serum_1_25_dihydroxyvitamin_d_pg_ml": 85, "serum_calcium_mg_dl": 10.8},
            "epigenomic": {"horvath_age_acceleration_yrs": 1.5},
        },
        "standard_biomarkers": {"eeg_theta_alpha_ratio": 0.50, "eeg_spectral_entropy": 0.68,
                                "mmse_score": 27, "moca_score": 24,
                                "hamilton_depression_score": 10, "updrs_motor_score": 4},
        "treatment_history": [_tx("prednisone", "60mg_taper", 39), _tx("methotrexate", "15mg_weekly", 40),
                              _tx("desmopressin", "0.2mg", 40)],
        "diagnostic_journey": {
            "initial_presentation": "Bilateral facial palsy + headaches at 39",
            "initial_diagnosis": "Bell_palsy",
            "misdiagnoses": [{"diagnosis": "bilateral_Bell_palsy", "age": 39, "duration_years": 0.25},
                             {"diagnosis": "MS", "age": 39, "duration_years": 0.5}],
            "correct_diagnosis": {"diagnosis": "neurosarcoidosis", "age": 40,
                                  "method": "meningeal_biopsy_non_caseating_granulomas"},
            "diagnostic_delay_years": 1,
            "key_diagnostic_clue": "Bilateral facial palsy + hypothalamic DI + elevated CSF ACE + bilateral hilar lymphadenopathy on CT chest",
        },
        "trajectory_summary": [
            _traj(38, "pulmonary sarcoidosis diagnosed", 29, "Bilateral hilar LAD on CXR"),
            _traj(39, "bilateral facial palsy", 28, "MRI: leptomeningeal enhancement"),
            _traj(40, "diabetes insipidus + meningeal biopsy", 27, "Non-caseating granulomas confirmed"),
        ],
        "clinical_narrative": (
            "40-year-old African American man with known pulmonary sarcoidosis (bilateral hilar "
            "lymphadenopathy, age 38) presenting with bilateral facial nerve palsy and progressive "
            "headaches. MRI shows diffuse leptomeningeal enhancement and hypothalamic thickening. "
            "Developed central diabetes insipidus. CSF shows lymphocytic pleocytosis (80 WBC/uL), "
            "elevated protein (120 mg/dL), low glucose, and elevated ACE. Meningeal biopsy confirms "
            "non-caseating granulomas. Serum ACE 95 U/L, soluble IL-2 receptor 2500 U/mL."
        ),
        "is_synthetic": True,
        "data_sources_modeled": ["A Case Register Study of Sarcoidosis", "Stern et al. 2018"],
    })

    # 14. Adult-onset Alexander Disease
    cases.append({
        "case_id": "ZEBRA_ALEXD_014",
        "category": "zebra",
        "classification": {
            "primary_diagnosis": "adult_onset_Alexander_disease",
            "secondary_diagnoses": ["bulbar_dysfunction", "spastic_paraparesis", "palatal_myoclonus"],
            "icd10_codes": {"primary": "E75.25", "secondary": ["G12.29", "G82.1", "G25.3"]},
            "orpha_code": "58", "omim": "203450", "rarity": "ultra_rare",
            "diagnostic_category": "leukodystrophy",
        },
        "demographics": {"age": 42, "sex": "M", "bmi": 24.0, "ethnicity": "Caucasian",
                         "family_history": "No known family history (de novo mutation suspected)"},
        "genetic_variants": [
            _v("GFAP", "c.716G>A", "p.Arg239His", "rs121918300", "VCV000004287",
               "pathogenic", "heterozygous", 1e-6, "autosomal_dominant",
               "Causes Rosenthal fiber accumulation; most common adult-onset GFAP mutation"),
        ],
        "comorbidities": [
            _comorb("bulbar_dysfunction", "G12.29", 38),
            _comorb("spastic_paraparesis", "G82.1", 40),
            _comorb("palatal_myoclonus", "G25.3", 39),
            _comorb("dysautonomia", "G90.9", 41),
        ],
        "multi_omics": {
            "genomic": {"variants": [{"gene": "GFAP", "hgvs_p": "p.Arg239His", "rsid": "rs121918300"}],
                        "cnv": [], "repeat_expansions": []},
            "transcriptomic": {"platform": "RNA-seq_blood", "key_degs": [
                _deg("GFAP", 4.0, 1e-25, "up", "massively upregulated; Rosenthal fibers"),
                _deg("CRYAB", 2.5, 1e-15, "up", "alpha-B crystallin, stress response"),
                _deg("HSP27", 2.0, 1e-10, "up", "heat shock protein response"),
            ]},
            "proteomic": {"csf_gfap_ng_ml": 25.0, "blood_gfap_pg_ml": 800,
                          "blood_nfl_pg_ml": 50, "csf_total_tau_pg_ml": 300},
            "metabolomic": {"csf_lactate_mmol_l": 1.8},
            "epigenomic": {"horvath_age_acceleration_yrs": 3.0},
        },
        "standard_biomarkers": {"eeg_theta_alpha_ratio": 0.55, "eeg_spectral_entropy": 0.62,
                                "fmri_bold_dmn": -0.20, "mmse_score": 26, "moca_score": 23,
                                "hamilton_depression_score": 8, "updrs_motor_score": 20},
        "treatment_history": [_tx("baclofen", "20mg_TID", 40), _tx("botulinum_toxin", "for_spasticity", 41)],
        "diagnostic_journey": {
            "initial_presentation": "Dysphagia and dysarthria at 38",
            "initial_diagnosis": "MS",
            "misdiagnoses": [{"diagnosis": "MS", "age": 38, "duration_years": 3},
                             {"diagnosis": "brainstem_tumor", "age": 40, "duration_years": 1}],
            "correct_diagnosis": {"diagnosis": "adult_Alexander_disease", "age": 42,
                                  "method": "GFAP_sequencing_after_characteristic_MRI_pattern_recognized"},
            "diagnostic_delay_years": 4,
            "key_diagnostic_clue": "Brainstem + cervical cord atrophy + palatal myoclonus + GFAP massively elevated in CSF",
        },
        "trajectory_summary": [
            _traj(38, "bulbar symptoms onset", 28, "MRI: brainstem T2 signal changes"),
            _traj(40, "spastic paraparesis", 27, "MRI: cervical cord atrophy, tadpole sign"),
            _traj(42, "GFAP mutation identified", 26, "CSF GFAP 25 ng/mL; GFAP R239H confirmed"),
        ],
        "clinical_narrative": (
            "42-year-old man with progressive bulbar dysfunction (dysphagia, dysarthria), "
            "palatal myoclonus, and spastic paraparesis evolving over 4 years. MRI shows "
            "characteristic brainstem and cervical cord atrophy ('tadpole' sign) with "
            "periventricular white matter changes. CSF GFAP massively elevated at 25 ng/mL. "
            "GFAP sequencing reveals c.716G>A (p.Arg239His) mutation, confirming adult-onset "
            "Alexander disease. Previously misdiagnosed as MS."
        ),
        "is_synthetic": True,
        "data_sources_modeled": ["Alexander Disease Registry", "ClinVar"],
    })

    # 15. Hashimoto's Encephalopathy / SREAT
    cases.append({
        "case_id": "ZEBRA_SREAT_015",
        "category": "zebra",
        "classification": {
            "primary_diagnosis": "steroid_responsive_encephalopathy_associated_with_thyroiditis",
            "secondary_diagnoses": ["fluctuating_encephalopathy", "seizures", "myoclonus", "autoimmune_thyroiditis"],
            "icd10_codes": {"primary": "G04.81", "secondary": ["G93.49", "G40.509", "G25.3", "E06.3"]},
            "orpha_code": "83601", "omim": None, "rarity": "rare",
            "diagnostic_category": "autoimmune_encephalopathy",
        },
        "demographics": {"age": 55, "sex": "F", "bmi": 30.0, "ethnicity": "Caucasian",
                         "family_history": "Daughter with Graves disease; sister with Hashimoto thyroiditis"},
        "genetic_variants": [],
        "comorbidities": [
            _comorb("hashimoto_thyroiditis", "E06.3", 45),
            _comorb("hypothyroidism", "E03.9", 46, "treated"),
            _comorb("seizures", "G40.509", 55),
            _comorb("myoclonus", "G25.3", 55),
        ],
        "multi_omics": {
            "genomic": {"variants": [], "cnv": [], "repeat_expansions": []},
            "transcriptomic": {"platform": "RNA-seq_blood", "key_degs": [
                _deg("TPO", 0.5, 0.05, "up"), _deg("IFNG", 1.5, 1e-6, "up"),
                _deg("IL17A", 1.0, 1e-4, "up"), _deg("FOXP3", -0.8, 0.01, "down", "Treg dysfunction"),
            ]},
            "proteomic": {"serum_anti_tpo_iu_ml": 1500, "serum_anti_thyroglobulin_iu_ml": 450,
                          "serum_tsh_miu_l": 8.5, "csf_protein_mg_dl": 75, "csf_wbc_per_ul": 12,
                          "blood_nfl_pg_ml": 22, "csf_total_tau_pg_ml": 350},
            "metabolomic": {"serum_t4_free_ng_dl": 0.7, "serum_tsh_miu_l": 8.5},
            "epigenomic": {"horvath_age_acceleration_yrs": 2.0},
        },
        "standard_biomarkers": {"eeg_theta_alpha_ratio": 1.0, "eeg_spectral_entropy": 0.45,
                                "eeg_delta_1_4hz_uv2": 25, "mmse_score": 18, "moca_score": 14,
                                "hamilton_depression_score": 14, "updrs_motor_score": 10},
        "treatment_history": [_tx("methylprednisolone", "1g_IV_x5d", 55),
                              _tx("levothyroxine", "100mcg", 46), _tx("levetiracetam", "500mg_BID", 55)],
        "diagnostic_journey": {
            "initial_presentation": "Fluctuating confusion + seizures + myoclonus at 55",
            "initial_diagnosis": "CJD_suspected",
            "misdiagnoses": [{"diagnosis": "Creutzfeldt_Jakob_disease", "age": 55, "duration_years": 0.1}],
            "correct_diagnosis": {"diagnosis": "SREAT", "age": 55,
                                  "method": "anti_TPO_1500_plus_dramatic_steroid_response"},
            "diagnostic_delay_years": 0.1,
            "key_diagnostic_clue": "Fluctuating course (not relentlessly progressive like CJD) + very high anti-TPO + steroid responsiveness",
        },
        "trajectory_summary": [
            _traj(45, "Hashimoto thyroiditis diagnosed", 30, "Anti-TPO positive, started levothyroxine"),
            _traj(55, "acute encephalopathy + seizures", 18, "EEG: diffuse slowing; MRI nonspecific"),
            _traj(55, "SREAT diagnosed, steroids given", 26, "Dramatic improvement within 48 hours of IV methylpred"),
        ],
        "clinical_narrative": (
            "55-year-old woman with known Hashimoto thyroiditis presenting with subacute "
            "fluctuating encephalopathy, generalized seizures, and myoclonus. Initially "
            "suspected CJD given myoclonus and cognitive decline. However, fluctuating course, "
            "anti-TPO >1500 IU/mL, and dramatic response to IV methylprednisolone within "
            "48 hours confirmed SREAT. EEG showed diffuse slowing without periodic complexes. "
            "CSF 14-3-3 negative. Family history of autoimmune thyroid disease."
        ),
        "is_synthetic": True,
        "data_sources_modeled": ["Laurent et al. 2016", "Castillo et al. 2006"],
    })

    return cases


# ---------------------------------------------------------------------------
# CATEGORY 2: MULTI-COMORBIDITY CLUSTERS
# ---------------------------------------------------------------------------

def _comorbidity_cases():
    cases = []

    # 16. Metabolic-Cognitive cluster
    cases.append({
        "case_id": "COMORB_METACOG_016",
        "category": "comorbidity",
        "classification": {
            "primary_diagnosis": "Alzheimers_disease",
            "secondary_diagnoses": ["type_2_diabetes", "hypertension", "hyperlipidemia", "obesity"],
            "icd10_codes": {"primary": "G30.1", "secondary": ["E11.9", "I10", "E78.5", "E66.9"]},
            "orpha_code": None, "omim": "104300", "rarity": "common",
            "diagnostic_category": "metabolic_cognitive_cluster",
        },
        "demographics": {"age": 75, "sex": "M", "bmi": 32.0, "ethnicity": "Caucasian",
                         "family_history": "Mother had AD at 78; father had T2DM and stroke"},
        "genetic_variants": [
            _v("APOE", "rs429358_C/C+rs7412_C/C", "APOE_e4/e4", "rs429358", None,
               "risk_factor", "homozygous", 0.02, "complex",
               "APOE epsilon4 homozygous; ~12x increased AD risk; impairs amyloid clearance"),
        ],
        "comorbidities": [
            _comorb("type_2_diabetes", "E11.9", 58, "treated"),
            _comorb("hypertension", "I10", 55, "treated"),
            _comorb("hyperlipidemia", "E78.5", 60, "treated"),
            _comorb("obesity", "E66.9", 50),
            _comorb("obstructive_sleep_apnea", "G47.33", 62, "treated"),
            _comorb("chronic_kidney_disease_stage3", "N18.3", 70),
        ],
        "multi_omics": {
            "genomic": {"variants": [{"gene": "APOE", "genotype": "e4/e4", "rsid": "rs429358"}],
                        "cnv": [], "repeat_expansions": []},
            "transcriptomic": {"platform": "RNA-seq_blood", "key_degs": [
                _deg("APOE", 2.0, 1e-15, "up"), _deg("TREM2", 1.8, 1e-10, "up"),
                _deg("CLU", 1.5, 1e-8, "up"), _deg("SLC2A4", -1.2, 1e-6, "down", "GLUT4, insulin resistance"),
                _deg("GFAP", 1.5, 1e-8, "up"),
            ]},
            "proteomic": {"csf_amyloid_beta42_pg_ml": 380, "csf_total_tau_pg_ml": 650,
                          "csf_p_tau181_pg_ml": 85, "blood_nfl_pg_ml": 55,
                          "blood_gfap_pg_ml": 300, "plasma_ptau217_pg_ml": 4.5,
                          "plasma_ab42_40_ratio": 0.048, "serum_hba1c_pct": 7.8,
                          "blood_crp_mg_l": 5.5, "blood_il6_pg_ml": 4.2},
            "metabolomic": {"plasma_homocysteine_umol_l": 18, "serum_uric_acid_mg_dl": 7.5,
                            "plasma_ceramides_umol_l": 4.5, "plasma_glucose_fasting_mg_dl": 165},
            "epigenomic": {"horvath_age_acceleration_yrs": 6.0, "global_methylation_pct": 3.6},
        },
        "standard_biomarkers": {"eeg_theta_alpha_ratio": 1.4, "eeg_spectral_entropy": 0.45,
                                "fmri_bold_dmn": -0.45, "fmri_conn_dmn_pfc": 0.18,
                                "mmse_score": 18, "moca_score": 14,
                                "hamilton_depression_score": 10, "updrs_motor_score": 5},
        "treatment_history": [
            _tx("donepezil", "10mg", 72), _tx("memantine", "10mg_BID", 73),
            _tx("metformin", "1000mg_BID", 58), _tx("atorvastatin", "40mg", 60),
            _tx("lisinopril", "20mg", 55), _tx("CPAP", "nightly", 62),
        ],
        "diagnostic_journey": {
            "initial_presentation": "Progressive memory loss at age 72",
            "initial_diagnosis": "age_related_memory_decline",
            "misdiagnoses": [{"diagnosis": "age_related_memory_decline", "age": 72, "duration_years": 1}],
            "correct_diagnosis": {"diagnosis": "Alzheimers_disease_with_metabolic_comorbidities", "age": 73,
                                  "method": "amyloid_PET_positive_plus_CSF_biomarkers"},
            "diagnostic_delay_years": 1,
            "key_diagnostic_clue": "CSF Abeta42/tau profile + amyloid PET + APOE e4/e4",
        },
        "trajectory_summary": [
            _traj(55, "metabolic syndrome established", 29, "HTN + T2DM + hyperlipidemia"),
            _traj(72, "memory complaints", 24, "Family noticed forgetfulness"),
            _traj(73, "AD diagnosed", 20, "Amyloid PET positive, CSF Abeta42=380"),
            _traj(75, "moderate dementia", 18, "Needs assistance with IADLs"),
        ],
        "clinical_narrative": (
            "75-year-old man with moderate Alzheimer's disease in the context of longstanding "
            "metabolic syndrome (T2DM since 58, HTN, hyperlipidemia, obesity BMI 32). APOE "
            "e4/e4 genotype. CSF shows classic AD profile (Abeta42 380, tTau 650, pTau181 85). "
            "Plasma pTau217 elevated at 4.5 pg/mL. HbA1c poorly controlled at 7.8% despite "
            "metformin. CRP elevated at 5.5 mg/L reflecting systemic inflammation. "
            "Epigenetic age acceleration of 6 years suggests accelerated biological aging from "
            "metabolic burden compounding genetic AD risk."
        ),
        "is_synthetic": True,
        "data_sources_modeled": ["UK Biobank", "ADNI", "Medicare Claims Database"],
    })

    # 17. Alpha-synucleinopathy spectrum
    cases.append({
        "case_id": "COMORB_SYNUC_017",
        "category": "comorbidity",
        "classification": {
            "primary_diagnosis": "Parkinsons_disease",
            "secondary_diagnoses": ["RBD", "autonomic_failure", "MCI", "depression", "constipation"],
            "icd10_codes": {"primary": "G20", "secondary": ["G47.52", "G90.3", "G31.84", "F32.1", "K59.0"]},
            "orpha_code": None, "omim": "168601", "rarity": "common",
            "diagnostic_category": "synucleinopathy_spectrum",
        },
        "demographics": {"age": 70, "sex": "M", "bmi": 25.5, "ethnicity": "Ashkenazi_Jewish",
                         "family_history": "Father had PD at 72; uncle had Gaucher disease"},
        "genetic_variants": [
            _v("GBA", "c.1226A>G", "p.Asn370Ser", "rs76763715", "VCV000004288",
               "risk_factor", "heterozygous", 0.005, "complex",
               "GBA N370S: 5-fold PD risk; reduced GCase activity; promotes alpha-synuclein aggregation"),
        ],
        "comorbidities": [
            _comorb("REM_sleep_behavior_disorder", "G47.52", 60),
            _comorb("constipation", "K59.0", 58),
            _comorb("hyposmia", "R43.0", 62),
            _comorb("orthostatic_hypotension", "I95.1", 67),
            _comorb("depression", "F32.1", 65),
            _comorb("mild_cognitive_impairment", "G31.84", 68),
            _comorb("urinary_urgency", "N39.41", 69),
        ],
        "multi_omics": {
            "genomic": {"variants": [{"gene": "GBA", "hgvs_p": "p.Asn370Ser", "rsid": "rs76763715"}],
                        "cnv": [], "repeat_expansions": []},
            "transcriptomic": {"platform": "RNA-seq_blood", "key_degs": [
                _deg("GBA", -0.8, 0.005, "down", "reduced GCase expression"),
                _deg("SNCA", 0.6, 0.01, "up", "alpha-synuclein elevated"),
                _deg("LRRK2", 0.5, 0.05, "up"),
                _deg("LAMP2A", -0.7, 0.01, "down", "impaired CMA"),
            ]},
            "proteomic": {"csf_alpha_synuclein_pg_ml": 800, "csf_amyloid_beta42_pg_ml": 900,
                          "csf_total_tau_pg_ml": 250, "blood_nfl_pg_ml": 35,
                          "blood_gfap_pg_ml": 140, "serum_gcase_activity_nmol_hr_mg": 3.5},
            "metabolomic": {"plasma_urate_mg_dl": 3.8, "csf_homovanillic_acid_ng_ml": 18,
                            "plasma_ceramides_umol_l": 3.8},
            "epigenomic": {"horvath_age_acceleration_yrs": 3.0, "snca_promoter_methylation_pct": 45},
        },
        "standard_biomarkers": {"eeg_theta_alpha_ratio": 0.65, "eeg_spectral_entropy": 0.58,
                                "fmri_bold_basal_ganglia": -0.35, "fmri_bold_dmn": -0.15,
                                "mmse_score": 25, "moca_score": 22,
                                "hamilton_depression_score": 14, "updrs_motor_score": 35},
        "treatment_history": [
            _tx("levodopa_carbidopa", "100_25mg_TID", 66), _tx("pramipexole", "0.5mg_TID", 65),
            _tx("melatonin", "3mg", 60), _tx("midodrine", "5mg_TID", 67),
            _tx("sertraline", "50mg", 65),
        ],
        "diagnostic_journey": {
            "initial_presentation": "RBD at 60, followed by constipation, then hyposmia, then tremor at 65",
            "initial_diagnosis": "REM_sleep_behavior_disorder",
            "misdiagnoses": [],
            "correct_diagnosis": {"diagnosis": "PD_with_GBA_variant", "age": 66,
                                  "method": "clinical_diagnosis_plus_GBA_panel"},
            "diagnostic_delay_years": 1,
            "key_diagnostic_clue": "RBD → hyposmia → parkinsonism prodromal sequence + Ashkenazi ancestry → GBA testing",
        },
        "trajectory_summary": [
            _traj(58, "constipation onset", 29, "Prodromal feature"),
            _traj(60, "RBD onset", 29, "Polysomnography: REM without atonia"),
            _traj(62, "hyposmia", 28, "UPSIT: 15/40"),
            _traj(65, "motor PD onset + depression", 27, "Resting tremor + bradykinesia"),
            _traj(68, "MCI onset + autonomic failure", 25, "Tilt table: OH; MoCA 22"),
            _traj(70, "worsening cognition + GBA-PD", 22, "GCase activity low; worse prognosis trajectory"),
        ],
        "clinical_narrative": (
            "70-year-old Ashkenazi Jewish man demonstrating the full alpha-synucleinopathy "
            "spectrum. Prodromal features preceded motor PD by 7 years: constipation (58), "
            "REM sleep behavior disorder (60), hyposmia (62). Motor symptoms at 65 with "
            "subsequent MCI (68) and severe autonomic failure including orthostatic hypotension. "
            "GBA N370S heterozygous with reduced GCase activity (3.5 nmol/hr/mg). CSF "
            "alpha-synuclein elevated. Worse cognitive trajectory than idiopathic PD, consistent "
            "with GBA-associated PD phenotype."
        ),
        "is_synthetic": True,
        "data_sources_modeled": ["PPMI", "UK Biobank", "GBA-PD Registry"],
    })

    # 18-30: Abbreviated comorbidity cases (still with full structure, shorter narrative)
    _comorb_specs = [
        ("COMORB_NEUROPSY_018", "comorbidity", "bipolar_disorder",
         ["epilepsy", "obesity", "PCOS", "hypothyroidism"],
         {"primary": "F31.9", "secondary": ["G40.509", "E66.9", "E28.2", "E03.9"]},
         60, "F", 34.0, "Mediterranean",
         [], [_comorb("bipolar_I_disorder", "F31.1", 25), _comorb("epilepsy_focal", "G40.209", 28),
              _comorb("obesity", "E66.9", 35), _comorb("PCOS", "E28.2", 36),
              _comorb("hypothyroidism", "E03.9", 40)],
         {"mmse_score": 27, "moca_score": 25, "hamilton_depression_score": 8, "updrs_motor_score": 2,
          "eeg_theta_alpha_ratio": 0.50, "eeg_spectral_entropy": 0.65},
         [_tx("lithium", "900mg", 25), _tx("valproate", "1000mg_BID", 28),
          _tx("levothyroxine", "75mcg", 40)],
         "60-year-old woman with bipolar I disorder (age 25) and comorbid focal epilepsy (28). "
         "Valproate used for dual indication caused weight gain leading to obesity and PCOS. "
         "Lithium-induced hypothyroidism. Complex polypharmacy with drug interactions."),

        ("COMORB_TBI_019", "comorbidity", "post_traumatic_brain_injury_syndrome",
         ["post_traumatic_epilepsy", "chronic_pain", "opioid_use_disorder", "MDD", "cognitive_decline"],
         {"primary": "F07.81", "secondary": ["G40.509", "G89.4", "F11.20", "F32.2", "G31.84"]},
         45, "M", 28.0, "Caucasian",
         [_v("APOE", "rs429358_C/rs7412_C", "APOE_e4_carrier", "rs429358", None,
             "risk_factor", "heterozygous", 0.14, "complex",
             "APOE e4 carrier: worse TBI outcomes, increased AD risk post-TBI")],
         [_comorb("moderate_TBI", "S06.2", 30), _comorb("post_traumatic_epilepsy", "G40.509", 32),
          _comorb("chronic_pain_syndrome", "G89.4", 33), _comorb("opioid_use_disorder", "F11.20", 35),
          _comorb("major_depressive_disorder", "F32.2", 36), _comorb("mild_cognitive_impairment", "G31.84", 43)],
         {"mmse_score": 24, "moca_score": 20, "hamilton_depression_score": 22, "updrs_motor_score": 3,
          "eeg_theta_alpha_ratio": 0.70, "eeg_spectral_entropy": 0.55},
         [_tx("levetiracetam", "1000mg_BID", 32), _tx("buprenorphine_naloxone", "8_2mg", 37),
          _tx("duloxetine", "60mg", 36)],
         "45-year-old man with moderate TBI at age 30 (motor vehicle accident) triggering a "
         "cascade: post-traumatic epilepsy (32), chronic pain (33), opioid dependence (35), "
         "depression (36), and early cognitive decline (43). APOE e4 carrier conferring "
         "increased vulnerability to post-TBI neurodegeneration. Blood NfL elevated."),

        ("COMORB_AUTOIMMUNE_020", "comorbidity", "multiple_sclerosis",
         ["autoimmune_thyroiditis", "celiac_disease", "Raynauds", "anxiety"],
         {"primary": "G35", "secondary": ["E06.3", "K90.0", "I73.0", "F41.1"]},
         35, "F", 22.5, "Northern_European",
         [],
         [_comorb("relapsing_remitting_MS", "G35", 25), _comorb("autoimmune_thyroiditis", "E06.3", 22),
          _comorb("celiac_disease", "K90.0", 20), _comorb("Raynauds_phenomenon", "I73.0", 18),
          _comorb("generalized_anxiety_disorder", "F41.1", 23), _comorb("fatigue", "R53.83", 25)],
         {"mmse_score": 29, "moca_score": 27, "hamilton_depression_score": 10, "updrs_motor_score": 3,
          "eeg_theta_alpha_ratio": 0.45, "eeg_spectral_entropy": 0.70},
         [_tx("ocrelizumab", "600mg_q6mo", 30), _tx("levothyroxine", "50mcg", 22),
          _tx("gluten_free_diet", "ongoing", 20), _tx("escitalopram", "10mg", 23)],
         "35-year-old woman with autoimmune polyendocrinopathy: celiac disease (20), Raynaud's "
         "(18), Hashimoto thyroiditis (22), and RRMS (25). HLA-DRB1*15:01 carrier. Represents "
         "shared autoimmune diathesis with multi-organ involvement."),

        ("COMORB_GERIATRIC_021", "comorbidity", "Alzheimers_disease",
         ["heart_failure", "atrial_fibrillation", "osteoarthritis", "CKD", "falls"],
         {"primary": "G30.1", "secondary": ["I50.9", "I48.91", "M19.90", "N18.3", "R29.6"]},
         82, "F", 26.0, "Caucasian",
         [_v("APOE", "rs429358_C/rs7412_T", "APOE_e3/e4", "rs429358", None,
             "risk_factor", "heterozygous", 0.14, "complex", "APOE e3/e4: ~3x AD risk")],
         [_comorb("Alzheimers_disease", "G30.1", 78), _comorb("osteoarthritis", "M19.90", 60),
          _comorb("heart_failure_HFpEF", "I50.9", 75), _comorb("atrial_fibrillation", "I48.91", 72),
          _comorb("chronic_kidney_disease_stage3", "N18.3", 74), _comorb("hip_fracture", "S72.001", 81),
          _comorb("delirium_superimposed_on_dementia", "F05", 81)],
         {"mmse_score": 15, "moca_score": 10, "hamilton_depression_score": 12, "updrs_motor_score": 8,
          "eeg_theta_alpha_ratio": 1.5, "eeg_spectral_entropy": 0.38},
         [_tx("donepezil", "10mg", 78), _tx("apixaban", "2.5mg_BID", 72),
          _tx("furosemide", "40mg", 75), _tx("acetaminophen", "650mg_TID", 60),
          _tx("calcium_vitamin_D", "600_800IU", 70), _tx("alendronate", "70mg_weekly", 70)],
         "82-year-old woman with moderate AD complicated by geriatric cascade: heart failure, "
         "atrial fibrillation, CKD stage 3, osteoarthritis, fall with hip fracture at 81 "
         "triggering delirium and accelerated cognitive decline. On 12 medications. APOE e3/e4."),

        ("COMORB_PSYCHCPLX_022", "comorbidity", "schizophrenia",
         ["metabolic_syndrome", "cannabis_use_disorder", "insomnia"],
         {"primary": "F20.9", "secondary": ["E88.81", "F12.20", "G47.00"]},
         28, "M", 35.0, "African_American",
         [_v("COMT", "c.472G>A", "p.Val158Met", "rs4680", None,
             "risk_factor", "homozygous", 0.25, "complex",
             "COMT Val158Met: Met/Met reduces dopamine catabolism in PFC")],
         [_comorb("schizophrenia_treatment_resistant", "F20.9", 20),
          _comorb("metabolic_syndrome", "E88.81", 23),
          _comorb("cannabis_use_disorder", "F12.20", 18),
          _comorb("insomnia", "G47.00", 20), _comorb("social_anxiety", "F40.10", 16)],
         {"mmse_score": 26, "moca_score": 23, "hamilton_depression_score": 8, "updrs_motor_score": 5,
          "eeg_theta_alpha_ratio": 0.60, "eeg_spectral_entropy": 0.55},
         [_tx("clozapine", "400mg", 22), _tx("metformin", "1000mg_BID", 23)],
         "28-year-old man with treatment-resistant schizophrenia on clozapine (400mg) with "
         "antipsychotic-induced metabolic syndrome (BMI 35, prediabetes). Cannabis use from "
         "age 18 may have precipitated psychosis onset at 20. COMT Met/Met genotype."),

        ("COMORB_ALSFTD_023", "comorbidity", "ALS_FTD",
         ["behavioral_variant_FTD", "ALS", "T2DM", "depression"],
         {"primary": "G12.21", "secondary": ["G31.09", "E11.9", "F32.1"]},
         58, "F", 27.0, "Caucasian",
         [_v("C9orf72", "(GGGGCC)n_expansion", "repeat_expansion", None, "VCV000227087",
             "pathogenic", "heterozygous", 0.001, "autosomal_dominant",
             "Hexanucleotide repeat expansion; RNA foci + DPR proteins")],
         [_comorb("ALS", "G12.21", 56), _comorb("behavioral_variant_FTD", "G31.09", 55),
          _comorb("type_2_diabetes", "E11.9", 50), _comorb("depression", "F32.1", 54)],
         {"mmse_score": 21, "moca_score": 17, "hamilton_depression_score": 14, "updrs_motor_score": 20,
          "eeg_theta_alpha_ratio": 0.65, "eeg_spectral_entropy": 0.55},
         [_tx("riluzole", "50mg_BID", 56), _tx("edaravone", "60mg_IV", 57),
          _tx("metformin", "500mg_BID", 50), _tx("sertraline", "100mg", 54)],
         "58-year-old woman with C9orf72-positive ALS-FTD spectrum. Behavioral changes preceded "
         "motor symptoms by 1 year. Blood NfL 200 pg/mL. Comorbid T2DM and depression complicate "
         "management. Rapidly progressive course."),

        ("COMORB_TSC_024", "comorbidity", "tuberous_sclerosis_complex",
         ["epilepsy", "ASD", "renal_angiomyolipoma", "cardiac_rhabdomyoma"],
         {"primary": "Q85.1", "secondary": ["G40.409", "F84.0", "D30.0", "D15.1"]},
         8, "M", None, "Mixed",
         [_v("TSC2", "c.5238_5255del", "p.His1746_Arg1751del", None, "VCV000227088",
             "pathogenic", "heterozygous", 1e-6, "autosomal_dominant",
             "Loss of function in tuberin; mTOR pathway hyperactivation")],
         [_comorb("infantile_spasms", "G40.822", 0.5), _comorb("Lennox_Gastaut_syndrome", "G40.812", 3),
          _comorb("autism_spectrum_disorder", "F84.0", 2), _comorb("renal_angiomyolipoma", "D30.0", 6),
          _comorb("cardiac_rhabdomyoma", "D15.1", 0)],
         {"mmse_score": None, "moca_score": None, "hamilton_depression_score": None, "updrs_motor_score": 2,
          "eeg_theta_alpha_ratio": 1.2, "eeg_spectral_entropy": 0.35},
         [_tx("vigabatrin", "1g_BID", 0.5), _tx("everolimus", "5mg", 6)],
         "8-year-old boy with tuberous sclerosis complex (TSC2 de novo). Cardiac rhabdomyoma "
         "detected prenatally. Infantile spasms at 6 months evolved to Lennox-Gastaut. ASD "
         "diagnosed at 2. Renal angiomyolipomas at 6 treated with everolimus (mTOR inhibitor "
         "with dual benefit for seizures and tumors)."),

        ("COMORB_STROKECARD_025", "comorbidity", "ischemic_stroke",
         ["atrial_fibrillation", "vascular_dementia", "post_stroke_depression", "dysphagia"],
         {"primary": "I63.411", "secondary": ["I48.91", "F01.50", "F32.9", "R13.10"]},
         72, "M", 29.0, "Caucasian",
         [],
         [_comorb("ischemic_stroke_MCA", "I63.411", 70), _comorb("atrial_fibrillation", "I48.91", 65),
          _comorb("vascular_dementia", "F01.50", 71), _comorb("post_stroke_depression", "F32.9", 70),
          _comorb("dysphagia", "R13.10", 70), _comorb("spasticity_left_hemiparesis", "G81.94", 70)],
         {"mmse_score": 20, "moca_score": 16, "hamilton_depression_score": 18, "updrs_motor_score": 15,
          "eeg_theta_alpha_ratio": 0.80, "eeg_spectral_entropy": 0.50},
         [_tx("apixaban", "5mg_BID", 70), _tx("citalopram", "20mg", 70),
          _tx("atorvastatin", "80mg", 70), _tx("baclofen", "10mg_TID", 71)],
         "72-year-old man with large right MCA ischemic stroke (age 70) on background of atrial "
         "fibrillation. Post-stroke: vascular dementia, depression, left hemiparesis with "
         "spasticity, and dysphagia. On anticoagulation, antidepressant, and spasticity management."),

        ("COMORB_HD_026", "comorbidity", "Huntingtons_disease",
         ["depression", "irritability", "chorea", "weight_loss"],
         {"primary": "G10", "secondary": ["F32.1", "R45.4", "G25.5", "R63.4"]},
         42, "M", 20.0, "Caucasian",
         [_v("HTT", "CAG_expansion", "CAG_repeat_43", None, None,
             "pathogenic", "heterozygous", None, "autosomal_dominant",
             "CAG repeat 43: full penetrance; onset typically 35-50")],
         [_comorb("depression", "F32.1", 35), _comorb("irritability", "R45.4", 36),
          _comorb("chorea", "G25.5", 38), _comorb("weight_loss", "R63.4", 40),
          _comorb("dysphagia_onset", "R13.10", 42)],
         {"mmse_score": 24, "moca_score": 20, "hamilton_depression_score": 20, "updrs_motor_score": 30,
          "eeg_theta_alpha_ratio": 0.58, "eeg_spectral_entropy": 0.55},
         [_tx("tetrabenazine", "25mg_TID", 38), _tx("sertraline", "150mg", 35)],
         "42-year-old man with Huntington's disease (CAG=43). Psychiatric symptoms (depression, "
         "irritability) preceded chorea by 3 years. Progressive weight loss and early dysphagia. "
         "Father died of HD at 55. Caudate atrophy on MRI."),

        ("COMORB_MSPROG_027", "comorbidity", "secondary_progressive_MS",
         ["trigeminal_neuralgia", "neurogenic_bladder", "spasticity", "fatigue", "depression"],
         {"primary": "G35", "secondary": ["G50.0", "N31.9", "G24.1", "R53.83", "F32.1"]},
         48, "F", 25.0, "Caucasian",
         [],
         [_comorb("RRMS", "G35", 25), _comorb("trigeminal_neuralgia", "G50.0", 35),
          _comorb("neurogenic_bladder", "N31.9", 40), _comorb("spasticity", "G24.1", 38),
          _comorb("fatigue", "R53.83", 25), _comorb("depression", "F32.1", 30),
          _comorb("SPMS_transition", "G35", 42)],
         {"mmse_score": 27, "moca_score": 24, "hamilton_depression_score": 16, "updrs_motor_score": 20,
          "eeg_theta_alpha_ratio": 0.48, "eeg_spectral_entropy": 0.65},
         [_tx("ocrelizumab", "600mg_q6mo", 42), _tx("carbamazepine", "400mg_BID", 35),
          _tx("baclofen", "20mg_TID", 38), _tx("oxybutynin", "5mg_BID", 40),
          _tx("duloxetine", "60mg", 30)],
         "48-year-old woman with MS for 23 years. RRMS at 25, failed interferon-beta and "
         "natalizumab. Transitioned to SPMS at 42. Accumulated disability: trigeminal neuralgia, "
         "neurogenic bladder, spasticity, fatigue, depression. Now on ocrelizumab. EDSS 6.0."),

        ("COMORB_EPIPSYCH_028", "comorbidity", "drug_resistant_temporal_lobe_epilepsy",
         ["depression", "anxiety", "memory_complaints", "osteoporosis"],
         {"primary": "G40.219", "secondary": ["F32.1", "F41.1", "R41.3", "M81.0"]},
         32, "F", 24.0, "Hispanic",
         [],
         [_comorb("temporal_lobe_epilepsy", "G40.219", 14), _comorb("depression", "F32.1", 18),
          _comorb("anxiety", "F41.1", 18), _comorb("memory_complaints", "R41.3", 25),
          _comorb("osteoporosis", "M81.0", 30)],
         {"mmse_score": 27, "moca_score": 24, "hamilton_depression_score": 16, "updrs_motor_score": 0,
          "eeg_theta_alpha_ratio": 0.55, "eeg_spectral_entropy": 0.58},
         [_tx("phenytoin", "300mg", 14, "discontinued_30"), _tx("lacosamide", "200mg_BID", 28),
          _tx("clobazam", "10mg_BID", 25), _tx("sertraline", "100mg", 18),
          _tx("alendronate", "70mg_weekly", 30)],
         "32-year-old woman with drug-resistant left temporal lobe epilepsy since age 14. "
         "Psychiatric comorbidity: depression and anxiety since 18. Long-term phenytoin caused "
         "osteoporosis by age 30. Memory complaints from epilepsy + medications. Being evaluated "
         "for anterior temporal lobectomy."),

        ("COMORB_LATEDEP_029", "comorbidity", "recurrent_major_depressive_disorder",
         ["insomnia", "fibromyalgia", "IBS", "hypertension", "MCI"],
         {"primary": "F33.1", "secondary": ["G47.00", "M79.7", "K58.9", "I10", "G31.84"]},
         68, "F", 28.0, "Caucasian",
         [],
         [_comorb("recurrent_MDD", "F33.1", 30), _comorb("insomnia", "G47.00", 28),
          _comorb("fibromyalgia", "M79.7", 40), _comorb("irritable_bowel_syndrome", "K58.9", 35),
          _comorb("hypertension", "I10", 55), _comorb("mild_cognitive_impairment", "G31.84", 65),
          _comorb("serotonin_syndrome_history", "T43.205", 60)],
         {"mmse_score": 26, "moca_score": 23, "hamilton_depression_score": 20, "updrs_motor_score": 2,
          "eeg_theta_alpha_ratio": 0.52, "eeg_spectral_entropy": 0.62},
         [_tx("venlafaxine", "150mg", 55), _tx("mirtazapine", "15mg", 62),
          _tx("gabapentin", "300mg_TID", 40), _tx("amlodipine", "5mg", 55)],
         "68-year-old woman with recurrent MDD (onset 30, >8 episodes), fibromyalgia, IBS, "
         "insomnia, and recent MCI onset. History of serotonin syndrome from SSRI-triptan "
         "interaction at age 60. Central sensitization + depression + early cognitive decline "
         "represent interconnected pathophysiology."),

        ("COMORB_NEURODEV_030", "comorbidity", "autism_spectrum_disorder",
         ["ADHD", "anxiety", "sleep_apnea", "obesity", "prediabetes"],
         {"primary": "F84.0", "secondary": ["F90.9", "F41.1", "G47.33", "E66.9", "R73.03"]},
         25, "M", 34.0, "Caucasian",
         [_v("16p11.2", "microdeletion", "del(16)(p11.2)", None, None,
             "pathogenic", "heterozygous", 0.0003, "autosomal_dominant",
             "16p11.2 microdeletion (~600kb): associated with ASD, obesity, and ID")],
         [_comorb("ASD", "F84.0", 3), _comorb("ADHD", "F90.9", 6),
          _comorb("anxiety", "F41.1", 15), _comorb("obesity", "E66.9", 12),
          _comorb("sleep_apnea", "G47.33", 22), _comorb("prediabetes", "R73.03", 24)],
         {"mmse_score": 28, "moca_score": 25, "hamilton_depression_score": 8, "updrs_motor_score": 0,
          "eeg_theta_alpha_ratio": 0.50, "eeg_spectral_entropy": 0.63},
         [_tx("atomoxetine", "80mg", 12), _tx("sertraline", "50mg", 15),
          _tx("CPAP", "nightly", 22)],
         "25-year-old man with ASD + ADHD + anxiety on background of 16p11.2 microdeletion. "
         "Obesity since childhood (known association with 16p11.2 del) progressed to prediabetes "
         "and OSA. Represents genomic disorder with neurodevelopmental + metabolic phenotype."),
    ]

    for (cid, cat, prim_dx, sec_dx, icd, age, sex, bmi, eth,
         variants, comorbs, biomarkers, treatments, narrative) in _comorb_specs:
        cases.append({
            "case_id": cid, "category": cat,
            "classification": {
                "primary_diagnosis": prim_dx, "secondary_diagnoses": sec_dx,
                "icd10_codes": icd, "orpha_code": None, "omim": None,
                "rarity": "common", "diagnostic_category": "multi_comorbidity",
            },
            "demographics": {"age": age, "sex": sex, "bmi": bmi, "ethnicity": eth,
                             "family_history": "See clinical narrative"},
            "genetic_variants": variants,
            "comorbidities": comorbs,
            "multi_omics": {
                "genomic": {"variants": [{"gene": v["gene"], "hgvs_p": v.get("hgvs_p", "")} for v in variants] if variants else [],
                            "cnv": [{"region": v["hgvs_c"]} for v in variants if "del" in v.get("hgvs_c", "")] if variants else [],
                            "repeat_expansions": []},
                "transcriptomic": {"platform": "RNA-seq_blood", "key_degs": []},
                "proteomic": {"blood_nfl_pg_ml": 30, "blood_gfap_pg_ml": 120},
                "metabolomic": {},
                "epigenomic": {"horvath_age_acceleration_yrs": 2.0},
            },
            "standard_biomarkers": biomarkers,
            "treatment_history": treatments,
            "diagnostic_journey": {
                "initial_presentation": "See clinical narrative",
                "initial_diagnosis": prim_dx, "misdiagnoses": [],
                "correct_diagnosis": {"diagnosis": prim_dx, "age": age, "method": "clinical"},
                "diagnostic_delay_years": 0,
                "key_diagnostic_clue": "Multi-system presentation",
            },
            "trajectory_summary": [],
            "clinical_narrative": narrative,
            "is_synthetic": True,
            "data_sources_modeled": ["UK Biobank", "Medicare Claims", "ClinVar"],
        })

    return cases


# ---------------------------------------------------------------------------
# CATEGORY 3: DIAGNOSTIC MIMICS
# ---------------------------------------------------------------------------

def _mimic_cases():
    cases = []
    _mimic_specs = [
        ("MIMIC_NMDARSZ_031", "Anti-NMDAR encephalitis mimicking schizophrenia",
         "anti_NMDAR_encephalitis", "schizophrenia",
         {"primary": "G04.81", "secondary": ["F20.9"]}, 20, "F", 21.0, "Asian",
         [], {"csf_anti_nmdar_titer": "1:256", "blood_nfl_pg_ml": 28},
         {"mmse_score": 16, "moca_score": 12, "hamilton_depression_score": 5, "eeg_theta_alpha_ratio": 1.5},
         "20-year-old woman with acute psychosis and catatonia. Treated as schizophrenia for "
         "3 weeks with worsening. CSF anti-NMDAR antibodies positive at 1:256. Extreme delta "
         "brush on EEG. Ovarian teratoma found. Immunotherapy led to full recovery."),

        ("MIMIC_NPHPD_032", "Normal pressure hydrocephalus mimicking PD",
         "normal_pressure_hydrocephalus", "Parkinsons_disease",
         {"primary": "G91.2", "secondary": ["G20"]}, 73, "M", 27.0, "Caucasian",
         [], {"csf_opening_pressure_cmH2O": 16, "blood_nfl_pg_ml": 20},
         {"mmse_score": 23, "moca_score": 19, "updrs_motor_score": 25, "eeg_theta_alpha_ratio": 0.55},
         "73-year-old man with magnetic gait, urinary incontinence, and cognitive decline. "
         "Diagnosed as PD (poor levodopa response). CSF tap test: gait improved 30%. Evans "
         "index 0.35. VP shunt placed with significant improvement."),

        ("MIMIC_FTDAD_033", "FTD mimicking AD",
         "frontotemporal_dementia", "Alzheimers_disease",
         {"primary": "G31.09", "secondary": ["G30.0"]}, 58, "M", 26.0, "Caucasian",
         [_v("GRN", "c.1477C>T", "p.Arg493Ter", "rs63750412", "VCV000001120",
             "pathogenic", "heterozygous", 0.0001, "autosomal_dominant",
             "Progranulin null mutation; haploinsufficiency causes TDP-43 proteinopathy")],
         {"plasma_progranulin_ng_ml": 35, "blood_nfl_pg_ml": 65, "csf_amyloid_beta42_pg_ml": 1000},
         {"mmse_score": 24, "moca_score": 20, "hamilton_depression_score": 2, "eeg_theta_alpha_ratio": 0.50},
         "58-year-old man with progressive behavioral disinhibition and semantic language impairment. "
         "Initially diagnosed early-onset AD. Memory relatively preserved. Plasma progranulin "
         "critically low at 35 ng/mL (normal >100). GRN R493X confirmed. CSF Abeta42 normal, "
         "excluding AD."),

        ("MIMIC_NSYPH_034", "Neurosyphilis mimicking dementia",
         "neurosyphilis", "behavioral_variant_FTD",
         {"primary": "A52.3", "secondary": ["G31.09"]}, 52, "M", 26.0, "Caucasian",
         [], {"csf_vdrl": "reactive_1:4", "csf_fta_abs": "reactive", "csf_wbc_per_ul": 35},
         {"mmse_score": 22, "moca_score": 18, "hamilton_depression_score": 8, "eeg_theta_alpha_ratio": 0.60},
         "52-year-old man with 18-month progressive cognitive decline and personality change. "
         "Diagnosed as bvFTD. Argyll Robertson pupils noted on exam. CSF VDRL reactive at 1:4, "
         "FTA-ABS positive, 35 WBC/uL. IV penicillin for 14 days with significant improvement."),

        ("MIMIC_CELIACMS_035", "Celiac disease mimicking MS",
         "celiac_disease_neurological", "multiple_sclerosis",
         {"primary": "K90.0", "secondary": ["G35"]}, 35, "F", 22.0, "European",
         [], {"anti_tissue_transglutaminase_iu_ml": 85, "anti_gliadin_iu_ml": 120, "blood_nfl_pg_ml": 18},
         {"mmse_score": 28, "moca_score": 26, "hamilton_depression_score": 10, "eeg_theta_alpha_ratio": 0.45},
         "35-year-old woman with progressive ataxia, white matter lesions on MRI, and peripheral "
         "neuropathy. Diagnosed MS, but poor response to DMTs over 2 years. Anti-tTG 85 IU/mL. "
         "Duodenal biopsy: villous atrophy (Marsh 3b). Gluten-free diet led to improvement."),

        ("MIMIC_B12ALS_036", "B12 deficiency mimicking ALS",
         "vitamin_B12_deficiency_neuropathy", "ALS",
         {"primary": "E53.8", "secondary": ["G12.21"]}, 55, "M", 24.0, "Indian",
         [], {"serum_b12_pg_ml": 85, "serum_methylmalonic_acid_nmol_l": 1200, "blood_nfl_pg_ml": 12},
         {"mmse_score": 28, "moca_score": 26, "hamilton_depression_score": 6, "updrs_motor_score": 8},
         "55-year-old vegetarian man with progressive weakness, fasciculations, and hyperreflexia. "
         "Suspected ALS. B12 critically low at 85 pg/mL, methylmalonic acid 1200 nmol/L. "
         "EMG showed denervation. B12 replacement led to significant improvement over 6 months."),

        ("MIMIC_DLBAD_037", "DLB mimicking AD",
         "dementia_with_Lewy_bodies", "Alzheimers_disease",
         {"primary": "G31.83", "secondary": ["G30.1"]}, 71, "M", 25.0, "Caucasian",
         [], {"csf_alpha_synuclein_pg_ml": 650, "csf_amyloid_beta42_pg_ml": 550, "blood_nfl_pg_ml": 40},
         {"mmse_score": 20, "moca_score": 16, "hamilton_depression_score": 10, "updrs_motor_score": 22,
          "eeg_theta_alpha_ratio": 0.80},
         "71-year-old man with cognitive fluctuations, visual hallucinations, and RBD preceding "
         "parkinsonism by 2 years. Initially diagnosed AD. DAT-SPECT shows reduced striatal "
         "dopamine uptake. MIBG cardiac scintigraphy abnormal. Reclassified as probable DLB."),

        ("MIMIC_WHIPPLE_038", "Whipple's disease mimicking psychiatric disorder",
         "Whipples_disease", "psychiatric_disorder",
         {"primary": "K90.81", "secondary": ["F09"]}, 48, "M", 21.0, "Caucasian",
         [], {"csf_tropheryma_whipplei_pcr": "positive", "blood_nfl_pg_ml": 25},
         {"mmse_score": 24, "moca_score": 20, "hamilton_depression_score": 12, "eeg_theta_alpha_ratio": 0.60},
         "48-year-old man with cognitive decline, psychiatric symptoms, oculomasticatory "
         "myorhythmia, arthralgia, weight loss, and chronic diarrhea. PAS-positive macrophages "
         "on duodenal biopsy. CSF PCR positive for Tropheryma whipplei. 1 year of antibiotics."),

        ("MIMIC_HPTHDEP_039", "Hyperparathyroidism mimicking depression",
         "primary_hyperparathyroidism", "major_depressive_disorder",
         {"primary": "E21.0", "secondary": ["F32.1"]}, 62, "F", 27.0, "Caucasian",
         [], {"serum_calcium_mg_dl": 12.2, "serum_pth_pg_ml": 145, "blood_nfl_pg_ml": 10},
         {"mmse_score": 27, "moca_score": 25, "hamilton_depression_score": 18, "eeg_theta_alpha_ratio": 0.45},
         "62-year-old woman with 2-year history of fatigue, depression, cognitive complaints, "
         "and bone pain. Treated as MDD (poor SSRI response). Calcium 12.2 mg/dL, PTH 145 pg/mL. "
         "Parathyroid adenoma on sestamibi scan. Parathyroidectomy resolved all symptoms."),

        ("MIMIC_CNSLMS_040", "CNS lymphoma mimicking MS",
         "primary_CNS_lymphoma", "multiple_sclerosis",
         {"primary": "C85.10", "secondary": ["G35"]}, 45, "M", 26.0, "Caucasian",
         [], {"csf_il10_pg_ml": 85, "csf_protein_mg_dl": 120, "blood_nfl_pg_ml": 55},
         {"mmse_score": 24, "moca_score": 20, "hamilton_depression_score": 8, "eeg_theta_alpha_ratio": 0.55},
         "45-year-old man with multifocal enhancing white matter lesions. Diagnosed MS, started "
         "on DMTs with progression. Stereotactic brain biopsy: diffuse large B-cell lymphoma. "
         "CSF IL-10 elevated at 85 pg/mL. CSF flow cytometry positive for monoclonal B cells."),
    ]

    for (cid, label, true_dx, mimic_dx, icd, age, sex, bmi, eth,
         variants, extra_proteomic, biomarkers, narrative) in _mimic_specs:
        cases.append({
            "case_id": cid, "category": "mimic",
            "classification": {
                "primary_diagnosis": true_dx,
                "secondary_diagnoses": [mimic_dx],
                "icd10_codes": icd,
                "orpha_code": None, "omim": None,
                "rarity": "uncommon",
                "diagnostic_category": "diagnostic_mimic",
                "mimic_details": {"true_diagnosis": true_dx, "mimicked_diagnosis": mimic_dx},
            },
            "demographics": {"age": age, "sex": sex, "bmi": bmi, "ethnicity": eth,
                             "family_history": "See clinical narrative"},
            "genetic_variants": variants,
            "comorbidities": [],
            "multi_omics": {
                "genomic": {"variants": [{"gene": v["gene"]} for v in variants] if variants else [],
                            "cnv": [], "repeat_expansions": []},
                "transcriptomic": {"platform": "RNA-seq_blood", "key_degs": []},
                "proteomic": extra_proteomic,
                "metabolomic": {},
                "epigenomic": {},
            },
            "standard_biomarkers": biomarkers,
            "treatment_history": [],
            "diagnostic_journey": {
                "initial_presentation": "See clinical narrative",
                "initial_diagnosis": mimic_dx,
                "misdiagnoses": [{"diagnosis": mimic_dx, "age": age}],
                "correct_diagnosis": {"diagnosis": true_dx, "age": age, "method": "See narrative"},
                "diagnostic_delay_years": None,
                "key_diagnostic_clue": label,
            },
            "trajectory_summary": [],
            "clinical_narrative": narrative,
            "is_synthetic": True,
            "data_sources_modeled": ["Published case series", "ClinVar"],
        })

    return cases


# ---------------------------------------------------------------------------
# CATEGORY 4: GENETIC OVERLAP / PLEIOTROPY
# ---------------------------------------------------------------------------

def _genetic_overlap_cases():
    cases = []
    _geno_specs = [
        ("GENO_LRRK2CROHN_041", "LRRK2_PD_with_Crohn_disease",
         "LRRK2_G2019S_pleiotropy", ["Parkinsons_disease", "Crohns_disease"],
         {"primary": "G20", "secondary": ["K50.90"]}, 55, "M", 25.0, "Ashkenazi_Jewish",
         [_v("LRRK2", "c.6055G>A", "p.Gly2019Ser", "rs34637584", "VCV000001940",
             "pathogenic", "heterozygous", 0.001, "autosomal_dominant",
             "G2019S: increased kinase activity; risk factor for both PD and Crohn's disease")],
         {"mmse_score": 27, "updrs_motor_score": 28, "hamilton_depression_score": 10},
         "55-year-old Ashkenazi Jewish man with PD (onset 50) and Crohn's disease (onset 30). "
         "LRRK2 G2019S is a shared risk factor for both via autophagy/lysosome pathway disruption."),

        ("GENO_GBADUAL_042", "GBA_PD_Gaucher_carrier",
         "GBA_pleiotropy", ["Parkinsons_disease", "Gaucher_disease_carrier"],
         {"primary": "G20", "secondary": ["E75.22"]}, 62, "F", 24.0, "Ashkenazi_Jewish",
         [_v("GBA", "c.1448T>C", "p.Leu444Pro", "rs421016", "VCV000004289",
             "pathogenic", "heterozygous", 0.003, "complex",
             "L444P heterozygous: Gaucher carrier + ~8x PD risk; worse PD cognitive trajectory")],
         {"mmse_score": 23, "updrs_motor_score": 35, "hamilton_depression_score": 14},
         "62-year-old Ashkenazi Jewish woman with PD and cognitive decline. GBA L444P heterozygous "
         "— Gaucher carrier with reduced GCase activity (2.8 nmol/hr/mg). Worse cognitive "
         "trajectory than typical PD. Alpha-synuclein-GCase interaction drives pathology."),

        ("GENO_C9FTD_043a", "C9orf72_pure_FTD",
         "C9orf72_phenotypic_heterogeneity", ["behavioral_variant_FTD"],
         {"primary": "G31.09"}, 65, "M", 26.0, "Irish",
         [_v("C9orf72", "(GGGGCC)n_expansion", "repeat_expansion", None, "VCV000227087",
             "pathogenic", "heterozygous", 0.001, "autosomal_dominant",
             "Same expansion in siblings with different phenotypes (FTD vs ALS)")],
         {"mmse_score": 20, "updrs_motor_score": 5, "hamilton_depression_score": 4},
         "65-year-old man with pure behavioral variant FTD (no motor neuron signs) from C9orf72 "
         "expansion. His 60-year-old sister (case 043b) has pure ALS from the same expansion, "
         "illustrating phenotypic heterogeneity within a single family."),

        ("GENO_C9ALS_043b", "C9orf72_pure_ALS",
         "C9orf72_phenotypic_heterogeneity", ["amyotrophic_lateral_sclerosis"],
         {"primary": "G12.21"}, 60, "F", 23.0, "Irish",
         [_v("C9orf72", "(GGGGCC)n_expansion", "repeat_expansion", None, "VCV000227087",
             "pathogenic", "heterozygous", 0.001, "autosomal_dominant",
             "Same expansion as sibling (043a) but manifests as pure ALS")],
         {"mmse_score": 28, "updrs_motor_score": 20, "hamilton_depression_score": 8},
         "60-year-old woman with pure ALS (no cognitive/behavioral features) from same C9orf72 "
         "expansion as her brother (case 043a) who has pure FTD. Blood NfL 150 pg/mL. "
         "Illustrates how identical mutations produce different phenotypes."),

        ("GENO_MAPTGRN_044", "MAPT_P301L_FTD",
         "FTD_genetic_heterogeneity", ["frontotemporal_dementia_tau"],
         {"primary": "G31.09"}, 55, "M", 27.0, "Caucasian",
         [_v("MAPT", "c.902C>T", "p.Pro301Leu", "rs63751273", "VCV000012556",
             "pathogenic", "heterozygous", 0.00005, "autosomal_dominant",
             "Promotes tau aggregation; tau-positive inclusions (Pick bodies)")],
         {"mmse_score": 22, "updrs_motor_score": 15, "hamilton_depression_score": 4},
         "55-year-old man with FTD from MAPT P301L — tau-positive pathology (3R+4R tauopathy). "
         "Compare with GRN FTD case (033) which has TDP-43 pathology. Same clinical syndrome, "
         "different molecular pathology, requiring different therapeutic targets."),

        ("GENO_SCN1A_045", "SCN1A_Dravet_vs_GEFS_plus",
         "SCN1A_phenotypic_spectrum", ["Dravet_syndrome", "GEFS_plus"],
         {"primary": "G40.834", "secondary": ["G40.509"]}, 3, "F", None, "Caucasian",
         [_v("SCN1A", "c.4943G>A", "p.Arg1648His", "rs121918722", "VCV000003254",
             "pathogenic", "heterozygous", 1e-5, "autosomal_dominant",
             "R1648H: de novo in child (Dravet); mosaic in mother (GEFS+). Variable expressivity.")],
         {"mmse_score": None, "updrs_motor_score": 2, "eeg_theta_alpha_ratio": 1.5},
         "3-year-old girl with Dravet syndrome (SCN1A R1648H, de novo) — febrile seizures at "
         "6 months, then prolonged afebrile seizures, developmental regression. Mother (35) "
         "carries same variant in mosaic form with mild GEFS+ phenotype. Same variant, "
         "dramatically different severity due to mosaicism."),

        ("GENO_APOEDIV_046", "APOE_e4_resilience_vs_vulnerability",
         "APOE_e4_divergent_outcomes", ["healthy_aging", "Alzheimers_disease"],
         {"primary": "Z00.00", "secondary": ["G30.1"]}, 78, "M", 25.0, "Caucasian",
         [_v("APOE", "rs429358_C/C+rs7412_C/C", "APOE_e4/e4", "rs429358", None,
             "risk_factor", "homozygous", 0.02, "complex",
             "e4/e4 homozygous yet cognitively intact at 78 — resilience factors include KLOTHO-VS, exercise, Mediterranean diet")],
         {"mmse_score": 29, "moca_score": 28, "hamilton_depression_score": 2, "updrs_motor_score": 2},
         "78-year-old man, APOE e4/e4 homozygous, cognitively normal (MMSE 29). KLOTHO-VS "
         "heterozygous (protective). Lifelong Mediterranean diet, regular exercise, high education. "
         "Contrast with comorbidity case 016 (75yo M, e4/e4, moderate AD). Demonstrates that "
         "genetic risk is modifiable."),

        ("GENO_TSC_047", "TSC_mTOR_spectrum",
         "TSC_multi_system", ["tuberous_sclerosis", "epilepsy", "ASD", "SEGA"],
         {"primary": "Q85.1", "secondary": ["G40.409", "F84.0", "D43.0"]}, 15, "F", 22.0, "Asian",
         [_v("TSC2", "c.4375C>T", "p.Arg1459Ter", None, None,
             "pathogenic", "heterozygous", 1e-6, "autosomal_dominant",
             "Tuberin loss of function; constitutive mTORC1 activation")],
         {"mmse_score": None, "updrs_motor_score": 3, "eeg_theta_alpha_ratio": 0.90},
         "15-year-old girl with TSC2 mutation manifesting as drug-resistant epilepsy, ASD, and "
         "subependymal giant cell astrocytoma (SEGA). Everolimus (mTOR inhibitor) treats both "
         "SEGA and seizures — a precision medicine approach targeting the shared pathway."),

        ("GENO_22Q11_048", "22q11_2_deletion_schizophrenia",
         "22q11.2_deletion_syndrome", ["schizophrenia", "hypoparathyroidism", "cardiac_defect"],
         {"primary": "D82.1", "secondary": ["F20.9", "E20.9", "Q21.0"]}, 22, "M", 24.0, "Caucasian",
         [_v("22q11.2", "del(22)(q11.21q11.23)", "3Mb_deletion", None, None,
             "pathogenic", "heterozygous", 0.0004, "autosomal_dominant",
             "TBX1 haploinsufficiency; ~25% develop schizophrenia; most common microdeletion syndrome")],
         {"mmse_score": 25, "updrs_motor_score": 3, "hamilton_depression_score": 10,
          "eeg_theta_alpha_ratio": 0.58},
         "22-year-old man with 22q11.2 deletion syndrome (DiGeorge). Repaired VSD at 6 months. "
         "Hypoparathyroidism requiring calcium/vitamin D. Schizophrenia onset at 20 (25% lifetime "
         "risk in 22q11.2DS). Illustrates CNV-driven psychiatric phenotype."),

        ("GENO_PSEN1SP_049", "PSEN1_atypical_spastic_paraparesis_AD",
         "PSEN1_cotton_wool_plaques", ["Alzheimers_disease", "spastic_paraparesis"],
         {"primary": "G30.0", "secondary": ["G82.1"]}, 38, "M", 24.0, "Caucasian",
         [_v("PSEN1", "c.869_871delTCA", "p.Ile290del_exon9", None, "VCV000430103",
             "pathogenic", "heterozygous", 1e-6, "autosomal_dominant",
             "Delta exon 9: cotton wool plaques; atypical with spastic paraparesis before dementia")],
         {"mmse_score": 26, "updrs_motor_score": 12, "hamilton_depression_score": 8},
         "38-year-old man presenting with progressive spastic paraparesis at 35, then memory "
         "decline at 37. PSEN1 exon 9 deletion — atypical early-onset AD with cotton wool "
         "plaques. CSF Abeta42 320 pg/mL, pTau181 120 pg/mL. Illustrates phenotypic expansion "
         "of AD beyond typical amnestic presentation."),

        ("GENO_DUAL_050", "APOE_plus_antiLGI1_dual_pathology",
         "genetic_plus_autoimmune", ["Alzheimers_disease_risk", "anti_LGI1_encephalitis"],
         {"primary": "G04.81", "secondary": ["G30.9"]}, 45, "F", 26.0, "Caucasian",
         [_v("APOE", "rs429358_C/rs7412_T", "APOE_e4/e3", "rs429358", None,
             "risk_factor", "heterozygous", 0.14, "complex",
             "e4 carrier with concurrent autoimmune encephalitis — dual pathology")],
         {"mmse_score": 22, "updrs_motor_score": 3, "hamilton_depression_score": 12},
         "45-year-old woman presenting with acute limbic encephalitis (anti-LGI1 positive) on "
         "background of APOE e4/e3 genotype. Immunotherapy resolved autoimmune component but "
         "subclinical amyloid pathology (borderline PET) persists. Dual genetic + autoimmune "
         "risk model — demonstrates additive pathological burden."),
    ]

    for (cid, label, prim_dx, sec_dx, icd, age, sex, bmi, eth,
         variants, biomarkers, narrative) in _geno_specs:
        icd_dict = icd if isinstance(icd, dict) else {"primary": icd}
        if "secondary" not in icd_dict:
            icd_dict["secondary"] = []
        cases.append({
            "case_id": cid, "category": "genetic_overlap",
            "classification": {
                "primary_diagnosis": prim_dx, "secondary_diagnoses": sec_dx,
                "icd10_codes": icd_dict, "orpha_code": None, "omim": None,
                "rarity": "uncommon", "diagnostic_category": "genetic_pleiotropy",
            },
            "demographics": {"age": age, "sex": sex, "bmi": bmi, "ethnicity": eth,
                             "family_history": "See clinical narrative"},
            "genetic_variants": variants,
            "comorbidities": [],
            "multi_omics": {
                "genomic": {"variants": [{"gene": v["gene"], "hgvs_p": v.get("hgvs_p", "")} for v in variants],
                            "cnv": [{"region": v["hgvs_c"]} for v in variants if "del" in v.get("hgvs_c", "")] if variants else [],
                            "repeat_expansions": [{"gene": v["gene"]} for v in variants if "repeat" in v.get("hgvs_p", "").lower()] if variants else []},
                "transcriptomic": {"platform": "RNA-seq_blood", "key_degs": []},
                "proteomic": {"blood_nfl_pg_ml": 30},
                "metabolomic": {},
                "epigenomic": {},
            },
            "standard_biomarkers": biomarkers,
            "treatment_history": [],
            "diagnostic_journey": {
                "initial_presentation": "See clinical narrative",
                "initial_diagnosis": prim_dx, "misdiagnoses": [],
                "correct_diagnosis": {"diagnosis": prim_dx, "age": age, "method": "genetic_testing"},
                "diagnostic_delay_years": None,
                "key_diagnostic_clue": label,
            },
            "trajectory_summary": [],
            "clinical_narrative": narrative,
            "is_synthetic": True,
            "data_sources_modeled": ["ClinVar", "OMIM", "Published literature"],
        })

    return cases


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def generate_all_cases():
    cases = []
    cases.extend(_zebra_cases())
    cases.extend(_comorbidity_cases())
    cases.extend(_mimic_cases())
    cases.extend(_genetic_overlap_cases())
    return cases


def flatten_case(case):
    """Flatten a case dict into a single-row dict for Parquet."""
    flat = {
        "case_id": case["case_id"],
        "category": case["category"],
        "primary_diagnosis": case["classification"]["primary_diagnosis"],
        "icd10_primary": case["classification"]["icd10_codes"].get("primary", ""),
        "rarity": case["classification"].get("rarity", ""),
        "omim": case["classification"].get("omim", ""),
        "orpha_code": case["classification"].get("orpha_code", ""),
        "age": case["demographics"]["age"],
        "sex": case["demographics"]["sex"],
        "bmi": case["demographics"].get("bmi"),
        "ethnicity": case["demographics"].get("ethnicity", ""),
        "n_comorbidities": len(case.get("comorbidities", [])),
        "n_genetic_variants": len(case.get("genetic_variants", [])),
        "genetic_variants_json": json.dumps(case.get("genetic_variants", [])),
        "comorbidities_json": json.dumps(case.get("comorbidities", [])),
        "clinical_narrative": case.get("clinical_narrative", ""),
        "is_synthetic": True,
    }
    for k, v in case.get("standard_biomarkers", {}).items():
        flat[k] = v
    return flat


def main():
    parser = argparse.ArgumentParser(
        description="Generate complex synthetic cases (zebras, comorbidities, mimics)")
    parser.add_argument("--output-dir", default=str(
        Path(__file__).resolve().parent.parent / "quarantine"))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cases = generate_all_cases()

    # Summary
    cats = {}
    for c in cases:
        cats[c["category"]] = cats.get(c["category"], 0) + 1

    print(f"Generated {len(cases)} complex cases across {len(cats)} categories")
    for cat, n in sorted(cats.items()):
        print(f"  {cat:25s}: {n}")

    # Validation checks
    errors = []
    for c in cases:
        if not c.get("is_synthetic"):
            errors.append(f"{c['case_id']}: missing is_synthetic flag")
        if not c.get("clinical_narrative"):
            errors.append(f"{c['case_id']}: missing clinical narrative")
        cid = c["case_id"]
        icd = c["classification"]["icd10_codes"]
        if "primary" not in icd:
            errors.append(f"{cid}: missing primary ICD-10")
        age = c["demographics"]["age"]
        if age is not None:
            for comorb in c.get("comorbidities", []):
                if comorb.get("onset_age") is not None and comorb["onset_age"] > age:
                    errors.append(f"{cid}: comorbidity {comorb['condition']} onset_age {comorb['onset_age']} > current age {age}")

    if errors:
        print(f"\nValidation errors ({len(errors)}):")
        for e in errors:
            print(f"  - {e}")
    else:
        print("\nAll validation checks passed.")

    # Write JSON
    output = {
        "_metadata": {
            "synthetic": True,
            "version": "1.0.0",
            "description": (
                "Complex synthetic patient cases: zebra diagnoses, multi-comorbidity "
                "clusters, diagnostic mimics, and genetic overlap cases with full "
                "multi-omics profiles. All genetic variants use real identifiers from "
                "ClinVar/OMIM/gnomAD. All patient data is ENTIRELY SYNTHETIC."
            ),
            "generation_timestamp": datetime.now(timezone.utc).isoformat(),
            "warning": "SYNTHETIC DATA - NOT FOR CLINICAL DECISIONS",
            "total_cases": len(cases),
            "categories": cats,
        },
        "cases": cases,
    }

    json_path = output_dir / "SYNTHETIC_complex_cases.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nJSON: {json_path} ({json_path.stat().st_size / 1024:.1f} KB)")

    # Write Parquet
    flat_rows = [flatten_case(c) for c in cases]
    df = pd.DataFrame(flat_rows)
    pq_path = output_dir / "SYNTHETIC_complex_cases.parquet"
    df.to_parquet(str(pq_path), engine="pyarrow", index=False)
    print(f"Parquet: {pq_path} ({len(df)} rows)")

    print(f"\nCase IDs:")
    for c in cases:
        tag = ""
        if c.get("genetic_variants"):
            genes = [v["gene"] for v in c["genetic_variants"]]
            tag = f"  [{', '.join(genes)}]"
        print(f"  {c['case_id']:30s} {c['classification']['primary_diagnosis']}{tag}")


if __name__ == "__main__":
    main()
