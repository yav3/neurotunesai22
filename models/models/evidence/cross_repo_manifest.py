"""Cross-repository integration manifest.

Documents the contracts between NeuroTunesAI and the other repositories:

  merlinaineuralplus2 (Merlin):
    - 65+ pre-registered study records in backend/records/studies/
    - RTP patent kernel in backend/app/patents/rtp/ (golden.json identity)
    - 14 data harvest sources
    - Eval harness: citation + calibration + truth-eval
    - Engine catalog: 12 physics/math engines
    - Patent estate: 3 granted (US 11,694,786; US 12,361,263; US 12,651,165)
      + 2 PCT (WO 2026/087924; WO 2026/087925)

  neurotunes (Flutter mobile):
    - Audio delivery layer (binaural/isochronic)
    - Patent documentation

  neurotunescopy4privatesharing (React product):
    - Therapeutic session UI
    - Audio analysis pipeline
    - Goal-based therapy workflows

Integration contracts:
  1. EntitySeries: shared data contract (entity_ids, T×N values, FNV-1a hash)
  2. Problem→Answer: typed math dispatch
  3. StudyRecord: pre-registered experiment format
  4. EvidenceRecord: hash-chained lineage
  5. CalibrationReport: confidence calibration format

Data flow:
  Merlin harvest → EntitySeries → NeuroTunesAI solver → GovernedAnswer
  → EvidenceRecord chain → Merlin study registry → Eval harness verdict
"""
from __future__ import annotations

from dataclasses import dataclass, field


PATENT_ESTATE = {
    "granted": [
        {
            "number": "US 11,694,786 B1",
            "title": "Recommendation Methods, Systems and Devices",
            "family": "affective",
            "filed": "2020-01-06",
            "issued": "2023-07-04",
        },
        {
            "number": "US 12,361,263 B1",
            "title": "AI Techniques Utilizing a Generative Relational Network",
            "family": "grn",
            "filed": "2024-10-22",
            "issued": "2025-07-15",
        },
        {
            "number": "US 12,651,165 B2",
            "title": "Techniques for Training a Generative Relational Network",
            "family": "grn",
            "filed": "2024-10-22",
            "issued": "2026-06-09",
        },
    ],
    "pending_international": [
        {"number": "WO 2026/087924 A1", "pct": "PCT/IB2024/060379"},
        {"number": "WO 2026/087925 A1", "pct": "PCT/IB2024/060380"},
    ],
    "bypass_continuation_deadline": "2027-04-22",
}

MERLIN_STUDY_SERIES = {
    "geo_replications": [
        "RB-010", "RB-011", "RB-012", "RB-013", "RB-014",
        "RB-015", "RB-016", "RB-017", "RB-018",
    ],
    "geo_controls": ["RB-019", "RB-020", "RB-021", "RB-022", "RB-023", "RB-024", "RB-025", "RB-027"],
    "clinical_frontiers": ["CF-001", "CF-002", "CF-003", "CF-004", "CF-005", "CF-006"],
    "evidence_retrieval": ["RC-001", "RC-002", "RC-003", "RC-011"],
    "golden_vectors": ["NA-001", "NA-002", "NA-007", "NA-009"],
    "kernel_trials": ["TP-001", "TP-002", "TP-003", "TP-004", "TP-005", "TP-006", "TP-007"],
}

MERLIN_ENGINE_CATALOG = [
    "tcs_engine",         # Temporal Causal Structure
    "si_engine",          # Susceptibility / Information
    "causal_graph_engine",
    "adl_engine",         # Activities of Daily Living
    "generative_engine",
    "stochastic_dynamics_engine",
    "vad_engine",         # Valence-Arousal-Dominance
    "vadee_engine",       # VADE Extended
    "deterministic_engine",
    "information_theory_engine",
    "entrainment_engine",
    "threshold_calibration",
]

MERLIN_HARVEST_SOURCES = [
    "clinicaltrials_gov", "sec_edgar", "cftc_cot", "bis_statistics",
    "fdic_banking", "fed_communications", "fred", "gdc_tcga",
    "ncbi_geo", "ken_french", "openfda", "pubmed_baseline",
    "tcia_imaging", "worldbank_imf",
]

MATH_KERNEL_COVERAGE = {
    "patent_263_claim_1": {
        "elements": ["dominance", "evolution", "simulate", "eliminate", "test", "identify"],
        "implemented_in": ["models/math_core/grn_solver.py", "models/math_core/rtp_solver.py"],
        "bridge": "models/evidence/rtp_bridge.py",
    },
    "patent_165_claim_1": {
        "elements": ["som_visualization", "train_grn_nodes"],
        "implemented_in": ["models/math/grn_dynamics.py"],
        "bridge": "models/evidence/rtp_bridge.py (planned)",
    },
    "patent_786": {
        "elements": ["media_selection", "recommendation"],
        "implemented_in": ["models/math_core/eif_solver.py", "models/math_core/therapeutic_solver.py"],
        "bridge": "separate estate — not co-located with GRN",
    },
}


@dataclass
class IntegrationStatus:
    """Current cross-repo integration status."""
    math_kernel_problem_types: int = 52
    math_core_solvers: int = 14
    evidence_modules: int = 6
    study_records_in_merlin: int = 65
    harvest_sources: int = 14
    patent_grants: int = 3
    pct_applications: int = 2
    eval_harness_loops: int = 3

    @property
    def summary(self) -> str:
        return (
            f"NeuroTunesAI: {self.math_kernel_problem_types} problem types, "
            f"{self.math_core_solvers} exact solvers, {self.evidence_modules} evidence modules. "
            f"Merlin: {self.study_records_in_merlin} study records, "
            f"{self.harvest_sources} data sources, {self.eval_harness_loops} eval loops. "
            f"IP: {self.patent_grants} granted + {self.pct_applications} PCT pending."
        )
