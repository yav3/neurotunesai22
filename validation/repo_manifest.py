"""Declarative manifest of what each repo in the ecosystem must provide.

Each RepoSpec names the repo, its role, and the modules/files/tests that
must exist for the cross-repo evidence discipline to hold.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ModuleSpec:
    """A required module or file within a repo."""
    path: str
    description: str
    must_export: list[str] = field(default_factory=list)


@dataclass
class RepoSpec:
    """What one repository must contain."""
    name: str
    role: str
    root_hint: str
    required_modules: list[ModuleSpec] = field(default_factory=list)
    required_tests: list[str] = field(default_factory=list)


REPO_MANIFEST: dict[str, RepoSpec] = {
    "NeuroTunesAI": RepoSpec(
        name="NeuroTunesAI",
        role="governed_solver",
        root_hint="/home/user/NeuroTunesAI",
        required_modules=[
            ModuleSpec(
                "models/evidence/lineage.py",
                "Hash-chained evidence lineage (SHA-256 chain)",
                ["LineageChain"],
            ),
            ModuleSpec(
                "models/evidence/registry.py",
                "Evidence registry — study declarations and verdicts",
                ["EvidenceRegistry"],
            ),
            ModuleSpec(
                "models/evidence/regime.py",
                "Species/regime firewall (declare or refuse)",
                ["Species", "Regime", "SpeciesDeclaration"],
            ),
            ModuleSpec(
                "models/inference/perturbation_inference.py",
                "Perturbation response inference (cVAE NT-001)",
                ["PerturbationInferenceService"],
            ),
            ModuleSpec(
                "models/training/study_runner.py",
                "Governed study runner with pre-registration",
                ["GovernedStudyRunner"],
            ),
            ModuleSpec(
                "models/training/governed_harness.py",
                "Training harness with evidence discipline",
            ),
        ],
        required_tests=[
            "tests/test_study_runner.py",
            "tests/test_perturbation_inference.py",
        ],
    ),
    "merlinaineuralplus2": RepoSpec(
        name="merlinaineuralplus2",
        role="evidence_engine",
        root_hint="/home/user/merlinaineuralplus2",
        required_modules=[
            ModuleSpec(
                "backend/methods/expression_io.py",
                "Expression data I/O with gene-ID normalisation",
            ),
            ModuleSpec(
                "backend/methods/species.py",
                "Species firewall (never pool mouse and human)",
            ),
            ModuleSpec(
                "backend/methods/grn_benchmark.py",
                "Planted-network benchmark for GRN ranking",
            ),
            ModuleSpec(
                "backend/methods/target_nomination.py",
                "Target nomination with calibrated composite weights",
            ),
            ModuleSpec(
                "backend/methods/validation_discipline.py",
                "Reusable regression utilities for evidence discipline",
            ),
        ],
        required_tests=[
            "backend/tests/test_endpoints.py",
            "backend/tests/test_prediction_engine.py",
        ],
    ),
    "neurotunes3": RepoSpec(
        name="neurotunes3",
        role="shared_schemas",
        root_hint="/home/user/neurotunes3",
        required_modules=[
            ModuleSpec(
                "evidence_protocol/schemas.py",
                "Canonical dataclass definitions for evidence records",
                [
                    "EvidenceRecord",
                    "StudyDeclaration",
                    "Verdict",
                    "PerturbationSpec",
                    "PredictionEnvelope",
                    "DatasetManifest",
                ],
            ),
            ModuleSpec(
                "evidence_protocol/firewall.py",
                "Species and regime validation rules",
                ["classify_regime", "validate_species"],
            ),
            ModuleSpec(
                "evidence_protocol/hashing.py",
                "Canonical SHA-256 hashing for lineage chains",
                ["chain_hash", "content_hash", "prediction_hash"],
            ),
        ],
        required_tests=[
            "tests/test_schemas.py",
        ],
    ),
    "neurotunes": RepoSpec(
        name="neurotunes",
        role="mobile_therapeutic",
        root_hint="/home/user/neurotunes",
        required_modules=[
            ModuleSpec(
                "lib/services/governed_inference_service.dart",
                "Flutter client with species firewall and trial records",
            ),
        ],
        required_tests=[],
    ),
    "neurotunescopy4privatesharing": RepoSpec(
        name="neurotunescopy4privatesharing",
        role="web_product",
        root_hint="/home/user/neurotunescopy4privatesharing",
        required_modules=[
            ModuleSpec(
                "supabase/functions/audio-analysis/index.ts",
                "Audio analysis edge function with perturbation interfaces",
            ),
        ],
        required_tests=[],
    ),
}
