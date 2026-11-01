"""Evidence discipline bridge — connects math solvers to Merlin's governed evidence system.

Implements the record-first evidence pattern from merlinaineuralplus2:
  - Pre-registered criteria (declare the criterion, then run)
  - Hash-chained lineage (SHA-256 chain over inputs + outputs)
  - Replay determinism (same inputs → same outputs, verified)
  - Species/regime firewall (declare or refuse, never default)
  - Confidence calibration (ECE, MCE, Brier score)
  - Truth evaluation against labelled gold sets
  - GovernedSolver wrapper enforcing discipline on every call
  - RTP bridge mapping to Merlin's patent kernel contract
"""
from models.evidence.lineage import EvidenceRecord, LineageChain
from models.evidence.registry import StudyRegistry, StudyRecord
from models.evidence.regime import (
    RegimeFirewall, SpeciesDeclaration, Species, Regime,
)
from models.evidence.eval_harness import (
    CalibrationEntry, calibration_report,
    TruthEntry, truth_eval,
)
from models.evidence.governed_solver import GovernedSolver, GovernedAnswer
from models.evidence.rtp_bridge import (
    EntitySeries, RTPRunManifest, ElementTrace, fnv1a64,
    run_263_bridge,
)
from models.evidence.run_manifest import RunManifest, build_manifest
from models.evidence.validation_discipline import (
    DegeneracyCheck,
    is_degenerate,
    assert_signal_survives_transform,
)

__all__ = [
    "EvidenceRecord",
    "LineageChain",
    "StudyRegistry",
    "StudyRecord",
    "RegimeFirewall",
    "SpeciesDeclaration",
    "Species",
    "Regime",
    "CalibrationEntry",
    "calibration_report",
    "TruthEntry",
    "truth_eval",
    "GovernedSolver",
    "GovernedAnswer",
    "EntitySeries",
    "RTPRunManifest",
    "ElementTrace",
    "fnv1a64",
    "run_263_bridge",
    "RunManifest",
    "build_manifest",
    "DegeneracyCheck",
    "is_degenerate",
    "assert_signal_survives_transform",
]
