"""Safe deployment layer for the music foundation model.

Implements point 4 of the robustness roadmap:
- Action layer with rollback tied to SHA-256 snapshots
- Versioned typed interfaces per engine (SI→TCS→VADE→GRN→OIS)
- Model registry with immutable artifacts
- Canary deployment support

Pure numpy/scipy — no external ML framework dependencies.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# SHA-256 snapshot system
# ---------------------------------------------------------------------------

def sha256_params(params: Dict[str, np.ndarray]) -> str:
    """Compute deterministic SHA-256 digest of model parameters."""
    h = hashlib.sha256()
    for key in sorted(params.keys()):
        h.update(key.encode("utf-8"))
        arr = params[key]
        h.update(arr.tobytes())
        h.update(np.array(arr.shape).tobytes())
    return h.hexdigest()


def sha256_file(path: str) -> str:
    """SHA-256 of a file on disk."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class ModelSnapshot:
    """Immutable record of a model state at a point in time."""
    sha256: str
    timestamp: float
    step: int
    loss: float
    params_path: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ModelSnapshot":
        return cls(**d)


class SnapshotStore:
    """Append-only store of model snapshots with rollback capability."""

    def __init__(self, base_dir: str = "checkpoints"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._manifest_path = self.base_dir / "manifest.json"
        self._snapshots: List[ModelSnapshot] = []
        self._load_manifest()

    def _load_manifest(self) -> None:
        if self._manifest_path.exists():
            with open(self._manifest_path, "r") as f:
                data = json.load(f)
            self._snapshots = [ModelSnapshot.from_dict(s) for s in data]

    def _save_manifest(self) -> None:
        with open(self._manifest_path, "w") as f:
            json.dump([s.to_dict() for s in self._snapshots], f, indent=2)

    def save(
        self,
        params: Dict[str, np.ndarray],
        step: int,
        loss: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ModelSnapshot:
        """Save a snapshot. Returns the snapshot record."""
        digest = sha256_params(params)
        fname = f"snapshot_{step:06d}_{digest[:12]}.npz"
        path = str(self.base_dir / fname)

        np.savez_compressed(path, **params)

        snap = ModelSnapshot(
            sha256=digest,
            timestamp=time.time(),
            step=step,
            loss=loss,
            params_path=path,
            metadata=metadata or {},
        )
        self._snapshots.append(snap)
        self._save_manifest()
        return snap

    def load(self, sha256: Optional[str] = None, step: Optional[int] = None) -> Dict[str, np.ndarray]:
        """Load parameters by SHA-256 or step number."""
        snap = self.get_snapshot(sha256=sha256, step=step)
        if snap is None:
            raise ValueError(f"No snapshot found for sha256={sha256}, step={step}")
        data = np.load(snap.params_path)
        return dict(data)

    def get_snapshot(self, sha256: Optional[str] = None, step: Optional[int] = None) -> Optional[ModelSnapshot]:
        if sha256 is not None:
            for s in reversed(self._snapshots):
                if s.sha256 == sha256 or s.sha256.startswith(sha256):
                    return s
        if step is not None:
            for s in reversed(self._snapshots):
                if s.step == step:
                    return s
        return None

    @property
    def latest(self) -> Optional[ModelSnapshot]:
        return self._snapshots[-1] if self._snapshots else None

    @property
    def best(self) -> Optional[ModelSnapshot]:
        if not self._snapshots:
            return None
        return min(self._snapshots, key=lambda s: s.loss)

    def rollback(self, to_sha256: str) -> Dict[str, np.ndarray]:
        """Rollback to a specific snapshot. Returns the parameters."""
        snap = self.get_snapshot(sha256=to_sha256)
        if snap is None:
            raise ValueError(f"Snapshot {to_sha256} not found")
        return self.load(sha256=to_sha256)

    def history(self) -> List[Dict[str, Any]]:
        return [s.to_dict() for s in self._snapshots]


# ---------------------------------------------------------------------------
# Versioned engine interfaces
# ---------------------------------------------------------------------------

@dataclass
class EngineVersion:
    """Semantic version for an engine interface."""
    major: int
    minor: int
    patch: int

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"

    def is_compatible(self, other: "EngineVersion") -> bool:
        return self.major == other.major

    @classmethod
    def parse(cls, s: str) -> "EngineVersion":
        parts = s.split(".")
        return cls(int(parts[0]), int(parts[1]), int(parts[2]))


@dataclass
class EngineSpec:
    """Typed specification of an engine's input/output contract."""
    name: str
    version: EngineVersion
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    description: str = ""
    dependencies: List[str] = field(default_factory=list)

    def validate_input(self, data: Dict[str, Any]) -> List[str]:
        """Validate input against schema. Returns list of errors (empty = valid)."""
        errors = []
        for key, spec in self.input_schema.items():
            if spec.get("required", False) and key not in data:
                errors.append(f"Missing required field: {key}")
            if key in data and "type" in spec:
                expected = spec["type"]
                val = data[key]
                if expected == "ndarray" and not isinstance(val, np.ndarray):
                    errors.append(f"{key}: expected ndarray, got {type(val).__name__}")
                elif expected == "float" and not isinstance(val, (int, float)):
                    errors.append(f"{key}: expected float, got {type(val).__name__}")
                elif expected == "int" and not isinstance(val, int):
                    errors.append(f"{key}: expected int, got {type(val).__name__}")
            if key in data and "shape" in spec and isinstance(data[key], np.ndarray):
                expected_ndim = len(spec["shape"])
                if data[key].ndim != expected_ndim:
                    errors.append(f"{key}: expected {expected_ndim}D, got {data[key].ndim}D")
        return errors

    def validate_output(self, data: Dict[str, Any]) -> List[str]:
        errors = []
        for key, spec in self.output_schema.items():
            if spec.get("required", False) and key not in data:
                errors.append(f"Missing required output: {key}")
        return errors


# Standard engine pipeline: SI → TCS → VADE → GRN → OIS
SIGNAL_INGESTION_SPEC = EngineSpec(
    name="SignalIngestion",
    version=EngineVersion(1, 0, 0),
    input_schema={
        "signal": {"type": "ndarray", "shape": ("T",), "required": True},
        "sr": {"type": "int", "required": True},
    },
    output_schema={
        "features": {"type": "dict", "required": True},
        "metadata": {"type": "dict", "required": True},
    },
    description="Raw audio → feature streams (mel, MFCC, chroma, ...)",
)

TEMPORAL_CONTEXT_SPEC = EngineSpec(
    name="TemporalContextSynthesis",
    version=EngineVersion(1, 0, 0),
    input_schema={
        "features": {"type": "dict", "required": True},
    },
    output_schema={
        "fused": {"type": "ndarray", "shape": ("d_model", "T"), "required": True},
    },
    description="Feature streams → fused representation via multi-stream integrator",
    dependencies=["SignalIngestion"],
)

VARIATIONAL_ENCODING_SPEC = EngineSpec(
    name="VariationalDiscreteEncoding",
    version=EngineVersion(1, 0, 0),
    input_schema={
        "fused": {"type": "ndarray", "shape": ("d_model", "T"), "required": True},
    },
    output_schema={
        "embeddings": {"type": "ndarray", "shape": ("T", "d_model"), "required": True},
        "quantized": {"type": "ndarray", "required": True},
        "indices": {"type": "ndarray", "required": True},
    },
    description="Fused features → transformer encoder → quantized codes",
    dependencies=["TemporalContextSynthesis"],
)

GRN_SPEC = EngineSpec(
    name="GRNDynamics",
    version=EngineVersion(1, 0, 0),
    input_schema={
        "embeddings": {"type": "ndarray", "required": True},
    },
    output_schema={
        "state": {"type": "ndarray", "required": True},
        "adjacency": {"type": "ndarray", "required": True},
    },
    description="Embeddings → regulatory network dynamics",
    dependencies=["VariationalDiscreteEncoding"],
)

OUTPUT_INTERFACE_SPEC = EngineSpec(
    name="OutputInterfaceSpec",
    version=EngineVersion(1, 0, 0),
    input_schema={
        "state": {"type": "ndarray", "required": True},
        "request_type": {"type": "str", "required": True},
    },
    output_schema={
        "result": {"type": "dict", "required": True},
        "confidence": {"type": "float", "required": True},
        "provenance": {"type": "dict", "required": True},
    },
    description="Internal state → user-facing output with provenance",
    dependencies=["GRNDynamics"],
)

ENGINE_PIPELINE = [
    SIGNAL_INGESTION_SPEC,
    TEMPORAL_CONTEXT_SPEC,
    VARIATIONAL_ENCODING_SPEC,
    GRN_SPEC,
    OUTPUT_INTERFACE_SPEC,
]


class EngineRegistry:
    """Registry of versioned engine implementations."""

    def __init__(self) -> None:
        self._engines: Dict[str, Tuple[EngineSpec, Callable]] = {}

    def register(self, spec: EngineSpec, impl: Callable) -> None:
        self._engines[spec.name] = (spec, impl)

    def get(self, name: str) -> Tuple[EngineSpec, Callable]:
        if name not in self._engines:
            raise KeyError(f"Engine {name} not registered")
        return self._engines[name]

    def run(self, name: str, data: Dict[str, Any]) -> Dict[str, Any]:
        spec, impl = self.get(name)
        errors = spec.validate_input(data)
        if errors:
            raise ValueError(f"Input validation failed for {name}: {errors}")
        result = impl(data)
        out_errors = spec.validate_output(result)
        if out_errors:
            raise ValueError(f"Output validation failed for {name}: {out_errors}")
        return result

    def run_pipeline(self, initial_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run the full engine pipeline SI→TCS→VADE→GRN→OIS."""
        data = dict(initial_data)
        for spec_template in ENGINE_PIPELINE:
            if spec_template.name in self._engines:
                result = self.run(spec_template.name, data)
                data.update(result)
        return data

    def list_engines(self) -> List[Dict[str, str]]:
        return [
            {"name": spec.name, "version": str(spec.version), "description": spec.description}
            for spec, _ in self._engines.values()
        ]


# ---------------------------------------------------------------------------
# Model artifact registry
# ---------------------------------------------------------------------------

@dataclass
class ModelArtifact:
    """Immutable model artifact with full provenance."""
    name: str
    version: str
    sha256: str
    created_at: float
    training_config: Dict[str, Any]
    metrics: Dict[str, float]
    params_path: str
    engine_versions: Dict[str, str] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)


class ModelRegistry:
    """Registry of trained model artifacts."""

    def __init__(self, base_dir: str = "model_registry"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._catalog_path = self.base_dir / "catalog.json"
        self._artifacts: Dict[str, ModelArtifact] = {}
        self._load()

    def _load(self) -> None:
        if self._catalog_path.exists():
            with open(self._catalog_path, "r") as f:
                data = json.load(f)
            for name, d in data.items():
                self._artifacts[name] = ModelArtifact(**d)

    def _save(self) -> None:
        data = {name: asdict(a) for name, a in self._artifacts.items()}
        with open(self._catalog_path, "w") as f:
            json.dump(data, f, indent=2)

    def register(
        self,
        name: str,
        version: str,
        params: Dict[str, np.ndarray],
        training_config: Dict[str, Any],
        metrics: Dict[str, float],
        tags: Optional[List[str]] = None,
    ) -> ModelArtifact:
        key = f"{name}@{version}"
        digest = sha256_params(params)
        fname = f"{name}_{version}_{digest[:12]}.npz"
        path = str(self.base_dir / fname)
        np.savez_compressed(path, **params)

        artifact = ModelArtifact(
            name=name,
            version=version,
            sha256=digest,
            created_at=time.time(),
            training_config=training_config,
            metrics=metrics,
            params_path=path,
            tags=tags or [],
        )
        self._artifacts[key] = artifact
        self._save()
        return artifact

    def load(self, name: str, version: str) -> Tuple[ModelArtifact, Dict[str, np.ndarray]]:
        key = f"{name}@{version}"
        if key not in self._artifacts:
            raise KeyError(f"Artifact {key} not found")
        artifact = self._artifacts[key]
        data = np.load(artifact.params_path)
        return artifact, dict(data)

    def list_artifacts(self, name: Optional[str] = None) -> List[ModelArtifact]:
        if name:
            return [a for a in self._artifacts.values() if a.name == name]
        return list(self._artifacts.values())

    def latest(self, name: str) -> Optional[ModelArtifact]:
        arts = self.list_artifacts(name)
        if not arts:
            return None
        return max(arts, key=lambda a: a.created_at)


# ---------------------------------------------------------------------------
# Canary deployment
# ---------------------------------------------------------------------------

@dataclass
class CanaryConfig:
    """Configuration for canary deployment."""
    traffic_fraction: float = 0.1
    metric_name: str = "loss"
    threshold: float = 0.05
    min_samples: int = 100
    rollback_on_regression: bool = True


class CanaryDeployment:
    """A/B test between current and candidate model."""

    def __init__(
        self,
        current_params: Dict[str, np.ndarray],
        candidate_params: Dict[str, np.ndarray],
        config: CanaryConfig = CanaryConfig(),
        seed: int = 42,
    ):
        self.current = current_params
        self.candidate = candidate_params
        self.config = config
        self.rng = np.random.RandomState(seed)
        self.current_metrics: List[float] = []
        self.candidate_metrics: List[float] = []

    def route(self) -> str:
        """Route a request to current or candidate."""
        if self.rng.random() < self.config.traffic_fraction:
            return "candidate"
        return "current"

    def record(self, variant: str, metric: float) -> None:
        if variant == "current":
            self.current_metrics.append(metric)
        else:
            self.candidate_metrics.append(metric)

    def should_promote(self) -> Optional[bool]:
        """Check if candidate should be promoted. None = insufficient data."""
        if (len(self.current_metrics) < self.config.min_samples or
                len(self.candidate_metrics) < self.config.min_samples):
            return None

        curr_mean = np.mean(self.current_metrics)
        cand_mean = np.mean(self.candidate_metrics)
        relative_change = (cand_mean - curr_mean) / (abs(curr_mean) + 1e-10)

        if relative_change < -self.config.threshold:
            return True
        if relative_change > self.config.threshold and self.config.rollback_on_regression:
            return False
        return None

    def summary(self) -> Dict[str, Any]:
        return {
            "current_n": len(self.current_metrics),
            "candidate_n": len(self.candidate_metrics),
            "current_mean": float(np.mean(self.current_metrics)) if self.current_metrics else None,
            "candidate_mean": float(np.mean(self.candidate_metrics)) if self.candidate_metrics else None,
            "decision": self.should_promote(),
        }


# ---------------------------------------------------------------------------
# Action layer with automatic rollback
# ---------------------------------------------------------------------------

class ActionLayer:
    """Execute model updates with automatic rollback on regression.

    Ties every parameter update to a SHA-256 snapshot. If a post-update
    evaluation shows regression beyond threshold, rolls back automatically.
    """

    def __init__(
        self,
        snapshot_store: SnapshotStore,
        regression_threshold: float = 0.1,
    ):
        self.store = snapshot_store
        self.threshold = regression_threshold
        self._rollback_count = 0

    def apply_update(
        self,
        params: Dict[str, np.ndarray],
        new_params: Dict[str, np.ndarray],
        eval_fn: Callable[[Dict[str, np.ndarray]], float],
        step: int,
    ) -> Tuple[Dict[str, np.ndarray], bool]:
        """Apply parameter update with rollback safety.

        Args:
            params: current parameters
            new_params: proposed new parameters
            eval_fn: function that evaluates params → loss (lower = better)
            step: current training step

        Returns:
            (accepted_params, was_rolled_back)
        """
        current_loss = eval_fn(params)
        self.store.save(params, step, current_loss, {"type": "pre_update"})

        new_loss = eval_fn(new_params)
        relative_change = (new_loss - current_loss) / (abs(current_loss) + 1e-10)

        if relative_change > self.threshold:
            self._rollback_count += 1
            return params, True

        self.store.save(new_params, step, new_loss, {"type": "post_update"})
        return new_params, False

    @property
    def rollback_count(self) -> int:
        return self._rollback_count


# ---------------------------------------------------------------------------
# Method card generator
# ---------------------------------------------------------------------------

@dataclass
class MethodCard:
    """One-page method card for an engine, following Model Cards (Mitchell et al. 2019)."""
    engine_name: str
    version: str
    description: str
    intended_use: str
    input_format: str
    output_format: str
    training_data: str
    eval_metrics: Dict[str, float]
    limitations: List[str]
    ethical_considerations: str = ""
    citation: str = ""

    def to_markdown(self) -> str:
        lines = [
            f"# {self.engine_name} v{self.version}",
            "",
            f"**Description:** {self.description}",
            "",
            f"**Intended Use:** {self.intended_use}",
            "",
            "## Input/Output",
            f"- **Input:** {self.input_format}",
            f"- **Output:** {self.output_format}",
            "",
            f"## Training Data",
            self.training_data,
            "",
            "## Evaluation Metrics",
        ]
        for k, v in self.eval_metrics.items():
            lines.append(f"- {k}: {v:.4f}")
        lines.extend([
            "",
            "## Limitations",
        ])
        for lim in self.limitations:
            lines.append(f"- {lim}")
        if self.ethical_considerations:
            lines.extend(["", "## Ethical Considerations", self.ethical_considerations])
        if self.citation:
            lines.extend(["", "## Citation", self.citation])
        return "\n".join(lines)


def generate_method_cards(registry: EngineRegistry) -> List[MethodCard]:
    """Generate a method card for each registered engine."""
    cards = []
    for info in registry.list_engines():
        spec, _ = registry.get(info["name"])
        card = MethodCard(
            engine_name=spec.name,
            version=str(spec.version),
            description=spec.description,
            intended_use="Music audio analysis and representation learning",
            input_format=json.dumps(spec.input_schema, indent=2),
            output_format=json.dumps(spec.output_schema, indent=2),
            training_data="Proprietary music corpus (Drive + local) + public domain datasets",
            eval_metrics={},
            limitations=["Pure numpy implementation — CPU only", "No streaming inference"],
        )
        cards.append(card)
    return cards
