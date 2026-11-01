"""Governed training harness — evidence-disciplined model training.

Every training run is pre-registered with declared criteria before any
parameter update. Lineage is hash-chained epoch-by-epoch. Checkpoints
are content-addressed. Species and regime boundaries are enforced.
Controls run in the same invocation as the result.

Mirrors merlinaineuralplus2/backend/app/inventions/harness.py conventions:
content-addressed records, pre-registered criteria, Criterion.evaluate().

Mirrors merlinaineuralplus2/backend/methods/method_router.py regimes:
data-driven (≥30 samples), prior-contrast, prior-mean, longitudinal.

Mirrors merlinaineuralplus2/backend/methods/species.py firewall:
human and mouse never pooled; declare or refuse, never default.
"""
from __future__ import annotations

import hashlib
import io
import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

from models.evidence.lineage import LineageChain, EvidenceRecord
from models.evidence.registry import StudyRegistry, StudyRecord
from models.evidence.regime import RegimeFirewall, SpeciesDeclaration, Species, Regime
from models.training.trainer import Trainer, cosine_warmup_schedule

# ── constants (Merlin-aligned) ────────────────────────────────────────

MIN_DATA_DRIVEN_SAMPLES = 30
VALID_SPECIES = ("human", "mouse", "in_silico", "synthetic", "not_applicable")
Op = Literal[">", ">=", "<", "<="]
DataKind = Literal["synthetic", "computed", "real"]
TrainingRegime = Literal[
    "data-driven", "prior-contrast", "prior-mean", "longitudinal", "benchmark",
]


# ── dataclasses ───────────────────────────────────────────────────────

@dataclass(frozen=True)
class TrainingCriterion:
    """Pre-registered pass/fail criterion evaluated after training.

    Semantics match Merlin's Criterion: ``metric <op> threshold``,
    optionally comparing two arms (``versus``) or requiring a p-value
    below ``alpha``.
    """
    metric: str
    op: Op
    threshold: float | None = None
    versus: str | None = None
    p_metric: str | None = None
    alpha: float = 0.05
    description: str = ""

    def evaluate(
        self,
        arms: dict[str, dict[str, float]],
        stats: dict[str, float] | None = None,
        arm: str = "model",
    ) -> tuple[bool, str]:
        stats = stats or {}
        lhs = stats[self.metric] if arm == "@stats" else arms[arm][self.metric]
        rhs = arms[self.versus][self.metric] if self.versus else self.threshold
        if rhs is None:
            raise ValueError("criterion needs threshold or versus")
        ok = {">": lhs > rhs, ">=": lhs >= rhs, "<": lhs < rhs, "<=": lhs <= rhs}[self.op]
        detail = f"{arm}.{self.metric}={lhs:.6g} {self.op} {rhs:.6g}"
        if self.p_metric is not None:
            import math
            p = stats.get(self.p_metric, math.nan)
            ok = ok and p < self.alpha
            detail += f" (p={p:.3g} < {self.alpha})"
        return bool(ok), detail


@dataclass(frozen=True)
class TrainingSpec:
    """Immutable training specification — hashed before any training begins."""
    model_id: str
    dataset_id: str
    seed: int
    held_out_fraction: float
    species: str
    regime: TrainingRegime
    criteria: tuple[TrainingCriterion, ...]
    lr: float = 1e-4
    max_epochs: int = 100
    batch_size: int = 32
    patience: int = 10
    data_kind: DataKind = "synthetic"
    description: str = ""

    def __post_init__(self):
        if self.species not in VALID_SPECIES:
            raise ValueError(f"Species '{self.species}' not in {VALID_SPECIES}")
        if not (0 < self.held_out_fraction < 1):
            raise ValueError(f"held_out_fraction must be in (0, 1), got {self.held_out_fraction}")
        if not self.criteria:
            raise ValueError("At least one criterion must be pre-registered")

    def content_hash(self) -> str:
        payload = json.dumps(
            {
                "model_id": self.model_id,
                "dataset_id": self.dataset_id,
                "seed": self.seed,
                "held_out_fraction": self.held_out_fraction,
                "species": self.species,
                "regime": self.regime,
                "criteria": [asdict(c) for c in self.criteria],
                "lr": self.lr,
                "max_epochs": self.max_epochs,
                "batch_size": self.batch_size,
                "patience": self.patience,
                "data_kind": self.data_kind,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()


@dataclass
class TrainingRecord:
    """Content-addressed training outcome record (Merlin-compatible JSON)."""
    study_id: str
    spec_hash: str
    spec: dict[str, Any]
    arms: dict[str, dict[str, float]]
    stats: dict[str, float]
    criteria_results: list[dict[str, Any]]
    verdict: str
    lineage_head: str
    checkpoint_digest: str
    run_date: str = ""
    elapsed_seconds: float = 0.0
    record_hash: str = ""

    def __post_init__(self):
        if not self.run_date:
            self.run_date = time.strftime("%Y-%m-%d")
        if not self.record_hash:
            self.record_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        payload = json.dumps(
            {
                "study_id": self.study_id,
                "spec_hash": self.spec_hash,
                "arms": self.arms,
                "stats": self.stats,
                "criteria_results": self.criteria_results,
                "verdict": self.verdict,
                "lineage_head": self.lineage_head,
                "checkpoint_digest": self.checkpoint_digest,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    def to_dict(self) -> dict:
        return {
            "study": self.study_id,
            "spec_hash": self.spec_hash,
            "spec": self.spec,
            "arms": self.arms,
            "stats": self.stats,
            "criteria_results": self.criteria_results,
            "verdict": self.verdict,
            "lineage_head": self.lineage_head,
            "checkpoint_digest": self.checkpoint_digest,
            "run_date": self.run_date,
            "elapsed_seconds": round(self.elapsed_seconds, 2),
            "record_hash": self.record_hash,
        }


# ── checkpoint store ──────────────────────────────────────────────────

class CheckpointStore:
    """Content-addressed checkpoint storage.

    Each checkpoint is named by the SHA-256 of its serialized bytes,
    so identical model states produce identical paths and any bit flip
    is detectable.
    """

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def save(self, state_dict: dict[str, Any], metadata: dict[str, Any] | None = None) -> str:
        buf = io.BytesIO()
        torch.save({"model_state_dict": state_dict, "metadata": metadata or {}}, buf)
        data = buf.getvalue()
        digest = hashlib.sha256(data).hexdigest()
        path = self.root / f"{digest}.pt"
        if not path.exists():
            path.write_bytes(data)
        return digest

    def load(self, digest: str) -> dict[str, Any]:
        path = self.root / f"{digest}.pt"
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint {digest} not found in {self.root}")
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != digest:
            raise ValueError(f"Checkpoint {digest} failed integrity check")
        return torch.load(io.BytesIO(data), weights_only=False)

    def exists(self, digest: str) -> bool:
        return (self.root / f"{digest}.pt").exists()


# ── regime selection (Merlin method_router.py) ────────────────────────

def select_training_regime(
    *,
    n_samples: int,
    n_conditions: int = 0,
    has_time_axis: bool = False,
) -> TrainingRegime:
    if has_time_axis:
        return "longitudinal"
    if n_samples >= MIN_DATA_DRIVEN_SAMPLES:
        return "data-driven"
    if n_conditions >= 2:
        return "prior-contrast"
    return "prior-mean"


# ── species inference (Merlin species.py) ─────────────────────────────

def infer_species_from_genes(gene_names: list[str]) -> str:
    human = mouse = 0
    for g in gene_names:
        s = str(g).strip()
        if s.upper().startswith("ENSG") and not s.upper().startswith("ENSMUSG"):
            human += 1
        elif s.upper().startswith("ENSMUSG"):
            mouse += 1
        elif len(s) > 1 and s[0].isupper():
            tail = s[1:]
            has_letter = any(c.isalpha() for c in tail)
            if has_letter and any(c.islower() for c in tail):
                mouse += 1
            else:
                pass  # ambiguous (all-caps or digits-only): no vote
    if human > 0 and mouse > 0:
        raise ValueError(
            f"Mixed species signal: {human} human-like, {mouse} mouse-like genes. "
            "Declare species explicitly."
        )
    if human > 0:
        return "human"
    if mouse > 0:
        return "mouse"
    raise ValueError("Cannot infer species from gene names — declare explicitly.")


# ── governed trainer ──────────────────────────────────────────────────

class GovernedTrainer:
    """Wraps model training with full evidence discipline.

    Lifecycle:
        1. ``pre_register()`` — hash the spec, write pre-registration
        2. ``run()`` — train model + control, hash-chain each epoch
        3. ``evaluate()`` — check pre-registered criteria, produce record
        4. ``export_record()`` — write content-addressed JSON

    Controls run in the same ``run()`` call as the model arm.
    """

    def __init__(
        self,
        spec: TrainingSpec,
        model_factory: Callable[[], nn.Module],
        dataset: Dataset,
        records_dir: Path | None = None,
        checkpoints_dir: Path | None = None,
        control_factory: Callable[[], nn.Module] | None = None,
        forward_fn: Callable[[nn.Module, dict], dict[str, torch.Tensor]] | None = None,
    ):
        self.spec = spec
        self.model_factory = model_factory
        self.control_factory = control_factory
        self.dataset = dataset
        self.forward_fn = forward_fn
        self.records_dir = records_dir or Path("records")
        self.records_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_store = CheckpointStore(
            checkpoints_dir or Path("checkpoints") / "governed"
        )
        self.lineage = LineageChain()
        self.registry = StudyRegistry(self.records_dir)
        self._spec_hash = spec.content_hash()
        self._record: TrainingRecord | None = None
        self._arms: dict[str, dict[str, float]] = {}
        self._stats: dict[str, float] = {}
        self._checkpoint_digest = ""

    def pre_register(self, study_id: str) -> StudyRecord:
        criterion_text = "; ".join(
            f"{c.metric} {c.op} {c.threshold}" + (f" vs {c.versus}" if c.versus else "")
            for c in self.spec.criteria
        )
        study = self.registry.register(
            study_id=study_id,
            title=self.spec.description or f"Training {self.spec.model_id} on {self.spec.dataset_id}",
            criterion=criterion_text,
            data_kind=self.spec.data_kind,
            synthetic=self.spec.data_kind == "synthetic",
        )
        self.lineage.append(
            record_id=f"{study_id}/pre-registration",
            problem_spec={"spec_hash": self._spec_hash, "spec": asdict(self.spec)},
            answer_values={"status": "pre-registered"},
        )
        return study

    def run(self, study_id: str) -> TrainingRecord:
        study = self.registry.get(study_id)
        if study is None:
            raise ValueError(f"Study {study_id} not pre-registered — call pre_register() first")
        if study.verdict != "pending":
            raise ValueError(f"Study {study_id} already completed with verdict '{study.verdict}'")

        self._enforce_species_firewall()

        t0 = time.time()
        g = torch.Generator().manual_seed(self.spec.seed)
        n = len(self.dataset)  # type: ignore[arg-type]
        n_held = int(n * self.spec.held_out_fraction)
        n_train = n - n_held
        train_ds, val_ds = random_split(self.dataset, [n_train, n_held], generator=g)

        train_loader = DataLoader(train_ds, batch_size=self.spec.batch_size, shuffle=True, generator=g)
        val_loader = DataLoader(val_ds, batch_size=self.spec.batch_size)

        model_metrics = self._train_arm("model", self.model_factory, train_loader, val_loader)
        self._arms["model"] = model_metrics

        if self.control_factory is not None:
            control_metrics = self._train_arm("control", self.control_factory, train_loader, val_loader)
            self._arms["control"] = control_metrics

        elapsed = time.time() - t0

        criteria_results = []
        all_met = True
        for c in self.spec.criteria:
            ok, detail = c.evaluate(self._arms, self._stats)
            criteria_results.append({
                "metric": c.metric,
                "op": c.op,
                "threshold": c.threshold,
                "versus": c.versus,
                "met": ok,
                "detail": detail,
                "description": c.description,
            })
            if not ok:
                all_met = False

        verdict = "met" if all_met else "not_met"

        self.lineage.append(
            record_id=f"{study_id}/verdict",
            problem_spec={"criteria_count": len(self.spec.criteria)},
            answer_values={"verdict": verdict, "criteria_results": criteria_results},
        )

        self._record = TrainingRecord(
            study_id=study_id,
            spec_hash=self._spec_hash,
            spec=asdict(self.spec),
            arms=self._arms,
            stats=self._stats,
            criteria_results=criteria_results,
            verdict=verdict,
            lineage_head=self.lineage.head_hash,
            checkpoint_digest=self._checkpoint_digest,
            elapsed_seconds=elapsed,
        )

        n_met = sum(1 for r in criteria_results if r["met"])
        summary = f"{n_met}/{len(criteria_results)} criteria met"
        self.registry.record_verdict(
            study_id=study_id,
            measured=self._arms,
            summary=summary,
            verdict=verdict,
        )

        return self._record

    def export_record(self, path: Path | None = None) -> Path:
        if self._record is None:
            raise RuntimeError("No training record — call run() first")
        out = path or self.records_dir / f"{self._record.study_id}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(self._record.to_dict(), f, indent=2, default=str)
        return out

    def _enforce_species_firewall(self):
        species_enum = {
            "human": Species.HUMAN,
            "mouse": Species.MOUSE,
            "in_silico": Species.IN_SILICO,
            "synthetic": Species.SYNTHETIC,
            "not_applicable": Species.IN_SILICO,
        }
        regime_enum = {
            "data-driven": Regime.EXPERIMENTAL,
            "prior-contrast": Regime.OBSERVATIONAL,
            "prior-mean": Regime.OBSERVATIONAL,
            "longitudinal": Regime.EXPERIMENTAL,
            "benchmark": Regime.BENCHMARK,
        }
        decl = SpeciesDeclaration(
            species=species_enum.get(self.spec.species, Species.SYNTHETIC),
            regime=regime_enum.get(self.spec.regime, Regime.SYNTHETIC),
            justification=f"Training spec {self._spec_hash[:12]}",
        )
        ok, msg = RegimeFirewall.validate(decl)
        if not ok:
            raise ValueError(f"Species/regime firewall blocked: {msg}")

    def _train_arm(
        self,
        arm_name: str,
        factory: Callable[[], nn.Module],
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> dict[str, float]:
        torch.manual_seed(self.spec.seed)
        model = factory()

        trainer = _GovernedArmTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            lr=self.spec.lr,
            max_epochs=self.spec.max_epochs,
            patience=self.spec.patience,
            fp16=False,
            device="cuda" if torch.cuda.is_available() else "cpu",
            forward_fn=self.forward_fn,
        )

        history = trainer.train()

        for epoch_idx, loss_val in enumerate(history.get("train_loss", [])):
            self.lineage.append(
                record_id=f"{arm_name}/epoch/{epoch_idx}",
                problem_spec={"arm": arm_name, "epoch": epoch_idx},
                answer_values={"train_loss": loss_val},
            )

        final_metrics: dict[str, float] = {}
        for key in ("train_loss", "val_loss"):
            vals = history.get(key, [])
            if vals:
                final_metrics[f"final_{key}"] = vals[-1]
                final_metrics[f"best_{key}"] = min(vals)
        final_metrics["epochs_trained"] = len(history.get("train_loss", []))
        final_metrics["global_steps"] = trainer.global_step

        if arm_name == "model":
            self._checkpoint_digest = self.checkpoint_store.save(
                model.state_dict(),
                metadata={
                    "spec_hash": self._spec_hash,
                    "arm": arm_name,
                    "final_metrics": final_metrics,
                },
            )
            self.lineage.append(
                record_id=f"{arm_name}/checkpoint",
                problem_spec={"arm": arm_name},
                answer_values={"checkpoint_digest": self._checkpoint_digest},
            )

        return final_metrics


class _GovernedArmTrainer(Trainer):
    """Thin Trainer subclass that delegates forward() to a user-supplied fn."""

    def __init__(self, forward_fn: Callable | None = None, **kwargs):
        super().__init__(**kwargs, checkpoint_dir="/tmp/governed_arm_ckpt")
        self._fwd = forward_fn

    def _forward(self, batch: dict) -> dict[str, torch.Tensor]:
        if self._fwd is not None:
            return self._fwd(self.model, batch)
        return self.model(batch)
