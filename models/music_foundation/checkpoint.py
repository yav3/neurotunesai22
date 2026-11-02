"""Model checkpoint: save/load, versioning, migration, differential snapshots.

Supports JSON, NPZ, and a custom binary format. Includes checkpoint
averaging (SWA/EMA merge), parameter diffing, and migration between
config versions.
"""
from __future__ import annotations

import hashlib
import json
import struct
import time
from pathlib import Path
from typing import Any

import numpy as np

from .encoder import MusicFoundationModel, EncoderConfig


CHECKPOINT_VERSION = 3


# ──────────────────────────────────────────────────────────────────────
# Core save / load (JSON)
# ──────────────────────────────────────────────────────────────────────

def save_checkpoint(
    model: MusicFoundationModel,
    path: str | Path,
    metadata: dict[str, Any] | None = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    params = model.params()
    arrays = {k: v.tolist() for k, v in params.items()}

    checkpoint = {
        "version": CHECKPOINT_VERSION,
        "config": {
            "audio_dim": model.config.audio_dim,
            "semantic_dim": model.config.semantic_dim,
            "hidden_dim": model.config.hidden_dim,
            "latent_dim": model.config.latent_dim,
            "n_heads": model.config.n_heads,
            "dropout": model.config.dropout,
            "alpha": model.config.alpha,
        },
        "params": arrays,
        "metadata": metadata or {},
        "timestamp": time.time(),
        "param_hash": _hash_params(params),
    }

    with open(path, "w") as f:
        json.dump(checkpoint, f)

    return path


def load_checkpoint(path: str | Path) -> tuple[MusicFoundationModel, dict[str, Any]]:
    path = Path(path)
    with open(path) as f:
        checkpoint = json.load(f)

    checkpoint = _migrate(checkpoint)

    cfg = EncoderConfig(**checkpoint["config"])
    model = MusicFoundationModel(cfg)

    params = model.params()
    for k, v in checkpoint["params"].items():
        if k in params:
            arr = np.array(v)
            if arr.shape == params[k].shape:
                params[k][:] = arr

    return model, checkpoint.get("metadata", {})


# ──────────────────────────────────────────────────────────────────────
# NPZ (compressed numpy archive)
# ──────────────────────────────────────────────────────────────────────

def save_npz(model: MusicFoundationModel, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    params = model.params()
    np.savez_compressed(str(path), **params)
    return path


def load_npz(model: MusicFoundationModel, path: str | Path) -> None:
    path = Path(path)
    data = np.load(str(path))
    params = model.params()
    for k in data.files:
        if k in params and data[k].shape == params[k].shape:
            params[k][:] = data[k]


# ──────────────────────────────────────────────────────────────────────
# Binary checkpoint (fast, compact)
# ──────────────────────────────────────────────────────────────────────

_MAGIC = b"MFCK"


def save_binary(params: dict[str, np.ndarray], path: str | Path) -> Path:
    """Save parameters in a compact binary format.

    Layout: MAGIC(4) | version(4) | n_params(4) | [name_len(4) name(N) ndim(4) shape(ndim*4) data(flat)] ...
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(_MAGIC)
        f.write(struct.pack("<II", CHECKPOINT_VERSION, len(params)))
        for name, arr in params.items():
            name_bytes = name.encode("utf-8")
            f.write(struct.pack("<I", len(name_bytes)))
            f.write(name_bytes)
            f.write(struct.pack("<I", arr.ndim))
            for s in arr.shape:
                f.write(struct.pack("<I", s))
            data = arr.astype(np.float64).tobytes()
            f.write(data)
    return path


def load_binary(path: str | Path) -> dict[str, np.ndarray]:
    path = Path(path)
    params: dict[str, np.ndarray] = {}
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != _MAGIC:
            raise ValueError(f"Invalid checkpoint magic: {magic!r}")
        version, n_params = struct.unpack("<II", f.read(8))
        for _ in range(n_params):
            name_len = struct.unpack("<I", f.read(4))[0]
            name = f.read(name_len).decode("utf-8")
            ndim = struct.unpack("<I", f.read(4))[0]
            shape = tuple(struct.unpack("<I", f.read(4))[0] for _ in range(ndim))
            n_elements = 1
            for s in shape:
                n_elements *= s
            data = np.frombuffer(f.read(n_elements * 8), dtype=np.float64).copy()
            params[name] = data.reshape(shape)
    return params


# ──────────────────────────────────────────────────────────────────────
# Parameter hashing
# ──────────────────────────────────────────────────────────────────────

def _hash_params(params: dict[str, np.ndarray]) -> str:
    h = hashlib.sha256()
    for k in sorted(params.keys()):
        h.update(k.encode("utf-8"))
        h.update(params[k].astype(np.float64).tobytes())
    return h.hexdigest()[:16]


def params_equal(a: dict[str, np.ndarray], b: dict[str, np.ndarray], atol: float = 1e-12) -> bool:
    if set(a.keys()) != set(b.keys()):
        return False
    return all(
        a[k].shape == b[k].shape and np.allclose(a[k], b[k], atol=atol)
        for k in a
    )


# ──────────────────────────────────────────────────────────────────────
# Differential checkpoint (save only changed params)
# ──────────────────────────────────────────────────────────────────────

def diff_params(
    base: dict[str, np.ndarray],
    updated: dict[str, np.ndarray],
    atol: float = 1e-12,
) -> dict[str, np.ndarray]:
    delta: dict[str, np.ndarray] = {}
    for k in updated:
        if k not in base or base[k].shape != updated[k].shape:
            delta[k] = updated[k].copy()
        elif not np.allclose(base[k], updated[k], atol=atol):
            delta[k] = updated[k] - base[k]
    return delta


def apply_diff(
    base: dict[str, np.ndarray],
    delta: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    result = {k: v.copy() for k, v in base.items()}
    for k, d in delta.items():
        if k in result and result[k].shape == d.shape:
            result[k] = result[k] + d
        else:
            result[k] = d.copy()
    return result


def save_diff_checkpoint(
    base_path: str | Path,
    updated_params: dict[str, np.ndarray],
    diff_path: str | Path,
    atol: float = 1e-12,
) -> int:
    base = load_binary(base_path)
    delta = diff_params(base, updated_params, atol)
    save_binary(delta, diff_path)
    return len(delta)


# ──────────────────────────────────────────────────────────────────────
# Checkpoint averaging (SWA / EMA merge)
# ──────────────────────────────────────────────────────────────────────

def average_checkpoints(
    param_list: list[dict[str, np.ndarray]],
    weights: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Weighted average of multiple parameter dictionaries (SWA)."""
    n = len(param_list)
    if n == 0:
        return {}
    if weights is None:
        weights = np.ones(n) / n
    else:
        weights = np.asarray(weights, dtype=np.float64)
        weights = weights / weights.sum()

    keys = list(param_list[0].keys())
    averaged: dict[str, np.ndarray] = {}
    for k in keys:
        acc = np.zeros_like(param_list[0][k])
        for i, p in enumerate(param_list):
            if k in p and p[k].shape == acc.shape:
                acc += weights[i] * p[k]
        averaged[k] = acc
    return averaged


def exponential_average(
    param_list: list[dict[str, np.ndarray]],
    decay: float = 0.999,
) -> dict[str, np.ndarray]:
    """EMA over an ordered list of parameter snapshots."""
    if not param_list:
        return {}
    n = len(param_list)
    weights = np.array([(1.0 - decay) * decay ** (n - 1 - i) for i in range(n)])
    weights[-1] += 1.0 - weights.sum()
    return average_checkpoints(param_list, weights)


def interpolate_params(
    a: dict[str, np.ndarray],
    b: dict[str, np.ndarray],
    alpha: float = 0.5,
) -> dict[str, np.ndarray]:
    """Linear interpolation: (1-α)·a + α·b."""
    result: dict[str, np.ndarray] = {}
    for k in a:
        if k in b and a[k].shape == b[k].shape:
            result[k] = (1.0 - alpha) * a[k] + alpha * b[k]
        else:
            result[k] = a[k].copy()
    return result


# ──────────────────────────────────────────────────────────────────────
# Parameter statistics
# ──────────────────────────────────────────────────────────────────────

def param_stats(params: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
    stats: dict[str, dict[str, float]] = {}
    for k, v in params.items():
        stats[k] = {
            "mean": float(v.mean()),
            "std": float(v.std()),
            "min": float(v.min()),
            "max": float(v.max()),
            "norm": float(np.linalg.norm(v)),
            "numel": int(v.size),
        }
    return stats


def total_parameters(params: dict[str, np.ndarray]) -> int:
    return sum(v.size for v in params.values())


def param_norm(params: dict[str, np.ndarray]) -> float:
    return float(np.sqrt(sum(float(np.sum(v ** 2)) for v in params.values())))


# ──────────────────────────────────────────────────────────────────────
# Migration
# ──────────────────────────────────────────────────────────────────────

def _migrate(checkpoint: dict) -> dict:
    """Migrate older checkpoint formats to current version."""
    v = checkpoint.get("version", 1)
    if v < 2:
        cfg = checkpoint.get("config", {})
        cfg.setdefault("n_heads", 4)
        cfg.setdefault("dropout", 0.1)
        cfg.setdefault("alpha", 0.5)
        checkpoint["config"] = cfg
        checkpoint["version"] = 2
    if v < 3:
        checkpoint.setdefault("param_hash", "")
        checkpoint["version"] = 3
    return checkpoint


# ──────────────────────────────────────────────────────────────────────
# Checkpoint directory manager
# ──────────────────────────────────────────────────────────────────────

class CheckpointManager:
    """Manages a directory of training checkpoints with rotation."""

    def __init__(self, directory: str | Path, max_keep: int = 5) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.max_keep = max_keep
        self._history: list[dict[str, Any]] = []

    def save(
        self,
        params: dict[str, np.ndarray],
        step: int,
        metrics: dict[str, float] | None = None,
    ) -> Path:
        name = f"ckpt_step{step:08d}.bin"
        path = self.directory / name
        save_binary(params, path)
        entry = {
            "step": step,
            "path": str(path),
            "timestamp": time.time(),
            "metrics": metrics or {},
            "hash": _hash_params(params),
        }
        self._history.append(entry)
        self._rotate()
        self._save_index()
        return path

    def load_latest(self) -> tuple[dict[str, np.ndarray], dict[str, Any]] | None:
        if not self._history:
            self._load_index()
        if not self._history:
            return None
        entry = self._history[-1]
        params = load_binary(entry["path"])
        return params, entry

    def load_best(self, metric: str = "loss", lower_is_better: bool = True) -> tuple[dict[str, np.ndarray], dict[str, Any]] | None:
        if not self._history:
            self._load_index()
        candidates = [e for e in self._history if metric in e.get("metrics", {})]
        if not candidates:
            return self.load_latest()
        key_fn = lambda e: e["metrics"][metric]
        best = min(candidates, key=key_fn) if lower_is_better else max(candidates, key=key_fn)
        return load_binary(best["path"]), best

    def _rotate(self) -> None:
        while len(self._history) > self.max_keep:
            old = self._history.pop(0)
            p = Path(old["path"])
            if p.exists():
                p.unlink()

    def _save_index(self) -> None:
        idx_path = self.directory / "index.json"
        with open(idx_path, "w") as f:
            json.dump(self._history, f)

    def _load_index(self) -> None:
        idx_path = self.directory / "index.json"
        if idx_path.exists():
            with open(idx_path) as f:
                self._history = json.load(f)

    @property
    def history(self) -> list[dict[str, Any]]:
        if not self._history:
            self._load_index()
        return list(self._history)


# ──────────────────────────────────────────────────────────────────────
# Self-test
# ──────────────────────────────────────────────────────────────────────

def _self_test() -> None:
    import tempfile
    rng = np.random.RandomState(42)

    params_a = {"w": rng.randn(4, 3), "b": rng.randn(3)}
    params_b = {"w": params_a["w"] + 0.01 * rng.randn(4, 3), "b": params_a["b"].copy()}
    params_c = {"w": rng.randn(4, 3), "b": rng.randn(3)}

    with tempfile.TemporaryDirectory() as tmpdir:
        # Binary save/load
        p = Path(tmpdir) / "test.bin"
        save_binary(params_a, p)
        loaded = load_binary(p)
        assert params_equal(params_a, loaded), "Binary roundtrip"

        # Diff checkpoint
        diff = diff_params(params_a, params_b)
        assert "w" in diff, "Diff should contain changed param"
        assert "b" not in diff, "Diff should skip unchanged param"
        restored = apply_diff(params_a, diff)
        assert np.allclose(restored["w"], params_b["w"]), "Diff apply"

        # Save diff
        base_p = Path(tmpdir) / "base.bin"
        diff_p = Path(tmpdir) / "diff.bin"
        save_binary(params_a, base_p)
        n_changed = save_diff_checkpoint(base_p, params_b, diff_p)
        assert n_changed == 1, f"Expected 1 changed, got {n_changed}"

        # Averaging
        avg = average_checkpoints([params_a, params_c])
        expected_w = (params_a["w"] + params_c["w"]) / 2
        assert np.allclose(avg["w"], expected_w), "Uniform average"

        weighted = average_checkpoints([params_a, params_c], np.array([0.3, 0.7]))
        assert np.allclose(weighted["w"], 0.3 * params_a["w"] + 0.7 * params_c["w"]), "Weighted average"

        # EMA
        ema = exponential_average([params_a, params_b, params_c], decay=0.9)
        assert ema["w"].shape == (4, 3), "EMA shape"

        # Interpolation
        mid = interpolate_params(params_a, params_c, 0.5)
        assert np.allclose(mid["w"], expected_w), "Interpolation"

        # Stats
        st = param_stats(params_a)
        assert "w" in st and "mean" in st["w"], "Stats keys"
        assert total_parameters(params_a) == 15, f"Total params: {total_parameters(params_a)}"

        # Hashing
        h1 = _hash_params(params_a)
        h2 = _hash_params(params_a)
        h3 = _hash_params(params_b)
        assert h1 == h2, "Same params same hash"
        assert h1 != h3, "Different params different hash"

        # Checkpoint manager
        mgr = CheckpointManager(Path(tmpdir) / "ckpts", max_keep=3)
        for i in range(5):
            p_i = {"w": rng.randn(4, 3), "b": rng.randn(3)}
            mgr.save(p_i, step=i * 100, metrics={"loss": 1.0 / (i + 1)})
        assert len(mgr.history) == 3, f"Rotation: {len(mgr.history)}"
        latest = mgr.load_latest()
        assert latest is not None, "Load latest"
        best = mgr.load_best("loss", lower_is_better=True)
        assert best is not None, "Load best"

        # Migration
        old_ckpt = {"config": {"audio_dim": 128, "semantic_dim": 64, "hidden_dim": 256, "latent_dim": 128}}
        migrated = _migrate(old_ckpt)
        assert migrated["version"] == CHECKPOINT_VERSION, "Migration version"
        assert "n_heads" in migrated["config"], "Migration adds n_heads"

    parts = [
        "Binary", "Diff", "DiffSave", "Average", "WeightedAvg",
        "EMA", "Interpolate", "Stats", "Hash", "Manager", "Migration",
    ]
    print(f"Self-test passed: {', '.join(parts)}.")
