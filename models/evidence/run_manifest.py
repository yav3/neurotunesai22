"""Deterministic run manifest — hash-addressed, immutable record of every run.

Ported from merlinaineuralplus2/backend/runs.py. Every analysis run through the
math kernel produces a manifest with two identities:

  manifest_id = sha256(analysis, request, params, seed, preprocess, output, runtime)
  input_identity = sha256(analysis, request, params, seed, preprocess, output)

manifest_id changes with code version; input_identity does not.
An auditor resolves against input_identity; strict reproducibility uses manifest_id.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

MANIFEST_VERSION = "1.0.0"


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False, default=_default_ser)


def _default_ser(o: Any) -> Any:
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if hasattr(o, "to_dict"):
        return o.to_dict()
    if hasattr(o, "__dict__"):
        return {k: v for k, v in o.__dict__.items() if not k.startswith("_")}
    return str(o)


def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def fingerprint(obj: Any) -> str:
    return sha256_hex(canonical_json(obj))


def _git_sha() -> str:
    env = os.environ.get("GIT_SHA") or os.environ.get("SOURCE_COMMIT")
    if env:
        return env.strip()
    here = os.path.dirname(os.path.abspath(__file__))
    git_dir = os.path.join(here, "..", "..", ".git")
    try:
        with open(os.path.join(git_dir, "HEAD"), encoding="utf-8") as fh:
            head = fh.read().strip()
        if head.startswith("ref:"):
            ref = head.split(" ", 1)[1].strip()
            with open(os.path.join(git_dir, ref), encoding="utf-8") as fh:
                return fh.read().strip()
        return head
    except OSError:
        return "unknown"


def _library_versions() -> dict[str, str]:
    versions: dict[str, str] = {"python": platform.python_version()}
    for mod in ("numpy", "scipy", "torch"):
        try:
            versions[mod] = sys.modules[mod].__version__ if mod in sys.modules else __import__(mod).__version__
        except Exception:
            versions[mod] = "unknown"
    return versions


def _runtime_fingerprint() -> dict[str, str]:
    return {
        "os": platform.system().lower(),
        "platform": platform.platform(),
        "libraries": _library_versions(),
        "code_version": _git_sha(),
    }


@dataclass
class RunManifest:
    """Immutable record of a single analysis run."""
    manifest_id: str
    input_identity: str
    analysis: str
    request_fingerprint: str
    parameters: dict[str, Any]
    seed: int | None
    input_fingerprint: str
    output_fingerprint: str
    preprocess_steps: list[dict[str, Any]]
    runtime: dict[str, Any]
    created_at: float
    species: str = "unknown"
    regime: str = "unknown"
    study_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest_id": self.manifest_id,
            "input_identity": self.input_identity,
            "analysis": self.analysis,
            "request_fingerprint": self.request_fingerprint,
            "parameters": self.parameters,
            "seed": self.seed,
            "input_fingerprint": self.input_fingerprint,
            "output_fingerprint": self.output_fingerprint,
            "preprocess_steps": self.preprocess_steps,
            "runtime": self.runtime,
            "created_at": self.created_at,
            "species": self.species,
            "regime": self.regime,
            "study_id": self.study_id,
        }


def build_manifest(
    *,
    analysis: str,
    request_payload: dict[str, Any],
    parameters: dict[str, Any],
    seed: int | None,
    input_fingerprint: str,
    preprocess_steps: list[dict[str, Any]] | None,
    output_payload: dict[str, Any],
    species: str = "unknown",
    regime: str = "unknown",
    study_id: str | None = None,
    created_at: float | None = None,
) -> RunManifest:
    """Build a run manifest with dual identity (strict + input-only)."""
    runtime = _runtime_fingerprint()
    hashable = {
        "manifest_version": MANIFEST_VERSION,
        "analysis": analysis,
        "request_fingerprint": fingerprint(request_payload),
        "parameters": parameters,
        "seed": seed,
        "input_fingerprint": input_fingerprint,
        "preprocess_steps": preprocess_steps or [],
        "output_fingerprint": fingerprint(output_payload),
        "runtime": runtime,
    }
    manifest_id = "run_" + sha256_hex(canonical_json(hashable))

    input_hashable = {k: v for k, v in hashable.items() if k != "runtime"}
    input_identity = "in_" + sha256_hex(canonical_json(input_hashable))

    return RunManifest(
        manifest_id=manifest_id,
        input_identity=input_identity,
        analysis=analysis,
        request_fingerprint=fingerprint(request_payload),
        parameters=parameters,
        seed=seed,
        input_fingerprint=input_fingerprint,
        output_fingerprint=fingerprint(output_payload),
        preprocess_steps=preprocess_steps or [],
        runtime=runtime,
        created_at=created_at or time.time(),
        species=species,
        regime=regime,
        study_id=study_id,
    )
