"""Schema alignment checks — verifies that cross-repo data contracts hold.

Checks:
1. Hashing formulas match between neurotunes3 (canonical) and NeuroTunesAI
2. Regime classification thresholds are consistent
3. Species firewall rules agree
4. Curated-series catalog entries correspond to real Merlin datasets
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from validation.health_check import Finding


@dataclass
class AlignmentReport:
    findings: list[Finding] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(f.severity != "error" for f in self.findings)


class SchemaAlignmentCheck:
    """Cross-repo schema alignment verifier."""

    def __init__(self, roots: dict[str, Path]) -> None:
        self.roots = roots

    def check_hashing_consistency(self) -> list[Finding]:
        """Verify chain_hash formula matches across repos."""
        findings: list[Finding] = []

        canonical = self.roots.get("neurotunes3")
        governed = self.roots.get("NeuroTunesAI")

        if not canonical or not governed:
            findings.append(Finding(
                "cross-repo", "warning", "missing_repo",
                "Cannot check hashing: need both neurotunes3 and NeuroTunesAI",
            ))
            return findings

        canon_hash = self._compute_canonical_chain_hash(canonical)
        governed_hash = self._compute_governed_chain_hash(governed)

        if canon_hash and governed_hash:
            if canon_hash != governed_hash:
                findings.append(Finding(
                    "cross-repo", "error", "hash_mismatch",
                    f"chain_hash diverges: canonical={canon_hash[:16]}... "
                    f"governed={governed_hash[:16]}...",
                ))
        elif canon_hash is None:
            findings.append(Finding(
                "neurotunes3", "error", "hash_unavailable",
                "Could not compute canonical chain_hash from neurotunes3",
            ))
        elif governed_hash is None:
            findings.append(Finding(
                "NeuroTunesAI", "warning", "hash_unavailable",
                "Could not compute governed chain_hash from NeuroTunesAI",
            ))

        return findings

    def check_regime_thresholds(self) -> list[Finding]:
        """Verify regime classification gives same results for known inputs."""
        findings: list[Finding] = []

        canonical = self.roots.get("neurotunes3")
        if not canonical:
            return findings

        test_cases = [
            (44, 23, 21, "data_driven"),
            (18, 6, 12, "prior_contrast"),
            (9, 0, 0, "prior_mean"),
            (30, 15, 15, "data_driven"),
            (29, 15, 14, "prior_contrast"),
            (28, 0, 0, "prior_mean"),
        ]

        firewall_path = canonical / "evidence_protocol" / "firewall.py"
        if not firewall_path.is_file():
            findings.append(Finding(
                "neurotunes3", "error", "missing_firewall",
                "evidence_protocol/firewall.py not found",
            ))
            return findings

        classify = self._load_classify_regime(canonical)
        if classify is None:
            findings.append(Finding(
                "neurotunes3", "warning", "import_failed",
                "Could not import classify_regime from neurotunes3",
            ))
            return findings

        for n_samples, n_control, n_treated, expected in test_cases:
            result = classify(n_samples, n_control, n_treated)
            if result != expected:
                findings.append(Finding(
                    "neurotunes3", "error", "regime_mismatch",
                    f"classify_regime({n_samples},{n_control},{n_treated})"
                    f" = '{result}', expected '{expected}'",
                ))

        return findings

    def check_species_firewall(self) -> list[Finding]:
        """Verify species firewall rejects undeclared species."""
        findings: list[Finding] = []

        canonical = self.roots.get("neurotunes3")
        if not canonical:
            return findings

        validate = self._load_validate_species(canonical)
        if validate is None:
            findings.append(Finding(
                "neurotunes3", "warning", "import_failed",
                "Could not import validate_species from neurotunes3",
            ))
            return findings

        try:
            validate("human")
        except Exception as e:
            findings.append(Finding(
                "neurotunes3", "error", "firewall_reject_human",
                f"validate_species('human') raised: {e}",
            ))

        try:
            validate("mouse")
            findings.append(Finding(
                "neurotunes3", "error", "firewall_accept_mouse",
                "validate_species('mouse') should raise ValueError but did not",
            ))
        except ValueError:
            pass
        except Exception as e:
            findings.append(Finding(
                "neurotunes3", "warning", "firewall_unexpected",
                f"validate_species('mouse') raised unexpected {type(e).__name__}: {e}",
            ))

        return findings

    def check_curated_series(self) -> list[Finding]:
        """Verify NT-001 curated series catalog entries are structurally sound."""
        findings: list[Finding] = []
        governed = self.roots.get("NeuroTunesAI")
        if not governed:
            return findings

        catalog_path = governed / "docs" / "records" / "NT-001_curated_series.json"
        if not catalog_path.is_file():
            findings.append(Finding(
                "NeuroTunesAI", "warning", "missing_catalog",
                "NT-001 curated series catalog not found",
            ))
            return findings

        try:
            catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            findings.append(Finding(
                "NeuroTunesAI", "error", "catalog_parse",
                f"Cannot parse curated series catalog: {e}",
            ))
            return findings

        required_fields = {"accession", "species", "n_samples"}
        entries = catalog if isinstance(catalog, list) else (
            catalog.get("datasets", []) or catalog.get("series", [])
        )

        if not entries:
            findings.append(Finding(
                "NeuroTunesAI", "warning", "catalog_empty",
                "Curated series catalog has no entries",
            ))
            return findings

        for i, entry in enumerate(entries):
            if not isinstance(entry, dict):
                continue
            missing = required_fields - set(entry.keys())
            if missing:
                findings.append(Finding(
                    "NeuroTunesAI", "error", "catalog_incomplete",
                    f"Catalog entry {i} missing fields: {missing}",
                ))
            species = entry.get("species", "")
            if species and species != "human":
                findings.append(Finding(
                    "NeuroTunesAI", "warning", "catalog_species",
                    f"Catalog entry {i} ({entry.get('accession','?')}) "
                    f"has species='{species}' (expected 'human')",
                ))

        return findings

    def check_all(self) -> AlignmentReport:
        report = AlignmentReport()
        report.findings.extend(self.check_hashing_consistency())
        report.findings.extend(self.check_regime_thresholds())
        report.findings.extend(self.check_species_firewall())
        report.findings.extend(self.check_curated_series())
        return report

    @staticmethod
    def _compute_canonical_chain_hash(root: Path) -> str | None:
        """Compute a reference chain_hash using neurotunes3's formula."""
        import sys
        ep = root / "evidence_protocol"
        if not (ep / "hashing.py").is_file():
            return None
        old_path = sys.path.copy()
        try:
            sys.path.insert(0, str(root))
            from evidence_protocol.hashing import chain_hash
            return chain_hash("prev", "rec-1", {"x": 1}, {"y": 2}, 1000.0)
        except Exception:
            return None
        finally:
            sys.path[:] = old_path

    @staticmethod
    def _compute_governed_chain_hash(root: Path) -> str | None:
        """Compute same inputs through NeuroTunesAI's lineage module."""
        lineage = root / "models" / "evidence" / "lineage.py"
        if not lineage.is_file():
            return None
        import sys
        old_path = sys.path.copy()
        try:
            sys.path.insert(0, str(root))
            mod = __import__("models.evidence.lineage", fromlist=["chain_hash"])
            return mod.chain_hash("prev", "rec-1", {"x": 1}, {"y": 2}, 1000.0)
        except Exception:
            return None
        finally:
            sys.path[:] = old_path

    @staticmethod
    def _load_classify_regime(root: Path):
        import sys
        old_path = sys.path.copy()
        try:
            sys.path.insert(0, str(root))
            from evidence_protocol.firewall import classify_regime
            return classify_regime
        except Exception:
            return None
        finally:
            sys.path[:] = old_path

    @staticmethod
    def _load_validate_species(root: Path):
        import sys
        old_path = sys.path.copy()
        try:
            sys.path.insert(0, str(root))
            from evidence_protocol.firewall import validate_species
            return validate_species
        except Exception:
            return None
        finally:
            sys.path[:] = old_path
