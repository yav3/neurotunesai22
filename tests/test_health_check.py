"""Tests for the cross-repo validation harness.

These tests run against the real repos at /home/user/* and verify
that the validation harness correctly detects both healthy and
unhealthy states.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from validation.repo_manifest import REPO_MANIFEST, RepoSpec, ModuleSpec
from validation.health_check import RepoHealthCheck, HealthReport, Finding
from validation.schema_alignment import SchemaAlignmentCheck
from validation.evidence_chain import EvidenceChainCheck


ROOTS = {
    "NeuroTunesAI": Path("/home/user/NeuroTunesAI"),
    "merlinaineuralplus2": Path("/home/user/merlinaineuralplus2"),
    "neurotunes3": Path("/home/user/neurotunes3"),
    "neurotunes": Path("/home/user/neurotunes"),
    "neurotunescopy4privatesharing": Path("/home/user/neurotunescopy4privatesharing"),
}


class TestManifest:

    def test_all_repos_in_manifest(self):
        assert "NeuroTunesAI" in REPO_MANIFEST
        assert "merlinaineuralplus2" in REPO_MANIFEST
        assert "neurotunes3" in REPO_MANIFEST
        assert "neurotunes" in REPO_MANIFEST
        assert "neurotunescopy4privatesharing" in REPO_MANIFEST

    def test_roles_are_distinct(self):
        roles = [s.role for s in REPO_MANIFEST.values()]
        assert len(roles) == len(set(roles))

    def test_module_specs_have_descriptions(self):
        for spec in REPO_MANIFEST.values():
            for mod in spec.required_modules:
                assert mod.description, f"{spec.name}/{mod.path} has no description"


class TestHealthCheck:

    def test_missing_repo_reports_error(self):
        spec = RepoSpec(
            name="nonexistent",
            role="test",
            root_hint="/tmp/does-not-exist-xyz",
        )
        checker = RepoHealthCheck()
        findings = checker.check_repo(spec)
        assert any(f.category == "missing_repo" for f in findings)

    def test_neurotunes3_is_healthy(self):
        if not ROOTS["neurotunes3"].is_dir():
            pytest.skip("neurotunes3 not available")
        checker = RepoHealthCheck()
        findings = checker.check_repo(REPO_MANIFEST["neurotunes3"])
        errors = [f for f in findings if f.severity == "error"]
        assert errors == [], f"Errors: {[f.message for f in errors]}"

    def test_neurotunes3_exports_chain_hash(self):
        if not ROOTS["neurotunes3"].is_dir():
            pytest.skip("neurotunes3 not available")
        checker = RepoHealthCheck()
        findings = checker.check_repo(REPO_MANIFEST["neurotunes3"])
        export_errors = [f for f in findings if f.category == "missing_export"]
        assert export_errors == [], f"Missing exports: {[f.message for f in export_errors]}"

    def test_neurotunesai_is_healthy(self):
        if not ROOTS["NeuroTunesAI"].is_dir():
            pytest.skip("NeuroTunesAI not available")
        checker = RepoHealthCheck()
        findings = checker.check_repo(REPO_MANIFEST["NeuroTunesAI"])
        errors = [f for f in findings if f.severity == "error"]
        assert errors == [], f"Errors: {[f.message for f in errors]}"

    def test_merlin_is_healthy(self):
        if not ROOTS["merlinaineuralplus2"].is_dir():
            pytest.skip("merlinaineuralplus2 not available")
        checker = RepoHealthCheck()
        findings = checker.check_repo(REPO_MANIFEST["merlinaineuralplus2"])
        errors = [f for f in findings if f.severity == "error"]
        assert errors == [], f"Errors: {[f.message for f in errors]}"

    def test_flutter_is_healthy(self):
        if not ROOTS["neurotunes"].is_dir():
            pytest.skip("neurotunes not available")
        checker = RepoHealthCheck()
        findings = checker.check_repo(REPO_MANIFEST["neurotunes"])
        errors = [f for f in findings if f.severity == "error"]
        assert errors == [], f"Errors: {[f.message for f in errors]}"

    def test_web_product_is_healthy(self):
        if not ROOTS["neurotunescopy4privatesharing"].is_dir():
            pytest.skip("neurotunescopy4privatesharing not available")
        checker = RepoHealthCheck()
        findings = checker.check_repo(REPO_MANIFEST["neurotunescopy4privatesharing"])
        errors = [f for f in findings if f.severity == "error"]
        assert errors == [], f"Errors: {[f.message for f in errors]}"

    def test_check_all_returns_report(self):
        checker = RepoHealthCheck()
        report = checker.check_all()
        assert isinstance(report, HealthReport)
        assert report.repos_checked == len(REPO_MANIFEST)

    def test_missing_export_detected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mod_dir = Path(tmpdir) / "evidence_protocol"
            mod_dir.mkdir()
            (mod_dir / "__init__.py").write_text("")
            (mod_dir / "hashing.py").write_text("def content_hash(): pass\n")
            spec = RepoSpec(
                name="test_repo",
                role="test",
                root_hint=tmpdir,
                required_modules=[
                    ModuleSpec(
                        "evidence_protocol/hashing.py",
                        "test",
                        ["content_hash", "chain_hash", "missing_func"],
                    ),
                ],
            )
            checker = RepoHealthCheck()
            findings = checker.check_repo(spec)
            missing = [f for f in findings if f.category == "missing_export"]
            assert len(missing) == 2
            names = {f.message.split("'")[1] for f in missing}
            assert "chain_hash" in names
            assert "missing_func" in names


class TestSchemaAlignment:

    def test_regime_thresholds(self):
        if not ROOTS["neurotunes3"].is_dir():
            pytest.skip("neurotunes3 not available")
        checker = SchemaAlignmentCheck(ROOTS)
        findings = checker.check_regime_thresholds()
        errors = [f for f in findings if f.severity == "error"]
        assert errors == [], f"Errors: {[f.message for f in errors]}"

    def test_species_firewall(self):
        if not ROOTS["neurotunes3"].is_dir():
            pytest.skip("neurotunes3 not available")
        checker = SchemaAlignmentCheck(ROOTS)
        findings = checker.check_species_firewall()
        errors = [f for f in findings if f.severity == "error"]
        assert errors == [], f"Errors: {[f.message for f in errors]}"

    def test_curated_series_catalog(self):
        if not ROOTS["NeuroTunesAI"].is_dir():
            pytest.skip("NeuroTunesAI not available")
        checker = SchemaAlignmentCheck(ROOTS)
        findings = checker.check_curated_series()
        errors = [f for f in findings if f.severity == "error"]
        assert errors == [], f"Errors: {[f.message for f in errors]}"

    def test_full_alignment(self):
        checker = SchemaAlignmentCheck(ROOTS)
        report = checker.check_all()
        assert hasattr(report, "ok")


class TestEvidenceChain:

    def test_valid_chain(self):
        schema_root = ROOTS.get("neurotunes3")
        if not schema_root or not schema_root.is_dir():
            pytest.skip("neurotunes3 not available")

        sys.path.insert(0, str(schema_root))
        try:
            from evidence_protocol.hashing import chain_hash
        except ImportError:
            pytest.skip("Cannot import evidence_protocol")
        finally:
            if str(schema_root) in sys.path:
                sys.path.remove(str(schema_root))

        records = []
        prev = "genesis"
        for i in range(5):
            h = chain_hash(prev, f"r{i}", {"step": i}, {"val": i * 10}, float(i))
            records.append({
                "record_id": f"r{i}",
                "prev_hash": prev,
                "problem_spec": {"step": i},
                "answer_values": {"val": i * 10},
                "timestamp": float(i),
                "hash": h,
            })
            prev = h

        checker = EvidenceChainCheck(schema_root=schema_root)
        report = checker.verify_chain(records, source="test")
        assert report.ok
        assert report.records_checked == 5

    def test_broken_link_detected(self):
        records = [
            {"record_id": "r0", "prev_hash": "genesis", "hash": "aaa", "timestamp": 0.0},
            {"record_id": "r1", "prev_hash": "WRONG", "hash": "bbb", "timestamp": 1.0},
        ]
        checker = EvidenceChainCheck()
        report = checker.verify_chain(records, source="test")
        assert any(f.category == "broken_link" for f in report.findings)

    def test_tampered_hash_detected(self):
        schema_root = ROOTS.get("neurotunes3")
        if not schema_root or not schema_root.is_dir():
            pytest.skip("neurotunes3 not available")

        sys.path.insert(0, str(schema_root))
        try:
            from evidence_protocol.hashing import chain_hash
        except ImportError:
            pytest.skip("Cannot import evidence_protocol")
        finally:
            if str(schema_root) in sys.path:
                sys.path.remove(str(schema_root))

        h = chain_hash("genesis", "r0", {"a": 1}, {"b": 2}, 0.0)
        records = [{
            "record_id": "r0",
            "prev_hash": "genesis",
            "problem_spec": {"a": 1},
            "answer_values": {"b": 2},
            "timestamp": 0.0,
            "hash": "TAMPERED" + "0" * 56,
        }]
        checker = EvidenceChainCheck(schema_root=schema_root)
        report = checker.verify_chain(records, source="test")
        assert any(f.category == "hash_mismatch" for f in report.findings)

    def test_timestamp_regression_warned(self):
        records = [
            {"record_id": "r0", "prev_hash": "g", "hash": "a", "timestamp": 10.0},
            {"record_id": "r1", "prev_hash": "a", "hash": "b", "timestamp": 5.0},
        ]
        checker = EvidenceChainCheck()
        report = checker.verify_chain(records, source="test")
        assert any(f.category == "timestamp_regression" for f in report.findings)

    def test_empty_chain_warned(self):
        checker = EvidenceChainCheck()
        report = checker.verify_chain([], source="test")
        assert any(f.category == "empty_chain" for f in report.findings)

    def test_verify_from_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump([
                {"record_id": "r0", "prev_hash": "g", "hash": "a", "timestamp": 0.0},
                {"record_id": "r1", "prev_hash": "a", "hash": "b", "timestamp": 1.0},
            ], f)
            f.flush()
            checker = EvidenceChainCheck()
            report = checker.verify_from_file(Path(f.name), source="test_file")
            assert report.records_checked == 2
        os.unlink(f.name)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
