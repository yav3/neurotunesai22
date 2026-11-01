"""Filesystem and import health checks against the repo manifest."""
from __future__ import annotations

import ast
import importlib.util
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from validation.repo_manifest import REPO_MANIFEST, RepoSpec, ModuleSpec


@dataclass
class Finding:
    """A single health-check finding."""
    repo: str
    severity: str  # "error" | "warning"
    category: str
    message: str


@dataclass
class HealthReport:
    """Aggregated health report across repos."""
    findings: list[Finding] = field(default_factory=list)
    repos_checked: int = 0
    repos_healthy: int = 0

    @property
    def ok(self) -> bool:
        return all(f.severity != "error" for f in self.findings)

    def errors(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == "error"]

    def warnings(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == "warning"]

    def summary(self) -> dict[str, Any]:
        return {
            "repos_checked": self.repos_checked,
            "repos_healthy": self.repos_healthy,
            "errors": len(self.errors()),
            "warnings": len(self.warnings()),
            "ok": self.ok,
        }


class RepoHealthCheck:
    """Checks a set of repos against the manifest."""

    def __init__(self, root_overrides: dict[str, str] | None = None) -> None:
        self.root_overrides = root_overrides or {}

    def _resolve_root(self, spec: RepoSpec) -> Path:
        override = self.root_overrides.get(spec.name)
        return Path(override) if override else Path(spec.root_hint)

    def check_repo(self, spec: RepoSpec) -> list[Finding]:
        findings: list[Finding] = []
        root = self._resolve_root(spec)

        if not root.is_dir():
            findings.append(Finding(
                spec.name, "error", "missing_repo",
                f"Repo root not found: {root}",
            ))
            return findings

        for mod in spec.required_modules:
            findings.extend(self._check_module(spec.name, root, mod))

        for test_path in spec.required_tests:
            full = root / test_path
            if not full.is_file():
                findings.append(Finding(
                    spec.name, "error", "missing_test",
                    f"Required test not found: {test_path}",
                ))

        return findings

    def _check_module(
        self, repo: str, root: Path, mod: ModuleSpec
    ) -> list[Finding]:
        findings: list[Finding] = []
        full = root / mod.path

        if not full.is_file():
            findings.append(Finding(
                repo, "error", "missing_module",
                f"Required module not found: {mod.path} ({mod.description})",
            ))
            return findings

        if mod.must_export:
            exported = self._extract_top_level_names(full)
            for name in mod.must_export:
                if name not in exported:
                    findings.append(Finding(
                        repo, "error", "missing_export",
                        f"{mod.path} must export '{name}' but it was not found",
                    ))

        return findings

    @staticmethod
    def _extract_top_level_names(path: Path) -> set[str]:
        """Parse a Python file's AST and return top-level class/function/assignment names."""
        try:
            source = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return set()

        if path.suffix == ".dart":
            return _extract_dart_names(source)
        if path.suffix == ".ts":
            return _extract_ts_names(source)

        try:
            tree = ast.parse(source, filename=str(path))
        except SyntaxError:
            return set()

        names: set[str] = set()
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                names.add(node.name)
            elif isinstance(node, ast.ClassDef):
                names.add(node.name)
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        names.add(target.id)
        return names

    def check_all(self, repos: list[str] | None = None) -> HealthReport:
        report = HealthReport()
        specs = (
            [REPO_MANIFEST[r] for r in repos if r in REPO_MANIFEST]
            if repos
            else list(REPO_MANIFEST.values())
        )
        for spec in specs:
            findings = self.check_repo(spec)
            report.findings.extend(findings)
            report.repos_checked += 1
            if not any(f.severity == "error" for f in findings):
                report.repos_healthy += 1
        return report


def _extract_dart_names(source: str) -> set[str]:
    """Heuristic extraction of top-level Dart class/enum/function names."""
    names: set[str] = set()
    for line in source.splitlines():
        stripped = line.strip()
        for keyword in ("class ", "enum ", "mixin "):
            if stripped.startswith(keyword):
                rest = stripped[len(keyword):]
                name = rest.split("{")[0].split("<")[0].split(" ")[0].strip()
                if name:
                    names.add(name)
    return names


def _extract_ts_names(source: str) -> set[str]:
    """Heuristic extraction of top-level TypeScript export names."""
    names: set[str] = set()
    for line in source.splitlines():
        stripped = line.strip()
        if stripped.startswith("export "):
            rest = stripped[len("export "):]
            for keyword in ("class ", "function ", "const ", "interface ", "type ", "enum "):
                if rest.startswith(keyword):
                    token = rest[len(keyword):].split("(")[0].split("<")[0].split("=")[0].split(":")[0].strip()
                    if token:
                        names.add(token)
    return names
