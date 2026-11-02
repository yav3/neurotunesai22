"""Schema validation for music metadata — structural validation, cross-field consistency, format compliance."""

from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Set
from enum import Enum
import numpy as np


class Severity(str, Enum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class MetadataStandard(str, Enum):
    DDEX = "ddex"
    ID3V2 = "id3v2"
    VORBIS = "vorbis"
    MUSICBRAINZ = "musicbrainz"
    CUSTOM = "custom"


@dataclass
class ValidationIssue:
    field: str
    message: str
    severity: Severity
    rule: str = ""
    suggestion: Optional[str] = None


@dataclass
class SchemaField:
    name: str
    field_type: str
    required: bool = False
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    pattern: Optional[str] = None
    allowed_values: Optional[List[str]] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None


@dataclass
class MetadataSchema:
    name: str
    standard: MetadataStandard
    fields: List[SchemaField] = field(default_factory=list)
    required_groups: List[List[str]] = field(default_factory=list)


@dataclass
class ValidationResult:
    valid: bool
    issues: List[ValidationIssue] = field(default_factory=list)
    completeness: float = 0.0
    n_errors: int = 0
    n_warnings: int = 0


ISRC_PATTERN = r'^[A-Z]{2}[A-Z0-9]{3}\d{7}$'
ISWC_PATTERN = r'^T-\d{9}-\d$'
UPC_PATTERN = r'^\d{12,13}$'
ISNI_PATTERN = r'^\d{15}[\dX]$'


def validate_isrc(code: str) -> Optional[ValidationIssue]:
    if not re.match(ISRC_PATTERN, code.replace('-', '').upper()):
        return ValidationIssue(
            field="isrc", message=f"Invalid ISRC format: {code}",
            severity=Severity.ERROR, rule="isrc_format",
            suggestion="ISRC must be CC-XXX-YY-NNNNN format",
        )
    return None


def validate_iswc(code: str) -> Optional[ValidationIssue]:
    if not re.match(ISWC_PATTERN, code):
        return ValidationIssue(
            field="iswc", message=f"Invalid ISWC format: {code}",
            severity=Severity.ERROR, rule="iswc_format",
        )
    digits = re.sub(r'[^0-9]', '', code)
    if len(digits) == 10:
        total = sum(int(d) * (i + 1) for i, d in enumerate(digits[:9]))
        check = total % 10
        if check != int(digits[9]):
            return ValidationIssue(
                field="iswc", message="ISWC check digit mismatch",
                severity=Severity.ERROR, rule="iswc_checksum",
            )
    return None


def validate_field(
    value: Any,
    schema_field: SchemaField,
) -> List[ValidationIssue]:
    issues = []
    if value is None or (isinstance(value, str) and not value.strip()):
        if schema_field.required:
            issues.append(ValidationIssue(
                field=schema_field.name, message=f"Required field '{schema_field.name}' is missing",
                severity=Severity.ERROR, rule="required",
            ))
        return issues

    if schema_field.field_type == "string" and isinstance(value, str):
        if schema_field.min_length and len(value) < schema_field.min_length:
            issues.append(ValidationIssue(
                field=schema_field.name,
                message=f"'{schema_field.name}' too short (min {schema_field.min_length})",
                severity=Severity.WARNING, rule="min_length",
            ))
        if schema_field.max_length and len(value) > schema_field.max_length:
            issues.append(ValidationIssue(
                field=schema_field.name,
                message=f"'{schema_field.name}' too long (max {schema_field.max_length})",
                severity=Severity.ERROR, rule="max_length",
            ))
        if schema_field.pattern and not re.match(schema_field.pattern, value):
            issues.append(ValidationIssue(
                field=schema_field.name,
                message=f"'{schema_field.name}' doesn't match pattern",
                severity=Severity.ERROR, rule="pattern",
            ))
        if schema_field.allowed_values and value not in schema_field.allowed_values:
            issues.append(ValidationIssue(
                field=schema_field.name,
                message=f"'{value}' not in allowed values",
                severity=Severity.ERROR, rule="allowed_values",
                suggestion=f"Allowed: {', '.join(schema_field.allowed_values[:5])}",
            ))
    elif schema_field.field_type == "number":
        try:
            num = float(value)
            if schema_field.min_value is not None and num < schema_field.min_value:
                issues.append(ValidationIssue(
                    field=schema_field.name,
                    message=f"'{schema_field.name}' below minimum {schema_field.min_value}",
                    severity=Severity.ERROR, rule="min_value",
                ))
            if schema_field.max_value is not None and num > schema_field.max_value:
                issues.append(ValidationIssue(
                    field=schema_field.name,
                    message=f"'{schema_field.name}' above maximum {schema_field.max_value}",
                    severity=Severity.ERROR, rule="max_value",
                ))
        except (ValueError, TypeError):
            issues.append(ValidationIssue(
                field=schema_field.name,
                message=f"'{schema_field.name}' is not a valid number",
                severity=Severity.ERROR, rule="type",
            ))
    return issues


def validate_metadata(
    metadata: Dict[str, Any],
    schema: MetadataSchema,
) -> ValidationResult:
    issues = []
    fields_present = 0
    fields_total = len(schema.fields)

    for sf in schema.fields:
        value = metadata.get(sf.name)
        field_issues = validate_field(value, sf)
        issues.extend(field_issues)
        if value is not None and (not isinstance(value, str) or value.strip()):
            fields_present += 1

    for group in schema.required_groups:
        present = [f for f in group if f in metadata and metadata[f]]
        if not present:
            issues.append(ValidationIssue(
                field=", ".join(group),
                message=f"At least one of {group} must be present",
                severity=Severity.ERROR, rule="required_group",
            ))

    n_errors = sum(1 for i in issues if i.severity == Severity.ERROR)
    n_warnings = sum(1 for i in issues if i.severity == Severity.WARNING)

    return ValidationResult(
        valid=n_errors == 0,
        issues=issues,
        completeness=fields_present / max(fields_total, 1),
        n_errors=n_errors,
        n_warnings=n_warnings,
    )


def cross_field_consistency(
    metadata: Dict[str, Any],
) -> List[ValidationIssue]:
    issues = []
    duration = metadata.get("duration_ms")
    bpm = metadata.get("bpm")
    if duration is not None and bpm is not None:
        try:
            d_sec = float(duration) / 1000.0
            b = float(bpm)
            if b > 0 and d_sec > 0:
                beats = d_sec * b / 60.0
                if beats < 4:
                    issues.append(ValidationIssue(
                        field="bpm+duration",
                        message=f"Only {beats:.1f} beats in track — BPM may be wrong",
                        severity=Severity.WARNING, rule="bpm_duration_cross",
                    ))
        except (ValueError, TypeError):
            pass

    year = metadata.get("year")
    if year is not None:
        try:
            y = int(year)
            if y < 1900 or y > 2030:
                issues.append(ValidationIssue(
                    field="year", message=f"Suspicious year: {y}",
                    severity=Severity.WARNING, rule="year_range",
                ))
        except (ValueError, TypeError):
            pass

    return issues


def completeness_score(
    metadata: Dict[str, Any],
    weights: Optional[Dict[str, float]] = None,
) -> float:
    if not metadata:
        return 0.0
    if weights is None:
        weights = {k: 1.0 for k in metadata}
    total_weight = sum(weights.get(k, 1.0) for k in weights)
    present_weight = sum(
        weights.get(k, 1.0) for k in weights
        if k in metadata and metadata[k] is not None and (not isinstance(metadata[k], str) or metadata[k].strip())
    )
    return present_weight / max(total_weight, 1e-9)


def detect_duplicates(
    records: List[Dict[str, Any]],
    match_fields: List[str],
    threshold: float = 0.8,
) -> List[Tuple[int, int, float]]:
    duplicates = []
    for i in range(len(records)):
        for j in range(i + 1, len(records)):
            matches = 0
            total = 0
            for f in match_fields:
                v1 = records[i].get(f)
                v2 = records[j].get(f)
                if v1 is not None and v2 is not None:
                    total += 1
                    if isinstance(v1, str) and isinstance(v2, str):
                        if v1.lower().strip() == v2.lower().strip():
                            matches += 1
                    elif v1 == v2:
                        matches += 1
            sim = matches / max(total, 1)
            if sim >= threshold:
                duplicates.append((i, j, sim))
    return duplicates
