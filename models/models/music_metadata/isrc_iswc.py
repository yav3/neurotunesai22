"""Standard music identifiers: ISRC, ISWC, EAN validation."""
from __future__ import annotations

import re


def validate_isrc(code: str) -> bool:
    """ISRC format: CC-XXX-YY-NNNNN (12 alphanum without hyphens)."""
    clean = code.replace("-", "").strip().upper()
    return bool(re.match(r"^[A-Z]{2}[A-Z0-9]{3}\d{7}$", clean))


def parse_isrc(code: str) -> dict[str, str]:
    clean = code.replace("-", "").strip().upper()
    if not validate_isrc(code):
        raise ValueError(f"invalid ISRC: {code}")
    return {
        "country": clean[:2],
        "registrant": clean[2:5],
        "year": clean[5:7],
        "designation": clean[7:12],
    }


def validate_iswc(code: str) -> bool:
    """ISWC format: T-NNNNNNNNN-C."""
    clean = code.replace("-", "").replace(".", "").strip().upper()
    if not re.match(r"^T\d{10}$", clean):
        return False
    return _iswc_check_digit(clean[:10]) == clean[10]


def _iswc_check_digit(partial: str) -> str:
    weights = [1] + list(range(3, 12))
    total = sum(int(c) * w for c, w in zip(partial[1:], weights))
    remainder = total % 10
    return str((10 - remainder) % 10)


def parse_iswc(code: str) -> dict[str, str]:
    clean = code.replace("-", "").replace(".", "").strip().upper()
    if not validate_iswc(code):
        raise ValueError(f"invalid ISWC: {code}")
    return {
        "prefix": clean[0],
        "work_id": clean[1:10],
        "check_digit": clean[10],
    }


def validate_ean(barcode: str) -> bool:
    """EAN-13 check digit validation."""
    clean = barcode.strip()
    if not re.match(r"^\d{13}$", clean):
        return False
    return generate_check_digit(clean[:12]) == clean[12]


def generate_check_digit(partial: str) -> str:
    """Generate EAN-13 check digit from first 12 digits."""
    if len(partial) != 12 or not partial.isdigit():
        raise ValueError("need exactly 12 digits")
    total = sum(int(d) * (1 if i % 2 == 0 else 3) for i, d in enumerate(partial))
    return str((10 - total % 10) % 10)
