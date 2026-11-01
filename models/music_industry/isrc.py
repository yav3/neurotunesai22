"""ISRC (International Standard Recording Code) handling."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class ISRC:
    country_code: str
    registrant: str
    year: int
    designation: int


_ISRC_RE = re.compile(
    r"^([A-Z]{2})-?([A-Z0-9]{3})-?(\d{2})-?(\d{5})$", re.IGNORECASE
)


def parse_isrc(code_str: str) -> ISRC:
    m = _ISRC_RE.match(code_str.strip())
    if not m:
        raise ValueError(f"Invalid ISRC: {code_str}")
    return ISRC(
        country_code=m.group(1).upper(),
        registrant=m.group(2).upper(),
        year=int(m.group(3)),
        designation=int(m.group(4)),
    )


def format_isrc(isrc: ISRC) -> str:
    return f"{isrc.country_code}-{isrc.registrant}-{isrc.year:02d}-{isrc.designation:05d}"


def validate_isrc(code_str: str) -> bool:
    try:
        parse_isrc(code_str)
        return True
    except ValueError:
        return False


def generate_isrc(
    country: str, registrant: str, year: int, designation: int
) -> str:
    if len(country) != 2 or not country.isalpha():
        raise ValueError("Country code must be 2 letters")
    if len(registrant) != 3 or not registrant.isalnum():
        raise ValueError("Registrant must be 3 alphanumeric chars")
    if not (0 <= year <= 99):
        raise ValueError("Year must be 0-99")
    if not (0 <= designation <= 99999):
        raise ValueError("Designation must be 0-99999")
    return format_isrc(ISRC(country.upper(), registrant.upper(), year, designation))
