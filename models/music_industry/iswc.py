"""ISWC (International Standard Musical Work Code) handling."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class ISWC:
    prefix: str = "T"
    work_id: int = 0
    check_digit: int = 0


_ISWC_RE = re.compile(r"^T-?(\d{9})-?(\d)$", re.IGNORECASE)


def compute_check_digit(work_id: int) -> int:
    digits = f"{work_id:09d}"
    total = 1  # T = position 0, weight 1
    for i, ch in enumerate(digits):
        total += int(ch) * (i + 2)
    return (10 - (total % 10)) % 10


def parse_iswc(code_str: str) -> ISWC:
    m = _ISWC_RE.match(code_str.strip())
    if not m:
        raise ValueError(f"Invalid ISWC: {code_str}")
    wid = int(m.group(1))
    cd = int(m.group(2))
    return ISWC(prefix="T", work_id=wid, check_digit=cd)


def validate_iswc(code_str: str) -> bool:
    try:
        iswc = parse_iswc(code_str)
        return compute_check_digit(iswc.work_id) == iswc.check_digit
    except ValueError:
        return False


def format_iswc(iswc: ISWC) -> str:
    return f"T-{iswc.work_id:09d}-{iswc.check_digit}"
