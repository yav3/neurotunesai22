"""Fair use analysis for music — four-factor test scoring."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Purpose(Enum):
    COMMERCIAL = "commercial"
    EDUCATIONAL = "educational"
    PARODY = "parody"
    CRITICISM = "criticism"
    NEWS = "news"
    TRANSFORMATIVE = "transformative"
    PERSONAL = "personal"


PURPOSE_SCORES: dict[Purpose, float] = {
    Purpose.PARODY: 0.85,
    Purpose.CRITICISM: 0.80,
    Purpose.EDUCATIONAL: 0.75,
    Purpose.NEWS: 0.70,
    Purpose.TRANSFORMATIVE: 0.65,
    Purpose.PERSONAL: 0.50,
    Purpose.COMMERCIAL: 0.20,
}


@dataclass
class FairUseFactors:
    """The four statutory fair-use factors (17 USC §107)."""
    purpose: Purpose
    nature_creative: bool = True  # creative vs. factual
    nature_published: bool = True  # published vs. unpublished
    amount_fraction: float = 0.0  # fraction of original used, 0-1
    is_heart_of_work: bool = False  # qualitative substantiality
    market_harm: float = 0.0  # 0-1, estimated market impact


def factor1_score(purpose: Purpose) -> float:
    return PURPOSE_SCORES.get(purpose, 0.5)


def factor2_score(creative: bool, published: bool) -> float:
    base = 0.4 if creative else 0.7
    if not published:
        base -= 0.15
    return max(0.0, min(1.0, base))


def factor3_score(amount_fraction: float, is_heart: bool) -> float:
    score = 1.0 - amount_fraction
    if is_heart:
        score *= 0.5
    return max(0.0, min(1.0, score))


def factor4_score(market_harm: float) -> float:
    return max(0.0, 1.0 - market_harm)


FACTOR_WEIGHTS = (0.30, 0.15, 0.25, 0.30)


def composite_fair_use_score(factors: FairUseFactors) -> float:
    """Weighted composite of four factors. Higher = stronger fair-use argument."""
    f1 = factor1_score(factors.purpose)
    f2 = factor2_score(factors.nature_creative, factors.nature_published)
    f3 = factor3_score(factors.amount_fraction, factors.is_heart_of_work)
    f4 = factor4_score(factors.market_harm)
    return (FACTOR_WEIGHTS[0] * f1 + FACTOR_WEIGHTS[1] * f2 +
            FACTOR_WEIGHTS[2] * f3 + FACTOR_WEIGHTS[3] * f4)


def fair_use_risk_level(score: float) -> str:
    if score >= 0.70:
        return "low_risk"
    elif score >= 0.45:
        return "moderate_risk"
    else:
        return "high_risk"


@dataclass
class FairUseReport:
    factors: FairUseFactors
    factor_scores: tuple[float, float, float, float]
    composite: float
    risk_level: str


def analyze_fair_use(factors: FairUseFactors) -> FairUseReport:
    f1 = factor1_score(factors.purpose)
    f2 = factor2_score(factors.nature_creative, factors.nature_published)
    f3 = factor3_score(factors.amount_fraction, factors.is_heart_of_work)
    f4 = factor4_score(factors.market_harm)
    composite = composite_fair_use_score(factors)
    return FairUseReport(
        factors=factors,
        factor_scores=(f1, f2, f3, f4),
        composite=composite,
        risk_level=fair_use_risk_level(composite),
    )
