from __future__ import annotations

import math
from enum import Enum
from dataclasses import dataclass


class EmotionIntensity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class PlutchikEmotion(Enum):
    # 8 primary emotions (medium intensity)
    JOY = "joy"
    TRUST = "trust"
    FEAR = "fear"
    SURPRISE = "surprise"
    SADNESS = "sadness"
    DISGUST = "disgust"
    ANGER = "anger"
    ANTICIPATION = "anticipation"

    # High-intensity variants
    ECSTASY = "ecstasy"
    ADMIRATION = "admiration"
    TERROR = "terror"
    AMAZEMENT = "amazement"
    GRIEF = "grief"
    LOATHING = "loathing"
    RAGE = "rage"
    VIGILANCE = "vigilance"

    # Low-intensity variants
    SERENITY = "serenity"
    ACCEPTANCE = "acceptance"
    APPREHENSION = "apprehension"
    DISTRACTION = "distraction"
    PENSIVENESS = "pensiveness"
    BOREDOM = "boredom"
    ANNOYANCE = "annoyance"
    INTEREST = "interest"


_PRIMARY = [
    PlutchikEmotion.JOY,
    PlutchikEmotion.TRUST,
    PlutchikEmotion.FEAR,
    PlutchikEmotion.SURPRISE,
    PlutchikEmotion.SADNESS,
    PlutchikEmotion.DISGUST,
    PlutchikEmotion.ANGER,
    PlutchikEmotion.ANTICIPATION,
]

_INTENSITY_MAP: dict[PlutchikEmotion, tuple[PlutchikEmotion, EmotionIntensity]] = {
    PlutchikEmotion.JOY: (PlutchikEmotion.JOY, EmotionIntensity.MEDIUM),
    PlutchikEmotion.TRUST: (PlutchikEmotion.TRUST, EmotionIntensity.MEDIUM),
    PlutchikEmotion.FEAR: (PlutchikEmotion.FEAR, EmotionIntensity.MEDIUM),
    PlutchikEmotion.SURPRISE: (PlutchikEmotion.SURPRISE, EmotionIntensity.MEDIUM),
    PlutchikEmotion.SADNESS: (PlutchikEmotion.SADNESS, EmotionIntensity.MEDIUM),
    PlutchikEmotion.DISGUST: (PlutchikEmotion.DISGUST, EmotionIntensity.MEDIUM),
    PlutchikEmotion.ANGER: (PlutchikEmotion.ANGER, EmotionIntensity.MEDIUM),
    PlutchikEmotion.ANTICIPATION: (PlutchikEmotion.ANTICIPATION, EmotionIntensity.MEDIUM),
    PlutchikEmotion.ECSTASY: (PlutchikEmotion.JOY, EmotionIntensity.HIGH),
    PlutchikEmotion.ADMIRATION: (PlutchikEmotion.TRUST, EmotionIntensity.HIGH),
    PlutchikEmotion.TERROR: (PlutchikEmotion.FEAR, EmotionIntensity.HIGH),
    PlutchikEmotion.AMAZEMENT: (PlutchikEmotion.SURPRISE, EmotionIntensity.HIGH),
    PlutchikEmotion.GRIEF: (PlutchikEmotion.SADNESS, EmotionIntensity.HIGH),
    PlutchikEmotion.LOATHING: (PlutchikEmotion.DISGUST, EmotionIntensity.HIGH),
    PlutchikEmotion.RAGE: (PlutchikEmotion.ANGER, EmotionIntensity.HIGH),
    PlutchikEmotion.VIGILANCE: (PlutchikEmotion.ANTICIPATION, EmotionIntensity.HIGH),
    PlutchikEmotion.SERENITY: (PlutchikEmotion.JOY, EmotionIntensity.LOW),
    PlutchikEmotion.ACCEPTANCE: (PlutchikEmotion.TRUST, EmotionIntensity.LOW),
    PlutchikEmotion.APPREHENSION: (PlutchikEmotion.FEAR, EmotionIntensity.LOW),
    PlutchikEmotion.DISTRACTION: (PlutchikEmotion.SURPRISE, EmotionIntensity.LOW),
    PlutchikEmotion.PENSIVENESS: (PlutchikEmotion.SADNESS, EmotionIntensity.LOW),
    PlutchikEmotion.BOREDOM: (PlutchikEmotion.DISGUST, EmotionIntensity.LOW),
    PlutchikEmotion.ANNOYANCE: (PlutchikEmotion.ANGER, EmotionIntensity.LOW),
    PlutchikEmotion.INTEREST: (PlutchikEmotion.ANTICIPATION, EmotionIntensity.LOW),
}


def _primary_index(e: PlutchikEmotion) -> int:
    base, _ = _INTENSITY_MAP[e]
    return _PRIMARY.index(base)


def primary_base(e: PlutchikEmotion) -> PlutchikEmotion:
    return _INTENSITY_MAP[e][0]


def emotion_intensity(e: PlutchikEmotion) -> EmotionIntensity:
    return _INTENSITY_MAP[e][1]


@dataclass
class PlutchikDyad:
    name: str
    component_a: PlutchikEmotion
    component_b: PlutchikEmotion


class PlutchikWheel:
    OPPOSITES: dict[PlutchikEmotion, PlutchikEmotion] = {
        PlutchikEmotion.JOY: PlutchikEmotion.SADNESS,
        PlutchikEmotion.SADNESS: PlutchikEmotion.JOY,
        PlutchikEmotion.TRUST: PlutchikEmotion.DISGUST,
        PlutchikEmotion.DISGUST: PlutchikEmotion.TRUST,
        PlutchikEmotion.FEAR: PlutchikEmotion.ANGER,
        PlutchikEmotion.ANGER: PlutchikEmotion.FEAR,
        PlutchikEmotion.SURPRISE: PlutchikEmotion.ANTICIPATION,
        PlutchikEmotion.ANTICIPATION: PlutchikEmotion.SURPRISE,
    }

    DYADS: dict[str, PlutchikDyad] = {
        "love": PlutchikDyad("love", PlutchikEmotion.JOY, PlutchikEmotion.TRUST),
        "submission": PlutchikDyad("submission", PlutchikEmotion.TRUST, PlutchikEmotion.FEAR),
        "awe": PlutchikDyad("awe", PlutchikEmotion.FEAR, PlutchikEmotion.SURPRISE),
        "disapproval": PlutchikDyad("disapproval", PlutchikEmotion.SURPRISE, PlutchikEmotion.SADNESS),
        "remorse": PlutchikDyad("remorse", PlutchikEmotion.SADNESS, PlutchikEmotion.DISGUST),
        "contempt": PlutchikDyad("contempt", PlutchikEmotion.DISGUST, PlutchikEmotion.ANGER),
        "aggressiveness": PlutchikDyad("aggressiveness", PlutchikEmotion.ANGER, PlutchikEmotion.ANTICIPATION),
        "optimism": PlutchikDyad("optimism", PlutchikEmotion.ANTICIPATION, PlutchikEmotion.JOY),
    }

    @staticmethod
    def emotion_distance(e1: PlutchikEmotion, e2: PlutchikEmotion) -> float:
        """Angular distance on the wheel, normalised to [0, 1].

        0 = same sector, 0.5 = opposite, intermediate values for
        adjacent/non-adjacent pairs.  Intensity variants map to their
        primary base before computing distance.
        """
        i1 = _primary_index(e1)
        i2 = _primary_index(e2)
        raw = abs(i1 - i2)
        steps = min(raw, 8 - raw)
        return steps / 4.0

    @staticmethod
    def blend_emotions(
        emotions: list[tuple[PlutchikEmotion, float]],
    ) -> dict[str, float]:
        """Weighted blend of emotions into a composite profile.

        Returns a dict mapping each primary emotion *name* to its
        blended score in [0, 1].  Input weights need not sum to 1;
        they are normalised internally.
        """
        totals: dict[str, float] = {e.value: 0.0 for e in _PRIMARY}
        weight_sum = sum(w for _, w in emotions)
        if weight_sum <= 0:
            return totals
        for emotion, weight in emotions:
            base = primary_base(emotion)
            totals[base.value] += weight / weight_sum
        mx = max(totals.values()) or 1.0
        return {k: v / mx for k, v in totals.items()}
