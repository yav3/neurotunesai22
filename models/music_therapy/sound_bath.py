"""Sound bath design: singing bowl frequencies, binaural beats, drone layering, session structure."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


class BowlType(str, Enum):
    CRYSTAL = "crystal"
    TIBETAN = "tibetan"
    GONG = "gong"
    TUNING_FORK = "tuning_fork"


class BrainwaveState(str, Enum):
    DELTA = "delta"
    THETA = "theta"
    ALPHA = "alpha"
    BETA = "beta"
    GAMMA = "gamma"


BRAINWAVE_RANGES: Dict[BrainwaveState, Tuple[float, float]] = {
    BrainwaveState.DELTA: (0.5, 4.0),
    BrainwaveState.THETA: (4.0, 8.0),
    BrainwaveState.ALPHA: (8.0, 13.0),
    BrainwaveState.BETA: (13.0, 30.0),
    BrainwaveState.GAMMA: (30.0, 50.0),
}

CHAKRA_FREQUENCIES: Dict[str, float] = {
    "root": 256.0,
    "sacral": 288.0,
    "solar_plexus": 320.0,
    "heart": 341.3,
    "throat": 384.0,
    "third_eye": 426.7,
    "crown": 480.0,
}


@dataclass
class SingingBowl:
    fundamental_hz: float
    bowl_type: BowlType = BowlType.CRYSTAL
    partials_ratios: List[float] = field(default_factory=lambda: [1.0, 2.0, 3.0, 4.71, 6.28])
    decay_seconds: float = 15.0

    def partial_frequencies(self) -> List[float]:
        return [self.fundamental_hz * r for r in self.partials_ratios]


@dataclass
class BinauralBeatConfig:
    base_frequency_hz: float
    beat_frequency_hz: float
    target_state: BrainwaveState

    @property
    def left_hz(self) -> float:
        return self.base_frequency_hz - self.beat_frequency_hz / 2.0

    @property
    def right_hz(self) -> float:
        return self.base_frequency_hz + self.beat_frequency_hz / 2.0


@dataclass
class DroneLayer:
    frequency_hz: float
    amplitude: float = 0.5
    modulation_rate_hz: float = 0.0
    modulation_depth: float = 0.0


@dataclass
class SoundBathPhase:
    name: str
    duration_minutes: float
    bowls: List[SingingBowl] = field(default_factory=list)
    drones: List[DroneLayer] = field(default_factory=list)
    binaural: Optional[BinauralBeatConfig] = None
    target_state: BrainwaveState = BrainwaveState.ALPHA


@dataclass
class SoundBathSession:
    phases: List[SoundBathPhase]

    @property
    def total_duration_minutes(self) -> float:
        return sum(p.duration_minutes for p in self.phases)


def binaural_beat_for_state(target: BrainwaveState, base_hz: float = 200.0) -> BinauralBeatConfig:
    lo, hi = BRAINWAVE_RANGES[target]
    beat = (lo + hi) / 2.0
    return BinauralBeatConfig(base_frequency_hz=base_hz, beat_frequency_hz=beat, target_state=target)


def isochronic_tone_frequency(target: BrainwaveState) -> float:
    lo, hi = BRAINWAVE_RANGES[target]
    return (lo + hi) / 2.0


def bowl_harmony_score(bowls: List[SingingBowl]) -> float:
    if len(bowls) < 2:
        return 1.0
    consonant_ratios = {1.0, 2.0, 1.5, 4 / 3, 5 / 4, 5 / 3, 3 / 2}
    scores: List[float] = []
    for i in range(len(bowls)):
        for j in range(i + 1, len(bowls)):
            f1, f2 = bowls[i].fundamental_hz, bowls[j].fundamental_hz
            ratio = max(f1, f2) / min(f1, f2) if min(f1, f2) > 0 else 1.0
            best = min(abs(ratio - r) for r in consonant_ratios)
            scores.append(math.exp(-best * 5.0))
    return sum(scores) / len(scores) if scores else 1.0


def drone_layer_from_fundamental(fundamental_hz: float, n_layers: int = 3) -> List[DroneLayer]:
    layers: List[DroneLayer] = []
    for i in range(n_layers):
        ratio = [1.0, 1.5, 2.0][i % 3]
        layers.append(DroneLayer(
            frequency_hz=fundamental_hz * ratio,
            amplitude=0.6 / (i + 1),
            modulation_rate_hz=0.1 * (i + 1),
            modulation_depth=0.05,
        ))
    return layers


def design_relaxation_session(duration_minutes: float = 30.0) -> SoundBathSession:
    intro = SoundBathPhase(
        name="opening", duration_minutes=duration_minutes * 0.15,
        bowls=[SingingBowl(fundamental_hz=CHAKRA_FREQUENCIES["heart"])],
        target_state=BrainwaveState.ALPHA,
        binaural=binaural_beat_for_state(BrainwaveState.ALPHA),
    )
    deep = SoundBathPhase(
        name="deepening", duration_minutes=duration_minutes * 0.55,
        bowls=[SingingBowl(fundamental_hz=CHAKRA_FREQUENCIES["root"]),
               SingingBowl(fundamental_hz=CHAKRA_FREQUENCIES["crown"])],
        drones=drone_layer_from_fundamental(CHAKRA_FREQUENCIES["heart"]),
        target_state=BrainwaveState.THETA,
        binaural=binaural_beat_for_state(BrainwaveState.THETA),
    )
    closing = SoundBathPhase(
        name="return", duration_minutes=duration_minutes * 0.30,
        bowls=[SingingBowl(fundamental_hz=CHAKRA_FREQUENCIES["throat"])],
        target_state=BrainwaveState.ALPHA,
        binaural=binaural_beat_for_state(BrainwaveState.ALPHA),
    )
    return SoundBathSession(phases=[intro, deep, closing])


def session_frequency_map(session: SoundBathSession) -> Dict[str, List[float]]:
    freq_map: Dict[str, List[float]] = {}
    for phase in session.phases:
        freqs: List[float] = []
        for bowl in phase.bowls:
            freqs.extend(bowl.partial_frequencies())
        for drone in phase.drones:
            freqs.append(drone.frequency_hz)
        if phase.binaural:
            freqs.extend([phase.binaural.left_hz, phase.binaural.right_hz])
        freq_map[phase.name] = sorted(set(freqs))
    return freq_map
