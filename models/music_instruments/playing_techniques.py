"""Playing techniques and their spectral characteristics."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class TechniqueFamily(Enum):
    BOWING = "bowing"
    PLUCKING = "plucking"
    WIND = "wind"
    PERCUSSION = "percussion"
    KEYBOARD = "keyboard"
    VOCAL = "vocal"


class BowingTechnique(Enum):
    ARCO = "arco"
    SUL_PONTICELLO = "sul_ponticello"
    SUL_TASTO = "sul_tasto"
    COL_LEGNO = "col_legno"
    TREMOLO = "tremolo"
    SPICCATO = "spiccato"
    PIZZICATO = "pizzicato"
    NATURAL_HARMONIC = "natural_harmonic"
    ARTIFICIAL_HARMONIC = "artificial_harmonic"
    RICOCHET = "ricochet"
    FLAUTANDO = "flautando"


class PluckingTechnique(Enum):
    TIRANDO = "tirando"
    APOYANDO = "apoyando"
    RASGUEADO = "rasgueado"
    TREMOLO = "tremolo"
    HARMONICS = "harmonics"
    SLAP = "slap"
    POP = "pop"
    PALM_MUTE = "palm_mute"
    HAMMER_ON = "hammer_on"
    PULL_OFF = "pull_off"
    SLIDE = "slide"
    BEND = "bend"


class WindTechnique(Enum):
    NORMAL = "normal"
    OVERBLOWING = "overblowing"
    FLUTTER_TONGUE = "flutter_tongue"
    MULTIPHONICS = "multiphonics"
    CIRCULAR_BREATHING = "circular_breathing"
    DOUBLE_TONGUE = "double_tongue"
    GROWL = "growl"
    SUBTONE = "subtone"
    SLAP_TONGUE = "slap_tongue"


class PercussionTechnique(Enum):
    CENTER_STRIKE = "center_strike"
    EDGE_STRIKE = "edge_strike"
    RIMSHOT = "rimshot"
    CROSS_STICK = "cross_stick"
    BRUSH = "brush"
    MALLET = "mallet"
    ROLL = "roll"
    FLAM = "flam"
    DRAG = "drag"
    MUTED = "muted"
    BELL = "bell"


@dataclass
class SpectralModification:
    harmonic_emphasis: list[float] = field(default_factory=lambda: [1.0])
    formant_shift_hz: float = 0.0
    noise_floor_db: float = -60.0
    attack_sharpness: float = 0.5
    brightness: float = 0.5
    description: str = ""


BOWING_SPECTRA: dict[BowingTechnique, SpectralModification] = {
    BowingTechnique.ARCO: SpectralModification(
        harmonic_emphasis=[1.0, 0.8, 0.5, 0.3, 0.2],
        brightness=0.5, description="Normal bowing, rich harmonic content",
    ),
    BowingTechnique.SUL_PONTICELLO: SpectralModification(
        harmonic_emphasis=[0.3, 0.5, 0.8, 1.0, 0.9, 0.7, 0.5],
        brightness=0.9, noise_floor_db=-40,
        description="Near bridge: glassy, high partials emphasized",
    ),
    BowingTechnique.SUL_TASTO: SpectralModification(
        harmonic_emphasis=[1.0, 0.4, 0.1],
        brightness=0.2, description="Over fingerboard: flute-like, fundamental dominant",
    ),
    BowingTechnique.COL_LEGNO: SpectralModification(
        harmonic_emphasis=[0.2, 0.1], noise_floor_db=-20,
        attack_sharpness=0.9, brightness=0.3,
        description="Wood of bow: percussive, pitched noise",
    ),
    BowingTechnique.NATURAL_HARMONIC: SpectralModification(
        harmonic_emphasis=[0.0, 1.0] if False else [1.0],
        brightness=0.7, description="Isolated harmonic: pure sine-like tone",
    ),
    BowingTechnique.PIZZICATO: SpectralModification(
        harmonic_emphasis=[1.0, 0.6, 0.4, 0.2],
        attack_sharpness=0.95, brightness=0.6,
        description="Plucked string: sharp attack, fast decay",
    ),
    BowingTechnique.FLAUTANDO: SpectralModification(
        harmonic_emphasis=[1.0, 0.2, 0.05],
        brightness=0.15, description="Extremely light bow: whisper tone",
    ),
}


PLUCKING_SPECTRA: dict[PluckingTechnique, SpectralModification] = {
    PluckingTechnique.TIRANDO: SpectralModification(
        harmonic_emphasis=[1.0, 0.7, 0.5, 0.3],
        brightness=0.5, description="Free stroke: balanced tone",
    ),
    PluckingTechnique.APOYANDO: SpectralModification(
        harmonic_emphasis=[1.0, 0.8, 0.6, 0.4, 0.2],
        brightness=0.6, description="Rest stroke: fuller, louder",
    ),
    PluckingTechnique.PALM_MUTE: SpectralModification(
        harmonic_emphasis=[1.0, 0.3, 0.1],
        brightness=0.2, description="Damped: short, dark",
    ),
    PluckingTechnique.SLAP: SpectralModification(
        harmonic_emphasis=[0.5, 0.3, 0.2], noise_floor_db=-15,
        attack_sharpness=0.98, brightness=0.7,
        description="Percussive slap: strong transient + fundamental",
    ),
    PluckingTechnique.HARMONICS: SpectralModification(
        harmonic_emphasis=[1.0], brightness=0.8,
        description="Isolated harmonic: bell-like, pure",
    ),
}


WIND_SPECTRA: dict[WindTechnique, SpectralModification] = {
    WindTechnique.NORMAL: SpectralModification(
        harmonic_emphasis=[1.0, 0.6, 0.3, 0.15],
        brightness=0.5, description="Standard embouchure",
    ),
    WindTechnique.OVERBLOWING: SpectralModification(
        harmonic_emphasis=[0.1, 1.0, 0.5, 0.3],
        brightness=0.7, description="Register break: 2nd partial dominant",
    ),
    WindTechnique.FLUTTER_TONGUE: SpectralModification(
        harmonic_emphasis=[1.0, 0.5, 0.3],
        brightness=0.6, noise_floor_db=-30,
        description="Rolled R: amplitude modulation at 15-25 Hz",
    ),
    WindTechnique.MULTIPHONICS: SpectralModification(
        harmonic_emphasis=[1.0, 0.9, 0.8, 0.7, 0.6],
        brightness=0.8, noise_floor_db=-25,
        description="Multiple simultaneous pitches: complex spectrum",
    ),
    WindTechnique.SUBTONE: SpectralModification(
        harmonic_emphasis=[1.0, 0.2, 0.05],
        brightness=0.15, noise_floor_db=-35,
        description="Breathy, quiet: mostly fundamental + air noise",
    ),
}


def spectral_envelope(technique_spectrum: SpectralModification, n_harmonics: int = 16) -> np.ndarray:
    env = np.zeros(n_harmonics)
    emphasis = technique_spectrum.harmonic_emphasis
    for i in range(min(len(emphasis), n_harmonics)):
        env[i] = emphasis[i]
    for i in range(len(emphasis), n_harmonics):
        env[i] = env[max(0, i - 1)] * 0.7
    noise = 10 ** (technique_spectrum.noise_floor_db / 20)
    env = np.maximum(env, noise)
    return env


def apply_technique_to_spectrum(
    base_harmonics: np.ndarray, modification: SpectralModification,
) -> np.ndarray:
    n = len(base_harmonics)
    env = spectral_envelope(modification, n)
    return base_harmonics * env
