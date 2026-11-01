"""Cross-cultural melodic ornamentation: gamakas, melisma, mordents, trills, slides.

Models ornamental practices across Indian (gamaka), Arabic (tahrir/zalzal),
Flamenco (melisma/floreo), Celtic (rolls/crans), and Western classical
(Baroque ornament tables) traditions. Each tradition defines ornaments as
pitch-time curves that can be applied to a base melody.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
from enum import Enum


class OrnamentTradition(Enum):
    INDIAN_GAMAKA = "indian_gamaka"
    ARABIC_TAHRIR = "arabic_tahrir"
    FLAMENCO_MELISMA = "flamenco_melisma"
    CELTIC_ROLL = "celtic_roll"
    WESTERN_BAROQUE = "western_baroque"
    JAZZ_BEND = "jazz_bend"


class GamakaType(Enum):
    KAMPITA = "kampita"        # oscillation around a note
    JARU = "jaru"              # slide (ascending/descending)
    SPURITA = "spurita"        # touch upper note
    PRATYAHATA = "pratyahata"  # bounce off lower note
    ANDOLITA = "andolita"      # slow wave
    NOKKU = "nokku"            # stress/emphasis
    ODUKKAL = "odukkal"        # slide down and hold


@dataclass
class OrnamentCurve:
    """Pitch-time curve for an ornament."""
    times: np.ndarray      # normalised [0, 1] within note duration
    pitches: np.ndarray    # cents deviation from base pitch
    dynamics: np.ndarray   # amplitude envelope [0, 1]
    tradition: OrnamentTradition = OrnamentTradition.WESTERN_BAROQUE

    @property
    def duration_ratio(self) -> float:
        return float(self.times[-1] - self.times[0]) if len(self.times) > 1 else 0.0

    @property
    def max_deviation(self) -> float:
        return float(np.max(np.abs(self.pitches)))


GAMAKA_PROFILES: Dict[GamakaType, Dict] = {
    GamakaType.KAMPITA: {"cycles": 3, "depth_cents": 50, "shape": "sine"},
    GamakaType.JARU: {"direction": "up", "range_cents": 200, "shape": "linear"},
    GamakaType.SPURITA: {"touch_cents": 100, "return_frac": 0.3},
    GamakaType.PRATYAHATA: {"touch_cents": -100, "return_frac": 0.3},
    GamakaType.ANDOLITA: {"cycles": 1.5, "depth_cents": 80, "shape": "sine"},
    GamakaType.NOKKU: {"accent_frac": 0.15, "decay": 0.7},
    GamakaType.ODUKKAL: {"slide_cents": -150, "hold_frac": 0.6},
}


def generate_gamaka(gamaka_type: GamakaType, n_points: int = 64,
                    depth_override: Optional[float] = None) -> OrnamentCurve:
    t = np.linspace(0, 1, n_points)
    profile = GAMAKA_PROFILES[gamaka_type]

    if gamaka_type in (GamakaType.KAMPITA, GamakaType.ANDOLITA):
        cycles = profile["cycles"]
        depth = depth_override or profile["depth_cents"]
        pitches = depth * np.sin(2 * np.pi * cycles * t)
        dynamics = np.ones(n_points)
    elif gamaka_type == GamakaType.JARU:
        rng = depth_override or profile["range_cents"]
        if profile["direction"] == "up":
            pitches = rng * t
        else:
            pitches = rng * (1 - t)
        dynamics = np.ones(n_points)
    elif gamaka_type == GamakaType.SPURITA:
        touch = depth_override or profile["touch_cents"]
        rf = profile["return_frac"]
        pitches = np.where(t < rf, touch * np.sin(np.pi * t / rf), 0.0)
        dynamics = np.ones(n_points)
    elif gamaka_type == GamakaType.PRATYAHATA:
        touch = depth_override or profile["touch_cents"]
        rf = profile["return_frac"]
        pitches = np.where(t < rf, touch * np.sin(np.pi * t / rf), 0.0)
        dynamics = np.ones(n_points)
    elif gamaka_type == GamakaType.NOKKU:
        af = profile["accent_frac"]
        decay = profile["decay"]
        pitches = np.zeros(n_points)
        dynamics = np.where(t < af, 1.0, decay + (1 - decay) * np.exp(-5 * (t - af)))
    elif gamaka_type == GamakaType.ODUKKAL:
        slide = depth_override or profile["slide_cents"]
        hf = profile["hold_frac"]
        pitches = np.where(t < (1 - hf), slide * t / (1 - hf), slide)
        dynamics = np.ones(n_points)
    else:
        pitches = np.zeros(n_points)
        dynamics = np.ones(n_points)

    return OrnamentCurve(t, pitches, dynamics, OrnamentTradition.INDIAN_GAMAKA)


def arabic_tahrir(n_oscillations: int = 5, depth_cents: float = 40,
                  n_points: int = 64) -> OrnamentCurve:
    """Tahrir: rapid throat oscillation in Arabic maqam singing."""
    t = np.linspace(0, 1, n_points)
    env = 1 - np.abs(2 * t - 1)  # triangle envelope, peak at center
    pitches = depth_cents * env * np.sin(2 * np.pi * n_oscillations * t)
    dynamics = 0.7 + 0.3 * env
    return OrnamentCurve(t, pitches, dynamics, OrnamentTradition.ARABIC_TAHRIR)


def flamenco_melisma(scale_degrees: List[int], duration_weights: Optional[List[float]] = None,
                     cents_per_degree: float = 100.0, n_points: int = 128) -> OrnamentCurve:
    """Flamenco melismatic passage: rapid scale-degree traversal."""
    n = len(scale_degrees)
    if duration_weights is None:
        duration_weights = [1.0] * n
    total = sum(duration_weights)
    norm_w = [w / total for w in duration_weights]

    t_all, p_all, d_all = [], [], []
    t_cursor = 0.0
    for i, (deg, w) in enumerate(zip(scale_degrees, norm_w)):
        pts = max(2, int(n_points * w))
        t_seg = np.linspace(t_cursor, t_cursor + w, pts)
        p_seg = np.full(pts, deg * cents_per_degree)
        d_seg = np.ones(pts) * (0.8 + 0.2 * (i == 0 or i == n - 1))
        t_all.append(t_seg)
        p_all.append(p_seg)
        d_all.append(d_seg)
        t_cursor += w

    return OrnamentCurve(
        np.concatenate(t_all), np.concatenate(p_all),
        np.concatenate(d_all), OrnamentTradition.FLAMENCO_MELISMA
    )


def celtic_roll(base_cents: float = 0.0, upper_cents: float = 200.0,
                lower_cents: float = -200.0, n_points: int = 64) -> OrnamentCurve:
    """Celtic roll: upper grace → main → lower grace → main."""
    t = np.linspace(0, 1, n_points)
    pitches = np.zeros(n_points)
    q1, q2, q3 = n_points // 4, n_points // 2, 3 * n_points // 4
    pitches[:q1] = np.linspace(upper_cents, base_cents, q1)
    pitches[q1:q2] = base_cents
    pitches[q2:q3] = np.linspace(base_cents, lower_cents, n_points // 4)
    pitches[q3:] = np.linspace(lower_cents, base_cents, n_points - q3)
    dynamics = np.ones(n_points)
    dynamics[:q1] = 0.6
    dynamics[q2:q3] = 0.6
    return OrnamentCurve(t, pitches, dynamics, OrnamentTradition.CELTIC_ROLL)


def baroque_trill(interval_cents: float = 200.0, n_alternations: int = 4,
                  nachschlag: bool = True, n_points: int = 64) -> OrnamentCurve:
    """Baroque trill with optional nachschlag (termination)."""
    t = np.linspace(0, 1, n_points)
    if nachschlag:
        trill_end = 0.85
        after = int(n_points * (1 - trill_end))
    else:
        trill_end = 1.0
        after = 0
    trill_pts = n_points - after
    phase = np.linspace(0, n_alternations * 2 * np.pi, trill_pts)
    trill_pitches = interval_cents * 0.5 * (1 + np.sign(np.sin(phase)))
    if after > 0:
        nach_pitches = np.linspace(interval_cents * 0.5, 0, after)
        pitches = np.concatenate([trill_pitches, nach_pitches])
    else:
        pitches = trill_pitches
    dynamics = np.ones(n_points)
    return OrnamentCurve(t, pitches, dynamics, OrnamentTradition.WESTERN_BAROQUE)


def jazz_bend(target_cents: float = 100.0, bend_type: str = "scoop",
              n_points: int = 64) -> OrnamentCurve:
    """Jazz pitch bend: scoop (below→target), fall (target→below), doit (target→above)."""
    t = np.linspace(0, 1, n_points)
    if bend_type == "scoop":
        pitches = target_cents * (t ** 0.5 - 1)
    elif bend_type == "fall":
        pitches = -target_cents * t ** 2
    elif bend_type == "doit":
        pitches = target_cents * t ** 0.5
    else:
        pitches = np.zeros(n_points)
    dynamics = np.ones(n_points)
    return OrnamentCurve(t, pitches, dynamics, OrnamentTradition.JAZZ_BEND)


def apply_ornament(base_pitch_hz: float, ornament: OrnamentCurve,
                   sr: int = 44100, duration_s: float = 0.5) -> np.ndarray:
    """Render ornamented pitch as a frequency trajectory in Hz."""
    n_samples = int(sr * duration_s)
    t_samples = np.linspace(0, 1, n_samples)
    cents_interp = np.interp(t_samples, ornament.times, ornament.pitches)
    dyn_interp = np.interp(t_samples, ornament.times, ornament.dynamics)
    freq_trajectory = base_pitch_hz * 2.0 ** (cents_interp / 1200.0)
    return freq_trajectory * dyn_interp


def ornament_complexity(ornament: OrnamentCurve) -> Dict[str, float]:
    """Measure ornament complexity: pitch range, velocity, direction changes."""
    pitches = ornament.pitches
    if len(pitches) < 2:
        return {"range_cents": 0.0, "mean_velocity": 0.0, "direction_changes": 0}
    dt = np.diff(ornament.times)
    dp = np.diff(pitches)
    velocity = np.abs(dp / np.maximum(dt, 1e-10))
    signs = np.sign(dp)
    dir_changes = int(np.sum(np.abs(np.diff(signs[signs != 0])) > 0))
    return {
        "range_cents": float(np.ptp(pitches)),
        "mean_velocity": float(np.mean(velocity)),
        "direction_changes": dir_changes,
        "max_deviation_cents": float(np.max(np.abs(pitches))),
    }


def compare_ornaments(a: OrnamentCurve, b: OrnamentCurve,
                      n_points: int = 128) -> float:
    """DTW-like distance between two ornament pitch curves (normalised time)."""
    t = np.linspace(0, 1, n_points)
    pa = np.interp(t, a.times, a.pitches)
    pb = np.interp(t, b.times, b.pitches)
    return float(np.sqrt(np.mean((pa - pb) ** 2)))


def tradition_ornament_catalog(tradition: OrnamentTradition) -> List[OrnamentCurve]:
    """Generate a catalog of representative ornaments for a tradition."""
    if tradition == OrnamentTradition.INDIAN_GAMAKA:
        return [generate_gamaka(gt) for gt in GamakaType]
    elif tradition == OrnamentTradition.ARABIC_TAHRIR:
        return [arabic_tahrir(n) for n in [3, 5, 7]]
    elif tradition == OrnamentTradition.CELTIC_ROLL:
        return [celtic_roll(upper_cents=c) for c in [100, 200, 300]]
    elif tradition == OrnamentTradition.WESTERN_BAROQUE:
        return [baroque_trill(n_alternations=n) for n in [3, 4, 6]]
    elif tradition == OrnamentTradition.JAZZ_BEND:
        return [jazz_bend(bend_type=bt) for bt in ["scoop", "fall", "doit"]]
    elif tradition == OrnamentTradition.FLAMENCO_MELISMA:
        return [flamenco_melisma([0, -1, -2, -1, 0]), flamenco_melisma([0, 1, 2, 3, 2, 1, 0])]
    return []
