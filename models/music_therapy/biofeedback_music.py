"""Biofeedback-driven music therapy: physiological signal → musical parameter mapping."""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Callable
from scipy import signal as sp_signal


@dataclass
class PhysiologicalSignal:
    name: str
    data: np.ndarray
    sample_rate: float
    unit: str = ""

    @property
    def duration(self) -> float:
        return len(self.data) / self.sample_rate if self.sample_rate > 0 else 0.0


@dataclass
class MusicalParameter:
    name: str
    value: float
    min_val: float = 0.0
    max_val: float = 1.0

    @property
    def normalized(self) -> float:
        rng = self.max_val - self.min_val
        if rng < 1e-12:
            return 0.5
        return (self.value - self.min_val) / rng


def heart_rate_from_ecg(ecg: np.ndarray, sample_rate: float,
                        peak_distance_s: float = 0.4) -> np.ndarray:
    filtered = sp_signal.sosfilt(
        sp_signal.butter(4, [0.5, 40.0], btype="band", fs=sample_rate, output="sos"),
        ecg,
    )
    min_dist = int(peak_distance_s * sample_rate)
    peaks, _ = sp_signal.find_peaks(filtered, distance=min_dist,
                                     height=np.std(filtered) * 0.5)
    if len(peaks) < 2:
        return np.array([60.0])
    rr_intervals = np.diff(peaks) / sample_rate
    hr = 60.0 / rr_intervals
    return hr


def hrv_rmssd(rr_intervals_s: np.ndarray) -> float:
    if len(rr_intervals_s) < 2:
        return 0.0
    diffs = np.diff(rr_intervals_s)
    return float(np.sqrt(np.mean(diffs ** 2)))


def hrv_sdnn(rr_intervals_s: np.ndarray) -> float:
    if len(rr_intervals_s) < 2:
        return 0.0
    return float(np.std(rr_intervals_s))


def respiration_rate(resp: np.ndarray, sample_rate: float) -> float:
    if len(resp) < sample_rate * 2:
        return 15.0
    filtered = sp_signal.sosfilt(
        sp_signal.butter(3, [0.1, 0.5], btype="band", fs=sample_rate, output="sos"),
        resp,
    )
    peaks, _ = sp_signal.find_peaks(filtered, distance=int(sample_rate * 1.5))
    if len(peaks) < 2:
        return 15.0
    mean_period = np.mean(np.diff(peaks)) / sample_rate
    return 60.0 / mean_period if mean_period > 0 else 15.0


def skin_conductance_level(eda: np.ndarray, sample_rate: float) -> Tuple[float, np.ndarray]:
    sos = sp_signal.butter(2, 0.05, btype="low", fs=sample_rate, output="sos")
    tonic = sp_signal.sosfilt(sos, eda)
    phasic = eda - tonic
    scl = float(np.mean(tonic))
    return scl, phasic


def emg_activation(emg: np.ndarray, sample_rate: float,
                   window_s: float = 0.1) -> np.ndarray:
    rectified = np.abs(emg)
    win_len = max(int(window_s * sample_rate), 1)
    if win_len > len(rectified):
        return np.array([float(np.mean(rectified))])
    kernel = np.ones(win_len) / win_len
    return np.convolve(rectified, kernel, mode="same")


@dataclass
class MappingRule:
    signal_name: str
    param_name: str
    signal_range: Tuple[float, float]
    param_range: Tuple[float, float]
    invert: bool = False
    curve: str = "linear"

    def map(self, value: float) -> float:
        lo, hi = self.signal_range
        rng = hi - lo
        if rng < 1e-12:
            t = 0.5
        else:
            t = np.clip((value - lo) / rng, 0.0, 1.0)
        if self.invert:
            t = 1.0 - t
        if self.curve == "exponential":
            t = t ** 2
        elif self.curve == "logarithmic":
            t = np.log1p(t * (np.e - 1)) / 1.0
        elif self.curve == "sigmoid":
            t = 1.0 / (1.0 + np.exp(-10 * (t - 0.5)))
        p_lo, p_hi = self.param_range
        return float(p_lo + t * (p_hi - p_lo))


DEFAULT_MAPPINGS: List[MappingRule] = [
    MappingRule("heart_rate", "tempo_bpm", (50, 120), (60, 140)),
    MappingRule("hrv_rmssd", "harmonic_complexity", (10, 80), (0.1, 0.9)),
    MappingRule("respiration_rate", "phrase_length", (8, 25), (8, 2), invert=True),
    MappingRule("skin_conductance", "dynamics", (1.0, 20.0), (0.2, 0.9)),
    MappingRule("emg_level", "rhythm_density", (0.0, 1.0), (0.2, 0.8)),
]


@dataclass
class BiofeedbackState:
    heart_rate: float = 72.0
    hrv_rmssd: float = 40.0
    respiration_rate: float = 15.0
    skin_conductance: float = 5.0
    emg_level: float = 0.3
    timestamp: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "heart_rate": self.heart_rate,
            "hrv_rmssd": self.hrv_rmssd,
            "respiration_rate": self.respiration_rate,
            "skin_conductance": self.skin_conductance,
            "emg_level": self.emg_level,
        }


@dataclass
class MusicalOutput:
    tempo_bpm: float = 80.0
    harmonic_complexity: float = 0.5
    phrase_length: float = 4.0
    dynamics: float = 0.5
    rhythm_density: float = 0.5
    mode: str = "major"


def apply_mappings(state: BiofeedbackState,
                   rules: Optional[List[MappingRule]] = None) -> MusicalOutput:
    if rules is None:
        rules = DEFAULT_MAPPINGS
    values = state.to_dict()
    params: Dict[str, float] = {}
    for rule in rules:
        sig_val = values.get(rule.signal_name, 0.0)
        params[rule.param_name] = rule.map(sig_val)
    return MusicalOutput(
        tempo_bpm=params.get("tempo_bpm", 80.0),
        harmonic_complexity=params.get("harmonic_complexity", 0.5),
        phrase_length=params.get("phrase_length", 4.0),
        dynamics=params.get("dynamics", 0.5),
        rhythm_density=params.get("rhythm_density", 0.5),
    )


def arousal_from_physiology(state: BiofeedbackState) -> float:
    hr_norm = np.clip((state.heart_rate - 50) / 70, 0, 1)
    scl_norm = np.clip((state.skin_conductance - 1) / 19, 0, 1)
    resp_norm = np.clip((state.respiration_rate - 8) / 17, 0, 1)
    return float(0.4 * hr_norm + 0.35 * scl_norm + 0.25 * resp_norm)


def valence_from_physiology(state: BiofeedbackState) -> float:
    hrv_norm = np.clip((state.hrv_rmssd - 10) / 70, 0, 1)
    emg_inv = 1.0 - np.clip(state.emg_level, 0, 1)
    return float(0.6 * hrv_norm + 0.4 * emg_inv)


def relaxation_index(state: BiofeedbackState) -> float:
    arousal = arousal_from_physiology(state)
    return 1.0 - arousal


@dataclass
class AdaptiveController:
    target_arousal: float = 0.4
    learning_rate: float = 0.1
    _history: List[Tuple[float, float]] = field(default_factory=list, repr=False)

    def step(self, current_state: BiofeedbackState,
             current_output: MusicalOutput) -> MusicalOutput:
        arousal = arousal_from_physiology(current_state)
        error = self.target_arousal - arousal
        self._history.append((arousal, error))
        tempo_adj = error * 20.0 * self.learning_rate
        dyn_adj = error * 0.2 * self.learning_rate
        new_tempo = np.clip(current_output.tempo_bpm + tempo_adj, 40, 180)
        new_dyn = np.clip(current_output.dynamics + dyn_adj, 0.1, 1.0)
        new_mode = "major" if valence_from_physiology(current_state) > 0.5 else "minor"
        return MusicalOutput(
            tempo_bpm=float(new_tempo),
            harmonic_complexity=current_output.harmonic_complexity,
            phrase_length=current_output.phrase_length,
            dynamics=float(new_dyn),
            rhythm_density=current_output.rhythm_density,
            mode=new_mode,
        )

    def convergence_rate(self) -> float:
        if len(self._history) < 4:
            return 0.0
        errors = [abs(e) for _, e in self._history[-10:]]
        if errors[0] < 1e-12:
            return 1.0
        return float(1.0 - errors[-1] / (errors[0] + 1e-12))


def coherence_score(hr_series: np.ndarray, resp_series: np.ndarray,
                    sample_rate: float) -> float:
    if len(hr_series) < 10 or len(resp_series) < 10:
        return 0.0
    min_len = min(len(hr_series), len(resp_series))
    hr = hr_series[:min_len]
    resp = resp_series[:min_len]
    hr_norm = (hr - np.mean(hr)) / (np.std(hr) + 1e-12)
    resp_norm = (resp - np.mean(resp)) / (np.std(resp) + 1e-12)
    corr = np.abs(np.correlate(hr_norm, resp_norm, mode="full"))
    return float(np.max(corr) / min_len)


def session_progress(states: List[BiofeedbackState]) -> Dict[str, float]:
    if len(states) < 2:
        return {"arousal_change": 0.0, "relaxation_trend": 0.0}
    arousals = [arousal_from_physiology(s) for s in states]
    half = len(arousals) // 2
    first_half = np.mean(arousals[:half])
    second_half = np.mean(arousals[half:])
    return {
        "arousal_change": float(second_half - first_half),
        "relaxation_trend": float(first_half - second_half),
        "final_arousal": float(arousals[-1]),
        "mean_arousal": float(np.mean(arousals)),
    }
