"""Modular synthesis routing: signal patching, CV/gate processing, module interconnection.

Models the signal flow of modular synthesizers:
- Patch matrix for arbitrary module interconnection
- CV (control voltage) scaling, offsetting, and quantization
- Gate/trigger generation and logic
- Signal mixing, splitting, and mult behavior
- Voltage-controlled parameter modulation
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum


class SignalType(Enum):
    AUDIO = "audio"
    CV = "cv"
    GATE = "gate"
    TRIGGER = "trigger"
    CLOCK = "clock"


class ModulePort(Enum):
    INPUT = "input"
    OUTPUT = "output"


@dataclass
class Port:
    name: str
    port_type: ModulePort
    signal_type: SignalType
    voltage_range: Tuple[float, float] = (-5.0, 5.0)


@dataclass
class Module:
    name: str
    ports: List[Port] = field(default_factory=list)

    def input_ports(self) -> List[Port]:
        return [p for p in self.ports if p.port_type == ModulePort.INPUT]

    def output_ports(self) -> List[Port]:
        return [p for p in self.ports if p.port_type == ModulePort.OUTPUT]


@dataclass
class PatchCable:
    source_module: str
    source_port: str
    dest_module: str
    dest_port: str
    attenuation: float = 1.0


class PatchMatrix:
    def __init__(self) -> None:
        self.modules: Dict[str, Module] = {}
        self.cables: List[PatchCable] = []
        self._signals: Dict[str, np.ndarray] = {}

    def add_module(self, module: Module) -> None:
        self.modules[module.name] = module

    def patch(self, source_module: str, source_port: str,
              dest_module: str, dest_port: str,
              attenuation: float = 1.0) -> PatchCable:
        cable = PatchCable(source_module, source_port,
                           dest_module, dest_port, attenuation)
        self.cables.append(cable)
        return cable

    def unpatch(self, source_module: str, source_port: str,
                dest_module: str, dest_port: str) -> bool:
        for i, cable in enumerate(self.cables):
            if (cable.source_module == source_module and
                cable.source_port == source_port and
                cable.dest_module == dest_module and
                cable.dest_port == dest_port):
                self.cables.pop(i)
                return True
        return False

    def get_connections_from(self, module_name: str) -> List[PatchCable]:
        return [c for c in self.cables if c.source_module == module_name]

    def get_connections_to(self, module_name: str) -> List[PatchCable]:
        return [c for c in self.cables if c.dest_module == module_name]

    def set_signal(self, module_name: str, port_name: str,
                   signal: np.ndarray) -> None:
        key = f"{module_name}:{port_name}"
        self._signals[key] = signal

    def get_signal(self, module_name: str, port_name: str) -> Optional[np.ndarray]:
        key = f"{module_name}:{port_name}"
        return self._signals.get(key)

    def route_signals(self) -> None:
        for cable in self.cables:
            src_key = f"{cable.source_module}:{cable.source_port}"
            dst_key = f"{cable.dest_module}:{cable.dest_port}"
            src_signal = self._signals.get(src_key)
            if src_signal is not None:
                existing = self._signals.get(dst_key)
                routed = src_signal * cable.attenuation
                if existing is not None:
                    self._signals[dst_key] = existing + routed
                else:
                    self._signals[dst_key] = routed.copy()

    def adjacency_matrix(self) -> Tuple[np.ndarray, List[str]]:
        names = sorted(self.modules.keys())
        n = len(names)
        idx = {name: i for i, name in enumerate(names)}
        mat = np.zeros((n, n))
        for cable in self.cables:
            if cable.source_module in idx and cable.dest_module in idx:
                mat[idx[cable.source_module], idx[cable.dest_module]] += cable.attenuation
        return mat, names


def cv_to_frequency(cv: np.ndarray, base_freq: float = 261.63) -> np.ndarray:
    return base_freq * np.power(2.0, cv)


def frequency_to_cv(freq: np.ndarray, base_freq: float = 261.63) -> np.ndarray:
    return np.log2(np.maximum(freq, 1e-10) / base_freq)


def quantize_cv(cv: np.ndarray, scale_semitones: Optional[List[int]] = None) -> np.ndarray:
    semitones = cv * 12.0
    if scale_semitones is None:
        quantized = np.round(semitones)
    else:
        scale = np.array(scale_semitones)
        quantized = np.zeros_like(semitones)
        for i, s in enumerate(semitones):
            octave = int(np.floor(s / 12.0))
            pc = s - octave * 12.0
            dists = np.abs(scale - pc)
            nearest = scale[np.argmin(dists)]
            quantized[i] = octave * 12.0 + nearest
    return quantized / 12.0


def cv_offset(cv: np.ndarray, semitones: float) -> np.ndarray:
    return cv + semitones / 12.0


def cv_scale(cv: np.ndarray, amount: float, center: float = 0.0) -> np.ndarray:
    return center + (cv - center) * amount


def gate_from_cv(cv: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    return (cv > threshold).astype(float)


def trigger_from_gate(gate: np.ndarray) -> np.ndarray:
    trigger = np.zeros_like(gate)
    trigger[1:] = np.maximum(0.0, np.diff(gate))
    if gate[0] > 0:
        trigger[0] = 1.0
    return trigger


def clock_signal(n_samples: int, sr: float, bpm: float,
                 division: float = 1.0) -> np.ndarray:
    period_samples = int(sr * 60.0 / (bpm * division))
    if period_samples <= 0:
        return np.zeros(n_samples)
    clock = np.zeros(n_samples)
    for i in range(0, n_samples, period_samples):
        pulse_len = min(int(sr * 0.001), period_samples // 2, n_samples - i)
        clock[i:i + pulse_len] = 1.0
    return clock


def clock_divider(clock: np.ndarray, ratio: int) -> np.ndarray:
    triggers = trigger_from_gate(clock)
    divided = np.zeros_like(clock)
    count = 0
    for i in range(len(triggers)):
        if triggers[i] > 0:
            if count % ratio == 0:
                divided[i] = 1.0
            count += 1
    return divided


def clock_multiplier(clock: np.ndarray, ratio: int, sr: float) -> np.ndarray:
    triggers = np.where(trigger_from_gate(clock) > 0)[0]
    if len(triggers) < 2:
        return clock.copy()
    multiplied = np.zeros_like(clock)
    for i in range(len(triggers) - 1):
        period = triggers[i + 1] - triggers[i]
        sub_period = period / ratio
        for j in range(ratio):
            idx = int(triggers[i] + j * sub_period)
            if 0 <= idx < len(multiplied):
                multiplied[idx] = 1.0
    return multiplied


def gate_logic_and(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.minimum(a, b)


def gate_logic_or(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.maximum(a, b)


def gate_logic_xor(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ga = (a > 0.5).astype(float)
    gb = (b > 0.5).astype(float)
    return np.abs(ga - gb)


def slew_limiter(cv: np.ndarray, rise_rate: float, fall_rate: float,
                 sr: float) -> np.ndarray:
    output = np.zeros_like(cv)
    output[0] = cv[0]
    rise_per_sample = rise_rate / sr
    fall_per_sample = fall_rate / sr
    for i in range(1, len(cv)):
        diff = cv[i] - output[i - 1]
        if diff > 0:
            output[i] = output[i - 1] + min(diff, rise_per_sample)
        else:
            output[i] = output[i - 1] + max(diff, -fall_per_sample)
    return output


def sample_and_hold(signal: np.ndarray, trigger: np.ndarray) -> np.ndarray:
    output = np.zeros_like(signal)
    held = signal[0]
    for i in range(len(signal)):
        if trigger[i] > 0.5:
            held = signal[i]
        output[i] = held
    return output


def voltage_controlled_mix(signals: List[np.ndarray],
                           cv_levels: List[np.ndarray]) -> np.ndarray:
    if not signals:
        return np.array([])
    n = min(len(s) for s in signals)
    output = np.zeros(n)
    for sig, cv in zip(signals, cv_levels):
        level = np.clip(cv[:n], 0.0, 1.0)
        output += sig[:n] * level
    return output


def ring_modulator(carrier: np.ndarray, modulator: np.ndarray) -> np.ndarray:
    n = min(len(carrier), len(modulator))
    return carrier[:n] * modulator[:n]


def rectifier(signal: np.ndarray, mode: str = "full") -> np.ndarray:
    if mode == "full":
        return np.abs(signal)
    elif mode == "half":
        return np.maximum(signal, 0.0)
    return signal


def wavefolder(signal: np.ndarray, gain: float = 2.0) -> np.ndarray:
    scaled = signal * gain
    folded = np.arcsin(np.sin(scaled * np.pi / 2.0)) * 2.0 / np.pi
    return folded


def envelope_follower(signal: np.ndarray, attack_ms: float = 10.0,
                      release_ms: float = 100.0, sr: float = 44100.0) -> np.ndarray:
    attack_coeff = np.exp(-1.0 / (attack_ms * sr / 1000.0))
    release_coeff = np.exp(-1.0 / (release_ms * sr / 1000.0))
    rectified = np.abs(signal)
    envelope = np.zeros_like(signal)
    envelope[0] = rectified[0]
    for i in range(1, len(signal)):
        if rectified[i] > envelope[i - 1]:
            envelope[i] = attack_coeff * envelope[i - 1] + (1.0 - attack_coeff) * rectified[i]
        else:
            envelope[i] = release_coeff * envelope[i - 1] + (1.0 - release_coeff) * rectified[i]
    return envelope
