"""Auditory scene analysis: stream segregation, grouping principles, auditory objects."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class AuditoryStream:
    """A coherent perceptual stream of sound events."""
    stream_id: int
    events: list[int] = field(default_factory=list)
    mean_freq: float = 0.0
    mean_time: float = 0.0
    freq_range: tuple[float, float] = (0.0, 0.0)

    @property
    def n_events(self) -> int:
        return len(self.events)


class StreamSegregation:
    """Frequency-based stream segregation (van Noorden, 1975; Bregman, 1990).

    Two interleaved tone sequences segregate into distinct perceptual streams
    when their frequency separation exceeds ~3-4 semitones at moderate tempi.
    The temporal coherence boundary (TCB) and fission boundary (FB) define
    the obligatory-integration and obligatory-segregation thresholds.
    """

    def __init__(
        self,
        fission_semitones: float = 3.5,
        coherence_semitones: float = 12.0,
    ) -> None:
        self.fission_semitones = fission_semitones
        self.coherence_semitones = coherence_semitones

    @staticmethod
    def freq_to_semitones(f1: float, f2: float) -> float:
        if f1 <= 0 or f2 <= 0:
            return 0.0
        return float(12.0 * np.log2(f2 / f1))

    def segregation_probability(
        self,
        freq_sep_semitones: float,
        ioi_seconds: float = 0.15,
    ) -> float:
        """Estimate probability of stream segregation.

        Based on van Noorden's (1975) temporal coherence / fission boundaries.
        Fast tempos (short IOI) and large frequency separations favour segregation.
        """
        tempo_factor = np.clip(1.0 - ioi_seconds / 0.5, 0.0, 1.0)
        sep = abs(freq_sep_semitones)
        if sep <= self.fission_semitones:
            return float(0.1 * tempo_factor)
        if sep >= self.coherence_semitones:
            return float(min(0.95, 0.8 + 0.15 * tempo_factor))
        t = (sep - self.fission_semitones) / (self.coherence_semitones - self.fission_semitones)
        base = 0.1 + 0.7 * t
        return float(np.clip(base + 0.15 * tempo_factor, 0.0, 1.0))

    def segregate(
        self,
        frequencies: np.ndarray,
        threshold_semitones: float | None = None,
    ) -> list[AuditoryStream]:
        """Assign events to streams by nearest-frequency clustering.

        Simple greedy approach: each event joins the stream whose mean frequency
        is closest (within threshold), or starts a new stream.
        """
        if len(frequencies) == 0:
            return []

        if threshold_semitones is None:
            threshold_semitones = (self.fission_semitones + self.coherence_semitones) / 2.0

        streams: list[AuditoryStream] = []
        for i, freq in enumerate(frequencies):
            if freq <= 0:
                continue
            best_stream = None
            best_dist = float("inf")
            for s in streams:
                if s.mean_freq <= 0:
                    continue
                dist = abs(self.freq_to_semitones(s.mean_freq, freq))
                if dist < best_dist:
                    best_dist = dist
                    best_stream = s

            if best_stream is not None and best_dist <= threshold_semitones:
                n = best_stream.n_events
                best_stream.mean_freq = (best_stream.mean_freq * n + freq) / (n + 1)
                best_stream.events.append(i)
                lo, hi = best_stream.freq_range
                best_stream.freq_range = (min(lo, freq), max(hi, freq))
            else:
                streams.append(AuditoryStream(
                    stream_id=len(streams),
                    events=[i],
                    mean_freq=freq,
                    freq_range=(freq, freq),
                ))

        return streams

    def buildup_rate(
        self,
        freq_sep_semitones: float,
        n_repetitions: int = 10,
    ) -> np.ndarray:
        """Model segregation buildup over repeated presentations.

        Stream segregation builds up over time: initially integrated, gradually
        segregating. Modelled as 1 - exp(-t/tau) with tau inversely proportional
        to frequency separation.
        """
        sep = abs(freq_sep_semitones)
        if sep < 0.5:
            return np.zeros(n_repetitions)
        tau = max(1.0, 15.0 - sep)
        t = np.arange(n_repetitions, dtype=float)
        asymptote = self.segregation_probability(sep)
        return asymptote * (1.0 - np.exp(-t / tau))


class AuditoryGrouping:
    """Gestalt grouping principles applied to auditory events (Bregman, 1990).

    Implements: proximity (time/frequency), similarity, common fate,
    good continuation, closure.
    """

    @staticmethod
    def proximity_score(
        dt: float,
        df_semitones: float,
        time_scale: float = 0.3,
        freq_scale: float = 6.0,
    ) -> float:
        """Gaussian proximity in time-frequency space."""
        return float(np.exp(-0.5 * ((dt / time_scale) ** 2 + (df_semitones / freq_scale) ** 2)))

    @staticmethod
    def similarity_score(
        timbre_a: np.ndarray,
        timbre_b: np.ndarray,
    ) -> float:
        """Cosine similarity between timbre descriptor vectors."""
        na = np.linalg.norm(timbre_a)
        nb = np.linalg.norm(timbre_b)
        if na < 1e-10 or nb < 1e-10:
            return 0.0
        return float(np.dot(timbre_a, timbre_b) / (na * nb))

    @staticmethod
    def common_fate_score(
        trajectory_a: np.ndarray,
        trajectory_b: np.ndarray,
    ) -> float:
        """Correlation of frequency/amplitude trajectories over time.

        Sounds that modulate together (vibrato, tremolo at same rate) group.
        """
        if len(trajectory_a) < 2 or len(trajectory_b) < 2:
            return 0.0
        n = min(len(trajectory_a), len(trajectory_b))
        a = trajectory_a[:n]
        b = trajectory_b[:n]
        da = np.diff(a)
        db = np.diff(b)
        na = np.linalg.norm(da)
        nb = np.linalg.norm(db)
        if na < 1e-10 or nb < 1e-10:
            return 0.0
        return float(np.dot(da, db) / (na * nb))

    @staticmethod
    def good_continuation_score(
        freqs: np.ndarray,
        times: np.ndarray,
    ) -> float:
        """How well a frequency trajectory follows a smooth contour.

        Measured as 1 - normalised second-derivative energy.
        """
        if len(freqs) < 3:
            return 1.0
        dt = np.diff(times)
        dt = np.maximum(dt, 1e-10)
        df = np.diff(freqs)
        slopes = df / dt
        curvature = np.diff(slopes) / dt[:-1]
        energy = np.mean(curvature ** 2)
        max_expected = (np.max(freqs) - np.min(freqs)) ** 2 if np.ptp(freqs) > 0 else 1.0
        return float(np.clip(1.0 - energy / (max_expected + 1e-10), 0.0, 1.0))

    def group_events(
        self,
        frequencies: np.ndarray,
        times: np.ndarray,
        proximity_threshold: float = 0.3,
    ) -> list[list[int]]:
        """Group events into auditory objects by proximity in time-frequency space."""
        n = len(frequencies)
        if n == 0:
            return []

        parent = list(range(n))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        for i in range(n):
            for j in range(i + 1, n):
                dt = abs(times[i] - times[j])
                fi, fj = frequencies[i], frequencies[j]
                if fi <= 0 or fj <= 0:
                    continue
                df = abs(12.0 * np.log2(fj / fi))
                score = self.proximity_score(dt, df)
                if score >= proximity_threshold:
                    union(i, j)

        groups: dict[int, list[int]] = {}
        for i in range(n):
            root = find(i)
            groups.setdefault(root, []).append(i)
        return list(groups.values())

    @staticmethod
    def closure_detect(
        magnitudes: np.ndarray,
        gap_start: int,
        gap_end: int,
        threshold_db: float = 6.0,
    ) -> bool:
        """Detect perceptual closure across a gap in a magnitude envelope.

        If the signal surrounding a gap is sufficiently loud relative to the
        gap depth, the auditory system 'fills in' the missing portion.
        """
        if gap_start <= 0 or gap_end >= len(magnitudes):
            return False
        before = magnitudes[max(0, gap_start - 5):gap_start]
        after = magnitudes[gap_end:min(len(magnitudes), gap_end + 5)]
        gap = magnitudes[gap_start:gap_end]
        if len(before) == 0 or len(after) == 0 or len(gap) == 0:
            return False
        surround_level = (np.mean(before) + np.mean(after)) / 2.0
        gap_level = np.mean(gap)
        if surround_level < 1e-10:
            return False
        ratio_db = 20.0 * np.log10(surround_level / max(gap_level, 1e-10))
        return bool(ratio_db <= threshold_db)
