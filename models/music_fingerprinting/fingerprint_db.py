"""In-memory fingerprint database for audio identification."""
from __future__ import annotations

from dataclasses import dataclass, field
from collections import defaultdict

import numpy as np


@dataclass
class TrackRecord:
    track_id: str
    title: str = ""
    duration_sec: float = 0.0


@dataclass
class FingerprintDB:
    """Hash-table-based fingerprint store for audio identification.

    Stores (hash_value → list of (track_id, time_offset)) pairs.
    Query: extract hashes from a clip, look up each hash, count
    time-coherent matches per track to identify the song.
    """
    _hash_table: dict[int, list[tuple[str, int]]] = field(default_factory=lambda: defaultdict(list))
    _tracks: dict[str, TrackRecord] = field(default_factory=dict)

    @property
    def n_tracks(self) -> int:
        return len(self._tracks)

    @property
    def n_hashes(self) -> int:
        return sum(len(v) for v in self._hash_table.values())

    def add_track(self, track: TrackRecord, hashes: list[tuple[int, int]]) -> None:
        """Insert a track's fingerprint hashes.

        hashes: list of (hash_value, time_offset_frames).
        """
        self._tracks[track.track_id] = track
        for h, t in hashes:
            self._hash_table[h].append((track.track_id, t))

    def remove_track(self, track_id: str) -> None:
        if track_id not in self._tracks:
            return
        del self._tracks[track_id]
        for h in list(self._hash_table.keys()):
            self._hash_table[h] = [(tid, t) for tid, t in self._hash_table[h] if tid != track_id]
            if not self._hash_table[h]:
                del self._hash_table[h]

    def query(
        self,
        query_hashes: list[tuple[int, int]],
        min_matches: int = 5,
        time_tolerance: int = 2,
    ) -> list[tuple[str, float]]:
        """Identify a track from query fingerprint hashes.

        Returns list of (track_id, confidence) sorted by confidence desc.
        Confidence = fraction of query hashes that matched time-coherently.
        """
        if not query_hashes:
            return []

        offset_counts: dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))

        for qh, qt in query_hashes:
            if qh not in self._hash_table:
                continue
            for track_id, db_t in self._hash_table[qh]:
                delta = db_t - qt
                offset_counts[track_id][delta] += 1

        results: list[tuple[str, float]] = []
        n_query = len(query_hashes)

        for track_id, offsets in offset_counts.items():
            best_count = 0
            for delta, count in offsets.items():
                total = count
                for d2 in range(-time_tolerance, time_tolerance + 1):
                    if d2 != 0 and (delta + d2) in offsets:
                        total += offsets[delta + d2]
                best_count = max(best_count, total)

            if best_count >= min_matches:
                confidence = min(best_count / n_query, 1.0)
                results.append((track_id, confidence))

        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def contains(self, track_id: str) -> bool:
        return track_id in self._tracks

    def get_track(self, track_id: str) -> TrackRecord | None:
        return self._tracks.get(track_id)


def generate_hashes_from_peaks(
    peak_times: np.ndarray,
    peak_freqs: np.ndarray,
    fan_out: int = 5,
    time_window: int = 50,
) -> list[tuple[int, int]]:
    """Generate combinatorial hashes from spectral peaks.

    Each hash encodes (anchor_freq, target_freq, time_delta).
    """
    n = len(peak_times)
    hashes: list[tuple[int, int]] = []

    for i in range(n):
        anchor_t = int(peak_times[i])
        anchor_f = int(peak_freqs[i]) & 0x1FF

        count = 0
        for j in range(i + 1, n):
            if count >= fan_out:
                break
            dt = int(peak_times[j] - peak_times[i])
            if dt > time_window:
                break
            if dt <= 0:
                continue
            target_f = int(peak_freqs[j]) & 0x1FF
            h = (anchor_f << 20) | (target_f << 11) | (dt & 0x7FF)
            hashes.append((h, anchor_t))
            count += 1

    return hashes
