"""Pure-Python MIDI representation: notes, tracks, sequences, piano-roll conversion."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class MIDINote:
    pitch: int = 60
    velocity: int = 100
    start_time: float = 0.0
    duration: float = 0.5
    channel: int = 0

    @property
    def end_time(self) -> float:
        return self.start_time + self.duration

    @property
    def note_name(self) -> str:
        names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        return f"{names[self.pitch % 12]}{self.pitch // 12 - 1}"


class MIDITrack:
    def __init__(self, tempo: float = 120.0, time_signature: str = "4/4"):
        self.notes: list[MIDINote] = []
        self.tempo = tempo
        self.time_signature = time_signature

    def add_note(self, note: MIDINote) -> None:
        self.notes.append(note)

    def add_notes(self, notes: list[MIDINote]) -> None:
        self.notes.extend(notes)

    @property
    def duration(self) -> float:
        if not self.notes:
            return 0.0
        return max(n.end_time for n in self.notes)

    def to_piano_roll(self, resolution: float = 0.01) -> np.ndarray:
        if not self.notes:
            return np.zeros((128, 0), dtype=np.float32)
        total_time = self.duration
        n_frames = max(1, int(np.ceil(total_time / resolution)))
        roll = np.zeros((128, n_frames), dtype=np.float32)
        for n in self.notes:
            start_idx = int(n.start_time / resolution)
            end_idx = min(int(n.end_time / resolution), n_frames)
            if 0 <= n.pitch < 128:
                roll[n.pitch, start_idx:end_idx] = n.velocity / 127.0
        return roll

    def quantize(self, grid_size: float = 0.25) -> MIDITrack:
        track = MIDITrack(tempo=self.tempo, time_signature=self.time_signature)
        for n in self.notes:
            q_start = round(n.start_time / grid_size) * grid_size
            q_dur = max(grid_size, round(n.duration / grid_size) * grid_size)
            track.add_note(MIDINote(
                pitch=n.pitch, velocity=n.velocity,
                start_time=q_start, duration=q_dur, channel=n.channel,
            ))
        return track

    def sort(self) -> None:
        self.notes.sort(key=lambda n: (n.start_time, n.pitch))


class MIDISequence:
    def __init__(self):
        self.tracks: list[MIDITrack] = []

    def add_track(self, track: MIDITrack) -> None:
        self.tracks.append(track)

    @property
    def duration(self) -> float:
        if not self.tracks:
            return 0.0
        return max(t.duration for t in self.tracks)

    def merge_tracks(self) -> MIDITrack:
        merged = MIDITrack(
            tempo=self.tracks[0].tempo if self.tracks else 120.0,
            time_signature=self.tracks[0].time_signature if self.tracks else "4/4",
        )
        for t in self.tracks:
            merged.add_notes(list(t.notes))
        merged.sort()
        return merged

    def transpose(self, semitones: int) -> MIDISequence:
        seq = MIDISequence()
        for t in self.tracks:
            new_track = MIDITrack(tempo=t.tempo, time_signature=t.time_signature)
            for n in t.notes:
                new_pitch = max(0, min(127, n.pitch + semitones))
                new_track.add_note(MIDINote(
                    pitch=new_pitch, velocity=n.velocity,
                    start_time=n.start_time, duration=n.duration, channel=n.channel,
                ))
            seq.add_track(new_track)
        return seq

    def tempo_scale(self, factor: float) -> MIDISequence:
        if factor <= 0:
            raise ValueError("factor must be positive")
        seq = MIDISequence()
        for t in self.tracks:
            new_track = MIDITrack(tempo=t.tempo * factor, time_signature=t.time_signature)
            for n in t.notes:
                new_track.add_note(MIDINote(
                    pitch=n.pitch, velocity=n.velocity,
                    start_time=n.start_time / factor, duration=n.duration / factor,
                    channel=n.channel,
                ))
            seq.add_track(new_track)
        return seq


class PianoRollConverter:
    @staticmethod
    def notes_to_piano_roll(
        notes: list[MIDINote], resolution: float = 0.01, duration: float | None = None,
    ) -> np.ndarray:
        if not notes:
            return np.zeros((128, 0), dtype=np.float32)
        total = duration or max(n.end_time for n in notes)
        n_frames = max(1, int(np.ceil(total / resolution)))
        roll = np.zeros((128, n_frames), dtype=np.float32)
        for n in notes:
            s = int(n.start_time / resolution)
            e = min(int(n.end_time / resolution), n_frames)
            if 0 <= n.pitch < 128:
                roll[n.pitch, s:e] = n.velocity / 127.0
        return roll

    @staticmethod
    def piano_roll_to_notes(
        roll: np.ndarray, resolution: float = 0.01, threshold: float = 0.1,
    ) -> list[MIDINote]:
        notes: list[MIDINote] = []
        for pitch in range(roll.shape[0]):
            row = roll[pitch]
            in_note = False
            start = 0
            vel = 0.0
            for i in range(len(row)):
                if row[i] >= threshold and not in_note:
                    in_note = True
                    start = i
                    vel = float(row[i])
                elif (row[i] < threshold or i == len(row) - 1) and in_note:
                    end = i + 1 if row[i] >= threshold else i
                    notes.append(MIDINote(
                        pitch=pitch,
                        velocity=int(min(127, vel * 127)),
                        start_time=start * resolution,
                        duration=(end - start) * resolution,
                    ))
                    in_note = False
        return notes
