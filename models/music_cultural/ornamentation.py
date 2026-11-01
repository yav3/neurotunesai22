"""Ornamentation: western, Indian, Arabic, Irish, Flamenco ornament libraries and detection."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


@dataclass
class Ornament:
    name: str
    pitch_pattern: list[float]
    timing: str
    region: str
    description: str = ""

    @property
    def n_notes(self) -> int:
        return len(self.pitch_pattern)

    def to_cents(self) -> list[float]:
        return [p * 100.0 for p in self.pitch_pattern]


# ---------------------------------------------------------------------------
# Western ornaments (pitch_pattern in semitones relative to main note = 0)
# ---------------------------------------------------------------------------

TRILL = Ornament("Trill", [0, 1, 0, 1, 0, 1, 0], "rapid alternation", "Western",
                 "Rapid alternation between main note and upper neighbor")
MORDENT = Ornament("Mordent", [0, -1, 0], "quick", "Western",
                   "Quick lower-neighbor grace (inverted = upper)")
UPPER_MORDENT = Ornament("Upper Mordent", [0, 1, 0], "quick", "Western",
                         "Quick upper-neighbor grace")
TURN = Ornament("Turn", [1, 0, -1, 0], "measured", "Western",
                "Upper-main-lower-main figure")
APPOGGIATURA = Ornament("Appoggiatura", [1, 0], "on-beat, stressed", "Western",
                        "Accented grace note resolving stepwise down")
ACCIACCATURA = Ornament("Acciaccatura", [-1, 0], "before-beat, crushed", "Western",
                        "Crushed grace note from below")
GRUPPETTO = Ornament("Gruppetto", [0, 1, 0, -1, 0], "measured", "Western",
                     "Turn figure starting on main note")
GLISSANDO = Ornament("Glissando", list(np.linspace(0, 12, 7).tolist()), "continuous slide", "Western",
                     "Continuous pitch slide between two notes")
TREMOLO = Ornament("Tremolo", [0, 0, 0, 0, 0, 0], "rapid rearticulation", "Western",
                   "Rapid repetition of the same note")

# Indian ornaments
GAMAK = Ornament("Gamak", [0, 2, 0, 2, 0], "oscillation", "South Asia",
                 "Heavy oscillation between adjacent notes")
MEEND = Ornament("Meend", list(np.linspace(0, 5, 8).tolist()), "smooth glide", "South Asia",
                 "Smooth continuous glide between notes")
KAN = Ornament("Kan", [2, 0], "fleeting touch", "South Asia",
               "Brief touch of adjacent note before main")
MURKI = Ornament("Murki", [0, -2, -1, 0], "rapid cluster", "South Asia",
                 "Rapid cluster of notes around main note")
KHATKA = Ornament("Khatka", [0, 2, 0, -1, 0], "quick shake", "South Asia",
                  "Quick shake involving multiple notes")
ANDOLAN = Ornament("Andolan", [0, 0.5, 0, -0.5, 0], "gentle oscillation", "South Asia",
                   "Slow gentle oscillation around a note (< semitone)")

# Arabic ornaments
RAQS = Ornament("Raqs", [0, 0.25, 0, -0.25, 0, 0.25, 0], "quarter-tone vibrato", "Middle East",
                "Characteristic Arabic vibrato with quarter-tone oscillation")
ZALAQA = Ornament("Zalaqa", list(np.linspace(0, 3, 5).tolist()), "slide", "Middle East",
                  "Ornamental slide between notes")
RASHA = Ornament("Rasha", [0, 1.5, 0, 1.5, 0], "shake", "Middle East",
                 "Rapid shake, often involving 3/4-tone intervals")

# Irish ornaments
CUT = Ornament("Cut", [3, 0], "very fast grace from above", "Ireland",
               "Quick grace note from above, cuts into duration of previous note")
STRIKE = Ornament("Strike", [-2, 0], "very fast grace from below", "Ireland",
                  "Quick hammer-on from below")
CRAN = Ornament("Cran", [0, 3, 0, 5, 0, 3, 0], "rapid fingered", "Ireland",
                "Rapid fingered ornament on D pipes/flute, uilleann-specific")
ROLL = Ornament("Roll", [0, 3, 0, -2, 0], "cut + strike", "Ireland",
                "Combination of cut and strike around main note")
SLIDE_IRISH = Ornament("Slide", list(np.linspace(-3, 0, 4).tolist()), "approach from below", "Ireland",
                       "Approach note from a third below")
TREBLE = Ornament("Treble", [0, 3, 0, 3, 0, -2, 0], "double cut + strike", "Ireland",
                  "Extended roll with double cuts")

# Flamenco ornaments
RASGUEADO = Ornament("Rasgueado", [0, 0, 0, 0, 0], "strummed fan", "Spain/Flamenco",
                     "Fan-like strumming across strings with individual fingers")
PICADO = Ornament("Picado", [0, 2, 4, 5, 7], "alternating fingers scale", "Spain/Flamenco",
                  "Rapid alternating-finger scale passage")
ALZAPUA = Ornament("Alzapua", [0, 0, 0], "thumb technique", "Spain/Flamenco",
                   "Thumb striking down-up-down pattern")
TREMOLO_FLAMENCO = Ornament("Tremolo (Flamenco)", [0, 0, 0, 0, 0], "p-a-m-i pattern", "Spain/Flamenco",
                            "5-note tremolo: thumb + 3 fingers + thumb")


ORNAMENT_LIBRARY: list[Ornament] = [
    TRILL, MORDENT, UPPER_MORDENT, TURN, APPOGGIATURA, ACCIACCATURA,
    GRUPPETTO, GLISSANDO, TREMOLO,
    GAMAK, MEEND, KAN, MURKI, KHATKA, ANDOLAN,
    RAQS, ZALAQA, RASHA,
    CUT, STRIKE, CRAN, ROLL, SLIDE_IRISH, TREBLE,
    RASGUEADO, PICADO, ALZAPUA, TREMOLO_FLAMENCO,
]


class OrnamentDetector:
    @staticmethod
    def detect(pitch_contour: np.ndarray,
               sr: int = 100,
               threshold_cents: float = 50.0) -> list[dict]:
        if len(pitch_contour) < 3:
            return []
        # Convert to cents relative to median
        median_pitch = float(np.median(pitch_contour))
        if median_pitch <= 0:
            return []
        cents = 1200.0 * np.log2(np.maximum(pitch_contour, 1e-6) / median_pitch)
        detections: list[dict] = []

        # Detect trill: rapid oscillation above threshold
        diff = np.diff(cents)
        sign_changes = np.diff(np.sign(diff))
        oscillation_points = np.where(np.abs(sign_changes) >= 1.5)[0]
        if len(oscillation_points) >= 4:
            # Check if oscillations are clustered
            gaps = np.diff(oscillation_points)
            clusters = np.where(gaps <= 3)[0]
            if len(clusters) >= 3:
                start = int(oscillation_points[clusters[0]])
                end = int(oscillation_points[clusters[-1] + 1]) + 2
                extent = float(np.max(np.abs(cents[start:end])))
                if extent > threshold_cents:
                    detections.append({
                        "ornament": "trill",
                        "start": start,
                        "end": min(end, len(cents)),
                        "extent_cents": extent,
                        "rate_hz": float(len(clusters)) * sr / max(end - start, 1),
                    })

        # Detect grace note: single quick deviation at the start
        if len(cents) > 4 and abs(cents[0]) > threshold_cents and abs(cents[2]) < threshold_cents * 0.5:
            direction = "above" if cents[0] > 0 else "below"
            ornament_name = "appoggiatura" if direction == "above" else "acciaccatura"
            detections.append({
                "ornament": ornament_name,
                "start": 0,
                "end": 2,
                "extent_cents": float(abs(cents[0])),
                "direction": direction,
            })

        # Detect glissando: monotonic pitch change over sustained segment
        if len(cents) > 5:
            total_change = cents[-1] - cents[0]
            if abs(total_change) > 200.0:
                monotonic = np.all(np.diff(cents) > -10) or np.all(np.diff(cents) < 10)
                if monotonic:
                    detections.append({
                        "ornament": "glissando",
                        "start": 0,
                        "end": len(cents),
                        "extent_cents": float(abs(total_change)),
                        "direction": "ascending" if total_change > 0 else "descending",
                    })

        # Detect mordent: quick dip/rise then return within first few samples
        if len(cents) >= 4:
            for i in range(len(cents) - 3):
                dev = cents[i + 1] - cents[i]
                ret = cents[i + 2] - cents[i + 1]
                if abs(dev) > threshold_cents and abs(ret) > threshold_cents * 0.7:
                    if dev * ret < 0:  # opposite directions
                        detections.append({
                            "ornament": "mordent" if dev < 0 else "upper_mordent",
                            "start": i,
                            "end": i + 3,
                            "extent_cents": float(abs(dev)),
                        })
                        break

        return detections

    @staticmethod
    def classify_vibrato(pitch_contour: np.ndarray, sr: int = 100) -> dict:
        if len(pitch_contour) < 10:
            return {"has_vibrato": False}
        median_pitch = float(np.median(pitch_contour))
        if median_pitch <= 0:
            return {"has_vibrato": False}
        cents = 1200.0 * np.log2(np.maximum(pitch_contour, 1e-6) / median_pitch)
        # Autocorrelation to find vibrato rate
        centered = cents - np.mean(cents)
        extent = float(np.std(centered))
        if extent < 10.0:
            return {"has_vibrato": False, "extent_cents": extent}
        corr = np.correlate(centered, centered, mode="full")
        corr = corr[len(corr) // 2:]
        if len(corr) < 4:
            return {"has_vibrato": False, "extent_cents": extent}
        corr = corr / (corr[0] + 1e-12)
        # Find first peak after zero crossing
        peaks = []
        for i in range(1, len(corr) - 1):
            if corr[i] > corr[i - 1] and corr[i] > corr[i + 1] and corr[i] > 0.3:
                peaks.append(i)
                break
        if not peaks:
            return {"has_vibrato": False, "extent_cents": extent}
        period_samples = peaks[0]
        rate_hz = float(sr / period_samples)
        width = "narrow" if extent < 30 else ("medium" if extent < 80 else "wide")
        return {
            "has_vibrato": True,
            "rate_hz": rate_hz,
            "extent_cents": extent,
            "width": width,
        }
