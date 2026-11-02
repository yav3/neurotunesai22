"""Room acoustic parameter extraction and quality metrics."""
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RoomAcousticAnalyzer:
    """Comprehensive room acoustic analysis from impulse response."""

    ir: np.ndarray
    sr: int

    def schroeder_curve(self) -> np.ndarray:
        squared = self.ir ** 2
        sc = np.cumsum(squared[::-1])[::-1]
        if sc[0] > 0:
            return 10.0 * np.log10(sc / sc[0] + 1e-30)
        return np.zeros(len(self.ir))

    def _rt_from_range(self, db_start: float, db_end: float) -> float:
        sc = self.schroeder_curve()
        idx_start = np.searchsorted(-sc, db_start)
        idx_end = np.searchsorted(-sc, db_end)
        if idx_end <= idx_start or idx_end >= len(sc):
            return 0.0
        slope = (sc[idx_end] - sc[idx_start]) / ((idx_end - idx_start) / self.sr)
        if slope >= 0:
            return 0.0
        return -60.0 / slope

    def parameters(self) -> dict:
        rt60 = self._rt_from_range(5.0, 35.0)
        edt = self._rt_from_range(0.0, 10.0)
        n50 = int(self.sr * 0.050)
        n80 = int(self.sr * 0.080)
        sq = self.ir ** 2
        total = np.sum(sq)
        if total < 1e-30:
            return {"rt60": 0, "edt": 0, "c50": 0, "c80": 0, "d50": 0, "d80": 0, "ts": 0}
        early50 = np.sum(sq[:n50])
        late50 = np.sum(sq[n50:])
        early80 = np.sum(sq[:n80])
        late80 = np.sum(sq[n80:])
        c50 = 10.0 * np.log10(early50 / max(late50, 1e-30))
        c80 = 10.0 * np.log10(early80 / max(late80, 1e-30))
        d50 = early50 / total
        d80 = early80 / total
        t = np.arange(len(self.ir)) / self.sr
        ts = np.sum(t * sq) / total
        return {"rt60": rt60, "edt": edt, "c50": c50, "c80": c80,
                "d50": d50, "d80": d80, "ts": ts}

    def frequency_dependent_rt60(
        self, bands: list = None
    ) -> dict:
        if bands is None:
            bands = [125, 250, 500, 1000, 2000, 4000]
        result = {}
        n_fft = len(self.ir)
        IR_fft = np.fft.rfft(self.ir, n=n_fft)
        freqs = np.fft.rfftfreq(n_fft, 1.0 / self.sr)
        for band in bands:
            low = band / np.sqrt(2)
            high = band * np.sqrt(2)
            mask = (freqs >= low) & (freqs <= high)
            filtered_fft = IR_fft.copy()
            filtered_fft[~mask] = 0
            filtered = np.fft.irfft(filtered_fft, n=n_fft)
            sc = np.cumsum(filtered[::-1] ** 2)[::-1]
            if sc[0] <= 0:
                result[band] = 0.0
                continue
            db = 10.0 * np.log10(sc / sc[0] + 1e-30)
            i5 = np.searchsorted(-db, 5.0)
            i25 = np.searchsorted(-db, 25.0)
            if i25 <= i5 or i25 >= len(db):
                result[band] = 0.0
                continue
            slope = (db[i25] - db[i5]) / ((i25 - i5) / self.sr)
            result[band] = -60.0 / slope if slope < 0 else 0.0
        return result


@dataclass
class SpeechIntelligibility:
    """Speech transmission index estimation from impulse response."""

    ir: np.ndarray
    sr: int

    def modulation_transfer_function(
        self, octave_band: float, mod_freqs: np.ndarray = None
    ) -> np.ndarray:
        if mod_freqs is None:
            mod_freqs = np.array([0.63, 0.8, 1.0, 1.25, 1.6, 2.0, 2.5,
                                  3.15, 4.0, 5.0, 6.3, 8.0, 10.0, 12.5])
        n_fft = len(self.ir)
        IR_fft = np.fft.rfft(self.ir, n=n_fft)
        freqs = np.fft.rfftfreq(n_fft, 1.0 / self.sr)
        low = octave_band / np.sqrt(2)
        high = octave_band * np.sqrt(2)
        mask = (freqs >= low) & (freqs <= high)
        filtered_fft = IR_fft.copy()
        filtered_fft[~mask] = 0
        filtered = np.fft.irfft(filtered_fft, n=n_fft)
        squared = filtered ** 2
        total_energy = np.sum(squared)
        if total_energy < 1e-30:
            return np.zeros(len(mod_freqs))
        t = np.arange(len(filtered)) / self.sr
        mtf = np.zeros(len(mod_freqs))
        for i, fm in enumerate(mod_freqs):
            cos_mod = np.cos(2 * np.pi * fm * t)
            sin_mod = np.sin(2 * np.pi * fm * t)
            real_part = np.sum(squared * cos_mod)
            imag_part = np.sum(squared * sin_mod)
            mtf[i] = np.sqrt(real_part ** 2 + imag_part ** 2) / total_energy
        return np.clip(mtf, 0, 1)

    def sti(self) -> float:
        octave_bands = [125, 250, 500, 1000, 2000, 4000, 8000]
        band_weights = np.array([0.13, 0.14, 0.11, 0.12, 0.19, 0.17, 0.14])
        band_stis = []
        for band in octave_bands:
            mtf = self.modulation_transfer_function(band)
            mtf = np.clip(mtf, 1e-10, 1.0)
            snr = 10.0 * np.log10(mtf / (1.0 - mtf + 1e-10))
            snr = np.clip(snr, -15.0, 15.0)
            ti = (snr + 15.0) / 30.0
            band_stis.append(np.mean(ti))
        return float(np.clip(np.sum(np.array(band_stis) * band_weights), 0, 1))

    def alcons(self) -> float:
        sti_val = self.sti()
        return float(np.clip(170.5405 * np.exp(-5.419 * sti_val), 0, 100))


@dataclass
class AcousticQualityMetrics:
    """Subjective room acoustic quality estimation."""

    def warmth(self, rt60_low: float, rt60_mid: float) -> float:
        if rt60_mid <= 0:
            return 0.0
        ratio = rt60_low / rt60_mid
        return float(np.clip(1.0 - abs(ratio - 1.2) / 0.8, 0, 1))

    def brilliance(self, rt60_high: float, rt60_mid: float) -> float:
        if rt60_mid <= 0:
            return 0.0
        ratio = rt60_high / rt60_mid
        return float(np.clip(ratio / 1.2, 0, 1))

    def intimacy(self, initial_time_delay_gap_ms: float) -> float:
        return float(np.clip(1.0 - initial_time_delay_gap_ms / 40.0, 0, 1))

    def envelopment(self, lateral_fraction: float) -> float:
        return float(np.clip(lateral_fraction, 0, 1))

    def overall_quality(self, metrics: dict) -> float:
        weights = {"warmth": 0.25, "brilliance": 0.2, "intimacy": 0.25, "envelopment": 0.3}
        total = 0.0
        w_sum = 0.0
        for key, w in weights.items():
            if key in metrics:
                total += w * metrics[key]
                w_sum += w
        return float(total / max(w_sum, 1e-10))


MATERIAL_ABSORPTION = {
    "concrete": {125: 0.01, 250: 0.01, 500: 0.02, 1000: 0.02, 2000: 0.02, 4000: 0.03},
    "carpet": {125: 0.08, 250: 0.24, 500: 0.57, 1000: 0.69, 2000: 0.71, 4000: 0.73},
    "glass": {125: 0.35, 250: 0.25, 500: 0.18, 1000: 0.12, 2000: 0.07, 4000: 0.04},
    "curtain": {125: 0.07, 250: 0.31, 500: 0.49, 1000: 0.75, 2000: 0.70, 4000: 0.60},
    "wood": {125: 0.15, 250: 0.11, 500: 0.10, 1000: 0.07, 2000: 0.06, 4000: 0.07},
    "acoustic_tile": {125: 0.20, 250: 0.40, 500: 0.70, 1000: 0.80, 2000: 0.60, 4000: 0.40},
}


@dataclass
class MaterialAbsorption:
    """Absorption coefficient database for common materials."""

    coefficients: dict = field(default_factory=lambda: dict(MATERIAL_ABSORPTION))

    def alpha(self, material: str, frequency: float) -> float:
        if material not in self.coefficients:
            return 0.1
        mat = self.coefficients[material]
        bands = sorted(mat.keys())
        if frequency <= bands[0]:
            return mat[bands[0]]
        if frequency >= bands[-1]:
            return mat[bands[-1]]
        for i in range(len(bands) - 1):
            if bands[i] <= frequency <= bands[i + 1]:
                t = (frequency - bands[i]) / (bands[i + 1] - bands[i])
                return mat[bands[i]] * (1 - t) + mat[bands[i + 1]] * t
        return 0.1

    def sabine_rt60(
        self, volume: float, surface_areas: dict, materials: dict
    ) -> float:
        total_absorption = 0.0
        for surface_name, area in surface_areas.items():
            mat = materials.get(surface_name, "concrete")
            avg_alpha = np.mean([self.alpha(mat, f) for f in [125, 250, 500, 1000, 2000, 4000]])
            total_absorption += area * avg_alpha
        if total_absorption < 1e-10:
            return 0.0
        return 0.161 * volume / total_absorption
