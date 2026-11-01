"""Impulse response analysis, synthesis, and convolution."""
import numpy as np
from dataclasses import dataclass


@dataclass
class ImpulseResponse:
    """Analyze a measured or synthetic impulse response."""

    ir: np.ndarray
    sr: int

    def _schroeder_curve(self) -> np.ndarray:
        squared = self.ir ** 2
        return np.cumsum(squared[::-1])[::-1]

    def rt60(self, freq_bands: list = None) -> float | dict:
        if freq_bands is not None:
            result = {}
            n_fft = len(self.ir)
            IR_fft = np.fft.rfft(self.ir, n=n_fft)
            freqs = np.fft.rfftfreq(n_fft, 1.0 / self.sr)
            for band in freq_bands:
                low = band / np.sqrt(2)
                high = band * np.sqrt(2)
                mask = (freqs >= low) & (freqs <= high)
                filtered_fft = IR_fft.copy()
                filtered_fft[~mask] = 0
                filtered = np.fft.irfft(filtered_fft, n=n_fft)
                result[band] = self._estimate_rt60(filtered)
            return result
        return self._estimate_rt60(self.ir)

    def _estimate_rt60(self, signal: np.ndarray) -> float:
        sc = np.cumsum(signal[::-1] ** 2)[::-1]
        if sc[0] <= 0:
            return 0.0
        db = 10.0 * np.log10(sc / sc[0] + 1e-30)
        idx_5 = np.searchsorted(-db, 5.0)
        idx_25 = np.searchsorted(-db, 25.0)
        if idx_25 <= idx_5 or idx_25 >= len(db):
            idx_25 = min(len(db) - 1, idx_5 + 1)
        if idx_25 == idx_5:
            return 0.0
        slope = (db[idx_25] - db[idx_5]) / ((idx_25 - idx_5) / self.sr)
        if slope >= 0:
            return 0.0
        return -60.0 / slope

    def edt(self) -> float:
        sc = np.cumsum(self.ir[::-1] ** 2)[::-1]
        if sc[0] <= 0:
            return 0.0
        db = 10.0 * np.log10(sc / sc[0] + 1e-30)
        idx_10 = np.searchsorted(-db, 10.0)
        if idx_10 == 0:
            return 0.0
        slope = db[idx_10] / (idx_10 / self.sr)
        if slope >= 0:
            return 0.0
        return -60.0 / slope

    def clarity(self, t_ms: float = 50.0) -> float:
        n_early = int(self.sr * t_ms / 1000.0)
        squared = self.ir ** 2
        early = np.sum(squared[:n_early])
        late = np.sum(squared[n_early:])
        if late < 1e-30:
            return 60.0
        return 10.0 * np.log10(early / late + 1e-30)

    def definition(self, t_ms: float = 50.0) -> float:
        n_early = int(self.sr * t_ms / 1000.0)
        squared = self.ir ** 2
        total = np.sum(squared)
        if total < 1e-30:
            return 0.0
        return float(np.sum(squared[:n_early]) / total)

    def centre_time(self) -> float:
        squared = self.ir ** 2
        total = np.sum(squared)
        if total < 1e-30:
            return 0.0
        t = np.arange(len(self.ir)) / self.sr
        return float(np.sum(t * squared) / total)

    def iacc(self, left_ir: np.ndarray, right_ir: np.ndarray) -> float:
        if len(left_ir) == 0 or len(right_ir) == 0:
            return 0.0
        n = max(len(left_ir), len(right_ir))
        left = np.pad(left_ir, (0, n - len(left_ir)))
        right = np.pad(right_ir, (0, n - len(right_ir)))
        cc = np.correlate(left, right, mode="full")
        norm = np.sqrt(np.sum(left ** 2) * np.sum(right ** 2))
        if norm < 1e-30:
            return 0.0
        max_lag_samples = int(0.001 * self.sr)
        center = len(cc) // 2
        lo = max(0, center - max_lag_samples)
        hi = min(len(cc), center + max_lag_samples + 1)
        return float(np.max(np.abs(cc[lo:hi])) / norm)


@dataclass
class IRSynthesizer:
    """Generate synthetic impulse responses."""

    def exponential_decay(
        self, rt60: float, sr: int, duration: float = None
    ) -> np.ndarray:
        if duration is None:
            duration = rt60 * 1.5
        n = int(sr * duration)
        t = np.arange(n) / sr
        tau = rt60 / (3.0 * np.log(10))
        ir = np.random.default_rng(42).standard_normal(n) * np.exp(-t / tau)
        return ir

    def room_model(
        self,
        dimensions: tuple,
        absorption: float,
        sr: int,
        n_reflections: int = 50,
    ) -> np.ndarray:
        lx, ly, lz = dimensions
        c = 343.0
        max_time = 2.0 * max(lx, ly, lz) * n_reflections / c
        n_samples = int(sr * max_time)
        n_samples = min(n_samples, sr * 3)
        ir = np.zeros(n_samples)
        rng = np.random.default_rng(42)
        source = np.array([lx * 0.3, ly * 0.4, lz * 0.5])
        receiver = np.array([lx * 0.7, ly * 0.6, lz * 0.3])
        direct_dist = np.linalg.norm(source - receiver)
        direct_sample = int(direct_dist / c * sr)
        if 0 <= direct_sample < n_samples:
            ir[direct_sample] = 1.0 / max(direct_dist, 0.01)
        for order in range(1, n_reflections + 1):
            for _ in range(6):
                nx = rng.integers(-order, order + 1)
                ny = rng.integers(-order, order + 1)
                nz = rng.integers(-order, order + 1)
                image = np.array([
                    ((-1) ** nx) * source[0] + 2 * nx * lx,
                    ((-1) ** ny) * source[1] + 2 * ny * ly,
                    ((-1) ** nz) * source[2] + 2 * nz * lz,
                ])
                dist = np.linalg.norm(image - receiver)
                if dist < 0.01:
                    continue
                n_walls = abs(nx) + abs(ny) + abs(nz)
                amp = ((1 - absorption) ** n_walls) / dist
                sample = int(dist / c * sr)
                if 0 <= sample < n_samples:
                    ir[sample] += amp
        return ir

    def diffuse_field(
        self, rt60: float, sr: int, density: int = 1000
    ) -> np.ndarray:
        duration = rt60 * 1.5
        n = int(sr * duration)
        rng = np.random.default_rng(42)
        tau = rt60 / (3.0 * np.log(10))
        intervals = rng.exponential(1.0 / density, size=density * int(duration + 1))
        times = np.cumsum(intervals)
        times = times[times < duration]
        ir = np.zeros(n)
        for t in times:
            sample = int(t * sr)
            if 0 <= sample < n:
                amp = np.exp(-t / tau) * rng.choice([-1, 1])
                ir[sample] += amp
        return ir


@dataclass
class Convolver:
    """Efficient FFT-based convolution."""

    ir: np.ndarray

    def process(self, signal: np.ndarray) -> np.ndarray:
        n = len(signal) + len(self.ir) - 1
        n_fft = 1
        while n_fft < n:
            n_fft *= 2
        sig_fft = np.fft.rfft(signal, n=n_fft)
        ir_fft = np.fft.rfft(self.ir, n=n_fft)
        result = np.fft.irfft(sig_fft * ir_fft, n=n_fft)
        return result[:len(signal)]

    def process_partitioned(
        self, signal: np.ndarray, block_size: int = 4096
    ) -> np.ndarray:
        n_blocks = (len(self.ir) + block_size - 1) // block_size
        ir_blocks = []
        for i in range(n_blocks):
            start = i * block_size
            end = min(start + block_size, len(self.ir))
            block = np.zeros(block_size)
            block[:end - start] = self.ir[start:end]
            ir_blocks.append(block)
        output = np.zeros(len(signal) + len(self.ir) - 1)
        n_fft = block_size * 2
        ir_ffts = [np.fft.rfft(b, n=n_fft) for b in ir_blocks]
        n_sig_blocks = (len(signal) + block_size - 1) // block_size
        for i in range(n_sig_blocks):
            start = i * block_size
            end = min(start + block_size, len(signal))
            sig_block = np.zeros(block_size)
            sig_block[:end - start] = signal[start:end]
            sig_fft = np.fft.rfft(sig_block, n=n_fft)
            for j, ir_fft in enumerate(ir_ffts):
                conv = np.fft.irfft(sig_fft * ir_fft, n=n_fft)
                out_start = (i + j) * block_size
                out_end = out_start + n_fft
                out_end = min(out_end, len(output))
                n_copy = out_end - out_start
                output[out_start:out_end] += conv[:n_copy]
        return output[:len(signal)]
