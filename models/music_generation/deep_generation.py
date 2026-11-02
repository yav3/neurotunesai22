"""Deep generative music: autoregressive transformer, variational sequences, tonal plans, groove, motivic development."""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict


# ---- §1 Autoregressive transformer (pure numpy) ----

@dataclass
class TransformerGenConfig:
    """Config for autoregressive transformer generation."""
    vocab_size: int = 128
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    d_ff: int = 128
    max_seq_len: int = 512
    dropout: float = 0.0


def _sinusoidal_pe(max_len: int, d_model: int) -> np.ndarray:
    """Sinusoidal positional encoding matrix (max_len × d_model)."""
    pe = np.zeros((max_len, d_model))
    pos = np.arange(max_len)[:, None]
    div = np.exp(np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
    pe[:, 0::2] = np.sin(pos * div)
    pe[:, 1::2] = np.cos(pos * div[:d_model // 2])
    return pe


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / (e.sum(axis=axis, keepdims=True) + 1e-30)


def _layer_norm(x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Layer normalization over last axis."""
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps)


def _causal_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Scaled dot-product attention with causal mask."""
    d_k = Q.shape[-1]
    scores = Q @ K.transpose(0, 2, 1) / np.sqrt(d_k)
    n = scores.shape[-1]
    mask = np.triu(np.full((n, n), -1e9), k=1)
    scores = scores + mask
    weights = _softmax(scores, axis=-1)
    return weights @ V


def _gelu(x: np.ndarray) -> np.ndarray:
    """Gaussian Error Linear Unit."""
    return 0.5 * x * (1 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))


@dataclass
class TransformerWeights:
    """Weight tensors for a transformer model."""
    embedding: np.ndarray = field(default_factory=lambda: np.array([]))
    pos_encoding: np.ndarray = field(default_factory=lambda: np.array([]))
    layers: List[Dict[str, np.ndarray]] = field(default_factory=list)
    output_proj: np.ndarray = field(default_factory=lambda: np.array([]))


def init_transformer_weights(config: TransformerGenConfig, seed: int = 42) -> TransformerWeights:
    """Initialize random transformer weights."""
    rng = np.random.RandomState(seed)
    scale = 0.02
    emb = rng.randn(config.vocab_size, config.d_model) * scale
    pe = _sinusoidal_pe(config.max_seq_len, config.d_model)
    layers = []
    for _ in range(config.n_layers):
        d = config.d_model
        layers.append({
            "Wq": rng.randn(d, d) * scale,
            "Wk": rng.randn(d, d) * scale,
            "Wv": rng.randn(d, d) * scale,
            "Wo": rng.randn(d, d) * scale,
            "W1": rng.randn(d, config.d_ff) * scale,
            "W2": rng.randn(config.d_ff, d) * scale,
        })
    out_proj = rng.randn(config.d_model, config.vocab_size) * scale
    return TransformerWeights(embedding=emb, pos_encoding=pe, layers=layers, output_proj=out_proj)


def transformer_forward(tokens: np.ndarray, weights: TransformerWeights, config: TransformerGenConfig) -> np.ndarray:
    """Forward pass returning logits (seq_len × vocab_size)."""
    seq_len = len(tokens)
    x = weights.embedding[tokens] + weights.pos_encoding[:seq_len]
    n_heads = config.n_heads
    d_head = config.d_model // n_heads
    for layer in weights.layers:
        residual = x
        x_norm = _layer_norm(x)
        Q = (x_norm @ layer["Wq"]).reshape(1, seq_len, n_heads, d_head).transpose(0, 2, 1, 3).reshape(n_heads, seq_len, d_head)
        K = (x_norm @ layer["Wk"]).reshape(1, seq_len, n_heads, d_head).transpose(0, 2, 1, 3).reshape(n_heads, seq_len, d_head)
        V = (x_norm @ layer["Wv"]).reshape(1, seq_len, n_heads, d_head).transpose(0, 2, 1, 3).reshape(n_heads, seq_len, d_head)
        attn = _causal_attention(Q, K, V)
        attn = attn.reshape(n_heads, seq_len, d_head).transpose(1, 0, 2).reshape(seq_len, config.d_model)
        x = residual + attn @ layer["Wo"]
        residual = x
        x_norm = _layer_norm(x)
        x = residual + _gelu(x_norm @ layer["W1"]) @ layer["W2"]
    logits = _layer_norm(x) @ weights.output_proj
    return logits


def temperature_sample(logits: np.ndarray, temperature: float = 1.0) -> int:
    """Sample from logits with temperature scaling."""
    if temperature <= 0:
        return int(np.argmax(logits))
    scaled = logits / temperature
    probs = _softmax(scaled)
    return int(np.random.choice(len(probs), p=probs))


def nucleus_sample(logits: np.ndarray, p: float = 0.9, temperature: float = 1.0) -> int:
    """Top-p (nucleus) sampling."""
    scaled = logits / max(temperature, 1e-8)
    probs = _softmax(scaled)
    sorted_idx = np.argsort(probs)[::-1]
    sorted_probs = probs[sorted_idx]
    cumsum = np.cumsum(sorted_probs)
    cutoff = np.searchsorted(cumsum, p) + 1
    kept = sorted_idx[:cutoff]
    kept_probs = probs[kept]
    kept_probs /= kept_probs.sum()
    return int(np.random.choice(kept, p=kept_probs))


def generate_sequence(config: TransformerGenConfig, weights: TransformerWeights,
                      prompt: np.ndarray, length: int = 64,
                      temperature: float = 0.8, top_p: float = 0.9) -> np.ndarray:
    """Autoregressively generate a token sequence."""
    tokens = list(prompt)
    for _ in range(length):
        ctx = np.array(tokens[-config.max_seq_len:])
        logits = transformer_forward(ctx, weights, config)
        next_token = nucleus_sample(logits[-1], p=top_p, temperature=temperature)
        tokens.append(next_token)
    return np.array(tokens)


# ---- §2 Variational sequence model ----

@dataclass
class VAEConfig:
    """Config for variational autoencoder on sequences."""
    input_dim: int = 128
    hidden_dim: int = 64
    latent_dim: int = 16


def vae_encode(x: np.ndarray, W_enc: np.ndarray, W_mu: np.ndarray, W_logvar: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Encode input to latent distribution parameters."""
    h = np.tanh(x @ W_enc)
    mu = h @ W_mu
    logvar = h @ W_logvar
    return h, mu, logvar


def reparameterize(mu: np.ndarray, logvar: np.ndarray, seed: int = 0) -> np.ndarray:
    """Reparameterization trick: z = mu + sigma * epsilon."""
    rng = np.random.RandomState(seed)
    std = np.exp(0.5 * logvar)
    eps = rng.randn(*mu.shape)
    return mu + std * eps


def vae_decode(z: np.ndarray, W_dec: np.ndarray, W_out: np.ndarray) -> np.ndarray:
    """Decode latent vector to reconstruction."""
    h = np.tanh(z @ W_dec)
    return h @ W_out


def kl_divergence(mu: np.ndarray, logvar: np.ndarray) -> float:
    """KL(q(z|x) || p(z)) for diagonal Gaussian."""
    return float(-0.5 * np.sum(1 + logvar - mu ** 2 - np.exp(logvar)))


def latent_interpolate(z1: np.ndarray, z2: np.ndarray, n_steps: int = 10) -> np.ndarray:
    """Spherical linear interpolation (slerp) between two latent vectors."""
    z1_norm = z1 / (np.linalg.norm(z1) + 1e-10)
    z2_norm = z2 / (np.linalg.norm(z2) + 1e-10)
    omega = np.arccos(np.clip(np.dot(z1_norm, z2_norm), -1, 1))
    if abs(omega) < 1e-6:
        ts = np.linspace(0, 1, n_steps)
        return np.array([(1 - t) * z1 + t * z2 for t in ts])
    sin_omega = np.sin(omega)
    ts = np.linspace(0, 1, n_steps)
    return np.array([
        np.sin((1 - t) * omega) / sin_omega * z1 + np.sin(t * omega) / sin_omega * z2
        for t in ts
    ])


# ---- §3 Tonal plan generator ----

CADENCE_TYPES = {
    "PAC": [4, 0],        # V → I (perfect authentic)
    "IAC": [4, 0],        # V → I (imperfect — weaker inversion)
    "HC": [0, 4],         # * → V (half cadence)
    "DC": [0, 3],         # * → IV (plagal direction)
    "PC": [3, 0],         # IV → I (plagal)
}

CHORD_TENSION = {
    0: 0.0,   # I — tonic (rest)
    1: 0.4,   # ii
    2: 0.5,   # iii
    3: 0.2,   # IV — subdominant
    4: 0.7,   # V — dominant
    5: 0.6,   # vi
    6: 0.9,   # vii°
}


def tension_arc(n_chords: int, shape: str = "arch") -> np.ndarray:
    """Generate a tension target curve."""
    t = np.linspace(0, 1, n_chords)
    if shape == "arch":
        return np.sin(np.pi * t)
    elif shape == "rising":
        return t
    elif shape == "falling":
        return 1.0 - t
    elif shape == "wave":
        return 0.5 * (1 + np.sin(2 * np.pi * t))
    return np.ones(n_chords) * 0.5


def generate_tonal_plan(n_chords: int = 16, key: int = 0,
                        shape: str = "arch", seed: int = 42) -> Dict[str, np.ndarray]:
    """Generate a chord progression following a tension/release arc."""
    rng = np.random.RandomState(seed)
    target = tension_arc(n_chords, shape)
    chords = np.zeros(n_chords, dtype=int)
    tensions = np.zeros(n_chords)
    chords[0] = 0
    chords[-1] = 0
    if n_chords > 1:
        chords[-2] = 4
    for i in range(1, n_chords - 2):
        best_chord = 0
        best_diff = float('inf')
        candidates = list(range(7))
        rng.shuffle(candidates)
        for c in candidates:
            diff = abs(CHORD_TENSION[c] - target[i])
            if diff < best_diff:
                best_diff = diff
                best_chord = c
        chords[i] = best_chord
    for i in range(n_chords):
        tensions[i] = CHORD_TENSION[chords[i]]
    degrees = (chords + key) % 7
    return {
        "degrees": degrees,
        "chords": chords,
        "tensions": tensions,
        "target_tension": target,
        "key": key,
    }


def insert_cadences(plan: Dict[str, np.ndarray],
                    cadence_positions: List[Tuple[int, str]]) -> Dict[str, np.ndarray]:
    """Insert cadential patterns at specified positions."""
    chords = plan["chords"].copy()
    for pos, cad_type in cadence_positions:
        pattern = CADENCE_TYPES.get(cad_type, [4, 0])
        for k, deg in enumerate(pattern):
            idx = pos - len(pattern) + 1 + k
            if 0 <= idx < len(chords):
                chords[idx] = deg
    plan["chords"] = chords
    plan["tensions"] = np.array([CHORD_TENSION[c] for c in chords])
    return plan


def modulate(plan: Dict[str, np.ndarray], target_key: int, pivot_pos: int) -> Dict[str, np.ndarray]:
    """Modulate to a new key at the pivot position."""
    chords = plan["chords"].copy()
    key_shift = target_key - plan["key"]
    for i in range(pivot_pos, len(chords)):
        chords[i] = (chords[i] + key_shift) % 7
    plan["chords"] = chords
    return plan


# ---- §4 Groove generator ----

@dataclass
class GrooveProfile:
    """Microtiming and velocity groove profile."""
    timing_offsets: np.ndarray
    velocities: np.ndarray
    swing_ratio: float = 0.5


def swing_groove(n_beats: int, subdivisions: int = 2,
                 swing: float = 0.6, seed: int = 42) -> GrooveProfile:
    """Generate swing groove with microtiming deviations."""
    rng = np.random.RandomState(seed)
    n_events = n_beats * subdivisions
    offsets = np.zeros(n_events)
    velocities = np.ones(n_events)
    for i in range(n_events):
        if i % subdivisions == 1:
            offsets[i] = (swing - 0.5) * (1.0 / subdivisions)
        offsets[i] += rng.randn() * 0.005
    for i in range(n_events):
        if i % subdivisions == 0:
            velocities[i] = 0.9 + rng.rand() * 0.1
        else:
            velocities[i] = 0.5 + rng.rand() * 0.3
    return GrooveProfile(timing_offsets=offsets, velocities=velocities, swing_ratio=swing)


def humanize(onsets: np.ndarray, velocities: np.ndarray,
             timing_std: float = 0.01, velocity_std: float = 0.05,
             seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Add controlled random deviations to onsets and velocities."""
    rng = np.random.RandomState(seed)
    h_onsets = onsets + rng.randn(len(onsets)) * timing_std
    h_velocities = np.clip(velocities + rng.randn(len(velocities)) * velocity_std, 0.0, 1.0)
    return h_onsets, h_velocities


def push_pull(onsets: np.ndarray, amount: float = 0.01, period: int = 4) -> np.ndarray:
    """Apply push/pull feel: alternate rushing and dragging by period."""
    out = onsets.copy()
    for i in range(len(out)):
        phase = (i % period) / period
        out[i] += amount * np.sin(2 * np.pi * phase)
    return out


# ---- §5 Euclidean rhythm extensions ----

def euclidean_rhythm(pulses: int, steps: int) -> np.ndarray:
    """Bjorklund's algorithm for Euclidean rhythms."""
    if steps <= 0:
        return np.array([], dtype=int)
    if pulses >= steps:
        return np.ones(steps, dtype=int)
    if pulses <= 0:
        return np.zeros(steps, dtype=int)
    groups: List[List[int]] = [[1]] * pulses + [[0]] * (steps - pulses)
    while True:
        n_full = min(len([g for g in groups if len(g) > 0 and g[0] == 1]),
                     len([g for g in groups if len(g) > 0 and g[0] == 0]))
        if n_full <= 1:
            break
        ones = [g for g in groups if g[0] == 1]
        zeros = [g for g in groups if g[0] == 0]
        merged = []
        for i in range(min(len(ones), len(zeros))):
            merged.append(ones[i] + zeros[i])
        remainder = ones[len(zeros):] + zeros[len(ones):]
        groups = merged + remainder
    result = []
    for g in groups:
        result.extend(g)
    return np.array(result[:steps], dtype=int)


def rotate_pattern(pattern: np.ndarray, rotation: int) -> np.ndarray:
    """Rotate a rhythm pattern (circular shift)."""
    return np.roll(pattern, -rotation)


def polymetric_combine(patterns: List[np.ndarray]) -> np.ndarray:
    """Combine multiple Euclidean patterns via LCM-length superposition."""
    if not patterns:
        return np.array([], dtype=int)
    lengths = [len(p) for p in patterns]
    lcm = lengths[0]
    for l in lengths[1:]:
        lcm = lcm * l // np.gcd(lcm, l)
    combined = np.zeros(lcm, dtype=int)
    for p in patterns:
        stretched = np.tile(p, lcm // len(p))
        combined = np.maximum(combined, stretched[:lcm])
    return combined


def optimal_rotation(pattern: np.ndarray) -> Tuple[int, float]:
    """Find the rotation maximizing onset-weighted syncopation."""
    best_rot = 0
    best_score = -np.inf
    n = len(pattern)
    for r in range(n):
        rotated = rotate_pattern(pattern, r)
        score = 0.0
        for i in range(n):
            if rotated[i] == 1:
                metric_weight = 1.0 / (1 + (i % 4))
                if i % 2 == 1:
                    score += metric_weight
        if score > best_score:
            best_score = score
            best_rot = r
    return best_rot, float(best_score)


def necklace_class(pattern: np.ndarray) -> np.ndarray:
    """Canonical (lexicographically smallest) rotation — necklace representative."""
    n = len(pattern)
    best = pattern.copy()
    for r in range(1, n):
        rotated = rotate_pattern(pattern, r)
        for i in range(n):
            if rotated[i] < best[i]:
                best = rotated.copy()
                break
            elif rotated[i] > best[i]:
                break
    return best


# ---- §6 Motivic development ----

def augmentation(motif: np.ndarray, factor: int = 2) -> np.ndarray:
    """Rhythmic augmentation: multiply durations by factor."""
    return np.repeat(motif, factor)


def diminution(motif: np.ndarray, factor: int = 2) -> np.ndarray:
    """Rhythmic diminution: take every factor-th element."""
    return motif[::factor]


def retrograde(motif: np.ndarray) -> np.ndarray:
    """Reverse the motif."""
    return motif[::-1].copy()


def inversion(motif: np.ndarray, axis: Optional[float] = None) -> np.ndarray:
    """Pitch inversion around an axis (default: first note)."""
    if axis is None:
        axis = float(motif[0])
    return 2.0 * axis - motif


def retrograde_inversion(motif: np.ndarray, axis: Optional[float] = None) -> np.ndarray:
    """Retrograde of the inversion."""
    return retrograde(inversion(motif, axis))


def transposition(motif: np.ndarray, semitones: int) -> np.ndarray:
    """Transpose motif by given semitones."""
    return motif + semitones


def fragmentation(motif: np.ndarray, n_fragments: int = 2) -> List[np.ndarray]:
    """Split motif into n equal fragments."""
    return [arr.copy() for arr in np.array_split(motif, n_fragments)]


def sequencing(motif: np.ndarray, intervals: List[int]) -> np.ndarray:
    """Repeat motif at successive transposition intervals."""
    parts = [motif + iv for iv in intervals]
    return np.concatenate(parts)


def rhythmic_displacement(motif: np.ndarray, offset: int) -> np.ndarray:
    """Shift motif onset by offset positions (zero-pad at start)."""
    if offset <= 0:
        return motif.copy()
    return np.concatenate([np.zeros(offset), motif])


def develop_motif(motif: np.ndarray, transformations: List[str],
                  params: Optional[Dict] = None) -> np.ndarray:
    """Apply a chain of named transformations to a motif."""
    if params is None:
        params = {}
    result = motif.copy()
    for t in transformations:
        if t == "retrograde":
            result = retrograde(result)
        elif t == "inversion":
            result = inversion(result, params.get("axis"))
        elif t == "augmentation":
            result = augmentation(result, params.get("factor", 2))
        elif t == "diminution":
            result = diminution(result, params.get("factor", 2))
        elif t == "transposition":
            result = transposition(result, params.get("semitones", 0))
        elif t == "retrograde_inversion":
            result = retrograde_inversion(result, params.get("axis"))
        elif t == "fragmentation":
            frags = fragmentation(result, params.get("n_fragments", 2))
            result = frags[0] if frags else result
    return result


# ---- §7 Generation pipeline ----

@dataclass
class GenerationResult:
    """Result of the full generation pipeline."""
    tonal_plan: Dict[str, np.ndarray]
    rhythm: np.ndarray
    melody: np.ndarray
    groove: GrooveProfile
    developed_motif: Optional[np.ndarray] = None


class GenerationPipeline:
    """Chain tonal plan → rhythm → melody → groove into a full generation pipeline."""

    def __init__(self, n_bars: int = 8, beats_per_bar: int = 4,
                 subdivisions: int = 2, key: int = 0, seed: int = 42):
        self.n_bars = n_bars
        self.beats_per_bar = beats_per_bar
        self.subdivisions = subdivisions
        self.key = key
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def generate(self, tension_shape: str = "arch",
                 swing: float = 0.55,
                 motif_length: int = 4) -> GenerationResult:
        """Run the full generation pipeline."""
        n_chords = self.n_bars * 2
        plan = generate_tonal_plan(n_chords, key=self.key,
                                   shape=tension_shape, seed=self.seed)

        n_beats = self.n_bars * self.beats_per_bar
        pulses = int(n_beats * 0.6)
        rhythm = euclidean_rhythm(pulses, n_beats * self.subdivisions)

        scale = np.array([0, 2, 4, 5, 7, 9, 11])
        melody = np.zeros(len(rhythm))
        current_pitch = 60
        for i in range(len(rhythm)):
            if rhythm[i] == 1:
                chord_idx = min(i * n_chords // len(rhythm), n_chords - 1)
                root = plan["chords"][chord_idx]
                chord_tones = (scale[root] + np.array([0, 4, 7])) % 12
                target = chord_tones[self.rng.randint(len(chord_tones))]
                current_pitch = 60 + target + 12 * self.rng.choice([-1, 0, 1], p=[0.1, 0.7, 0.2])
            melody[i] = current_pitch

        groove = swing_groove(n_beats, self.subdivisions, swing, self.seed)

        motif = melody[:motif_length * self.subdivisions]
        if len(motif) > 0:
            developed = develop_motif(motif, ["transposition", "retrograde"],
                                      {"semitones": 5})
        else:
            developed = None

        return GenerationResult(
            tonal_plan=plan,
            rhythm=rhythm,
            melody=melody,
            groove=groove,
            developed_motif=developed,
        )


# ---- Self-test ----

if __name__ == "__main__":
    config = TransformerGenConfig(vocab_size=32, d_model=16, n_heads=2,
                                  n_layers=1, d_ff=32, max_seq_len=64)
    weights = init_transformer_weights(config)
    prompt = np.array([0, 1, 2, 3])
    seq = generate_sequence(config, weights, prompt, length=8)
    assert len(seq) == 12
    print(f"Generated sequence: {seq}")

    plan = generate_tonal_plan(16, shape="arch")
    assert len(plan["chords"]) == 16
    assert plan["chords"][0] == 0 and plan["chords"][-1] == 0
    print(f"Tonal plan: {plan['chords']}")

    e = euclidean_rhythm(5, 8)
    assert e.sum() == 5 and len(e) == 8
    p1, p2 = euclidean_rhythm(3, 8), euclidean_rhythm(5, 12)
    poly = polymetric_combine([p1, p2])
    assert len(poly) == 24
    print(f"Euclidean(5,8): {e}, polymetric len: {len(poly)}")

    motif = np.array([60, 64, 67, 72])
    dev = develop_motif(motif, ["transposition", "inversion", "retrograde"],
                        {"semitones": 2, "axis": 65})
    assert len(dev) == len(motif)
    print(f"Motif: {motif} → developed: {dev}")

    pipe = GenerationPipeline(n_bars=4, seed=42)
    result = pipe.generate()
    assert len(result.rhythm) > 0
    assert len(result.melody) == len(result.rhythm)
    print(f"Pipeline: {result.rhythm.sum()} onsets in {len(result.rhythm)} steps")
    print("deep_generation OK")
