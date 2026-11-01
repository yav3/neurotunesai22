"""Music audio fingerprinting and identification algorithms."""
from models.music_fingerprinting.chromaprint import SpectralFingerprinter
from models.music_fingerprinting.landmarks import ConstellationMap, LandmarkIndex
from models.music_fingerprinting.matching import FingerprintMatcher, align_matches, score_match
from models.music_fingerprinting.hashing import MinHashFingerprint, LSHIndex

from models.music_fingerprinting.spectral_peaks import (
    Peak, FingerprintHash, SpectralPeakExtractor,
)

from models.music_fingerprinting.audio_codec import (
    CodecType, CodecArtifact, CodecRobustnessTest,
    simulate_codec, codec_distortion, analyze_codec_artifacts,
    quantize, pre_echo_simulation, CODEC_PARAMS,
)
from models.music_fingerprinting.robustness import (
    add_white_noise, time_stretch, pitch_shift_semitones,
    apply_lowpass, mp3_simulation, DegradationResult,
    evaluate_robustness,
)

__all__ = [
    "SpectralFingerprinter",
    "ConstellationMap",
    "LandmarkIndex",
    "FingerprintMatcher",
    "align_matches",
    "score_match",
    "MinHashFingerprint",
    "LSHIndex",
    "add_white_noise", "time_stretch", "pitch_shift_semitones",
    "apply_lowpass", "mp3_simulation", "DegradationResult",
    "evaluate_robustness",
    "CodecType", "CodecArtifact", "CodecRobustnessTest",
    "simulate_codec", "codec_distortion", "analyze_codec_artifacts",
    "quantize", "pre_echo_simulation", "CODEC_PARAMS",
    "AudioHash", "perceptual_hash", "hash_similarity",
    "content_id", "HashIndex", "bark_bands", "spectral_flatness",
    "Peak", "FingerprintHash", "SpectralPeakExtractor",
    "WatermarkConfig", "WatermarkResult",
    "spread_spectrum_embed", "spread_spectrum_extract",
    "qim_embed", "qim_extract",
    "watermark_robustness", "payload_capacity",
    "HashAlgorithm", "FingerprintConfig", "AudioFingerprint",
    "rf_bark_band_energies", "wavelet_fingerprint",
    "modulation_spectrum_hash", "chromaprint_style",
    "generate_fingerprint", "rf_time_offset_histogram",
    "fingerprint_match_score", "batch_match",
    "rf_false_positive_rate", "noise_robustness_test", "compression_robustness",
    "PerceptualHash", "HashMatch",
    "ph_bark_scale", "bark_band_edges", "ph_bark_band_energies",
    "compute_frame_hash", "compute_perceptual_hash",
    "ph_hamming_distance", "normalized_hamming", "ph_bit_error_rate",
    "ph_match_hashes", "find_offset",
    "tempo_invariant_hash", "ph_hash_similarity",
    "locality_sensitive_hash", "hash_database_search", "sub_fingerprint_sequence",
]

from models.music_fingerprinting.audio_hash import (
    AudioHash, perceptual_hash, hash_similarity,
    content_id, HashIndex, bark_bands, spectral_flatness,
)
from models.music_fingerprinting.robust_matching import (
    SpectralPeak as SpectralPeak2, find_spectral_peaks,
    generate_fingerprints, hash_fingerprint,
    FingerprintDB, time_offset_histogram, match_score as fp_match_score,
)
from models.music_fingerprinting.query_by_humming import (
    MelodyContour, MatchResult,
    extract_pitch_contour, pitch_to_midi_cents, normalize_contour,
    dtw_distance, subsequence_dtw,
    build_melody_database, search_melody_database, contour_similarity,
)
from models.music_fingerprinting.robust_hashing import (
    RobustHash, MatchResult as RobustMatchResult,
    bark_band_energies, energy_difference_hash,
    hamming_distance, bit_error_rate, majority_vote_hash,
    repetition_encode, repetition_decode,
    compute_robust_hash, match_hashes, sliding_match, hash_to_hex,
)
from models.music_fingerprinting.watermarking import (
    WatermarkConfig, WatermarkResult,
    spread_spectrum_embed, spread_spectrum_extract,
    qim_embed, qim_extract,
    watermark_robustness, payload_capacity,
)
from models.music_fingerprinting.acoustic_id import (
    IdentificationResult, AcousticProfile, compute_acoustic_profile,
    profile_distance, IdentificationIndex, time_coherence_check,
    confidence_score, spectral_contrast_features,
)
from models.music_fingerprinting.robust_fingerprint import (
    HashAlgorithm, FingerprintConfig, AudioFingerprint,
    bark_band_energies as rf_bark_band_energies,
    wavelet_fingerprint, modulation_spectrum_hash, chromaprint_style,
    generate_fingerprint,
    time_offset_histogram as rf_time_offset_histogram,
    fingerprint_match_score, batch_match,
    false_positive_rate as rf_false_positive_rate,
    noise_robustness_test, compression_robustness,
)
from models.music_fingerprinting.perceptual_hash import (
    PerceptualHash, HashMatch,
    bark_scale as ph_bark_scale,
    bark_band_edges,
    bark_band_energies as ph_bark_band_energies,
    compute_frame_hash, compute_perceptual_hash,
    hamming_distance as ph_hamming_distance,
    normalized_hamming,
    bit_error_rate as ph_bit_error_rate,
    match_hashes as ph_match_hashes,
    find_offset,
    tempo_invariant_hash,
    hash_similarity as ph_hash_similarity,
    locality_sensitive_hash, hash_database_search,
    sub_fingerprint_sequence,
)
