"""Music composition: counterpoint, form, orchestration, voice leading, Markov."""
from models.music_composition.counterpoint import (
    FirstSpecies, SecondSpecies, ThirdSpecies, FourthSpecies,
    check_parallel_motion, is_consonant, contrary_motion_score,
    classify_motion, motion_profile, generate_first_species,
    CONSONANT_INTERVALS, PERFECT_CONSONANCES,
)
from models.music_composition.form import (
    SongForm, FormGenerator, VariationEngine,
    transpose, invert, retrograde,
)
from models.music_composition.orchestration import (
    InstrumentRange, INSTRUMENTS, Orchestrator,
    fits_range, assign_voices,
)
from models.music_composition.voice_leading import (
    VoiceLeading, VoiceLeader, NeoRiemannian,
    minimal_voice_leading, voice_leading_smoothness,
)
from models.music_composition.markov_composer import (
    PitchMarkov, RhythmMarkov, MarkovComposer,
)
from models.music_composition.texture_analysis import (
    TextureType, TextureProfile, TextureSegment,
    estimate_voice_count, pitch_spread, onset_synchrony,
    rhythmic_independence_score, textural_density,
    classify_texture, analyze_texture, segment_texture,
)
from models.music_composition.rhythm_pattern import (
    RhythmPattern, PolymetricPattern,
    euclidean_rhythm, polyrhythm, swing_quantize,
    syncopation_score, onset_to_ioi, ioi_regularity,
    create_polymetric, groove_template, humanize,
)

from models.music_composition.motivic_transformation import (
    Motif, retrograde as motif_retrograde, inversion as motif_inversion,
    retrograde_inversion, augmentation, diminution, transposition,
    fragmentation, sequence as motif_sequence,
    motivic_similarity, interval_sequence, contour, contour_similarity,
    develop_motif, permute_rhythm, TRANSFORMS as MOTIF_TRANSFORMS,
)
from models.music_composition.cadence import (
    CadenceType, Cadence, detect_cadence, generate_cadence,
    cadential_strength, find_cadences, suggest_cadence,
    is_cadential_64, cadence_rate,
)

__all__ = [
    "FirstSpecies", "SecondSpecies",
    "check_parallel_motion", "is_consonant", "contrary_motion_score",
    "CONSONANT_INTERVALS", "PERFECT_CONSONANCES",
    "ThirdSpecies", "FourthSpecies",
    "classify_motion", "motion_profile", "generate_first_species",
    "SongForm", "FormGenerator", "VariationEngine",
    "transpose", "invert", "retrograde",
    "InstrumentRange", "INSTRUMENTS", "Orchestrator",
    "fits_range", "assign_voices",
    "VoiceLeading", "VoiceLeader", "NeoRiemannian",
    "minimal_voice_leading", "voice_leading_smoothness",
    "PitchMarkov", "RhythmMarkov", "MarkovComposer",
    "RhythmPattern", "PolymetricPattern",
    "euclidean_rhythm", "polyrhythm", "swing_quantize",
    "syncopation_score", "onset_to_ioi", "ioi_regularity",
    "create_polymetric", "groove_template", "humanize",
    "TextureType", "TextureProfile", "TextureSegment",
    "estimate_voice_count", "pitch_spread", "onset_synchrony",
    "rhythmic_independence_score", "textural_density",
    "classify_texture", "analyze_texture", "segment_texture",
    "Motif", "motif_retrograde", "motif_inversion",
    "retrograde_inversion", "augmentation", "diminution", "transposition",
    "fragmentation", "motif_sequence",
    "motivic_similarity", "interval_sequence", "contour", "contour_similarity",
    "develop_motif", "permute_rhythm", "MOTIF_TRANSFORMS",
    "CadenceType", "Cadence", "detect_cadence", "generate_cadence",
    "cadential_strength", "find_cadences", "suggest_cadence",
    "is_cadential_64", "cadence_rate",
]
