"""Music improvisation: scale navigation, motivic development, call-response, jazz voicings."""

from .scale_navigation import (
    ScaleMode, scale_degrees, chord_tones_in_scale,
    available_tensions, approach_note_options,
    target_note_gravity,
)
from .motivic_development import (
    Motif, transpose_motif, invert_motif, retrograde_motif,
    augment_motif, diminish_motif, sequence_motif,
    motivic_similarity,
)
from .call_response import (
    call_response_score, rhythmic_complement,
    melodic_answer, phrase_balance,
)
from .jazz_voicings import (
    VoicingType, voice_chord, rootless_voicing,
    drop2_voicing, shell_voicing, quartal_voicing,
    voice_leading_cost,
)

__all__ = [
    "ScaleMode", "scale_degrees", "chord_tones_in_scale",
    "available_tensions", "approach_note_options", "target_note_gravity",
    "Motif", "transpose_motif", "invert_motif", "retrograde_motif",
    "augment_motif", "diminish_motif", "sequence_motif", "motivic_similarity",
    "call_response_score", "rhythmic_complement", "melodic_answer", "phrase_balance",
    "VoicingType", "voice_chord", "rootless_voicing",
    "drop2_voicing", "shell_voicing", "quartal_voicing", "voice_leading_cost",
]
