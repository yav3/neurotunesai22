"""Music arrangement: orchestration, scoring, voicing, texture, reduction."""

from .orchestration import (
    InstrumentRange, ORCHESTRA_RANGES,
    fits_range, suggest_transposition,
    doubling_score, orchestral_balance,
)
from .scoring import (
    ScoreFormat, reduce_to_piano, expand_lead_sheet,
    part_extraction, transpose_part,
)
from .texture_mapping import (
    TextureType, classify_texture,
    texture_density, recommended_texture,
)
from .reharmonization import (
    tritone_substitution, backdoor_progression,
    modal_interchange, chromatic_mediant,
    reharmonize_melody_note,
)

__all__ = [
    "InstrumentRange", "ORCHESTRA_RANGES",
    "fits_range", "suggest_transposition", "doubling_score", "orchestral_balance",
    "ScoreFormat", "reduce_to_piano", "expand_lead_sheet",
    "part_extraction", "transpose_part",
    "TextureType", "classify_texture", "texture_density", "recommended_texture",
    "tritone_substitution", "backdoor_progression",
    "modal_interchange", "chromatic_mediant", "reharmonize_melody_note",
]
