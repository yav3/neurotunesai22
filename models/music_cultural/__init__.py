"""Music cultural analysis: scales, tunings, rhythmic systems, form, ornamentation across world traditions."""
from models.music_cultural.scales import (
    CulturalScale, ScaleDatabase,
    MAQAM_SCALES, RAGA_SCALES, GAMELAN_SCALES,
    interval_pattern, scale_similarity,
)
from models.music_cultural.tuning import (
    TuningSystem as TuningSystemBase, EqualTemperament as EqualTemperamentBase,
    JustIntonation as JustIntonationBase, PythagoreanTuning as PythagoreanTuningBase,
    MeantoneTemperament as MeantoneTemperamentBase, WellTemperament as WellTemperamentBase,
    cents_to_ratio, ratio_to_cents, frequency_ratio,
)
from models.music_cultural.rhythm import *  # noqa: F401,F403

# Deep world-scale library (35+ scales)
from models.music_cultural.world_scales import (
    WorldScale, ScaleMatcher, WORLD_SCALE_REGISTRY,
    MAQAM_RAST, MAQAM_BAYATI, MAQAM_HIJAZ, MAQAM_SABA,
    MAQAM_NAHAWAND, MAQAM_KURD, MAQAM_AJAM,
    RAGA_BILAWAL, RAGA_KAFI, RAGA_BHAIRAVI, RAGA_YAMAN,
    RAGA_KHAMAJ, RAGA_ASAVARI, RAGA_MARWA, RAGA_TODI, RAGA_PURVI,
    HIRAJOSHI, KUMOI, IN_SCALE, YO_SCALE, IWATO, MIYAKO_BUSHI,
    GONG, SHANG, JIAO, ZHI, YU,
    SLENDRO, PELOG,
    TIZITA_MAJOR, TIZITA_MINOR, BATI_MAJOR, BATI_MINOR, AMBASSEL,
    BLUES, WHOLE_TONE, HUNGARIAN_MINOR, PERSIAN, BYZANTINE,
)

# Deep tuning system library
from models.music_cultural.tuning_systems import (
    CentConverter, TuningSystem, EqualTemperament,
    TET_12, TET_19, TET_24, TET_31, TET_53,
    JustIntonation, PythagoreanTuning, MeantoneTemperament,
    QUARTER_COMMA_MEANTONE, THIRD_COMMA_MEANTONE, SIXTH_COMMA_MEANTONE,
    WellTemperament, TuningComparator,
)

# Deep rhythm system library
from models.music_cultural.rhythm_systems import (
    TimeSystem, Tala, Usul, ClaveSyncopation,
    TINTAL, JHAPTAL, RUPAK, EKTAL, DADRA, KEHERWA, TALA_REGISTRY,
    DUYEK, AKSAK, CIFTETELLI, KARSILAMA, USUL_REGISTRY,
    SON_32, SON_23, RUMBA, BOSSA_NOVA, CLAVE_REGISTRY,
    EuclideanRhythm, PolyrhythmGenerator,
)

# Musical form analysis (22 forms)
from models.music_cultural.form_analysis import (
    MusicalForm, FORM_LIBRARY, FormMatcher,
)

# Ornamentation library (28 ornaments, 5 traditions)
from models.music_cultural.ornamentation import (
    Ornament, ORNAMENT_LIBRARY, OrnamentDetector,
    TRILL, MORDENT, UPPER_MORDENT, TURN, APPOGGIATURA, ACCIACCATURA,
    GRUPPETTO, GLISSANDO, TREMOLO,
    GAMAK, MEEND, KAN, MURKI, KHATKA, ANDOLAN,
    RAQS, ZALAQA, RASHA,
    CUT, STRIKE, CRAN, ROLL, SLIDE_IRISH, TREBLE,
    RASGUEADO, PICADO, ALZAPUA, TREMOLO_FLAMENCO,
)

__all__ = [
    # Original scale module
    "CulturalScale", "ScaleDatabase",
    "MAQAM_SCALES", "RAGA_SCALES", "GAMELAN_SCALES",
    "interval_pattern", "scale_similarity",
    # Original tuning module (base classes)
    "TuningSystemBase", "EqualTemperamentBase", "JustIntonationBase",
    "PythagoreanTuningBase", "MeantoneTemperamentBase", "WellTemperamentBase",
    "cents_to_ratio", "ratio_to_cents", "frequency_ratio",
    # World scales (35+)
    "WorldScale", "ScaleMatcher", "WORLD_SCALE_REGISTRY",
    "MAQAM_RAST", "MAQAM_BAYATI", "MAQAM_HIJAZ", "MAQAM_SABA",
    "MAQAM_NAHAWAND", "MAQAM_KURD", "MAQAM_AJAM",
    "RAGA_BILAWAL", "RAGA_KAFI", "RAGA_BHAIRAVI", "RAGA_YAMAN",
    "RAGA_KHAMAJ", "RAGA_ASAVARI", "RAGA_MARWA", "RAGA_TODI", "RAGA_PURVI",
    "HIRAJOSHI", "KUMOI", "IN_SCALE", "YO_SCALE", "IWATO", "MIYAKO_BUSHI",
    "GONG", "SHANG", "JIAO", "ZHI", "YU",
    "SLENDRO", "PELOG",
    "TIZITA_MAJOR", "TIZITA_MINOR", "BATI_MAJOR", "BATI_MINOR", "AMBASSEL",
    "BLUES", "WHOLE_TONE", "HUNGARIAN_MINOR", "PERSIAN", "BYZANTINE",
    # Tuning systems
    "CentConverter", "TuningSystem", "EqualTemperament",
    "TET_12", "TET_19", "TET_24", "TET_31", "TET_53",
    "JustIntonation", "PythagoreanTuning", "MeantoneTemperament",
    "QUARTER_COMMA_MEANTONE", "THIRD_COMMA_MEANTONE", "SIXTH_COMMA_MEANTONE",
    "WellTemperament", "TuningComparator",
    # Rhythm systems
    "TimeSystem", "Tala", "Usul", "ClaveSyncopation",
    "TINTAL", "JHAPTAL", "RUPAK", "EKTAL", "DADRA", "KEHERWA", "TALA_REGISTRY",
    "DUYEK", "AKSAK", "CIFTETELLI", "KARSILAMA", "USUL_REGISTRY",
    "SON_32", "SON_23", "RUMBA", "BOSSA_NOVA", "CLAVE_REGISTRY",
    "EuclideanRhythm", "PolyrhythmGenerator",
    # Form analysis
    "MusicalForm", "FORM_LIBRARY", "FormMatcher",
    # Ornamentation
    "Ornament", "ORNAMENT_LIBRARY", "OrnamentDetector",
    "TRILL", "MORDENT", "UPPER_MORDENT", "TURN", "APPOGGIATURA", "ACCIACCATURA",
    "GRUPPETTO", "GLISSANDO", "TREMOLO",
    "GAMAK", "MEEND", "KAN", "MURKI", "KHATKA", "ANDOLAN",
    "RAQS", "ZALAQA", "RASHA",
    "CUT", "STRIKE", "CRAN", "ROLL", "SLIDE_IRISH", "TREBLE",
    "RASGUEADO", "PICADO", "ALZAPUA", "TREMOLO_FLAMENCO",
]
