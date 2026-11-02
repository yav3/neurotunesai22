from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class DataSource:
    name: str
    url: str
    license: str
    languages: tuple[str, ...]
    content_type: str  # "lyrics", "poetry", "both", "midi", "metadata"
    api_available: bool = False
    notes: str = ""


OPEN_SOURCE_REGISTRY: list[DataSource] = [
    # --- Poetry ---
    DataSource(
        name="Project Gutenberg",
        url="https://www.gutenberg.org",
        license="Public Domain",
        languages=("en", "fr", "de", "it", "es", "pt", "fi", "nl", "zh"),
        content_type="poetry",
        api_available=True,
        notes="60,000+ free ebooks. Poetry collections across many languages. "
              "Bulk download available. Robot-friendly.",
    ),
    DataSource(
        name="Poetry Foundation",
        url="https://www.poetryfoundation.org",
        license="Various (check per poem)",
        languages=("en",),
        content_type="poetry",
        api_available=False,
        notes="Curated English poetry archive. Major canonical poets. "
              "No bulk API; requires careful scraping with rate limits.",
    ),
    DataSource(
        name="Allpoetry",
        url="https://allpoetry.com",
        license="User-contributed (mixed)",
        languages=("en", "es", "fr", "de"),
        content_type="poetry",
        api_available=False,
        notes="User-contributed poetry. Large volume, variable quality. "
              "Licensing per contributor — check terms.",
    ),
    DataSource(
        name="Wikisource",
        url="https://wikisource.org",
        license="CC BY-SA / Public Domain",
        languages=("en", "fr", "de", "it", "es", "pt", "ru", "zh", "ja", "ar", "hi", "ko"),
        content_type="poetry",
        api_available=True,
        notes="Multilingual public-domain literature and poetry. "
              "MediaWiki API for structured access. ~5M pages total.",
    ),
    DataSource(
        name="Chinese Poetry Dataset",
        url="https://github.com/chinese-poetry/chinese-poetry",
        license="MIT",
        languages=("zh",),
        content_type="poetry",
        api_available=False,
        notes="5.5M+ Chinese poems: Tang (48K), Song (254K), Yuan (4K), "
              "Ming, Qing, ShiJing, ChuCi, Wudai, HuaJian, NanTang. "
              "JSON format with author, dynasty, paragraphs, notes.",
    ),
    DataSource(
        name="Sanskrit Heritage",
        url="https://sanskrit.inria.fr",
        license="Academic",
        languages=("sa",),
        content_type="poetry",
        api_available=False,
        notes="Classical Sanskrit poetry and literature. "
              "Metre analysis tools. Academic use.",
    ),
    DataSource(
        name="Aozora Bunko",
        url="https://www.aozora.gr.jp",
        license="Public Domain (Japanese copyright expired)",
        languages=("ja",),
        content_type="poetry",
        api_available=True,
        notes="16,000+ Japanese literary works. Includes tanka, haiku, "
              "modern poetry. XHTML format with furigana markup.",
    ),
    DataSource(
        name="Korean Classic Literature DB (KLDB)",
        url="https://www.krpia.co.kr",
        license="Academic / Government",
        languages=("ko",),
        content_type="poetry",
        api_available=False,
        notes="Classical Korean sijo, gasa, hansi. "
              "Government-funded database. Academic access.",
    ),
    DataSource(
        name="Arabic Poetry Corpus",
        url="https://www.aldiwan.net",
        license="Various (check per source)",
        languages=("ar",),
        content_type="poetry",
        api_available=False,
        notes="Pre-Islamic to modern Arabic poetry. Multiple academic "
              "datasets exist (APCD, Poem Comprehensive Dataset). "
              "Al-Diwan has 80K+ poems with metre and rhyme annotations.",
    ),
    DataSource(
        name="Kavitakosh",
        url="https://www.kavitakosh.org",
        license="CC BY-SA",
        languages=("hi", "ur", "bn", "gu", "mr", "pa", "ta", "te"),
        content_type="poetry",
        api_available=False,
        notes="Indian poetry in Hindi and other Indian languages. "
              "Wiki-style, ~100K poems. CC BY-SA license.",
    ),
    DataSource(
        name="Russian Poetry Corpus (RPC)",
        url="https://ruverse.ru",
        license="Academic",
        languages=("ru",),
        content_type="poetry",
        api_available=False,
        notes="Annotated Russian poetry with metre, rhyme scheme, stanza "
              "structure. Academic access. ~50K poems.",
    ),
    # --- Lyrics ---
    DataSource(
        name="Genius",
        url="https://genius.com",
        license="API Terms (non-commercial research)",
        languages=("en", "es", "fr", "de", "pt", "ja", "ko"),
        content_type="lyrics",
        api_available=True,
        notes="Largest lyrics database. API provides metadata + referents; "
              "full lyrics require page scraping. Rate-limited. "
              "Community annotations add interpretive value.",
    ),
    DataSource(
        name="Musixmatch",
        url="https://www.musixmatch.com",
        license="API Terms (30% lyrics snippet)",
        languages=("en", "es", "fr", "de", "it", "pt", "ja", "ko", "zh", "ar", "ru"),
        content_type="lyrics",
        api_available=True,
        notes="14M+ lyrics. API returns 30% snippet (commercial license "
              "required for full text). Best multilingual coverage. "
              "Word-level sync timestamps available.",
    ),
    DataSource(
        name="LAKH MIDI Dataset",
        url="https://colinraffel.com/projects/lmd/",
        license="Various (matched to MSD)",
        languages=(),
        content_type="midi",
        api_available=False,
        notes="176,581 unique MIDI files matched to Million Song Dataset "
              "entries. Useful for melody/harmony/rhythm analysis. "
              "No lyrics but rich structural music data.",
    ),
    DataSource(
        name="Million Song Dataset",
        url="http://millionsongdataset.com",
        license="Academic (free for research)",
        languages=(),
        content_type="metadata",
        api_available=False,
        notes="1M tracks with audio features (tempo, key, loudness, timbre), "
              "artist metadata, tags. No audio or lyrics directly. "
              "MSD + musiXmatch = lyrics-features alignment.",
    ),
    DataSource(
        name="OpenSubtitles",
        url="https://www.opensubtitles.org",
        license="Various (user-contributed)",
        languages=("en", "es", "fr", "de", "it", "pt", "ru", "zh", "ja", "ko", "ar", "hi"),
        content_type="both",
        api_available=True,
        notes="Dialogue subtitles in 60+ languages. Useful for "
              "spoken-word/rap cadence analysis, multilingual sentiment. "
              "OPUS parallel corpus available for aligned text.",
    ),
    DataSource(
        name="Deezer Mood Detection Dataset",
        url="https://github.com/deezer/deezer_mood_detection_dataset",
        license="CC BY-NC-SA 4.0",
        languages=("en",),
        content_type="metadata",
        api_available=False,
        notes="18K tracks with valence/arousal annotations. "
              "Audio features pre-extracted. Research use only.",
    ),
    DataSource(
        name="DEAM (MediaEval Database for Emotional Analysis in Music)",
        url="https://cvml.unige.ch/databases/DEAM/",
        license="Academic",
        languages=(),
        content_type="metadata",
        api_available=False,
        notes="2058 song excerpts with continuous valence/arousal annotations. "
              "Benchmark for music emotion recognition (MER). "
              "Per-second annotations available.",
    ),
    DataSource(
        name="PMEmo",
        url="https://github.com/HuiZhangDB/PMEmo",
        license="Academic",
        languages=(),
        content_type="metadata",
        api_available=False,
        notes="794 songs with both static and dynamic V-A annotations, "
              "plus EDA physiological signals. MER benchmark.",
    ),
]


def get_sources_by_language(lang_code: str) -> list[DataSource]:
    return [s for s in OPEN_SOURCE_REGISTRY if lang_code in s.languages]


def get_sources_by_license(license_type: str) -> list[DataSource]:
    lt = license_type.lower()
    return [s for s in OPEN_SOURCE_REGISTRY if lt in s.license.lower()]


def get_poetry_sources() -> list[DataSource]:
    return [s for s in OPEN_SOURCE_REGISTRY if s.content_type == "poetry"]


def get_lyrics_sources() -> list[DataSource]:
    return [s for s in OPEN_SOURCE_REGISTRY if s.content_type == "lyrics"]


def get_midi_sources() -> list[DataSource]:
    return [s for s in OPEN_SOURCE_REGISTRY if s.content_type == "midi"]


def get_emotion_benchmark_sources() -> list[DataSource]:
    return [
        s for s in OPEN_SOURCE_REGISTRY
        if "emotion" in s.name.lower()
        or "mood" in s.name.lower()
        or "valence" in s.notes.lower()
    ]
