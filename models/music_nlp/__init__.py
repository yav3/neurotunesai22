from __future__ import annotations

from models.music_nlp.lyrics_features import LyricsAnalyzer, LyricsFeatures
from models.music_nlp.lyrics_emotion import LyricsEmotionProfile, LyricsEmotionAnalyzer
from models.music_nlp.multilingual import MultilingualProcessor
from models.music_nlp.prosody import ProsodyFeatures, ProsodyAnalyzer
from models.music_nlp.scraper_registry import DataSource

__all__ = [
    "LyricsAnalyzer", "LyricsFeatures",
    "LyricsEmotionProfile", "LyricsEmotionAnalyzer",
    "MultilingualProcessor",
    "ProsodyFeatures", "ProsodyAnalyzer",
    "DataSource",
]
