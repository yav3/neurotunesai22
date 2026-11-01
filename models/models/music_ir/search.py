"""Music search: inverted index, track indexing, query expansion, ranking."""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np


class InvertedIndex:
    """Token-to-document inverted index with TF-IDF ranking."""

    def __init__(self) -> None:
        self._index: dict[str, set[str]] = defaultdict(set)
        self._doc_tokens: dict[str, list[str]] = {}

    def add(self, doc_id: str, tokens: list[str]) -> None:
        lower = [t.lower() for t in tokens]
        self._doc_tokens[doc_id] = lower
        for tok in set(lower):
            self._index[tok].add(doc_id)

    def search(self, query_tokens: list[str], top_k: int = 10) -> list[tuple[str, float]]:
        query = [t.lower() for t in query_tokens]
        if not query or not self._doc_tokens:
            return []
        n_docs = len(self._doc_tokens)
        scores: dict[str, float] = Counter()
        for tok in query:
            postings = self._index.get(tok, set())
            if not postings:
                continue
            idf = np.log(n_docs / len(postings)) + 1.0
            for doc_id in postings:
                tf = self._doc_tokens[doc_id].count(tok) / max(len(self._doc_tokens[doc_id]), 1)
                scores[doc_id] += tf * idf
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

    @property
    def num_docs(self) -> int:
        return len(self._doc_tokens)

    @property
    def vocab_size(self) -> int:
        return len(self._index)


@dataclass
class TrackIndex:
    """Composite track index: text fields + numeric features."""

    _text_index: InvertedIndex = field(default_factory=InvertedIndex)
    _features: dict[str, dict[str, float]] = field(default_factory=dict)

    def index_track(self, track_id: str, metadata: dict[str, Any]) -> None:
        tokens: list[str] = []
        for key in ("title", "artist", "album", "genre", "tags"):
            val = metadata.get(key, "")
            if isinstance(val, str):
                tokens.extend(val.split())
            elif isinstance(val, (list, tuple)):
                tokens.extend(str(v) for v in val)
        self._text_index.add(track_id, tokens)
        feats: dict[str, float] = {}
        for key in ("bpm", "valence", "arousal", "energy", "danceability"):
            if key in metadata:
                feats[key] = float(metadata[key])
        self._features[track_id] = feats

    def text_search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        return self._text_index.search(query.split(), top_k)

    def feature_search(
        self,
        target: dict[str, float],
        top_k: int = 10,
    ) -> list[tuple[str, float]]:
        results: list[tuple[str, float]] = []
        for track_id, feats in self._features.items():
            dist = 0.0
            count = 0
            for k, v in target.items():
                if k in feats:
                    dist += (feats[k] - v) ** 2
                    count += 1
            if count > 0:
                dist = np.sqrt(dist / count)
                results.append((track_id, dist))
        results.sort(key=lambda x: x[1])
        return results[:top_k]

    @property
    def num_tracks(self) -> int:
        return self._text_index.num_docs


class QueryExpander:
    """Expand search queries with synonyms and related terms."""

    SYNONYMS: dict[str, list[str]] = {
        "happy": ["joyful", "cheerful", "upbeat", "euphoric"],
        "sad": ["melancholic", "somber", "mournful", "blue"],
        "energetic": ["upbeat", "driving", "powerful", "intense"],
        "calm": ["peaceful", "serene", "relaxing", "tranquil"],
        "rock": ["alternative", "indie", "punk", "metal"],
        "electronic": ["edm", "techno", "house", "trance"],
        "jazz": ["swing", "bebop", "fusion", "smooth"],
        "classical": ["orchestral", "chamber", "symphonic", "baroque"],
        "fast": ["uptempo", "quick", "rapid"],
        "slow": ["downtempo", "ballad", "adagio"],
    }

    @classmethod
    def expand(cls, tokens: list[str], max_expansions: int = 3) -> list[str]:
        expanded = list(tokens)
        for tok in tokens:
            synonyms = cls.SYNONYMS.get(tok.lower(), [])
            expanded.extend(synonyms[:max_expansions])
        return expanded


class SearchRanker:
    """Re-rank search results by combining text relevance with feature proximity."""

    def __init__(self, text_weight: float = 0.6, feature_weight: float = 0.4):
        self.text_weight = text_weight
        self.feature_weight = feature_weight

    def rerank(
        self,
        text_results: list[tuple[str, float]],
        feature_results: list[tuple[str, float]],
        top_k: int = 10,
    ) -> list[tuple[str, float]]:
        text_scores = {tid: score for tid, score in text_results}
        feat_dists = {tid: dist for tid, dist in feature_results}

        max_text = max((s for _, s in text_results), default=1.0) or 1.0
        max_feat = max((d for _, d in feature_results), default=1.0) or 1.0

        all_ids = set(text_scores) | set(feat_dists)
        combined: list[tuple[str, float]] = []
        for tid in all_ids:
            t_score = text_scores.get(tid, 0.0) / max_text
            f_score = 1.0 - feat_dists.get(tid, max_feat) / max_feat
            combined.append((tid, self.text_weight * t_score + self.feature_weight * f_score))

        combined.sort(key=lambda x: x[1], reverse=True)
        return combined[:top_k]
