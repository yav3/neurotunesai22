"""Named entity recognition for music domain text.

Extracts artist names, song titles, album titles, genre labels, instrument
names, venue/location references, and temporal expressions from unstructured
music text (reviews, liner notes, social media).  Rule-based + statistical
(no external ML frameworks — uses n-gram frequency and gazetteer matching).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Set, Tuple

import numpy as np


class EntityType(Enum):
    ARTIST = auto()
    SONG = auto()
    ALBUM = auto()
    GENRE = auto()
    INSTRUMENT = auto()
    VENUE = auto()
    LABEL = auto()
    KEY_SIGNATURE = auto()
    TEMPO_MARKING = auto()
    TIME_PERIOD = auto()


@dataclass
class MusicEntity:
    text: str
    entity_type: EntityType
    start: int
    end: int
    confidence: float  # 0-1


@dataclass
class NERResult:
    entities: List[MusicEntity]
    text: str
    token_labels: List[Tuple[str, str]]  # (token, BIO label)


GENRE_GAZETTEER: Set[str] = {
    "jazz", "blues", "rock", "pop", "classical", "hip hop", "r&b", "soul",
    "funk", "reggae", "country", "folk", "metal", "punk", "electronic",
    "house", "techno", "ambient", "indie", "alternative", "grunge",
    "ska", "gospel", "bluegrass", "bossa nova", "flamenco", "salsa",
    "cumbia", "merengue", "samba", "afrobeat", "highlife", "soukous",
    "k-pop", "j-pop", "trap", "drill", "grime", "dancehall",
    "dubstep", "drum and bass", "trance", "progressive rock",
    "post-punk", "new wave", "shoegaze", "dream pop", "lo-fi",
    "neo-soul", "trip hop", "downtempo", "breakbeat", "garage",
    "psych rock", "math rock", "post-rock", "emo", "screamo",
    "black metal", "death metal", "doom metal", "stoner rock",
    "noise rock", "industrial", "synthwave", "vaporwave",
}

INSTRUMENT_GAZETTEER: Set[str] = {
    "guitar", "bass", "drums", "piano", "keyboard", "violin", "viola",
    "cello", "flute", "clarinet", "saxophone", "trumpet", "trombone",
    "tuba", "oboe", "bassoon", "harp", "banjo", "mandolin", "ukulele",
    "accordion", "harmonica", "organ", "synthesizer", "theremin",
    "sitar", "tabla", "oud", "kora", "djembe", "mbira", "erhu",
    "shamisen", "koto", "didgeridoo", "bagpipes", "steel drum",
    "marimba", "vibraphone", "xylophone", "timpani", "congas",
    "bongos", "tambourine", "cajon", "electric guitar", "acoustic guitar",
    "bass guitar", "double bass", "french horn", "english horn",
    "piccolo", "recorder", "dulcimer", "zither", "lute", "harpsichord",
}

TEMPO_MARKINGS: Set[str] = {
    "largo", "adagio", "andante", "moderato", "allegro", "vivace",
    "presto", "prestissimo", "lento", "grave", "allegretto",
    "andantino", "larghetto", "accelerando", "ritardando", "rubato",
    "a tempo", "rallentando",
}

KEY_PATTERN = re.compile(
    r'\b([A-G][#b♯♭]?\s*(?:major|minor|maj|min|m|M|dim|aug)?)\b'
)

QUOTED_TITLE_PATTERN = re.compile(r'["“”]([^"“”]{2,60})["“”]')

LABEL_INDICATORS = {
    "records", "recordings", "entertainment", "music group", "label",
    "productions", "sound", "audio",
}


def _tokenize(text: str) -> List[Tuple[str, int, int]]:
    """Split text into (token, start, end) triples."""
    tokens: List[Tuple[str, int, int]] = []
    for m in re.finditer(r'\S+', text):
        tokens.append((m.group(), m.start(), m.end()))
    return tokens


def _ngrams(
    tokens: List[Tuple[str, int, int]], n: int
) -> List[Tuple[str, int, int]]:
    """Generate n-gram spans from token list."""
    result = []
    for i in range(len(tokens) - n + 1):
        text = " ".join(t[0] for t in tokens[i : i + n])
        start = tokens[i][1]
        end = tokens[i + n - 1][2]
        result.append((text, start, end))
    return result


def gazetteer_match(
    text: str,
    gazetteer: Set[str],
    entity_type: EntityType,
    confidence: float = 0.85,
) -> List[MusicEntity]:
    """Find all gazetteer entries in text (case-insensitive, greedy longest)."""
    lower = text.lower()
    entities: List[MusicEntity] = []
    sorted_gaz = sorted(gazetteer, key=len, reverse=True)
    used: Set[Tuple[int, int]] = set()
    for term in sorted_gaz:
        start = 0
        while True:
            idx = lower.find(term, start)
            if idx == -1:
                break
            end = idx + len(term)
            overlap = any(
                not (end <= us or idx >= ue) for us, ue in used
            )
            if not overlap:
                if idx == 0 or not lower[idx - 1].isalpha():
                    if end == len(lower) or not lower[end].isalpha():
                        entities.append(MusicEntity(
                            text=text[idx:end],
                            entity_type=entity_type,
                            start=idx, end=end,
                            confidence=confidence,
                        ))
                        used.add((idx, end))
            start = end
    return entities


def extract_quoted_titles(text: str) -> List[MusicEntity]:
    """Extract quoted strings as potential song/album titles."""
    entities = []
    for m in QUOTED_TITLE_PATTERN.finditer(text):
        entities.append(MusicEntity(
            text=m.group(1),
            entity_type=EntityType.SONG,
            start=m.start(1),
            end=m.end(1),
            confidence=0.70,
        ))
    return entities


def extract_key_signatures(text: str) -> List[MusicEntity]:
    """Extract musical key references."""
    entities = []
    for m in KEY_PATTERN.finditer(text):
        entities.append(MusicEntity(
            text=m.group(1).strip(),
            entity_type=EntityType.KEY_SIGNATURE,
            start=m.start(1),
            end=m.end(1),
            confidence=0.75,
        ))
    return entities


def extract_tempo_markings(text: str) -> List[MusicEntity]:
    """Extract Italian tempo marking terms."""
    return gazetteer_match(text, TEMPO_MARKINGS, EntityType.TEMPO_MARKING, 0.90)


def capitalized_phrase_candidates(
    tokens: List[Tuple[str, int, int]],
) -> List[Tuple[str, int, int]]:
    """Find capitalized multi-word phrases (potential artist/venue names)."""
    phrases: List[Tuple[str, int, int]] = []
    i = 0
    while i < len(tokens):
        if tokens[i][0][0].isupper() and not tokens[i][0].isupper():
            j = i + 1
            while (
                j < len(tokens)
                and tokens[j][0][0].isupper()
                and len(tokens[j][0]) > 1
            ):
                j += 1
            if j - i >= 2:
                text = " ".join(t[0] for t in tokens[i:j])
                phrases.append((text, tokens[i][1], tokens[j - 1][2]))
            i = j
        else:
            i += 1
    return phrases


def classify_capitalized_phrase(
    phrase: str,
    left_context: str,
    right_context: str,
) -> Tuple[EntityType, float]:
    """Heuristically classify a capitalized phrase by surrounding context."""
    left_lower = left_context.lower().strip()
    right_lower = right_context.lower().strip()
    if any(w in left_lower for w in ("by", "featuring", "feat.", "ft.", "artist")):
        return EntityType.ARTIST, 0.80
    if any(w in left_lower for w in ("at", "venue", "theater", "theatre", "hall", "club")):
        return EntityType.VENUE, 0.75
    if any(w in left_lower for w in ("album", "record", "lp", "ep")):
        return EntityType.ALBUM, 0.75
    if any(w in right_lower for w in LABEL_INDICATORS):
        return EntityType.LABEL, 0.70
    if any(w in left_lower for w in ("on", "signed to", "released on")):
        return EntityType.LABEL, 0.65
    return EntityType.ARTIST, 0.50


def extract_entities(text: str) -> NERResult:
    """Full NER pipeline: gazetteer + pattern + statistical heuristics."""
    entities: List[MusicEntity] = []
    entities.extend(gazetteer_match(text, GENRE_GAZETTEER, EntityType.GENRE))
    entities.extend(gazetteer_match(text, INSTRUMENT_GAZETTEER, EntityType.INSTRUMENT))
    entities.extend(extract_quoted_titles(text))
    entities.extend(extract_key_signatures(text))
    entities.extend(extract_tempo_markings(text))

    tokens = _tokenize(text)
    cap_phrases = capitalized_phrase_candidates(tokens)
    occupied = {(e.start, e.end) for e in entities}
    for phrase, start, end in cap_phrases:
        if any(not (end <= os or start >= oe) for os, oe in occupied):
            continue
        left_ctx = text[max(0, start - 40):start]
        right_ctx = text[end:end + 40]
        etype, conf = classify_capitalized_phrase(phrase, left_ctx, right_ctx)
        entities.append(MusicEntity(
            text=phrase, entity_type=etype,
            start=start, end=end, confidence=conf,
        ))

    entities.sort(key=lambda e: (e.start, -e.confidence))

    # BIO labels
    token_labels: List[Tuple[str, str]] = []
    for tok, ts, te in tokens:
        label = "O"
        for ent in entities:
            if ts >= ent.start and te <= ent.end:
                prefix = "B" if ts == ent.start else "I"
                label = f"{prefix}-{ent.entity_type.name}"
                break
        token_labels.append((tok, label))

    return NERResult(entities=entities, text=text, token_labels=token_labels)


def entity_cooccurrence_matrix(
    documents: List[str],
) -> Tuple[np.ndarray, List[str]]:
    """Build entity co-occurrence matrix across a corpus of music texts."""
    all_entities: Dict[str, int] = {}
    doc_entities: List[Set[str]] = []
    for doc in documents:
        result = extract_entities(doc)
        names = set()
        for e in result.entities:
            key = f"{e.entity_type.name}:{e.text.lower()}"
            if key not in all_entities:
                all_entities[key] = len(all_entities)
            names.add(key)
        doc_entities.append(names)

    n = len(all_entities)
    cooc = np.zeros((n, n), dtype=float)
    for names in doc_entities:
        idxs = [all_entities[nm] for nm in names]
        for i in idxs:
            for j in idxs:
                if i != j:
                    cooc[i, j] += 1.0

    labels = [""] * n
    for key, idx in all_entities.items():
        labels[idx] = key
    return cooc, labels


def entity_frequency(documents: List[str]) -> Dict[str, int]:
    """Count entity frequencies across a corpus."""
    counts: Dict[str, int] = {}
    for doc in documents:
        result = extract_entities(doc)
        for e in result.entities:
            key = f"{e.entity_type.name}:{e.text.lower()}"
            counts[key] = counts.get(key, 0) + 1
    return counts
