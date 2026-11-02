"""Discourse structure analysis for music text: segmentation of music
reviews, liner notes, and music criticism into rhetorical units.

Identifies argumentative structure (claim → evidence → evaluation),
section roles (introduction, description, analysis, verdict), and
cross-referencing patterns in music writing.

Based on Rhetorical Structure Theory (Mann & Thompson 1988) adapted for
music criticism.  Pure numpy/regex — no external NLP libraries.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np


class SectionRole(Enum):
    INTRODUCTION = "introduction"
    DESCRIPTION = "description"
    ANALYSIS = "analysis"
    COMPARISON = "comparison"
    VERDICT = "verdict"
    TRACK_LIST = "track_list"
    BIOGRAPHICAL = "biographical"
    CONTEXTUAL = "contextual"


class RhetoricalRelation(Enum):
    ELABORATION = "elaboration"
    CONTRAST = "contrast"
    EVIDENCE = "evidence"
    EVALUATION = "evaluation"
    CONCESSION = "concession"
    CAUSE = "cause"
    SEQUENCE = "sequence"


@dataclass
class DiscourseUnit:
    text: str
    role: SectionRole
    position: int       # sentence index in document
    confidence: float   # [0, 1]


@dataclass
class RhetoricalEdge:
    source: int   # unit index
    target: int
    relation: RhetoricalRelation
    strength: float


@dataclass
class DiscourseTree:
    units: List[DiscourseUnit]
    edges: List[RhetoricalEdge]
    dominant_role: SectionRole


INTRO_PATTERNS = re.compile(
    r'(?:debut|first\s+album|latest\s+release|returns?\s+with|'
    r'opens?\s+with|begin|new\s+album|follow[\s-]up)',
    re.IGNORECASE,
)

VERDICT_PATTERNS = re.compile(
    r'(?:overall|in\s+conclusion|all\s+in\s+all|final\s*ly|'
    r'verdict|bottom\s+line|rating|score|stars?\s+out\s+of|'
    r'must[\s-]listen|recommended|essential)',
    re.IGNORECASE,
)

COMPARISON_PATTERNS = re.compile(
    r'(?:compared?\s+to|reminiscent|similar\s+to|unlike|'
    r'whereas|in\s+contrast|on\s+the\s+other\s+hand)',
    re.IGNORECASE,
)

ANALYSIS_PATTERNS = re.compile(
    r'(?:harmoni|melodic|rhythmic|structur|chord|progression|'
    r'mix(?:ing)?|production|arrangement|dynamic|tempo|meter|'
    r'orchestrat|modulation|texture)',
    re.IGNORECASE,
)

TRACK_LIST_PATTERN = re.compile(
    r'(?:track\s+\d|song\s+\d|#\d|opener|closer|highlight|standout)',
    re.IGNORECASE,
)

BIOGRAPHICAL_PATTERN = re.compile(
    r'(?:born\s+in|formed\s+in|hails?\s+from|career|'
    r'previously\s+(?:known|played)|founding\s+member)',
    re.IGNORECASE,
)


def _tokenize_sentences(text: str) -> List[str]:
    """Split text into sentences via punctuation heuristic."""
    sents = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sents if s.strip()]


def _score_role(sent: str) -> Dict[SectionRole, float]:
    scores: Dict[SectionRole, float] = {r: 0.0 for r in SectionRole}
    if INTRO_PATTERNS.search(sent):
        scores[SectionRole.INTRODUCTION] += 1.0
    if VERDICT_PATTERNS.search(sent):
        scores[SectionRole.VERDICT] += 1.0
    if COMPARISON_PATTERNS.search(sent):
        scores[SectionRole.COMPARISON] += 1.0
    if ANALYSIS_PATTERNS.search(sent):
        scores[SectionRole.ANALYSIS] += 1.0
    if TRACK_LIST_PATTERN.search(sent):
        scores[SectionRole.TRACK_LIST] += 1.0
    if BIOGRAPHICAL_PATTERN.search(sent):
        scores[SectionRole.BIOGRAPHICAL] += 1.0
    total = sum(scores.values())
    if total < 0.5:
        scores[SectionRole.DESCRIPTION] = 0.5
    return scores


def classify_sentence(sent: str) -> Tuple[SectionRole, float]:
    """Return best role and confidence for a single sentence."""
    scores = _score_role(sent)
    best = max(scores, key=scores.get)  # type: ignore[arg-type]
    total = sum(scores.values())
    conf = scores[best] / total if total > 0 else 0.0
    return best, conf


def segment_document(text: str) -> List[DiscourseUnit]:
    """Segment a music text into discourse units with role labels."""
    sents = _tokenize_sentences(text)
    units = []
    n = len(sents)
    for i, s in enumerate(sents):
        role, conf = classify_sentence(s)
        if i < n * 0.15 and role == SectionRole.DESCRIPTION:
            role = SectionRole.INTRODUCTION
            conf = max(conf, 0.4)
        elif i > n * 0.85 and role == SectionRole.DESCRIPTION:
            role = SectionRole.VERDICT
            conf = max(conf, 0.3)
        units.append(DiscourseUnit(text=s, role=role, position=i, confidence=conf))
    return units


def build_discourse_tree(text: str) -> DiscourseTree:
    """Construct a shallow discourse tree from music text."""
    units = segment_document(text)
    edges: List[RhetoricalEdge] = []
    for i in range(len(units) - 1):
        u, v = units[i], units[i + 1]
        if u.role == SectionRole.DESCRIPTION and v.role == SectionRole.ANALYSIS:
            edges.append(RhetoricalEdge(i, i + 1, RhetoricalRelation.ELABORATION, 0.7))
        elif u.role == SectionRole.ANALYSIS and v.role == SectionRole.VERDICT:
            edges.append(RhetoricalEdge(i, i + 1, RhetoricalRelation.EVALUATION, 0.8))
        elif v.role == SectionRole.COMPARISON:
            edges.append(RhetoricalEdge(i, i + 1, RhetoricalRelation.CONTRAST, 0.6))
        else:
            edges.append(RhetoricalEdge(i, i + 1, RhetoricalRelation.SEQUENCE, 0.3))
    role_counts = {}
    for u in units:
        role_counts[u.role] = role_counts.get(u.role, 0) + 1
    dominant = max(role_counts, key=role_counts.get) if role_counts else SectionRole.DESCRIPTION  # type: ignore[arg-type]
    return DiscourseTree(units=units, edges=edges, dominant_role=dominant)


def argument_chain(tree: DiscourseTree) -> List[List[int]]:
    """Extract chains of claim→evidence→evaluation from the discourse tree."""
    chains: List[List[int]] = []
    current: List[int] = []
    argument_roles = {SectionRole.ANALYSIS, SectionRole.COMPARISON, SectionRole.VERDICT}
    for u in tree.units:
        if u.role in argument_roles:
            current.append(u.position)
        else:
            if len(current) >= 2:
                chains.append(current)
            current = []
    if len(current) >= 2:
        chains.append(current)
    return chains


def section_proportions(tree: DiscourseTree) -> Dict[str, float]:
    """Fraction of document devoted to each section role."""
    n = len(tree.units)
    if n == 0:
        return {}
    counts: Dict[str, float] = {}
    for u in tree.units:
        counts[u.role.value] = counts.get(u.role.value, 0) + 1
    return {k: v / n for k, v in counts.items()}


def discourse_feature_vector(text: str) -> np.ndarray:
    """Numeric feature vector from discourse structure (9 features)."""
    tree = build_discourse_tree(text)
    props = section_proportions(tree)
    chains = argument_chain(tree)
    n_units = len(tree.units)
    n_edges = len(tree.edges)
    avg_conf = float(np.mean([u.confidence for u in tree.units])) if tree.units else 0.0
    n_chains = len(chains)
    avg_chain_len = float(np.mean([len(c) for c in chains])) if chains else 0.0
    analysis_frac = props.get("analysis", 0.0)
    verdict_frac = props.get("verdict", 0.0)
    comparison_frac = props.get("comparison", 0.0)
    return np.array([
        n_units, n_edges, avg_conf,
        n_chains, avg_chain_len,
        analysis_frac, verdict_frac, comparison_frac,
        len(set(u.role for u in tree.units)),
    ])
