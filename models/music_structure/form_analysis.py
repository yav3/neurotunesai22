"""Form analysis: large-scale musical structure recognition.

Identify and classify musical forms (binary, ternary, rondo, sonata,
strophic, through-composed) from section labels, key areas, and
thematic return patterns. Includes proportion analysis (golden ratio
placement) and formal function classification.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum


class FormType(str, Enum):
    BINARY = "binary"
    ROUNDED_BINARY = "rounded_binary"
    TERNARY = "ternary"
    RONDO = "rondo"
    SONATA = "sonata"
    STROPHIC = "strophic"
    THROUGH_COMPOSED = "through_composed"
    THEME_VARIATIONS = "theme_and_variations"
    ARCH = "arch"
    BAR_FORM = "bar_form"


class FormalFunction(str, Enum):
    PRESENTATION = "presentation"
    CONTINUATION = "continuation"
    CADENTIAL = "cadential"
    TRANSITION = "transition"
    STANDING_ON_DOMINANT = "standing_on_dominant"
    RETRANSITION = "retransition"
    CODETTA = "codetta"
    CODA = "coda"


@dataclass
class Section:
    label: str
    start_beat: float
    end_beat: float
    key: int = 0
    thematic_id: int = 0


@dataclass
class FormAnalysis:
    form_type: FormType
    confidence: float
    sections: List[Section]
    proportions: List[float]
    golden_ratio_proximity: float


def section_similarity_matrix(features_per_section: List[np.ndarray]) -> np.ndarray:
    n = len(features_per_section)
    sim = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            ni = np.linalg.norm(features_per_section[i])
            nj = np.linalg.norm(features_per_section[j])
            if ni > 1e-10 and nj > 1e-10:
                sim[i, j] = np.dot(features_per_section[i], features_per_section[j]) / (ni * nj)
    return sim


def label_sections(similarity: np.ndarray, threshold: float = 0.8) -> List[str]:
    n = similarity.shape[0]
    labels = [""] * n
    current_label = 0
    label_map = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    for i in range(n):
        if labels[i]:
            continue
        labels[i] = label_map[current_label % len(label_map)]
        for j in range(i + 1, n):
            if similarity[i, j] >= threshold and not labels[j]:
                labels[j] = labels[i]
        current_label += 1
    return labels


def classify_form(labels: List[str]) -> Tuple[FormType, float]:
    pattern = "".join(labels)
    if pattern in ("AB", "AABB"):
        return FormType.BINARY, 0.9
    if pattern in ("ABA", "AABA"):
        return FormType.TERNARY, 0.9
    if len(pattern) >= 3 and pattern.startswith("AB") and pattern.endswith("A"):
        return FormType.ROUNDED_BINARY, 0.8
    unique = set(labels)
    if len(unique) == 1:
        return FormType.STROPHIC, 0.85
    a_positions = [i for i, l in enumerate(labels) if l == "A"]
    if len(a_positions) >= 3 and all(a_positions[i + 1] - a_positions[i] == 2
                                      for i in range(len(a_positions) - 1)):
        return FormType.RONDO, 0.85
    if len(pattern) >= 5:
        half = len(pattern) // 2
        if pattern[:half] == pattern[half:half + len(pattern[:half])]:
            return FormType.STROPHIC, 0.7
    reversed_labels = labels[::-1]
    if labels == reversed_labels and len(labels) >= 5:
        return FormType.ARCH, 0.8
    if len(unique) == len(labels):
        return FormType.THROUGH_COMPOSED, 0.7
    return FormType.THROUGH_COMPOSED, 0.4


def golden_ratio_analysis(section_durations: np.ndarray) -> Dict[str, float]:
    total = float(np.sum(section_durations))
    if total <= 0:
        return {"phi_proximity": 0.0, "golden_point_beat": 0.0}
    phi = (1 + np.sqrt(5)) / 2
    golden_point = total / phi
    cumulative = np.cumsum(section_durations)
    distances = np.abs(cumulative - golden_point)
    nearest_idx = int(np.argmin(distances))
    proximity = 1.0 - distances[nearest_idx] / total
    return {
        "phi_proximity": float(proximity),
        "golden_point_beat": float(golden_point),
        "nearest_boundary_idx": nearest_idx,
        "nearest_boundary_beat": float(cumulative[nearest_idx]),
    }


def proportion_profile(section_durations: np.ndarray) -> np.ndarray:
    total = np.sum(section_durations)
    if total <= 0:
        return np.zeros_like(section_durations)
    return section_durations / total


def key_area_plan(sections: List[Section]) -> List[Tuple[str, int, int]]:
    plan = []
    for s in sections:
        plan.append((s.label, s.key, s.thematic_id))
    return plan


def sonata_form_template(sections: List[Section]) -> Dict[str, float]:
    if len(sections) < 3:
        return {"sonata_confidence": 0.0}
    labels = [s.label for s in sections]
    keys = [s.key for s in sections]
    has_exposition = "A" in labels
    has_development = len(set(labels)) > 2
    has_recap = labels.count("A") >= 2
    key_contrast = len(set(keys)) >= 2
    tonic_return = keys[0] == keys[-1] if keys else False
    score = sum([has_exposition, has_development, has_recap,
                 key_contrast, tonic_return]) / 5.0
    return {
        "sonata_confidence": score,
        "has_exposition": has_exposition,
        "has_development": has_development,
        "has_recapitulation": has_recap,
        "key_contrast": key_contrast,
        "tonic_return": tonic_return,
    }


def formal_function_sequence(sections: List[Section]) -> List[FormalFunction]:
    n = len(sections)
    funcs: List[FormalFunction] = []
    for i, s in enumerate(sections):
        pos = i / max(n - 1, 1)
        if pos < 0.15:
            funcs.append(FormalFunction.PRESENTATION)
        elif pos < 0.4:
            funcs.append(FormalFunction.CONTINUATION)
        elif pos < 0.55:
            funcs.append(FormalFunction.TRANSITION)
        elif pos < 0.7:
            funcs.append(FormalFunction.STANDING_ON_DOMINANT)
        elif pos < 0.85:
            funcs.append(FormalFunction.CADENTIAL)
        elif pos < 0.95:
            funcs.append(FormalFunction.CODETTA)
        else:
            funcs.append(FormalFunction.CODA)
    return funcs


def analyze_form(features_per_section: List[np.ndarray],
                 section_durations: np.ndarray,
                 keys: Optional[List[int]] = None) -> FormAnalysis:
    sim = section_similarity_matrix(features_per_section)
    labels = label_sections(sim)
    form_type, confidence = classify_form(labels)
    props = proportion_profile(section_durations)
    golden = golden_ratio_analysis(section_durations)
    sections = []
    cum = 0.0
    for i, l in enumerate(labels):
        sections.append(Section(
            label=l,
            start_beat=cum,
            end_beat=cum + section_durations[i],
            key=keys[i] if keys and i < len(keys) else 0,
            thematic_id=ord(l) - ord("A"),
        ))
        cum += section_durations[i]
    return FormAnalysis(
        form_type=form_type,
        confidence=confidence,
        sections=sections,
        proportions=props.tolist(),
        golden_ratio_proximity=golden["phi_proximity"],
    )
