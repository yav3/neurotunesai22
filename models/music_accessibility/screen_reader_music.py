"""Screen reader integration for music content — structured descriptions and navigation."""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional
import numpy as np


class MusicRole(str, Enum):
    SCORE = "score"
    STAFF = "staff"
    MEASURE = "measure"
    NOTE = "note"
    REST = "rest"
    CHORD = "chord"
    CLEF = "clef"
    KEY_SIGNATURE = "key_signature"
    TIME_SIGNATURE = "time_signature"
    DYNAMIC = "dynamic"
    TEMPO = "tempo"
    ARTICULATION = "articulation"
    BARLINE = "barline"
    SECTION = "section"


class NavigationLevel(str, Enum):
    PIECE = "piece"
    SECTION = "section"
    MEASURE = "measure"
    BEAT = "beat"
    NOTE = "note"


@dataclass
class MusicElement:
    role: MusicRole
    label: str
    description: str
    measure: int = 0
    beat: float = 0.0
    staff: int = 0
    children: List["MusicElement"] = field(default_factory=list)


@dataclass
class NavigationLandmark:
    level: NavigationLevel
    label: str
    index: int
    measure_start: int
    measure_end: int


@dataclass
class AccessibilityTree:
    root: MusicElement
    landmarks: List[NavigationLandmark]
    total_measures: int
    total_notes: int


NOTE_NAMES = ["C", "D", "E", "F", "G", "A", "B"]
ACCIDENTAL_NAMES = {-1: "flat", 0: "", 1: "sharp", 2: "double sharp", -2: "double flat"}
DURATION_NAMES = {
    4.0: "whole", 2.0: "half", 1.0: "quarter",
    0.5: "eighth", 0.25: "sixteenth", 0.125: "thirty-second",
}
DYNAMIC_LABELS = {
    "ppp": "pianississimo, very very soft",
    "pp": "pianissimo, very soft",
    "p": "piano, soft",
    "mp": "mezzo piano, moderately soft",
    "mf": "mezzo forte, moderately loud",
    "f": "forte, loud",
    "ff": "fortissimo, very loud",
    "fff": "fortississimo, very very loud",
}


def describe_note(
    pitch_name: str, octave: int, duration: float,
    accidental: int = 0, dots: int = 0,
) -> str:
    dur_name = DURATION_NAMES.get(duration, f"{duration}-beat")
    acc = ACCIDENTAL_NAMES.get(accidental, "")
    dot_str = " dotted" * dots
    parts = []
    if acc:
        parts.append(acc)
    parts.append(f"{pitch_name}{octave}")
    parts.append(f"{dur_name}{dot_str} note")
    return " ".join(parts)


def describe_chord(notes: List[str]) -> str:
    if not notes:
        return "empty chord"
    if len(notes) == 1:
        return notes[0]
    return f"chord of {', '.join(notes[:-1])} and {notes[-1]}"


def describe_rest(duration: float, dots: int = 0) -> str:
    dur_name = DURATION_NAMES.get(duration, f"{duration}-beat")
    dot_str = " dotted" * dots
    return f"{dur_name}{dot_str} rest"


def describe_key_signature(key: str, mode: str = "major") -> str:
    return f"key of {key} {mode}"


def describe_time_signature(numerator: int, denominator: int) -> str:
    if numerator == 4 and denominator == 4:
        return "common time, four-four"
    if numerator == 2 and denominator == 2:
        return "cut time, two-two"
    return f"{numerator} over {denominator} time"


def describe_dynamic(marking: str) -> str:
    return DYNAMIC_LABELS.get(marking, f"dynamic marking {marking}")


def describe_tempo(bpm: float, marking: Optional[str] = None) -> str:
    if marking:
        return f"{marking}, {int(bpm)} beats per minute"
    return f"{int(bpm)} beats per minute"


def describe_measure(
    measure_num: int, elements: List[MusicElement],
) -> str:
    if not elements:
        return f"measure {measure_num}, empty"
    descs = [e.description for e in elements]
    return f"measure {measure_num}: {'; '.join(descs)}"


def build_element(
    role: MusicRole, label: str, description: str,
    measure: int = 0, beat: float = 0.0, staff: int = 0,
) -> MusicElement:
    return MusicElement(
        role=role, label=label, description=description,
        measure=measure, beat=beat, staff=staff,
    )


def build_landmarks(
    section_boundaries: List[int], total_measures: int,
    section_labels: Optional[List[str]] = None,
) -> List[NavigationLandmark]:
    landmarks = [NavigationLandmark(
        level=NavigationLevel.PIECE, label="beginning",
        index=0, measure_start=1, measure_end=total_measures,
    )]
    boundaries = sorted(section_boundaries)
    for i, b in enumerate(boundaries):
        label = section_labels[i] if section_labels and i < len(section_labels) else f"section {i + 1}"
        end = boundaries[i + 1] - 1 if i + 1 < len(boundaries) else total_measures
        landmarks.append(NavigationLandmark(
            level=NavigationLevel.SECTION, label=label,
            index=i + 1, measure_start=b, measure_end=end,
        ))
    return landmarks


def build_accessibility_tree(
    measures: List[List[MusicElement]],
    section_boundaries: Optional[List[int]] = None,
    section_labels: Optional[List[str]] = None,
) -> AccessibilityTree:
    total_measures = len(measures)
    total_notes = sum(
        1 for m in measures for e in m
        if e.role in (MusicRole.NOTE, MusicRole.CHORD)
    )
    root = MusicElement(
        role=MusicRole.SCORE, label="score",
        description=f"score with {total_measures} measures and {total_notes} notes",
    )
    for i, m_elements in enumerate(measures):
        m_elem = MusicElement(
            role=MusicRole.MEASURE, label=f"measure {i + 1}",
            description=describe_measure(i + 1, m_elements),
            measure=i + 1,
            children=m_elements,
        )
        root.children.append(m_elem)
    landmarks = build_landmarks(
        section_boundaries or [], total_measures, section_labels,
    )
    return AccessibilityTree(
        root=root, landmarks=landmarks,
        total_measures=total_measures, total_notes=total_notes,
    )


def navigate_to(tree: AccessibilityTree, level: NavigationLevel, index: int) -> Optional[str]:
    if level == NavigationLevel.MEASURE:
        if 0 < index <= tree.total_measures:
            m = tree.root.children[index - 1]
            return m.description
    elif level == NavigationLevel.SECTION:
        for lm in tree.landmarks:
            if lm.level == NavigationLevel.SECTION and lm.index == index:
                return f"{lm.label}, measures {lm.measure_start} to {lm.measure_end}"
    return None


def score_summary(tree: AccessibilityTree) -> str:
    n_sections = sum(1 for lm in tree.landmarks if lm.level == NavigationLevel.SECTION)
    parts = [f"{tree.total_measures} measures", f"{tree.total_notes} notes"]
    if n_sections > 0:
        parts.append(f"{n_sections} sections")
    return f"Score: {', '.join(parts)}"


def element_to_aria(element: MusicElement) -> Dict[str, str]:
    return {
        "role": element.role.value,
        "aria-label": element.label,
        "aria-description": element.description,
    }
