"""Musical form analysis: classification, complexity scoring."""
from __future__ import annotations
from dataclasses import dataclass

MUSICAL_FORMS: dict[str, dict[str, str]] = {
    "binary": {"pattern": "AB", "description": "Two contrasting sections"},
    "ternary": {"pattern": "ABA", "description": "Statement-contrast-return"},
    "rondo": {"pattern": "ABACADA", "description": "Recurring refrain with episodes"},
    "sonata": {"pattern": "exposition-development-recapitulation", "description": "Tonal drama"},
    "theme_and_variations": {"pattern": "A A1 A2 A3 ...", "description": "Theme with transformations"},
    "strophic": {"pattern": "AAA...", "description": "Same music, different lyrics"},
    "through_composed": {"pattern": "ABCDE...", "description": "No repetition"},
    "verse_chorus": {"pattern": "ABABCAB", "description": "Pop song form"},
    "AABA": {"pattern": "AABA", "description": "32-bar standard"},
    "12_bar_blues": {"pattern": "I-I-I-I-IV-IV-I-I-V-IV-I-V", "description": "Blues progression"},
}


@dataclass
class FormSection:
    label: str
    start_bar: int
    end_bar: int
    key: str = "C"
    is_development: bool = False

    @property
    def length(self) -> int:
        return self.end_bar - self.start_bar


def classify_form(sections: list[FormSection]) -> str:
    if not sections:
        return "unknown"
    labels = [s.label.upper() for s in sections]
    pattern = "".join(labels)

    unique_count = len(set(labels))
    if unique_count == 1:
        return "strophic"
    if unique_count == len(labels) and len(labels) > 3:
        return "through_composed"

    if pattern == "AB":
        return "binary"
    if len(labels) == 3 and labels[0] == labels[2] and labels[0] != labels[1]:
        return "ternary"
    if len(labels) == 4 and labels[:3] == [labels[0], labels[0], labels[1], labels[0]]:
        return "AABA"

    has_verse = any(l in ("V", "VERSE") for l in labels)
    has_chorus = any(l in ("C", "CHORUS") for l in labels)
    if has_verse and has_chorus:
        return "verse_chorus"

    if len(labels) >= 5 and labels[0] == labels[2] == labels[4]:
        return "rondo"

    if any(s.is_development for s in sections):
        return "sonata"

    base = labels[0] if labels else ""
    if all(l.startswith(base) for l in labels) and unique_count > 1:
        return "theme_and_variations"

    return "unknown"


def form_complexity(form_name: str) -> float:
    complexity_map = {
        "strophic": 0.1, "binary": 0.2, "ternary": 0.3, "verse_chorus": 0.35,
        "AABA": 0.4, "12_bar_blues": 0.3, "rondo": 0.5,
        "theme_and_variations": 0.6, "sonata": 0.8, "through_composed": 0.7,
    }
    return complexity_map.get(form_name, 0.5)
