"""Musical form analysis: form templates, section detection, edit-distance matching."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


@dataclass
class MusicalForm:
    name: str
    section_labels: list[str]
    repeat_structure: str = ""
    region: str = ""
    era: str = ""
    description: str = ""


# ---------------------------------------------------------------------------
# Form library — 22 forms
# ---------------------------------------------------------------------------

FORM_LIBRARY: list[MusicalForm] = [
    MusicalForm("Sonata", ["Exposition", "Development", "Recapitulation"],
                "||: Exp :|| Dev | Recap", "Western", "Classical",
                "Three-part structure central to Classical instrumental music"),
    MusicalForm("Rondo", ["A", "B", "A", "C", "A"],
                "ABACA", "Western", "Classical",
                "Recurring refrain alternating with contrasting episodes"),
    MusicalForm("Rondo (7-part)", ["A", "B", "A", "C", "A", "B", "A"],
                "ABACABA", "Western", "Classical"),
    MusicalForm("Theme and Variations", ["Theme", "Var1", "Var2", "Var3", "Var4"],
                "A A' A'' A''' ...", "Western", "Classical"),
    MusicalForm("12-Bar Blues", ["I", "I", "I", "I", "IV", "IV", "I", "I", "V", "IV", "I", "V"],
                "AAB (lyric form)", "African-American", "20th century",
                "12-bar chord progression foundational to blues and rock"),
    MusicalForm("32-Bar AABA", ["A", "A", "B", "A"],
                "AABA (8+8+8+8)", "Western", "Tin Pan Alley",
                "Standard 32-bar song form of American popular music"),
    MusicalForm("Verse-Chorus", ["Verse", "Chorus", "Verse", "Chorus"],
                "V C V C", "Western", "Pop",
                "Most common popular-music form"),
    MusicalForm("Verse-Chorus-Bridge", ["Verse", "Chorus", "Verse", "Chorus", "Bridge", "Chorus"],
                "V C V C B C", "Western", "Pop"),
    MusicalForm("Strophic", ["A", "A", "A", "A"],
                "A A A A ...", "Global", "All eras",
                "Same music repeated for each stanza of text"),
    MusicalForm("Through-Composed", ["A", "B", "C", "D"],
                "ABCD (no repeats)", "Western", "Romantic"),
    MusicalForm("Binary", ["A", "B"],
                "||: A :||: B :||", "Western", "Baroque"),
    MusicalForm("Ternary", ["A", "B", "A"],
                "ABA", "Western", "Classical"),
    MusicalForm("Ritornello", ["R", "S1", "R", "S2", "R", "S3", "R"],
                "R S1 R S2 R ...", "Western", "Baroque",
                "Orchestral refrain alternating with solo episodes"),
    MusicalForm("Fugue", ["Subject", "Answer", "Episode", "Subject", "Stretto"],
                "Exposition → Episodes → Final entry", "Western", "Baroque"),
    MusicalForm("Raga Performance", ["Alap", "Jor", "Jhala", "Gat"],
                "Alap → Jor → Jhala → Gat/Bandish", "South Asia", "Traditional",
                "Progressive acceleration from free rhythm to fast rhythmic climax"),
    MusicalForm("Taqasim", ["Taqasim", "Wasla"],
                "Free improvisation → composed section", "Middle East", "Traditional",
                "Improvised modal exploration preceding composed music"),
    MusicalForm("Lancaran", ["A", "A", "A", "A"],
                "Short cyclical gending", "Indonesia", "Traditional"),
    MusicalForm("Ladrang", ["A", "B", "A", "B"],
                "Medium cyclical gending (32 beats)", "Indonesia", "Traditional"),
    MusicalForm("Ketawang", ["A", "B"],
                "Short refined gending (16 beats)", "Indonesia", "Traditional"),
    MusicalForm("Call-and-Response", ["Call", "Response", "Call", "Response"],
                "C R C R ...", "Sub-Saharan Africa", "Traditional",
                "Leader-group alternation in vocal or instrumental music"),
    MusicalForm("Baroque Suite", ["Allemande", "Courante", "Sarabande", "Gigue"],
                "A-C-S-G + optional Galanteries", "Western", "Baroque"),
    MusicalForm("Da Capo Aria", ["A", "B", "A"],
                "ABA with ornamented repeat", "Western", "Baroque"),
]


class FormMatcher:
    @staticmethod
    def edit_distance(a: Sequence[str], b: Sequence[str]) -> int:
        m, n = len(a), len(b)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                cost = 0 if a[i - 1].lower() == b[j - 1].lower() else 1
                dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
        return dp[m][n]

    @staticmethod
    def match(section_labels: Sequence[str],
              library: list[MusicalForm] | None = None,
              top_k: int = 3) -> list[tuple[MusicalForm, int]]:
        if library is None:
            library = FORM_LIBRARY
        if not section_labels:
            return []
        results: list[tuple[MusicalForm, int]] = []
        for form in library:
            d = FormMatcher.edit_distance(section_labels, form.section_labels)
            results.append((form, d))
        results.sort(key=lambda x: x[1])
        return results[:top_k]

    @staticmethod
    def by_region(region: str, library: list[MusicalForm] | None = None) -> list[MusicalForm]:
        if library is None:
            library = FORM_LIBRARY
        return [f for f in library if region.lower() in f.region.lower()]

    @staticmethod
    def by_era(era: str, library: list[MusicalForm] | None = None) -> list[MusicalForm]:
        if library is None:
            library = FORM_LIBRARY
        return [f for f in library if era.lower() in f.era.lower()]

    @staticmethod
    def detect_repetition(labels: Sequence[str]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for lbl in labels:
            key = lbl.lower()
            counts[key] = counts.get(key, 0) + 1
        return counts
