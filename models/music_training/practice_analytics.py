"""Practice session analytics: tempo/note accuracy, spaced repetition, progress curves."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class PracticeEvent:
    timestamp: float
    passage_id: str
    tempo_target: float
    tempo_actual: float
    note_errors: int
    total_notes: int
    duration_seconds: float


@dataclass
class PracticeSession:
    events: List[PracticeEvent] = field(default_factory=list)
    session_start: float = 0.0
    duration_sec: float = 0.0
    tempo_sequence: np.ndarray = field(default_factory=lambda: np.array([]))
    error_positions: List[int] = field(default_factory=list)
    repetition_count: int = 1
    piece_id: str = ""

    def add_event(self, event: PracticeEvent) -> None:
        self.events.append(event)

    @property
    def total_duration(self) -> float:
        if self.duration_sec > 0:
            return self.duration_sec
        return sum(e.duration_seconds for e in self.events)

    @property
    def total_notes(self) -> int:
        return sum(e.total_notes for e in self.events)

    @property
    def total_errors(self) -> int:
        return sum(e.note_errors for e in self.events)


@dataclass
class PracticeReport:
    accuracy: float
    tempo_stability: float
    improvement_rate: float
    problem_spots: List[Tuple[int, int]]
    estimated_mastery: float


def tempo_accuracy(target: float, actual: float) -> float:
    if target <= 0:
        return 0.0
    ratio = actual / target
    return max(0.0, 1.0 - abs(1.0 - ratio))


def tempo_stability(tempo_seq: np.ndarray) -> float:
    if len(tempo_seq) < 2:
        return 1.0
    mean_t = float(np.mean(tempo_seq))
    if mean_t < 1e-12:
        return 0.0
    cv = float(np.std(tempo_seq) / mean_t)
    return max(1.0 - cv, 0.0)


def note_accuracy(errors: int, total: int) -> float:
    if total <= 0:
        return 1.0
    return max(0.0, 1.0 - errors / total)


def note_accuracy_over_time(events: List[PracticeEvent]) -> List[Tuple[float, float]]:
    return [(e.timestamp, note_accuracy(e.note_errors, e.total_notes)) for e in events]


def tempo_accuracy_over_time(events: List[PracticeEvent]) -> List[Tuple[float, float]]:
    return [(e.timestamp, tempo_accuracy(e.tempo_target, e.tempo_actual)) for e in events]


def error_density(
    error_positions: List[int],
    total_notes: int,
    window: int = 10,
) -> np.ndarray:
    if total_notes <= 0:
        return np.array([])
    indicator = np.zeros(total_notes)
    for p in error_positions:
        if 0 <= p < total_notes:
            indicator[p] = 1.0
    if total_notes < window:
        return np.array([float(np.mean(indicator))])
    kernel = np.ones(window) / window
    return np.convolve(indicator, kernel, mode="valid")


def identify_problem_spots(
    error_positions: List[int],
    total_notes: int,
    threshold: float = 0.3,
    window: int = 10,
) -> List[Tuple[int, int]]:
    density = error_density(error_positions, total_notes, window)
    if len(density) == 0:
        return []
    above = density >= threshold
    spots = []
    in_spot = False
    start = 0
    for i in range(len(above)):
        if above[i] and not in_spot:
            start = i
            in_spot = True
        elif not above[i] and in_spot:
            spots.append((start, i + window - 1))
            in_spot = False
    if in_spot:
        spots.append((start, len(above) - 1 + window - 1))
    return spots


@dataclass
class PracticeStreak:
    dates: List[str] = field(default_factory=list)
    current_streak: int = 0
    longest_streak: int = 0
    total_days: int = 0

    def record_day(self, date: str) -> None:
        self.total_days += 1
        if not self.dates or self.dates[-1] != date:
            self.dates.append(date)
            if len(self.dates) >= 2:
                prev = self.dates[-2]
                if _consecutive_dates(prev, date):
                    self.current_streak += 1
                else:
                    self.current_streak = 1
            else:
                self.current_streak = 1
            self.longest_streak = max(self.longest_streak, self.current_streak)

    @property
    def consistency_score(self) -> float:
        if self.total_days == 0:
            return 0.0
        unique_days = len(set(self.dates))
        return min(unique_days / max(self.total_days, 1), 1.0)


def _consecutive_dates(a: str, b: str) -> bool:
    try:
        ya, ma, da = int(a[:4]), int(a[5:7]), int(a[8:10])
        yb, mb, db = int(b[:4]), int(b[5:7]), int(b[8:10])
        day_a = ya * 365 + ma * 30 + da
        day_b = yb * 365 + mb * 30 + db
        return abs(day_b - day_a) <= 1
    except (ValueError, IndexError):
        return False


@dataclass
class WeakSpot:
    passage_id: str
    error_rate: float
    n_attempts: int
    common_errors: List[str] = field(default_factory=list)


def detect_weak_spots(events: List[PracticeEvent],
                      min_attempts: int = 3,
                      error_threshold: float = 0.15) -> List[WeakSpot]:
    by_passage: Dict[str, List[PracticeEvent]] = {}
    for e in events:
        by_passage.setdefault(e.passage_id, []).append(e)

    weak = []
    for pid, evts in by_passage.items():
        if len(evts) < min_attempts:
            continue
        total_err = sum(e.note_errors for e in evts)
        total_notes = sum(e.total_notes for e in evts)
        rate = total_err / total_notes if total_notes > 0 else 0.0
        if rate >= error_threshold:
            weak.append(WeakSpot(passage_id=pid, error_rate=rate, n_attempts=len(evts)))
    weak.sort(key=lambda w: -w.error_rate)
    return weak


def improvement_trend(events: List[PracticeEvent], passage_id: str) -> Optional[float]:
    passage_events = [e for e in events if e.passage_id == passage_id]
    if len(passage_events) < 3:
        return None
    accuracies = [note_accuracy(e.note_errors, e.total_notes) for e in passage_events]
    x = np.arange(len(accuracies), dtype=float)
    y = np.array(accuracies)
    if x.std() < 1e-12:
        return 0.0
    slope = float(np.polyfit(x, y, 1)[0])
    return slope


@dataclass
class SM2Card:
    passage_id: str
    easiness: float = 2.5
    interval_days: int = 1
    repetitions: int = 0
    next_review: float = 0.0

    def review(self, quality: int, now: float) -> None:
        if quality < 0:
            quality = 0
        if quality > 5:
            quality = 5
        if quality >= 3:
            if self.repetitions == 0:
                self.interval_days = 1
            elif self.repetitions == 1:
                self.interval_days = 6
            else:
                self.interval_days = round(self.interval_days * self.easiness)
            self.repetitions += 1
        else:
            self.repetitions = 0
            self.interval_days = 1
        self.easiness = max(1.3, self.easiness + 0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02))
        self.next_review = now + self.interval_days * 86400.0


def schedule_review(cards: List[SM2Card], now: float) -> List[SM2Card]:
    due = [c for c in cards if c.next_review <= now]
    due.sort(key=lambda c: c.next_review)
    return due


def quality_from_accuracy(accuracy: float) -> int:
    if accuracy >= 0.95:
        return 5
    elif accuracy >= 0.85:
        return 4
    elif accuracy >= 0.7:
        return 3
    elif accuracy >= 0.5:
        return 2
    elif accuracy >= 0.3:
        return 1
    return 0


def learning_curve(session_accuracies: List[float]) -> Tuple[float, float]:
    if len(session_accuracies) < 2:
        return (0.0, session_accuracies[0] if session_accuracies else 0.0)
    y = np.array(session_accuracies)
    n = len(y)
    x = np.arange(n, dtype=float)
    asymptote = float(np.max(y))
    gap = asymptote - y[0]
    if gap < 0.01:
        return (0.0, asymptote)
    progress = (y - y[0]) / max(gap, 1e-12)
    progress = np.clip(progress, 1e-6, 1.0 - 1e-6)
    log_compl = np.log(1.0 - progress + 1e-12)
    valid = np.isfinite(log_compl) & (log_compl < 0)
    if np.sum(valid) < 2:
        return (0.1, asymptote)
    slope = float(-np.mean(log_compl[valid]) / max(np.mean(x[valid]), 1.0))
    return (max(slope, 0.0), asymptote)


def fit_learning_curve(accuracies: List[float]) -> Dict[str, float]:
    if len(accuracies) < 3:
        return {"learning_rate": 0.0, "asymptote": accuracies[-1] if accuracies else 0.0,
                "r_squared": 0.0}
    x = np.arange(len(accuracies), dtype=float)
    y = np.array(accuracies)
    coeffs = np.polyfit(x, y, 2)
    y_pred = np.polyval(coeffs, x)
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
    if len(accuracies) >= 5:
        early = np.mean(accuracies[:3])
        late = np.mean(accuracies[-3:])
        rate = (late - early) / len(accuracies)
    else:
        rate = float(coeffs[-2]) if len(coeffs) >= 2 else 0.0
    return {"learning_rate": rate, "asymptote": float(min(1.0, y_pred[-1] + rate * 5)),
            "r_squared": r2}


def spaced_repetition_schedule(
    mastery_level: float,
    last_practice_days_ago: int = 0,
) -> int:
    base_interval = 1.0
    if mastery_level >= 0.95:
        base_interval = 14.0
    elif mastery_level >= 0.85:
        base_interval = 7.0
    elif mastery_level >= 0.7:
        base_interval = 3.0
    elif mastery_level >= 0.5:
        base_interval = 2.0
    else:
        base_interval = 1.0
    decay = np.exp(-0.1 * last_practice_days_ago)
    effective_mastery = mastery_level * decay
    if effective_mastery < 0.5:
        return 1
    return max(int(base_interval), 1)


def session_efficiency(session: PracticeSession) -> float:
    if session.total_duration <= 0:
        return 0.0
    note_rate = session.total_notes / session.total_duration
    accuracy = note_accuracy(session.total_errors, session.total_notes)
    return note_rate * accuracy


def passage_mastery(events: List[PracticeEvent], passage_id: str,
                    accuracy_threshold: float = 0.95,
                    tempo_threshold: float = 0.95,
                    consecutive_required: int = 3) -> bool:
    relevant = [e for e in events if e.passage_id == passage_id]
    if len(relevant) < consecutive_required:
        return False
    recent = relevant[-consecutive_required:]
    for e in recent:
        if note_accuracy(e.note_errors, e.total_notes) < accuracy_threshold:
            return False
        if tempo_accuracy(e.tempo_target, e.tempo_actual) < tempo_threshold:
            return False
    return True


def analyze_practice(
    sessions: List[PracticeSession],
    total_notes: int = 100,
) -> PracticeReport:
    if not sessions:
        return PracticeReport(0.0, 0.0, 0.0, [], 0.0)
    accuracies = []
    all_errors: List[int] = []
    stabilities = []
    for s in sessions:
        acc = 1.0 - len(s.error_positions) / max(total_notes, 1)
        accuracies.append(max(acc, 0.0))
        all_errors.extend(s.error_positions)
        stabilities.append(tempo_stability(s.tempo_sequence))
    rate, asymptote = learning_curve(accuracies)
    spots = identify_problem_spots(all_errors, total_notes)
    latest_acc = accuracies[-1]
    avg_stability = float(np.mean(stabilities))
    mastery = 0.6 * latest_acc + 0.4 * avg_stability
    return PracticeReport(
        accuracy=latest_acc,
        tempo_stability=avg_stability,
        improvement_rate=rate,
        problem_spots=spots,
        estimated_mastery=mastery,
    )


def practice_summary(sessions: List[PracticeSession]) -> Dict[str, float]:
    if not sessions:
        return {"total_time": 0.0, "total_notes": 0, "avg_accuracy": 0.0, "n_sessions": 0}
    total_time = sum(s.total_duration for s in sessions)
    total_notes = sum(s.total_notes for s in sessions)
    total_errors = sum(s.total_errors for s in sessions)
    avg_acc = note_accuracy(total_errors, total_notes)
    return {
        "total_time": total_time,
        "total_notes": float(total_notes),
        "avg_accuracy": avg_acc,
        "n_sessions": float(len(sessions)),
        "avg_session_length": total_time / len(sessions),
    }
