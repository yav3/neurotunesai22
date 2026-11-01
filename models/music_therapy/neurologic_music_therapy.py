"""Neurologic Music Therapy (NMT) techniques and protocol modeling."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class NMTTechnique(Enum):
    RAS = "rhythmic_auditory_stimulation"
    PSE = "patterned_sensory_enhancement"
    TIMP = "therapeutic_instrumental_music_performance"
    MSIT = "musical_sensory_orientation_training"
    MEFT = "musical_executive_function_training"
    AST = "auditory_speech_training"
    MIT = "melodic_intonation_therapy"
    MPC = "music_psychotherapy_and_counseling"
    SYCOM = "symbolic_communication_through_music"


class ClinicalDomain(Enum):
    SENSORIMOTOR = "sensorimotor"
    SPEECH_LANGUAGE = "speech_language"
    COGNITIVE = "cognitive"


NMT_DOMAIN_MAP: dict[NMTTechnique, ClinicalDomain] = {
    NMTTechnique.RAS: ClinicalDomain.SENSORIMOTOR,
    NMTTechnique.PSE: ClinicalDomain.SENSORIMOTOR,
    NMTTechnique.TIMP: ClinicalDomain.SENSORIMOTOR,
    NMTTechnique.MSIT: ClinicalDomain.COGNITIVE,
    NMTTechnique.MEFT: ClinicalDomain.COGNITIVE,
    NMTTechnique.AST: ClinicalDomain.SPEECH_LANGUAGE,
    NMTTechnique.MIT: ClinicalDomain.SPEECH_LANGUAGE,
}


@dataclass
class NMTProtocol:
    technique: NMTTechnique
    domain: ClinicalDomain
    target_function: str
    tempo_range: tuple[float, float]
    duration_minutes: int
    repetitions: int
    progression_criteria: str


RAS_PROTOCOL = NMTProtocol(
    technique=NMTTechnique.RAS,
    domain=ClinicalDomain.SENSORIMOTOR,
    target_function="gait_training",
    tempo_range=(60, 120),
    duration_minutes=20,
    repetitions=3,
    progression_criteria="stable cadence within 5% of target for 2 consecutive sessions",
)


def ras_tempo_progression(
    baseline_cadence: float,
    target_cadence: float,
    n_sessions: int,
) -> list[float]:
    """Linear-then-plateau tempo curve for RAS gait training."""
    if n_sessions <= 0:
        return []
    ramp_sessions = max(1, int(n_sessions * 0.7))
    tempos = []
    for i in range(n_sessions):
        if i < ramp_sessions:
            t = baseline_cadence + (target_cadence - baseline_cadence) * (i / ramp_sessions)
        else:
            t = target_cadence
        tempos.append(round(t, 1))
    return tempos


def pse_movement_mapping(
    movement_phases: list[str],
    musical_elements: dict[str, str],
) -> dict[str, str]:
    """Map movement phases to musical cues for PSE."""
    mapping = {}
    defaults = {
        "preparation": "ascending melodic line",
        "execution": "rhythmic accent on downbeat",
        "hold": "sustained chord",
        "release": "descending resolution",
        "rest": "silence or soft drone",
    }
    for phase in movement_phases:
        mapping[phase] = musical_elements.get(phase, defaults.get(phase, "neutral accompaniment"))
    return mapping


def mit_syllable_intoning(
    text: str,
    base_pitch: int = 60,
) -> list[tuple[int, float]]:
    """Melodic Intonation Therapy: pitch contour from prosodic heuristic.

    Stressed syllables → higher pitch, unstressed → lower,
    clause-final → descending.
    """
    words = text.split()
    result: list[tuple[int, float]] = []
    for i, word in enumerate(words):
        syllables = max(1, (len(word) + 1) // 3)
        for s in range(syllables):
            is_stressed = (s == 0)
            is_final = (i == len(words) - 1 and s == syllables - 1)
            if is_final:
                pitch = base_pitch - 3
            elif is_stressed:
                pitch = base_pitch + 2
            else:
                pitch = base_pitch
            duration = 0.5 if is_stressed else 0.3
            result.append((pitch, duration))
    return result


def timp_instrument_assignment(motor_targets: list[str]) -> dict[str, str]:
    """Map motor rehabilitation goals to instrument choices."""
    instrument_map: dict[str, str] = {
        "grip_strength": "hand drum / djembe",
        "finger_dexterity": "piano / xylophone",
        "wrist_rotation": "maracas / tambourine",
        "bilateral_coordination": "bongos / conga pair",
        "shoulder_range": "overhead cymbal / windchimes",
        "elbow_flexion": "guitar strumming",
        "trunk_rotation": "standing drum kit",
        "lower_extremity": "bass drum pedal / stomp box",
    }
    return {target: instrument_map.get(target, "adaptive percussion") for target in motor_targets}
