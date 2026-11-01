"""Regime firewall — species/regime declaration enforcement.

Enforces: "declare or refuse, never default."
Every computation must declare its species and regime scope.
Mirrors Merlin convention where human/mouse are never pooled
without explicit declaration (see RB-020 study record).
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Species(Enum):
    HUMAN = "human"
    MOUSE = "mouse"
    RAT = "rat"
    IN_VITRO = "in_vitro"
    IN_SILICO = "in_silico"
    SYNTHETIC = "synthetic"


class Regime(Enum):
    THERAPEUTIC = "therapeutic"
    OBSERVATIONAL = "observational"
    EXPERIMENTAL = "experimental"
    BENCHMARK = "benchmark"
    SYNTHETIC = "synthetic"
    VALIDATION = "validation"


@dataclass
class SpeciesDeclaration:
    """Explicit species/regime declaration for a computation."""
    species: Species
    regime: Regime
    justification: str = ""
    cross_species: bool = False


MIN_DATA_DRIVEN_SAMPLES = 30
ALLOWED_SPECIES = {"human", "mouse"}


class RegimeFirewall:
    """Enforces species/regime declaration before computation.

    Two modes:
      1. Selection: check_species / select_regime (stateless, sample-count-based)
      2. Enforcement: declare / check / check_or_raise (stateful, pre-declaration required)
    """

    def __init__(self):
        self._declarations: set[tuple[str, str]] = set()
        self._bridges: list[tuple[str, str, str]] = []

    def declare(self, species: Species, regime: Regime, source: str = "") -> None:
        """Declare that (species, regime) is in scope for this computation."""
        self._declarations.add((species.value, regime.value))

    def check(self, species: Species, regime: Regime) -> bool:
        """Check if (species, regime) was declared."""
        return (species.value, regime.value) in self._declarations

    def check_or_raise(self, species: Species, regime: Regime) -> None:
        """Raise ValueError if (species, regime) was not declared."""
        if not self.check(species, regime):
            raise ValueError(
                f"Undeclared (species={species.value}, regime={regime.value}). "
                "Declare or refuse, never default."
            )

    def declare_bridge(self, from_species: Species, to_species: Species, justification: str = "") -> None:
        """Declare that from_species results can bridge to to_species context."""
        self._bridges.append((from_species.value, to_species.value, justification))

    def can_bridge(self, from_species: Species, to_species: Species) -> bool:
        """Check if a declared bridge exists."""
        return any(f == from_species.value and t == to_species.value for f, t, _ in self._bridges)

    def check_species(self, species_list: list[str]) -> bool:
        """All entries must be a single recognised species (human or mouse).

        Returns False on mixed species, unknown species, or empty list.
        """
        if not species_list:
            return False
        unique = set(species_list)
        return len(unique) == 1 and unique <= ALLOWED_SPECIES

    def select_regime(
        self, n_samples: int, n_conditions: int = 0, has_time_axis: bool = False,
    ) -> str:
        """Pick the training regime from sample/condition counts.

        Mirrors merlinaineuralplus2/backend/methods/method_router.py.
        """
        if has_time_axis:
            return "longitudinal"
        if n_samples >= MIN_DATA_DRIVEN_SAMPLES:
            return "data-driven"
        if n_conditions >= 2:
            return "prior-contrast"
        return "prior-mean"

    @staticmethod
    def validate(decl: SpeciesDeclaration) -> tuple[bool, str]:
        if decl.species == Species.HUMAN and decl.regime == Regime.EXPERIMENTAL:
            if not decl.justification:
                return False, "Human experimental data requires justification"
        if decl.cross_species and decl.species != Species.IN_SILICO:
            if not decl.justification:
                return False, "Cross-species comparison requires justification"
        return True, ""

    @staticmethod
    def check_compatibility(a: SpeciesDeclaration, b: SpeciesDeclaration) -> tuple[bool, str]:
        if a.species != b.species:
            if not (a.cross_species and b.cross_species):
                return False, (
                    f"Species mismatch: {a.species.value} vs {b.species.value}. "
                    "Set cross_species=True on both declarations to allow."
                )
        if a.regime != b.regime:
            return False, (
                f"Regime mismatch: {a.regime.value} vs {b.regime.value}. "
                "Analyses must share a regime unless explicitly bridged."
            )
        return True, ""
