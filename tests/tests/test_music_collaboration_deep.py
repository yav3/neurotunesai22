"""Tests for music collaboration: version control and role assignment."""
from __future__ import annotations

import pytest

from models.music_collaboration.version_control import (
    NoteEvent, DiffType, MusicalDiff, MergeConflict,
    compute_musical_diff, apply_diffs, merge_revisions,
    revision_distance, make_revision,
)
from models.music_collaboration.role_assignment import (
    CollaboratorRole, SkillProfile,
    role_compatibility, assign_roles, suggest_voice_ranges,
    workload_balance, ensemble_diversity,
)


class TestVersionControl:
    def test_empty_diff(self):
        assert compute_musical_diff([], []) == []

    def test_add_notes(self):
        new = [NoteEvent(60, 0.0, 1.0, 0)]
        diffs = compute_musical_diff([], new)
        assert len(diffs) == 1
        assert diffs[0].diff_type == DiffType.ADD

    def test_delete_notes(self):
        old = [NoteEvent(60, 0.0, 1.0, 0)]
        diffs = compute_musical_diff(old, [])
        assert len(diffs) == 1
        assert diffs[0].diff_type == DiffType.DELETE

    def test_modify_notes(self):
        old = [NoteEvent(60, 0.0, 1.0, 0)]
        new = [NoteEvent(64, 0.0, 1.0, 0)]
        diffs = compute_musical_diff(old, new)
        assert len(diffs) == 1
        assert diffs[0].diff_type == DiffType.MODIFY

    def test_no_change(self):
        events = [NoteEvent(60, 0.0, 1.0, 0), NoteEvent(64, 2.0, 1.0, 0)]
        diffs = compute_musical_diff(events, list(events))
        assert len(diffs) == 0

    def test_apply_add_diff(self):
        base = [NoteEvent(60, 0.0, 1.0, 0)]
        new_note = NoteEvent(67, 4.0, 1.0, 0)
        diffs = [MusicalDiff(1, 0, DiffType.ADD, [], [new_note])]
        result = apply_diffs(base, diffs)
        assert len(result) == 2

    def test_apply_delete_diff(self):
        base = [NoteEvent(60, 0.0, 1.0, 0), NoteEvent(64, 4.0, 1.0, 0)]
        diffs = [MusicalDiff(0, 0, DiffType.DELETE, [base[0]], [])]
        result = apply_diffs(base, diffs)
        assert len(result) == 1
        assert result[0].pitch == 64

    def test_roundtrip(self):
        old = [NoteEvent(60, 0.0, 1.0, 0)]
        new = [NoteEvent(60, 0.0, 1.0, 0), NoteEvent(64, 4.0, 2.0, 0)]
        diffs = compute_musical_diff(old, new)
        result = apply_diffs(old, diffs)
        assert len(result) == len(new)

    def test_merge_no_conflict(self):
        base = [NoteEvent(60, 0.0, 1.0, 0)]
        rev_a = [NoteEvent(60, 0.0, 1.0, 0), NoteEvent(64, 4.0, 1.0, 0)]
        rev_b = [NoteEvent(60, 0.0, 1.0, 0), NoteEvent(67, 8.0, 1.0, 1)]
        merged, conflicts = merge_revisions(base, rev_a, rev_b)
        assert len(conflicts) == 0
        assert len(merged) == 3

    def test_merge_with_conflict(self):
        base = [NoteEvent(60, 0.0, 1.0, 0)]
        rev_a = [NoteEvent(64, 0.0, 1.0, 0)]
        rev_b = [NoteEvent(67, 0.0, 1.0, 0)]
        merged, conflicts = merge_revisions(base, rev_a, rev_b)
        assert len(conflicts) == 1
        assert conflicts[0].measure == 0

    def test_revision_distance_identical(self):
        events = [NoteEvent(60, 0.0, 1.0, 0)]
        assert revision_distance(events, events) == 0

    def test_revision_distance_different(self):
        a = [NoteEvent(60, 0.0, 1.0, 0)]
        b = [NoteEvent(64, 0.0, 1.0, 0)]
        assert revision_distance(a, b) == 1

    def test_make_revision(self):
        old = [NoteEvent(60, 0.0, 1.0, 0)]
        new = [NoteEvent(64, 0.0, 1.0, 0)]
        rev = make_revision("alice", old, new)
        assert len(rev.revision_id) == 12
        assert rev.author == "alice"
        assert len(rev.diffs) == 1

    def test_multi_voice_diff(self):
        old = [NoteEvent(60, 0.0, 1.0, 0), NoteEvent(48, 0.0, 1.0, 1)]
        new = [NoteEvent(64, 0.0, 1.0, 0), NoteEvent(48, 0.0, 1.0, 1)]
        diffs = compute_musical_diff(old, new)
        assert len(diffs) == 1
        assert diffs[0].voice == 0


class TestRoleAssignment:
    def test_role_compatibility_melodist(self):
        p = SkillProfile(melodic_strength=0.9, harmonic_strength=0.3, rhythmic_strength=0.4)
        score = role_compatibility(CollaboratorRole.MELODIST, p)
        assert score > 0.5

    def test_role_compatibility_rhythmist(self):
        p = SkillProfile(melodic_strength=0.2, harmonic_strength=0.2, rhythmic_strength=0.95)
        score = role_compatibility(CollaboratorRole.RHYTHMIST, p)
        assert score > 0.5

    def test_assign_roles_empty(self):
        assert assign_roles([]) == []

    def test_assign_roles_single(self):
        profiles = [SkillProfile(0.9, 0.3, 0.2)]
        assignments = assign_roles(profiles, 1)
        assert len(assignments) == 1
        idx, role = assignments[0]
        assert idx == 0

    def test_assign_roles_diverse_team(self):
        profiles = [
            SkillProfile(0.9, 0.2, 0.1),
            SkillProfile(0.1, 0.9, 0.2),
            SkillProfile(0.1, 0.2, 0.9),
        ]
        assignments = assign_roles(profiles, 3)
        assert len(assignments) == 3
        indices = {a[0] for a in assignments}
        assert indices == {0, 1, 2}

    def test_suggest_voice_ranges(self):
        roles = [CollaboratorRole.MELODIST, CollaboratorRole.BASSIST]
        ranges = suggest_voice_ranges(roles)
        assert ranges[CollaboratorRole.BASSIST][0] < ranges[CollaboratorRole.MELODIST][0]

    def test_workload_balance_equal(self):
        assert workload_balance([1.0, 1.0, 1.0]) == pytest.approx(0.0)

    def test_workload_balance_unequal(self):
        g = workload_balance([0.0, 0.0, 10.0])
        assert g > 0.3

    def test_workload_balance_edge(self):
        assert workload_balance([]) == 0.0
        assert workload_balance([5.0]) == 0.0

    def test_ensemble_diversity_identical(self):
        p = SkillProfile(0.5, 0.5, 0.5, arranging_skill=0.5, leadership_skill=0.5)
        assert ensemble_diversity([p, p]) == pytest.approx(0.0)

    def test_ensemble_diversity_different(self):
        a = SkillProfile(1.0, 0.0, 0.0, arranging_skill=0.0, leadership_skill=0.0)
        b = SkillProfile(0.0, 1.0, 0.0, arranging_skill=0.0, leadership_skill=0.0)
        d = ensemble_diversity([a, b])
        assert d > 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
