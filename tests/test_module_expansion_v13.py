"""Tests for expansion round 13: collaboration version control, role assignment, arrangement reduction, pitch-class set theory."""
from __future__ import annotations

import pytest

from models.music_collaboration.version_control import (
    NoteEvent, DiffType, compute_musical_diff, apply_diffs,
    merge_revisions, revision_distance, make_revision,
)
from models.music_collaboration.role_assignment import (
    CollaboratorRole, SkillProfile,
    role_compatibility, assign_roles, suggest_voice_ranges,
    workload_balance, ensemble_diversity,
)
from models.music_arrangement.reduction import (
    ScoreNote, piano_reduce, extract_melody, extract_bass_line,
    condense_parts, lead_sheet, reduction_quality, voice_count_over_time,
)
from models.music_theory.set_theory import (
    pitch_class_set, normal_form, prime_form, interval_vector,
    transpose, invert, complement, is_subset, z_related,
    lookup_forte, similarity_index,
)


class TestVersionControl:
    def test_empty_diff(self):
        assert compute_musical_diff([], []) == []

    def test_add_diff(self):
        new = [NoteEvent(60, 0.0, 1.0, 0)]
        diffs = compute_musical_diff([], new)
        assert len(diffs) == 1
        assert diffs[0].diff_type == DiffType.ADD

    def test_delete_diff(self):
        old = [NoteEvent(60, 0.0, 1.0, 0)]
        diffs = compute_musical_diff(old, [])
        assert len(diffs) == 1
        assert diffs[0].diff_type == DiffType.DELETE

    def test_no_change(self):
        events = [NoteEvent(60, 0.0, 1.0, 0)]
        assert len(compute_musical_diff(events, list(events))) == 0

    def test_roundtrip(self):
        old = [NoteEvent(60, 0.0, 1.0, 0)]
        new = [NoteEvent(60, 0.0, 1.0, 0), NoteEvent(64, 4.0, 2.0, 0)]
        diffs = compute_musical_diff(old, new)
        result = apply_diffs(old, diffs)
        assert len(result) == 2

    def test_merge_no_conflict(self):
        base = [NoteEvent(60, 0.0, 1.0, 0)]
        rev_a = [NoteEvent(60, 0.0, 1.0, 0), NoteEvent(64, 4.0, 1.0, 0)]
        rev_b = [NoteEvent(60, 0.0, 1.0, 0), NoteEvent(67, 8.0, 1.0, 1)]
        merged, conflicts = merge_revisions(base, rev_a, rev_b)
        assert len(conflicts) == 0

    def test_merge_with_conflict(self):
        base = [NoteEvent(60, 0.0, 1.0, 0)]
        rev_a = [NoteEvent(64, 0.0, 1.0, 0)]
        rev_b = [NoteEvent(67, 0.0, 1.0, 0)]
        _, conflicts = merge_revisions(base, rev_a, rev_b)
        assert len(conflicts) == 1

    def test_revision_distance(self):
        a = [NoteEvent(60, 0.0, 1.0, 0)]
        b = [NoteEvent(64, 0.0, 1.0, 0)]
        assert revision_distance(a, b) >= 1

    def test_make_revision(self):
        old = [NoteEvent(60, 0.0, 1.0, 0)]
        new = [NoteEvent(64, 0.0, 1.0, 0)]
        rev = make_revision("alice", old, new)
        assert rev.author == "alice"
        assert len(rev.diffs) >= 1


class TestRoleAssignment:
    def test_melodist_compatibility(self):
        p = SkillProfile(0.9, 0.2, 0.1)
        assert role_compatibility(CollaboratorRole.MELODIST, p) > 0.5

    def test_assign_empty(self):
        assert assign_roles([]) == []

    def test_assign_single(self):
        assignments = assign_roles([SkillProfile(0.9, 0.3, 0.2)], 1)
        assert len(assignments) == 1

    def test_assign_diverse(self):
        profiles = [
            SkillProfile(0.9, 0.1, 0.1),
            SkillProfile(0.1, 0.9, 0.1),
            SkillProfile(0.1, 0.1, 0.9),
        ]
        assignments = assign_roles(profiles, 3)
        assert len(assignments) == 3

    def test_voice_ranges(self):
        ranges = suggest_voice_ranges([CollaboratorRole.MELODIST, CollaboratorRole.BASSIST])
        assert ranges[CollaboratorRole.BASSIST][0] < ranges[CollaboratorRole.MELODIST][0]

    def test_workload_equal(self):
        assert workload_balance([1.0, 1.0, 1.0]) == pytest.approx(0.0)

    def test_workload_unequal(self):
        assert workload_balance([0.0, 0.0, 10.0]) > 0.3

    def test_diversity(self):
        a = SkillProfile(1.0, 0.0, 0.0, arranging_skill=0.0, leadership_skill=0.0)
        b = SkillProfile(0.0, 1.0, 0.0, arranging_skill=0.0, leadership_skill=0.0)
        assert ensemble_diversity([a, b]) > 0.5


class TestArrangementReduction:
    def test_piano_reduce_empty(self):
        r = piano_reduce([])
        assert r.right_hand == [] and r.left_hand == []

    def test_piano_reduce_split(self):
        notes = [ScoreNote(72, 0.0, 1.0), ScoreNote(48, 0.0, 1.0)]
        r = piano_reduce(notes, split_point=60)
        assert len(r.right_hand) == 1
        assert len(r.left_hand) == 1
        assert r.right_hand[0].pitch == 72
        assert r.left_hand[0].pitch == 48

    def test_extract_melody(self):
        notes = [ScoreNote(60, 0.0, 1.0), ScoreNote(72, 0.0, 1.0), ScoreNote(64, 1.0, 1.0)]
        mel = extract_melody(notes)
        assert mel[0].pitch == 72
        assert mel[1].pitch == 64

    def test_extract_bass(self):
        notes = [ScoreNote(60, 0.0, 1.0), ScoreNote(48, 0.0, 1.0)]
        bass = extract_bass_line(notes)
        assert bass[0].pitch == 48

    def test_condense_parts(self):
        notes = [ScoreNote(60 + i, 0.0, 1.0) for i in range(8)]
        condensed = condense_parts(notes, max_voices=4)
        onset_0 = [n for n in condensed if n.onset == 0.0]
        assert len(onset_0) <= 4

    def test_lead_sheet(self):
        notes = [ScoreNote(72, 0.0, 1.0), ScoreNote(48, 0.0, 1.0), ScoreNote(60, 0.0, 1.0)]
        mel, bass = lead_sheet(notes)
        assert mel[0].pitch == 72
        assert bass[0].pitch == 48

    def test_reduction_quality(self):
        orig = [ScoreNote(60, 0.0, 1.0), ScoreNote(64, 0.0, 1.0)]
        assert reduction_quality(orig, orig) == 1.0
        assert reduction_quality(orig, []) == 0.0

    def test_voice_count(self):
        notes = [
            ScoreNote(60, 0.0, 2.0),
            ScoreNote(64, 0.0, 1.0),
            ScoreNote(67, 1.0, 1.0),
        ]
        counts = voice_count_over_time(notes, resolution=1.0)
        assert counts[0] == 2
        assert counts[1] == 2


class TestSetTheory:
    def test_pitch_class_set(self):
        pcs = pitch_class_set([60, 64, 67])
        assert pcs == frozenset({0, 4, 7})

    def test_normal_form_major_triad(self):
        pcs = frozenset({0, 4, 7})
        nf = normal_form(pcs)
        assert len(nf) == 3
        assert nf[0] == 0

    def test_prime_form_major_minor(self):
        major = frozenset({0, 4, 7})
        minor = frozenset({0, 3, 7})
        assert prime_form(major) == prime_form(minor)

    def test_interval_vector_major_triad(self):
        iv = interval_vector(frozenset({0, 4, 7}))
        assert iv == (0, 0, 1, 1, 1, 0)

    def test_transpose(self):
        pcs = frozenset({0, 4, 7})
        t = transpose(pcs, 3)
        assert t == frozenset({3, 7, 10})

    def test_invert(self):
        pcs = frozenset({0, 4, 7})
        inv = invert(pcs)
        assert 0 in inv
        assert (12 - 4) % 12 in inv

    def test_complement(self):
        pcs = frozenset({0, 4, 7})
        comp = complement(pcs)
        assert len(comp) == 9
        assert pcs & comp == frozenset()

    def test_subset(self):
        a = frozenset({0, 4})
        b = frozenset({0, 4, 7})
        assert is_subset(a, b)
        assert not is_subset(b, a)

    def test_forte_lookup_major(self):
        pcs = frozenset({0, 4, 7})
        forte = lookup_forte(pcs)
        assert forte == "3-11"

    def test_forte_lookup_diminished(self):
        pcs = frozenset({0, 3, 6})
        forte = lookup_forte(pcs)
        assert forte == "3-10"

    def test_similarity_identical(self):
        pcs = frozenset({0, 4, 7})
        assert similarity_index(pcs, pcs) == pytest.approx(1.0)

    def test_similarity_different(self):
        a = frozenset({0, 1, 2})
        b = frozenset({0, 4, 8})
        s = similarity_index(a, b)
        assert 0.0 <= s <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
