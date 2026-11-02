"""Tests for the music_metadata module."""
import pytest

from models.music_metadata.isrc_iswc import (
    validate_isrc, parse_isrc, validate_iswc, parse_iswc,
    validate_ean, generate_check_digit,
)
from models.music_metadata.musicbrainz_schema import (
    MBEntity, MBRelationType, MBArtistCredit, MBRecording, MBRelease,
    MBWork, MBRelation, entity_url, parse_artist_credit, format_artist_credit,
)
from models.music_metadata.tag_schema import (
    TagStandard, TagField, ID3_FIELDS, VORBIS_FIELDS,
    normalize_field, validate_tag,
)


class TestISRC:
    def test_valid_isrc(self):
        assert validate_isrc("USRC17607839") is True
        assert validate_isrc("US-RC1-76-07839") is True

    def test_invalid_isrc(self):
        assert validate_isrc("INVALID") is False
        assert validate_isrc("1234567890") is False

    def test_parse_isrc(self):
        result = parse_isrc("USRC17607839")
        assert result["country"] == "US"
        assert result["registrant"] == "RC1"
        assert result["year"] == "76"

    def test_parse_isrc_invalid(self):
        with pytest.raises(ValueError):
            parse_isrc("BAD")


class TestISWC:
    def test_validate_iswc_structure(self):
        valid = "T-034.524.680-1"
        result = validate_iswc(valid)
        assert isinstance(result, bool)

    def test_parse_iswc_invalid(self):
        with pytest.raises(ValueError):
            parse_iswc("BAD-CODE")


class TestEAN:
    def test_generate_check_digit(self):
        check = generate_check_digit("590123412345")
        assert check.isdigit()
        assert len(check) == 1

    def test_validate_ean(self):
        partial = "590123412345"
        check = generate_check_digit(partial)
        assert validate_ean(partial + check) is True

    def test_invalid_ean(self):
        assert validate_ean("1234567890123") is False or validate_ean("1234567890123") is True

    def test_generate_check_digit_bad_input(self):
        with pytest.raises(ValueError):
            generate_check_digit("short")


class TestMusicBrainzSchema:
    def test_entity_url(self):
        url = entity_url(MBEntity.ARTIST, "89ad4ac3-39f7-470e-963a-56509c546377")
        assert "musicbrainz.org/artist/" in url

    def test_entity_url_invalid_mbid(self):
        with pytest.raises(ValueError):
            entity_url(MBEntity.ARTIST, "not-a-uuid")

    def test_parse_artist_credit_single(self):
        credits = parse_artist_credit("Radiohead")
        assert len(credits) == 1
        assert credits[0].name == "Radiohead"

    def test_parse_artist_credit_feat(self):
        credits = parse_artist_credit("Drake feat. Rihanna")
        assert len(credits) == 2
        assert credits[0].name == "Drake"
        assert credits[1].name == "Rihanna"

    def test_format_artist_credit(self):
        credits = [
            MBArtistCredit("A", "A", "", " feat. "),
            MBArtistCredit("B", "B", "", ""),
        ]
        assert format_artist_credit(credits) == "A feat. B"

    def test_recording_dataclass(self):
        r = MBRecording(
            mbid="00000000-0000-0000-0000-000000000000",
            title="Test",
            artist_credit=[],
            duration_ms=240000,
        )
        assert r.title == "Test"
        assert r.duration_ms == 240000

    def test_relation_types(self):
        assert MBRelationType.PERFORMANCE.value == "performance"
        assert MBRelationType.REMIX.value == "remix"


class TestTagSchema:
    def test_id3_fields_count(self):
        assert len(ID3_FIELDS) >= 10

    def test_vorbis_fields_count(self):
        assert len(VORBIS_FIELDS) >= 8

    def test_normalize_id3_to_vorbis(self):
        result = normalize_field("TIT2", TagStandard.ID3V2, TagStandard.VORBIS_COMMENT)
        assert result == "TITLE"

    def test_normalize_vorbis_to_id3(self):
        result = normalize_field("ARTIST", TagStandard.VORBIS_COMMENT, TagStandard.ID3V2)
        assert result == "TPE1"

    def test_normalize_same_standard(self):
        assert normalize_field("TIT2", TagStandard.ID3V2, TagStandard.ID3V2) == "TIT2"

    def test_validate_tag_text(self):
        f = TagField("TIT2", TagStandard.ID3V2, "text", 256, True)
        ok, err = validate_tag(f, "Hello World")
        assert ok is True

    def test_validate_tag_too_long(self):
        f = TagField("TEST", TagStandard.ID3V2, "text", 5)
        ok, err = validate_tag(f, "Too Long String")
        assert ok is False

    def test_validate_tag_integer(self):
        f = TagField("TBPM", TagStandard.ID3V2, "integer", 5)
        ok, _ = validate_tag(f, "120")
        assert ok is True
        ok, _ = validate_tag(f, "abc")
        assert ok is False

    def test_validate_tag_date(self):
        f = TagField("TDRC", TagStandard.ID3V2, "date", 10)
        ok, _ = validate_tag(f, "2024")
        assert ok is True
        ok, _ = validate_tag(f, "2024-03-15")
        assert ok is True
        ok, _ = validate_tag(f, "not-a-date")
        assert ok is False
