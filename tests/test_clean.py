import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fetch_anilist import clean_text, clean_record


def test_clean_text_strips_html():
    assert clean_text("A<br>B <i>C</i> &amp; D") == "A B C & D"


def test_clean_text_handles_none():
    assert clean_text(None) == ""


def test_clean_text_strips_source_credit():
    assert clean_text("Great show. (Source: Crunchyroll)") == "Great show."
    assert clean_text("Great show. [Written by MAL Rewrite]") == "Great show."


def test_clean_record_handles_nulls_and_spoiler_tags():
    raw = {
        "id": 1, "idMal": None,
        "title": {"romaji": "Foo", "english": None},
        "description": None, "genres": None,
        "tags": [
            {"name": "Time Loop", "rank": 90, "isMediaSpoiler": False},
            {"name": "Twist", "rank": 95, "isMediaSpoiler": True},
            {"name": "Minor", "rank": 20, "isMediaSpoiler": False},
        ],
        "episodes": None, "seasonYear": None, "averageScore": None,
        "popularity": 10, "format": "TV", "coverImage": None, "relations": None,
    }
    rec = clean_record(raw)
    assert rec["title"] == "Foo"
    assert rec["mal_id"] is None
    assert rec["synopsis"] == ""
    assert rec["tags"] == ["Time Loop"]
    assert rec["spoiler_tags"] == ["Twist"]  # kept separately, for retrieval only
    assert rec["relations"] == []
