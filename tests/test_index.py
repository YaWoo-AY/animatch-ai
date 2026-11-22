import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from build_index import build_text, to_metadata, collection_name, MODELS
from search import build_where


def make_rec(**over):
    rec = {
        "anilist_id": 1, "mal_id": 11, "title": "Show", "synopsis": "A detective hunts a thief.",
        "genres": ["Mystery", "Drama"], "tags": ["Time Loop", "Primarily Teen Cast", "Detective"],
        "episodes": 12, "year": 2020, "score": 80, "popularity": 500, "format": "TV",
        "cover": "http://img", "franchise_id": 1,
    }
    rec.update(over)
    return rec


# ---- embedding text ----

def test_build_text_puts_title_genres_themes_before_synopsis():
    t = build_text(make_rec())
    assert t.index("Show.") < t.index("Genres:") < t.index("Themes:") < t.index("A detective")


def test_build_text_drops_structural_tags_but_keeps_themes():
    t = build_text(make_rec())
    assert "Primarily Teen Cast" not in t
    assert "Time Loop" in t and "Detective" in t


def test_build_text_handles_no_genres_and_no_tags():
    t = build_text(make_rec(genres=[], tags=[]))
    assert t == "Show. A detective hunts a thief."


def test_build_text_caps_the_number_of_tags():
    tags = [f"Tag{i}" for i in range(30)]
    t = build_text(make_rec(tags=tags))
    assert "Tag9" in t and "Tag10" not in t


# ---- metadata ----

def test_to_metadata_has_no_none_and_only_chroma_types():
    meta = to_metadata(make_rec(mal_id=None, episodes=None, year=None, score=None,
                                popularity=None, format=None, cover=None))
    assert all(isinstance(v, (str, int, float, bool)) for v in meta.values())
    assert meta["mal_id"] == -1 and meta["episodes"] == -1
    assert meta["year"] == -1 and meta["score"] == -1
    assert meta["format"] == "" and meta["cover"] == ""


def test_to_metadata_keeps_real_values_and_joins_genres():
    meta = to_metadata(make_rec())
    assert meta["episodes"] == 12 and meta["mal_id"] == 11
    assert meta["genres"] == "Mystery|Drama"
    assert meta["franchise_id"] == 1


# ---- naming and filters ----

def test_collection_names_are_per_model():
    assert collection_name("minilm") == "anime_minilm"
    assert set(MODELS) == {"minilm", "bge"}


def test_where_is_none_without_filters():
    assert build_where() is None


def test_episode_cap_excludes_unknown_episodes():
    w = build_where(max_episodes=26)
    assert w == {"$and": [{"episodes": {"$gte": 1}}, {"episodes": {"$lte": 26}}]}


def test_single_filter_is_not_wrapped_in_and():
    assert build_where(min_year=2010) == {"year": {"$gte": 2010}}


def test_several_filters_are_combined_with_and():
    w = build_where(max_episodes=13, min_year=2000, min_score=70)
    assert len(w["$and"]) == 4


# ---- variants: title off, spoiler tags on ----

def test_build_text_can_drop_the_title():
    from build_index import build_text as bt
    t = bt(make_rec(), title=False)
    assert "Show." not in t
    assert t.startswith("Genres:") and "A detective hunts a thief." in t


def test_spoiler_tags_are_off_by_default():
    rec = make_rec(tags=["Detective"], spoiler_tags=["Survival"])
    assert "Survival" not in build_text(rec)
    assert "Plot elements" not in build_text(rec)


def test_spoiler_tags_are_added_when_asked_and_before_the_synopsis():
    rec = make_rec(tags=["Detective"], spoiler_tags=["Survival", "Time Loop"])
    t = build_text(rec, spoilers=True)
    assert "Plot elements: Survival, Time Loop." in t
    assert t.index("Themes:") < t.index("Plot elements:") < t.index("A detective")


def test_spoiler_tags_use_the_stoplist_and_a_cap():
    rec = make_rec(tags=[], spoiler_tags=["Heterosexual"] + [f"S{i}" for i in range(20)])
    t = build_text(rec, spoilers=True)
    assert "Heterosexual" not in t
    assert "S7" in t and "S8" not in t


def test_records_without_spoiler_tags_still_work():
    rec = make_rec()
    rec.pop("spoiler_tags", None)
    assert build_text(rec, spoilers=True) == build_text(rec)


def test_variant_collection_names_keep_the_base_name():
    from build_index import VARIANTS
    assert collection_name("bge") == "anime_bge"
    assert collection_name("bge", "base") == "anime_bge"
    assert collection_name("bge", "notitle") == "anime_bge_notitle"
    assert set(VARIANTS) == {"base", "notitle", "spoilers", "notitle_spoilers"}
