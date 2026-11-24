import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from franchise import build_franchises


def rec(anilist_id, *relations, fmt="TV"):
    """Make a minimal record. Each relation is (type, target_id, media_type)."""
    return {
        "anilist_id": anilist_id,
        "format": fmt,
        "relations": [{"type": t, "anilist_id": i, "media_type": m} for t, i, m in relations],
    }


def test_sequel_chain_shares_one_franchise():
    records = [
        rec(1, ("SEQUEL", 2, "ANIME")),
        rec(2, ("PREQUEL", 1, "ANIME"), ("SEQUEL", 3, "ANIME")),
        rec(3, ("PREQUEL", 2, "ANIME")),
    ]
    m = build_franchises(records)
    assert m[1] == m[2] == m[3] == 1


def test_connected_through_an_uncrawled_title():
    # 1 and 3 are in our data; 2 (the link between them) is not.
    records = [rec(1, ("SEQUEL", 2, "ANIME")), rec(3, ("PREQUEL", 2, "ANIME"))]
    m = build_franchises(records)
    assert m[1] == m[3]


def test_unrelated_titles_stay_separate():
    m = build_franchises([rec(1), rec(2)])
    assert m[1] != m[2]


def test_manga_edges_are_ignored():
    # Two anime that merely share a manga source must not be merged.
    records = [rec(1, ("SEQUEL", 900, "MANGA")), rec(2, ("SEQUEL", 900, "MANGA"))]
    m = build_franchises(records)
    assert m[1] != m[2]


def test_spin_off_and_adaptation_are_not_linked():
    records = [rec(1, ("SPIN_OFF", 2, "ANIME"), ("ALTERNATIVE", 3, "ANIME")), rec(2), rec(3)]
    m = build_franchises(records)
    assert len({m[1], m[2], m[3]}) == 3


def test_side_story_and_summary_are_linked():
    records = [rec(1, ("SIDE_STORY", 2, "ANIME"), ("SUMMARY", 3, "ANIME")), rec(2, fmt="OVA"), rec(3)]
    m = build_franchises(records)
    assert m[1] == m[2] == m[3]


def test_franchise_id_is_the_smallest_id():
    records = [rec(50, ("SEQUEL", 7, "ANIME")), rec(7, ("PREQUEL", 50, "ANIME"))]
    m = build_franchises(records)
    assert m[50] == m[7] == 7


def test_timelines_with_own_chains_are_not_pulled_under_the_parent():
    # Mirrors Gundam: Wing and Seed each have a chain and both point PARENT -> original.
    records = [
        rec(1, ("SIDE_STORY", 10, "ANIME"), ("SIDE_STORY", 20, "ANIME")),   # original
        rec(10, ("PARENT", 1, "ANIME"), ("SEQUEL", 11, "ANIME")),           # Wing
        rec(11, ("PREQUEL", 10, "ANIME")),                                  # Endless Waltz
        rec(20, ("PARENT", 1, "ANIME"), ("SEQUEL", 21, "ANIME")),           # Seed
        rec(21, ("PREQUEL", 20, "ANIME")),                                  # Destiny
    ]
    m = build_franchises(records)
    assert m[10] == m[11]
    assert m[20] == m[21]
    assert len({m[1], m[10], m[20]}) == 3


def test_standalone_side_story_joins_its_parent():
    # Like Gundam 0080: no chain of its own, so it joins the original.
    records = [rec(1, ("SIDE_STORY", 5, "ANIME")), rec(5, ("PARENT", 1, "ANIME"), fmt="OVA")]
    m = build_franchises(records)
    assert m[1] == m[5]


def test_standalone_with_two_different_parents_stays_alone():
    # Two real parents (each with its own sequel), one standalone child pointing at both.
    records = [
        rec(1, ("SEQUEL", 3, "ANIME")), rec(3, ("PREQUEL", 1, "ANIME")),
        rec(2, ("SEQUEL", 4, "ANIME")), rec(4, ("PREQUEL", 2, "ANIME")),
        rec(5, ("PARENT", 1, "ANIME"), ("PARENT", 2, "ANIME"), fmt="OVA"),
    ]
    m = build_franchises(records)
    assert m[1] == m[3] and m[2] == m[4]
    assert len({m[1], m[2], m[5]}) == 3


def test_side_story_of_an_uncrawled_title_stays_alone():
    m = build_franchises([rec(5, ("PARENT", 999, "ANIME"), fmt="OVA")])
    assert m[5] == 5


def test_movie_only_chain_attaches_to_its_parent_series():
    # Like the Pokemon movies: a chain of movies, no TV series of its own.
    records = [
        rec(1, ("SEQUEL", 2, "ANIME")), rec(2, ("PREQUEL", 1, "ANIME")),        # parent TV chain
        rec(10, ("SEQUEL", 11, "ANIME"), ("PARENT", 1, "ANIME"), fmt="MOVIE"),
        rec(11, ("PREQUEL", 10, "ANIME"), fmt="MOVIE"),
    ]
    m = build_franchises(records)
    assert m[10] == m[11] == m[1] == m[2]


def test_movie_only_chain_with_two_different_parents_stays_alone():
    records = [
        rec(1, ("SEQUEL", 2, "ANIME")), rec(2, ("PREQUEL", 1, "ANIME")),
        rec(3, ("SEQUEL", 4, "ANIME")), rec(4, ("PREQUEL", 3, "ANIME")),
        rec(10, ("SEQUEL", 11, "ANIME"), ("PARENT", 1, "ANIME"), ("PARENT", 3, "ANIME"), fmt="MOVIE"),
        rec(11, ("PREQUEL", 10, "ANIME"), fmt="MOVIE"),
    ]
    m = build_franchises(records)
    assert m[10] == m[11]
    assert len({m[1], m[3], m[10]}) == 3


def test_chain_with_a_tv_series_still_stays_separate():
    records = [
        rec(1, ("SIDE_STORY", 10, "ANIME"), ("SEQUEL", 2, "ANIME")),   # original has its own chain
        rec(2, ("PREQUEL", 1, "ANIME")),
        rec(10, ("PARENT", 1, "ANIME"), ("SEQUEL", 11, "ANIME"), fmt="TV"),
        rec(11, ("PREQUEL", 10, "ANIME"), fmt="MOVIE"),
    ]
    m = build_franchises(records)
    assert m[10] == m[11] and m[10] != m[1] and m[1] == m[2]


def test_lone_tv_series_with_a_parent_link_stays_separate():
    # Like G Gundam or Turn A: a standalone TV show hung under the original by PARENT.
    records = [
        rec(1, ("SIDE_STORY", 5, "ANIME"), ("SEQUEL", 2, "ANIME")),
        rec(2, ("PREQUEL", 1, "ANIME")),
        rec(5, ("PARENT", 1, "ANIME"), fmt="TV"),
    ]
    m = build_franchises(records)
    assert m[5] != m[1] and m[1] == m[2]


def test_lone_ova_with_a_parent_link_joins_the_parent():
    # Like Gundam 0080.
    records = [
        rec(1, ("SIDE_STORY", 5, "ANIME"), ("SEQUEL", 2, "ANIME")),
        rec(2, ("PREQUEL", 1, "ANIME")),
        rec(5, ("PARENT", 1, "ANIME"), fmt="OVA"),
    ]
    m = build_franchises(records)
    assert m[5] == m[1] == m[2]


def test_franchise_of_lists_members_and_relations():
    import franchise_of
    recs = [
        {"anilist_id": 1, "title": "Show", "format": "TV", "year": 2021, "franchise_id": 1,
         "relations": [{"anilist_id": 2, "media_type": "ANIME", "type": "ALTERNATIVE"}]},
        {"anilist_id": 2, "title": "Show (2012 OVA)", "format": "OVA", "year": 2012, "franchise_id": 2,
         "relations": [{"anilist_id": 99, "media_type": "ANIME", "type": "SEQUEL"}]},
    ]
    out = "\n".join(franchise_of.describe(recs, "show (2012"))
    assert "franchise 2" in out and "same franchise: none" in out
    assert "SEQUEL" in out and "99 (not in the index)" in out
    assert "ALTERNATIVE" in "\n".join(franchise_of.describe(recs, "Show"))
