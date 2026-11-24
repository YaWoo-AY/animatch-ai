import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import evaluate as ev
import recommend
import scoring
import taste
from taste import build_taste, entry_weight, explain, planned_franchises, seen_franchises


def entry(aid, fid, status="completed", score=None, progress=12, title=None):
    return {"anilist_id": aid, "franchise_id": fid, "status": status, "score": score,
            "progress": progress, "title": title or f"T{aid}"}


def unit(*v):
    v = np.array(v, dtype=float)
    return v / np.linalg.norm(v)


# ---- weights ----

def test_ratings_set_the_weight():
    assert entry_weight(entry(1, 1, score=10)) == 1.0
    assert abs(entry_weight(entry(1, 1, score=8)) - 0.6) < 1e-9
    assert entry_weight(entry(1, 1, score=7)) == 0.0          # lukewarm: no opinion
    assert entry_weight(entry(1, 1, score=5)) == -0.25
    assert entry_weight(entry(1, 1, score=2)) == -1.0


def test_status_is_used_only_when_there_is_no_score():
    assert entry_weight(entry(1, 1, "completed")) == taste.UNRATED_COMPLETED
    assert entry_weight(entry(1, 1, "watching", progress=3)) == taste.UNRATED_WATCHING
    assert entry_weight(entry(1, 1, "watching", progress=0)) == 0.0
    assert entry_weight(entry(1, 1, "planning")) == 0.0
    assert entry_weight(entry(1, 1, "paused")) == 0.0
    # a score beats the status: dropped but rated 9 still counts as a like
    assert abs(entry_weight(entry(1, 1, "dropped", score=9)) - 0.8) < 1e-9
    # planning stays neutral even with a score
    assert entry_weight(entry(1, 1, "planning", score=9)) == 0.0


def test_dropping_late_is_a_weaker_no_than_dropping_early():
    early = entry_weight(entry(1, 1, "dropped", progress=2), episodes=24)
    late = entry_weight(entry(1, 1, "dropped", progress=20), episodes=24)
    unknown_length = entry_weight(entry(1, 1, "dropped", progress=20), episodes=None)
    assert early == taste.UNRATED_DROPPED_EARLY and late == taste.UNRATED_DROPPED_LATE
    assert unknown_length == taste.UNRATED_DROPPED_EARLY


# ---- exclusion ----

def test_seen_franchises_include_dropped_and_exclude_planned():
    matched = [entry(1, 10, "completed"), entry(2, 20, "dropped"), entry(3, 30, "planning"),
               entry(4, 40, "watching"), entry(5, 50, "paused")]
    assert seen_franchises(matched) == {10, 20, 40, 50}
    assert planned_franchises(matched) == {30}


def test_a_planned_title_in_a_seen_franchise_is_not_reported_as_planned():
    matched = [entry(1, 10, "completed"), entry(2, 10, "planning")]
    assert planned_franchises(matched) == set()


# ---- taste vector ----

def test_vector_moves_toward_liked_titles_and_away_from_disliked_ones():
    emb = {1: unit(1, 0, 0), 2: unit(0, 1, 0), 3: unit(0, 0, 1)}
    matched = [entry(1, 1, score=10), entry(2, 2, score=1), entry(3, 3, "planning")]
    t = build_taste(matched, emb, alpha=0.5)
    assert t.vector @ emb[1] > 0.85       # (1, -0.5) normalised is 0.894 along the liked axis
    assert t.vector @ emb[2] < 0          # pushed away from the disliked title
    assert abs(np.linalg.norm(t.vector) - 1.0) < 1e-9
    no_dislike = build_taste(matched, emb, alpha=0.0)
    assert no_dislike.vector @ emb[2] == 0


def test_nothing_positive_gives_no_taste():
    emb = {1: unit(1, 0), 2: unit(0, 1)}
    assert build_taste([entry(1, 1, score=2), entry(2, 2, "planning")], emb) is None
    assert build_taste([], emb) is None


def test_a_long_franchise_counts_once():
    # five seasons of "A" and one title "B", both rated 10: the vector should sit midway, not near A
    emb = {i: unit(1, 0) for i in range(1, 6)}
    emb[9] = unit(0, 1)
    matched = [entry(i, 1, score=10) for i in range(1, 6)] + [entry(9, 2, score=10)]
    t = build_taste(matched, emb)
    assert abs(t.vector @ unit(1, 0) - t.vector @ unit(0, 1)) < 1e-9
    assert len(t.positives) == 2


def test_missing_embeddings_are_skipped_not_fatal():
    emb = {1: unit(1, 0)}
    t = build_taste([entry(1, 1, score=9), entry(2, 2, score=9)], emb)
    assert len(t.positives) == 1


def test_unrated_titles_cannot_outvote_rated_ones():
    # 1 rated 10 along axis x; 30 unrated completed along axis y. Without a cap y would win.
    emb = {0: unit(1, 0)}
    matched = [entry(0, 0, score=10)]
    for i in range(1, 31):
        emb[i] = unit(0, 1)
        matched.append(entry(i, i, "completed"))
    capped = build_taste(matched, emb, unrated_share=0.35)
    uncapped = build_taste(matched, emb, unrated_share=0.999999)
    assert capped.unrated_scale < 1
    assert capped.vector @ unit(1, 0) > capped.vector @ unit(0, 1) * 1.5   # x leads, y still present
    assert uncapped.vector @ unit(0, 1) > uncapped.vector @ unit(1, 0)      # the cap is what changed it


def test_unrated_only_users_still_get_a_taste():
    emb = {1: unit(1, 0), 2: unit(1, 1)}
    t = build_taste([entry(1, 1, "completed"), entry(2, 2, "completed")], emb)
    assert t is not None and t.unrated_scale == 1.0


def test_explain_prefers_rated_franchises_close_to_the_recommendation():
    emb = {1: unit(1, 0, 0), 2: unit(0, 1, 0), 3: unit(0, 0, 1), 4: unit(1, 1, 0)}
    matched = [entry(1, 1, score=10, title="Near"), entry(2, 2, score=9, title="Far"),
               entry(3, 3, "completed", title="Unrated but close"), entry(4, 4, score=9, title="Mid")]
    t = build_taste(matched, emb)
    names = [n for n, _ in explain(t, unit(1, 0.2, 0), n=2)]
    assert names == ["Near", "Mid"]


# ---- recommending over a small catalog ----

def rec_row(aid, fid, fmt="TV", year=2010, pop=100, eps=12, score=70, genres=("Action",)):
    return {"anilist_id": aid, "mal_id": None, "title": f"T{aid}", "franchise_id": fid, "format": fmt,
            "episodes": eps, "year": year, "score": score, "popularity": pop, "genres": list(genres),
            "tags": [], "cover": None}


PLAIN = "minilm@center=0@knn=0"      # taste vector only: no centering, no nearest-liked, no popularity


def small_world():
    """Six franchises on two axes: 'liked' direction (1,0) and 'disliked' direction (0,1)."""
    rows = [rec_row(1, 1, pop=50), rec_row(2, 2, pop=60),
            rec_row(3, 3, year=2012, pop=100),                    # close to the liked title
            rec_row(30, 3, "OVA", year=2015, pop=10),             # same franchise, even closer
            rec_row(4, 4, pop=500),                               # close to the disliked title
            rec_row(5, 5, pop=300),                               # plan-to-watch, close to the liked one
            rec_row(6, 6, pop=200)]
    vecs = {1: [1, 0], 2: [0, 1], 3: [0.9, 0.1], 30: [0.95, 0.05], 4: [0.1, 0.9], 5: [0.8, 0.2], 6: [0.7, 0.3]}
    catalog = scoring.Catalog(rows)
    M = np.array([vecs[r["anilist_id"]] for r in catalog.records], dtype=float)
    G = ev.genre_matrix(catalog.records)

    def matrices(name, variant):
        return G if name == "genre" else M
    return catalog, matrices


def world_matched():
    return [entry(1, 1, score=10, title="Liked"), entry(2, 2, score=1, title="Disliked"),
            entry(5, 5, "planning", title="Planned")]


def ids(recs):
    return [r["anilist_id"] for r in recs]


def test_seen_franchises_are_never_recommended_and_the_closest_leads():
    catalog, matrices = small_world()
    model, recs = recommend.recommend(world_matched(), catalog, matrices, PLAIN, k=10)
    assert ids(recs) == [3, 5, 6, 4]                  # liked-like first, disliked-like last, seen hidden
    assert recs[1]["planned"] is True and recs[0]["planned"] is False
    assert recs[0]["because"][0][0] == "Liked"


def test_plan_to_watch_can_be_hidden():
    catalog, matrices = small_world()
    _, recs = recommend.recommend(world_matched(), catalog, matrices, PLAIN, k=10, hide_planned=True)
    assert 5 not in ids(recs)


def test_the_franchise_start_is_shown_not_the_closest_title():
    catalog, matrices = small_world()
    _, recs = recommend.recommend(world_matched(), catalog, matrices, PLAIN, k=1)
    assert recs[0]["anilist_id"] == 3 and recs[0]["format"] == "TV"    # not the OVA (30) that is closer


def test_limits_are_checked_on_the_franchise_start():
    catalog, matrices = small_world()
    _, recs = recommend.recommend(world_matched(), catalog, matrices, PLAIN, k=10, limits={"min_year": 2013})
    assert 3 not in ids(recs)                         # its first entry is from 2012
    _, recs = recommend.recommend(world_matched(), catalog, matrices, PLAIN, k=10, limits={"max_episodes": 6})
    assert recs == []


def test_skip_popular_leaves_out_the_most_popular_franchises():
    catalog, matrices = small_world()
    _, recs = recommend.recommend(world_matched(), catalog, matrices, PLAIN, k=10, skip_popular=1)
    assert 4 not in ids(recs)                         # franchise 4 is the most popular one


def test_a_huge_popularity_weight_orders_the_results_by_popularity():
    catalog, matrices = small_world()
    _, recs = recommend.recommend(world_matched(), catalog, matrices, PLAIN + "@pop=1000", k=10)
    assert ids(recs) == [4, 5, 6, 3]


def test_retellings_of_seen_franchises_are_hidden_but_not_merged():
    catalog, matrices = small_world()
    for r in catalog.records:
        if r["anilist_id"] == 1:
            r["relations"] = [{"anilist_id": 6, "media_type": "ANIME", "type": "ALTERNATIVE"}]
    links = recommend.retelling_links(catalog.records)
    assert links[1] == {6} and links[6] == {1}                       # symmetric
    _, shown = recommend.recommend(world_matched(), catalog, matrices, PLAIN, k=10)
    assert 6 in ids(shown)                                           # visible without the links
    _, recs = recommend.recommend(world_matched(), catalog, matrices, PLAIN, k=10, retellings=links)
    assert 6 not in ids(recs) and 3 in ids(recs)


def test_only_alternative_links_between_anime_count_as_retellings():
    rows = [{**rec_row(1, 1), "relations": [{"anilist_id": 2, "media_type": "ANIME", "type": "OTHER"},
                                           {"anilist_id": 3, "media_type": "MANGA", "type": "ALTERNATIVE"}]},
            rec_row(2, 2), rec_row(3, 3)]
    assert recommend.retelling_links(rows) == {}


def test_nothing_positive_returns_nothing():
    catalog, matrices = small_world()
    model, recs = recommend.recommend([entry(5, 5, "planning")], catalog, matrices, PLAIN)
    assert model is None and recs == []


def test_the_live_default_is_the_evaluated_configuration():
    assert scoring.parse_method(recommend.build_spec()) == scoring.parse_method(scoring.DEFAULT_SPEC)[:2] + (
        {**scoring.parse_method(scoring.DEFAULT_SPEC)[2], "alpha": 0.3, "share": 0.35},)


def test_live_ranking_matches_the_scorer_the_evaluation_uses():
    catalog, matrices = small_world()
    spec = "minilm@center=1@knn=2@genre=1@pop=0.5"
    matched = world_matched()
    scores = scoring.make_scorer(spec, catalog, matrices)(matched, np.random.default_rng(0))
    seen = seen_franchises(matched)
    expected = [int(f) for f in catalog.franchise_ids[np.argsort(-scores, kind="stable")] if int(f) not in seen]
    _, recs = recommend.recommend(matched, catalog, matrices, spec, k=10)
    assert [r["franchise_id"] for r in recs] == expected


# ---- franchise entry points ----

def test_entry_point_prefers_tv_then_earliest_year():
    rows = [rec_row(1, 7, "OVA", 1999), rec_row(2, 7, "TV", 2012), rec_row(3, 7, "TV", 2010),
            rec_row(4, 7, "MOVIE", 2005), rec_row(5, 8, "MOVIE", 2020), rec_row(6, 8, "OVA", 2018)]
    pts = recommend.entry_points(rows)
    assert pts[7]["anilist_id"] == 3          # earliest TV, not the 1999 OVA
    assert pts[8]["anilist_id"] == 5          # no TV: movie beats OVA


def test_entry_point_handles_missing_years():
    rows = [rec_row(1, 1, "TV", None), rec_row(2, 1, "TV", 2015)]
    assert recommend.entry_points(rows)[1]["anilist_id"] == 2


def test_within_limits_uses_the_stored_sentinels():
    ok = {"episodes": 12, "year": 2015, "score": 70}
    assert recommend.within_limits(ok, max_episodes=26, min_year=2015, min_score=70)
    assert not recommend.within_limits({**ok, "episodes": -1}, max_episodes=26)   # unknown length under a cap
    assert not recommend.within_limits({**ok, "year": 2014}, min_year=2015)
    assert recommend.within_limits({**ok, "episodes": -1})                         # no cap, unknown is fine
