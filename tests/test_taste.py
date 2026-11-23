import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import recommend
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


# ---- retrieval with a fake ChromaDB collection ----

class FakeCollection:
    """Brute-force stand-in for a Chroma collection (cosine distance, no real database)."""

    def __init__(self, items):
        self.items = items   # list of (anilist_id, franchise_id, vector, extra_metadata)
        self.queries = []

    def count(self):
        return len(self.items)

    def get(self, ids, include):
        found = [it for it in self.items if str(it[0]) in ids]
        return {"ids": [str(it[0]) for it in found], "embeddings": [it[2] for it in found]}

    def query(self, query_embeddings, n_results, where=None, include=None):
        self.queries.append({"n": n_results, "where": where})
        q = unit(*query_embeddings[0])
        rows = sorted(self.items, key=lambda it: -float(unit(*it[2]) @ q))[:n_results]
        return {"ids": [[str(it[0]) for it in rows]],
                "distances": [[1 - float(unit(*it[2]) @ q) for it in rows]],
                "embeddings": [[it[2] for it in rows]],
                "metadatas": [[{"anilist_id": it[0], "franchise_id": it[1], "title": f"T{it[0]}",
                                "episodes": 12, "score": 80, "year": 2020, "format": "TV", **it[3]}
                               for it in rows]]}


def test_retrieve_skips_hidden_franchises_and_keeps_one_per_franchise():
    items = [(1, 10, [1, 0], {}), (2, 10, [0.99, 0.1], {}),    # franchise 10, hidden
             (3, 20, [0.95, 0.3], {}), (4, 20, [0.94, 0.3], {}),  # franchise 20: only the best shows
             (5, 30, [0.5, 0.9], {})]
    col = FakeCollection(items)
    got = recommend.retrieve(col, [1, 0], hidden_franchises={10}, k=5)
    assert [r["anilist_id"] for r in got] == [3, 5]


def test_retrieve_asks_for_more_when_the_first_batch_is_all_hidden():
    items = [(i, i, [1, i * 0.01], {}) for i in range(1, 201)]
    col = FakeCollection(items)
    hidden = set(range(1, 151))                      # the 150 closest are all seen
    got = recommend.retrieve(col, [1, 0], hidden, k=5, start=20)
    assert [r["anilist_id"] for r in got] == [151, 152, 153, 154, 155]
    assert len(col.queries) > 1 and col.queries[-1]["n"] > 150


def test_retrieve_stops_when_the_collection_is_exhausted():
    items = [(1, 1, [1, 0], {}), (2, 2, [0, 1], {})]
    got = recommend.retrieve(FakeCollection(items), [1, 0], hidden_franchises={1, 2}, k=3)
    assert got == []


def test_recommend_end_to_end_with_fake_data():
    items = [(1, 1, [1, 0], {}),          # liked, seen
             (2, 2, [0, 1], {}),          # disliked, seen
             (3, 3, [0.9, 0.1], {}),      # close to the liked title: should lead
             (4, 4, [0.1, 0.9], {}),      # close to the disliked title: should trail
             (5, 5, [0.8, 0.2], {})]      # plan-to-watch, close to the liked one
    col = FakeCollection(items)
    matched = [entry(1, 1, score=10, title="Liked"), entry(2, 2, score=1, title="Disliked"),
               entry(5, 5, "planning", title="Planned")]
    model, recs = recommend.recommend(matched, col, {}, k=3, where={"year": {"$gte": 2000}})
    assert [r["anilist_id"] for r in recs] == [3, 5, 4]
    assert recs[1]["planned"] is True and recs[0]["planned"] is False
    assert recs[0]["because"][0][0] == "Liked"
    assert col.queries[0]["where"] == {"year": {"$gte": 2000}}      # filters reach the database
    _, hidden_planned = recommend.recommend(matched, col, {}, k=3, hide_planned=True)
    assert 5 not in [r["anilist_id"] for r in hidden_planned]


def test_recommend_returns_nothing_when_there_is_nothing_to_learn_from():
    col = FakeCollection([(1, 1, [1, 0], {})])
    model, recs = recommend.recommend([entry(1, 1, "planning")], col, {})
    assert model is None and recs == []


# ---- franchise entry points ----

def rec_row(aid, fid, fmt="TV", year=2010, pop=100, eps=12, score=70):
    return {"anilist_id": aid, "mal_id": None, "title": f"T{aid}", "franchise_id": fid, "format": fmt,
            "episodes": eps, "year": year, "score": score, "popularity": pop, "genres": [], "cover": None}


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


def test_recommend_shows_the_franchise_start_but_keeps_the_matched_title():
    items = [(1, 1, [1, 0], {}),                                  # liked and seen
             (3, 3, [0.9, 0.1], {"title": "Season 4"})]           # franchise 3, closest title
    col = FakeCollection(items)
    matched = [entry(1, 1, score=10, title="Liked")]
    start = {3: {"anilist_id": 30, "franchise_id": 3, "title": "Season 1", "episodes": 12, "year": 2012,
                 "score": 75, "format": "TV"}}
    _, recs = recommend.recommend(matched, col, {}, k=1, starts=start)
    assert recs[0]["title"] == "Season 1" and recs[0]["via"] == "Season 4" and recs[0]["anilist_id"] == 30


def test_the_franchise_start_is_skipped_when_it_breaks_the_limits():
    items = [(1, 1, [1, 0], {}), (3, 3, [0.9, 0.1], {"title": "Season 4", "year": 2020})]
    col = FakeCollection(items)
    matched = [entry(1, 1, score=10, title="Liked")]
    start = {3: {"anilist_id": 30, "franchise_id": 3, "title": "Season 1", "episodes": 12, "year": 2005,
                 "score": 75, "format": "TV"}}
    _, recs = recommend.recommend(matched, col, {}, k=1, starts=start, limits={"min_year": 2015})
    assert recs[0]["title"] == "Season 4" and "via" not in recs[0]
