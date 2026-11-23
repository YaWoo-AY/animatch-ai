import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import evaluate as ev
import taste


# ---- a tiny synthetic world: 3 topics x 10 franchises x 2 titles ----

def make_world(seed=0):
    rng = np.random.default_rng(seed)
    records, vecs = [], []
    aid = 0
    for topic in range(3):
        for k in range(10):
            fid = topic * 10 + k + 1
            for _ in range(2):
                aid += 1
                v = np.zeros(3)
                v[topic] = 1.0
                v = v + rng.normal(0, 0.05, 3)
                vecs.append(v / np.linalg.norm(v))
                # topic 0 is the LEAST popular, so a popularity baseline is bad for a topic-0 fan
                records.append({"anilist_id": aid, "mal_id": None, "franchise_id": fid, "title": f"T{aid}",
                                "popularity": 1000 * (topic + 1) + k, "episodes": 12,
                                "genres": [f"g{topic}"], "tags": [f"t{topic}-{k % 2}"]})
    return ev.Catalog(records), np.vstack(vecs)


def entry(catalog, row, score=9.0, status="completed"):
    r = catalog.records[row]
    return {"anilist_id": r["anilist_id"], "franchise_id": r["franchise_id"], "title": r["title"],
            "status": status, "score": score, "progress": 12, "mal_id": None}


def topic0_fan(catalog, uid="u001", n=9):
    """Rated 9 the first title of the first n franchises of topic 0."""
    return ev.User(uid, [entry(catalog, 2 * k) for k in range(n)])


def scorers(catalog, M, *specs):
    return {s: ev.make_scorer(s, catalog, lambda name, variant: M) for s in specs}


# ---- ranking ----

def test_rank_counts_only_candidates_and_breaks_ties_by_position():
    scores = np.array([5.0, 9.0, 7.0, 7.0, 1.0])
    mask = np.array([True, False, True, True, True])      # franchise 1 (score 9) is seen, so not a candidate
    assert ev.rank_of(scores, 2, mask) == 1               # 7.0 is the best candidate (tie, lower position wins)
    assert ev.rank_of(scores, 3, mask) == 2               # same score, later position
    assert ev.rank_of(scores, 4, mask) == 4


def test_constant_scores_get_no_free_luck():
    scores = np.zeros(10)
    mask = np.ones(10, dtype=bool)
    assert ev.rank_of(scores, 9, mask) == 10


# ---- the experiment ----

def test_taste_beats_random_and_popularity_on_a_world_where_it_should():
    catalog, M = make_world()
    users = [topic0_fan(catalog)]
    ranks, fb = ev.evaluate(users, catalog, scorers(catalog, M, "random", "popularity", "minilm"),
                            per_user=9, log=lambda *_: None)
    taste_ranks = ranks["minilm"]["u001"]
    assert max(taste_ranks) <= 2                          # only 2 topic-0 franchises are unseen at any time
    assert np.mean(ranks["popularity"]["u001"]) > 15      # the unpopular topic is buried
    assert np.mean(ranks["random"]["u001"]) > 3
    assert fb["minilm"] == 0


def test_the_hidden_franchise_is_fully_removed_from_what_the_method_sees():
    catalog, M = make_world()
    seen_by_method = []

    def spy(visible, rng):
        seen_by_method.append({e["franchise_id"] for e in visible})
        return catalog.popularity

    user = ev.User("u001", [entry(catalog, 0), entry(catalog, 1, 9.0), entry(catalog, 2), entry(catalog, 4)])
    # titles at rows 0 and 1 are two seasons of franchise 1
    ev.evaluate([user], catalog, {"spy": spy}, per_user=10, log=lambda *_: None)
    held = ev.pick_held_out(user, 10, 0)
    assert len(seen_by_method) == len(held)
    for fid, visible_franchises in zip(held, seen_by_method):
        assert fid not in visible_franchises


def test_the_hidden_franchise_is_a_candidate_even_though_the_user_watched_it():
    catalog, M = make_world()
    ranks, _ = ev.evaluate([topic0_fan(catalog, n=3)], catalog, scorers(catalog, M, "popularity"), per_user=3,
                           log=lambda *_: None)
    assert all(1 <= r <= catalog.n_franchises for r in ranks["popularity"]["u001"])


def test_nothing_positive_left_falls_back_to_popularity_and_is_counted():
    catalog, M = make_world()
    lone = ev.User("u001", [entry(catalog, 0)])           # the only liked franchise is the hidden one
    ranks, fb = ev.evaluate([lone], catalog, scorers(catalog, M, "minilm"), per_user=5, log=lambda *_: None)
    assert fb["minilm"] == 1 and len(ranks["minilm"]["u001"]) == 1


def test_heavy_users_are_capped_and_choices_are_repeatable():
    catalog, M = make_world()
    user = topic0_fan(catalog, n=10)
    a = ev.pick_held_out(user, 4, seed=1)
    assert len(a) == 4 and a == ev.pick_held_out(user, 4, seed=1)
    assert a != ev.pick_held_out(user, 4, seed=2)
    assert len(ev.pick_held_out(user, 50, seed=1)) == 10  # fewer liked franchises than the cap


def test_liked_franchises_need_a_rating_of_eight():
    catalog, M = make_world()
    user = ev.User("u001", [entry(catalog, 0, 9.0), entry(catalog, 2, 7.9), entry(catalog, 4, None),
                            entry(catalog, 6, 8.0)])
    assert ev.liked_franchises(user) == [1, 4]


# ---- methods ----

def test_parse_method():
    assert ev.parse_method("minilm") == ("minilm", "base", {})
    assert ev.parse_method("bge/notitle@alpha=0.0@share=1") == ("bge", "notitle", {"alpha": 0.0, "share": 1.0})
    with pytest.raises(SystemExit):
        ev.parse_method("minilm@beta=2")


def test_unrated_share_of_one_means_no_cap_and_no_crash():
    emb = {1: np.array([1.0, 0]), 2: np.array([0, 1.0])}
    matched = [{"anilist_id": 1, "franchise_id": 1, "status": "completed", "score": 10.0, "progress": 1, "title": "a"},
               {"anilist_id": 2, "franchise_id": 2, "status": "completed", "score": None, "progress": 1, "title": "b"}]
    t = taste.build_taste(matched, emb, unrated_share=1.0)
    assert t.unrated_scale == 1.0


def test_genre_matrix_has_unit_rows_and_rare_tags_weigh_more():
    recs = [{"genres": ["Common"], "tags": ["Rare"]}] + [{"genres": ["Common"], "tags": []} for _ in range(9)]
    m = ev.genre_matrix(recs)
    assert np.allclose(np.linalg.norm(m, axis=1), 1.0)
    assert m[0].max() > m[0].min() and m[0][1] > m[0][0]      # the rare tag outweighs the common genre


def test_genre_method_runs_end_to_end():
    catalog, _ = make_world()
    G = ev.genre_matrix(catalog.records)
    ranks, _ = ev.evaluate([topic0_fan(catalog)], catalog, scorers(catalog, G, "genre"), per_user=9,
                           log=lambda *_: None)
    assert np.mean(ranks["genre"]["u001"]) < 5


# ---- metrics ----

def test_averages_are_per_user_so_a_heavy_user_cannot_dominate():
    ranks = {"m": {"u001": [1] * 100, "u002": [500]}}
    s = ev.summarize(ranks, ks=(10,))
    assert s["m"]["recall@10"][0] == pytest.approx(0.5)       # (1.0 + 0.0) / 2, not 100/101
    assert s["m"]["trials"] == 101


def test_bootstrap_interval_is_repeatable_and_brackets_the_mean():
    vals = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    a, b = ev.bootstrap_ci(vals, seed=3), ev.bootstrap_ci(vals, seed=3)
    assert a == b
    mean, lo, hi = a
    assert lo <= mean <= hi and mean == pytest.approx(0.35)


def test_paired_difference_sign_and_pairing():
    ranks = {"good": {"u1": [1, 2], "u2": [1, 1], "u3": [3, 2]},
             "bad": {"u1": [90, 91], "u2": [80, 70], "u3": [60, 50]}}
    d, lo, hi = ev.paired_difference(ranks, "good", "bad", k=10)
    assert d == 1.0 and lo > 0


def test_summary_table_prints(capsys):
    ev.print_table(ev.summarize({"m": {"u001": [1, 20, 60]}}))
    out = capsys.readouterr().out
    assert "Recall@10" in out and "m" in out
