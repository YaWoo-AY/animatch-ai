import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import parsers
import sample_users as su


def entry(aid, score=None, status="completed"):
    return {"anilist_id": aid, "mal_id": None, "status": status, "score": score, "progress": 12}


def make_index(n=300):
    recs = [{"anilist_id": i, "mal_id": None, "franchise_id": i, "title": f"T{i}"} for i in range(1, n + 1)]
    return {"by_anilist": {r["anilist_id"]: r for r in recs}, "by_mal": {}}


def rich_list(n_rated=60, n_liked=20, n_plain=60):
    """A list with enough rated titles, enough 8+ ratings and extra unrated completed titles."""
    out = [entry(i, 9.0) for i in range(1, n_liked + 1)]
    out += [entry(i, 6.0) for i in range(n_liked + 1, n_rated + 1)]
    out += [entry(i) for i in range(n_rated + 1, n_rated + n_plain + 1)]
    return out


# ---- candidates ----

def test_extract_users_dedupes_and_skips_items_without_a_user():
    data = {"Page": {"activities": [
        {"user": {"id": 1, "name": "a"}}, {"user": {"id": 1, "name": "a"}},
        {},                                    # a union member that is not a ListActivity
        {"user": None}, {"user": {"id": 2, "name": "b"}}]}}
    assert su.extract_users(data, "activities") == [{"id": 1, "name": "a"}, {"id": 2, "name": "b"}]


def test_extract_users_for_the_users_strategy():
    data = {"Page": {"users": [{"id": 5, "name": "x"}, {"id": 5, "name": "x"}, {"id": 6, "name": "y"}]}}
    assert [u["id"] for u in su.extract_users(data, "users")] == [5, 6]


def test_gather_candidates_follows_pages_until_the_last():
    pages = {1: {"Page": {"pageInfo": {"hasNextPage": True}, "activities": [{"user": {"id": 1, "name": "a"}}]}},
             2: {"Page": {"pageInfo": {"hasNextPage": False}, "activities": [{"user": {"id": 2, "name": "b"}},
                                                                             {"user": {"id": 1, "name": "a"}}]}}}
    seen = []

    def post(query, variables):
        seen.append(variables["page"])
        return pages[variables["page"]]

    got = su.gather_candidates("activity", pages=5, post=post, sleep=lambda s: None)
    assert [u["id"] for u in got] == [1, 2] and seen == [1, 2]


# ---- qualification ----

def test_a_rich_list_qualifies():
    ok, stats, reason = su.assess(rich_list(), make_index())
    assert ok and reason == ""
    assert stats["rated"] == 60 and stats["liked"] == 20


def test_each_rule_can_reject():
    idx = make_index()
    ok, _, why = su.assess(rich_list(n_plain=0), idx)                         # 60 matched < 100
    assert not ok and "in our index" in why
    ok, _, why = su.assess(rich_list(n_rated=30, n_liked=20, n_plain=100), idx)
    assert not ok and "rated titles" in why
    ok, _, why = su.assess(rich_list(n_rated=60, n_liked=5), idx)
    assert not ok and "rated 8+" in why


def test_titles_outside_the_index_do_not_count():
    entries = rich_list() + [entry(9000 + i, 9.0) for i in range(50)]         # 50 high ratings, none in index
    ok, stats, _ = su.assess(entries, make_index())
    assert stats["liked"] == 20


# ---- the sampling loop ----

def run(tmp_path, candidates, fetch, **kw):
    kw.setdefault("log", lambda *_: None)
    return su.sample(candidates, fetch, make_index(), tmp_path, sleep=lambda s: None, **kw)


def test_sample_saves_qualifying_users_without_names_and_stops_at_the_target(tmp_path):
    cands = [{"id": i, "name": f"secret_name_{i}"} for i in range(1, 6)]
    fetched = []

    def fetch(name):
        fetched.append(name)
        return rich_list()

    have = run(tmp_path, cands, fetch, target=2)
    assert have == 2 and len(fetched) == 2
    files = su.saved_users(tmp_path)
    assert [f.name for f in files] == ["u001.json", "u002.json"]
    blob = "".join(f.read_text() for f in files) + (tmp_path / su.CHECKED_NAME).read_text()
    assert "secret_name" not in blob                      # no username anywhere on disk
    assert json.loads(files[0].read_text())["stats"]["liked"] == 20


def test_sample_skips_private_missing_and_thin_lists(tmp_path):
    cands = [{"id": 1, "name": "priv"}, {"id": 2, "name": "gone"}, {"id": 3, "name": "thin"},
             {"id": 4, "name": "good"}]

    def fetch(name):
        if name == "priv":
            raise parsers.PrivateList("private")
        if name == "gone":
            raise parsers.UserNotFound("missing")
        if name == "thin":
            return [entry(1, 9.0)]
        return rich_list()

    have = run(tmp_path, cands, fetch, target=5)
    assert have == 1 and su.load_checked(tmp_path) == {1, 2, 3, 4}


def test_sample_resumes_without_refetching_people_already_checked(tmp_path):
    cands = [{"id": 1, "name": "a"}, {"id": 2, "name": "b"}, {"id": 3, "name": "c"}]
    calls = []

    def fetch(name):
        calls.append(name)
        return rich_list()

    run(tmp_path, cands, fetch, target=1)
    first = list(calls)
    run(tmp_path, cands, fetch, target=3)
    assert len(first) == 1 and len(calls) == 3 and len(set(calls)) == 3      # nobody downloaded twice
    assert len(su.saved_users(tmp_path)) == 3


def test_excluded_names_are_never_fetched(tmp_path):
    cands = [{"id": 1, "name": "Me"}, {"id": 2, "name": "other"}]
    calls = []
    run(tmp_path, cands, lambda n: calls.append(n) or rich_list(), target=5, exclude=["me"])
    assert calls == ["other"]


def test_report_prints_a_table(tmp_path, capsys):
    su.save_user(tmp_path, 1, [], {"entries": 120, "matched": 110, "rated": 70, "liked": 25, "completed": 90})
    su.report(tmp_path)
    out = capsys.readouterr().out
    assert "u001" in out and "median rated: 70" in out
