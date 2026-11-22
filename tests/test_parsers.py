import json
import sys
import tempfile
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import parsers
from parsers import (ApiError, PrivateList, QueryRejected, UserNotFound, cache_path, classify_anilist_errors,
                     get_profile, load_cache, match_to_index, parse_anilist_collection, parse_mal_items,
                     save_cache)


class FakeResponse:
    """Stands in for requests.Response, so no test touches the network."""

    def __init__(self, status=200, body=None, headers=None):
        self.status_code = status
        self._body = body
        self.headers = headers or {}
        self.text = json.dumps(body)

    def json(self):
        if self._body is None:
            raise ValueError("no json")
        return self._body


def anilist_entry(media_id, status="COMPLETED", score=8.0, mal=None, progress=12):
    return {"status": status, "score": score, "progress": progress, "media": {"id": media_id, "idMal": mal}}


def collection(*entries, has_next=False):
    return {"data": {"MediaListCollection": {"hasNextChunk": has_next, "lists": [{"entries": list(entries)}]}}}


# ---- AniList parsing ----

def test_anilist_maps_statuses_and_scores():
    into = {}
    parse_anilist_collection({"lists": [{"entries": [
        anilist_entry(1, "COMPLETED", 9.5, mal=11),
        anilist_entry(2, "REPEATING", 8.0),
        anilist_entry(3, "CURRENT", 0),
        anilist_entry(4, "PAUSED", 7.0),
        anilist_entry(5, "DROPPED", 4.0),
        anilist_entry(6, "PLANNING", 0),
    ]}]}, into)
    assert [into[i]["status"] for i in range(1, 7)] == [
        "completed", "completed", "watching", "paused", "dropped", "planning"]
    assert into[1]["score"] == 9.5 and into[1]["mal_id"] == 11
    assert into[3]["score"] is None and into[6]["score"] is None  # 0 means not rated


def test_anilist_dedupes_entries_across_custom_lists():
    into = {}
    parse_anilist_collection({"lists": [
        {"entries": [anilist_entry(1)]},
        {"entries": [anilist_entry(1), anilist_entry(2)]},  # a custom list repeats entry 1
    ]}, into)
    assert sorted(into) == [1, 2]


def test_anilist_skips_unknown_status_and_missing_media():
    into = {}
    parse_anilist_collection({"lists": [{"entries": [
        anilist_entry(1, "SOMETHING_NEW"), {"status": "COMPLETED", "media": None}, anilist_entry(2)]}]}, into)
    assert list(into) == [2]


def test_anilist_handles_null_collection():
    into = {}
    parse_anilist_collection(None, into)
    assert into == {}


# ---- error classification ----

def test_errors_are_classified():
    assert isinstance(classify_anilist_errors(404, [{"message": "User not found", "status": 404}]), UserNotFound)
    assert isinstance(classify_anilist_errors(403, [{"message": "Private User", "status": 403}]), PrivateList)
    assert isinstance(classify_anilist_errors(400, [{"message": "Unknown argument", "status": 400}]), QueryRejected)
    assert type(classify_anilist_errors(500, [{"message": "boom"}])) is ApiError
    assert isinstance(classify_anilist_errors(200, [{"message": "Private User"}]), PrivateList)


# ---- AniList fetching (HTTP faked) ----

def test_rate_limit_is_retried_then_succeeds():
    replies = [FakeResponse(429, None, {"Retry-After": "2"}), FakeResponse(200, collection(anilist_entry(1)))]
    with mock.patch.object(parsers.requests, "post", side_effect=replies) as post, \
            mock.patch.object(parsers.time, "sleep") as sleep:
        entries = parsers.fetch_anilist("someone")
    assert post.call_count == 2
    sleep.assert_any_call(2)
    assert [e["anilist_id"] for e in entries] == [1]


def test_chunks_are_followed_and_merged():
    replies = [FakeResponse(200, collection(anilist_entry(1), has_next=True)),
               FakeResponse(200, collection(anilist_entry(1), anilist_entry(2), has_next=False))]
    with mock.patch.object(parsers.requests, "post", side_effect=replies) as post, \
            mock.patch.object(parsers.time, "sleep"):
        entries = parsers.fetch_anilist("someone")
    assert sorted(e["anilist_id"] for e in entries) == [1, 2]
    assert post.call_args_list[1].kwargs["json"]["variables"]["chunk"] == 2


def test_falls_back_to_a_plain_query_when_chunk_arguments_are_rejected():
    replies = [FakeResponse(400, {"errors": [{"message": "Unknown argument chunk", "status": 400}]}),
               FakeResponse(200, collection(anilist_entry(7)))]
    with mock.patch.object(parsers.requests, "post", side_effect=replies) as post, \
            mock.patch.object(parsers.time, "sleep"):
        entries = parsers.fetch_anilist("someone")
    assert [e["anilist_id"] for e in entries] == [7]
    assert "chunk" not in post.call_args_list[1].kwargs["json"]["variables"]


def test_private_and_missing_users_raise_clear_errors():
    private = FakeResponse(403, {"errors": [{"message": "Private User", "status": 403}]})
    missing = FakeResponse(404, {"errors": [{"message": "User not found", "status": 404}]})
    with mock.patch.object(parsers.requests, "post", return_value=private), mock.patch.object(parsers.time, "sleep"):
        try:
            parsers.fetch_anilist("x")
            assert False, "expected PrivateList"
        except PrivateList:
            pass
    with mock.patch.object(parsers.requests, "post", return_value=missing), mock.patch.object(parsers.time, "sleep"):
        try:
            parsers.fetch_anilist("x")
            assert False, "expected UserNotFound"
        except UserNotFound:
            pass


def test_gives_up_after_repeated_server_errors():
    with mock.patch.object(parsers.requests, "post", return_value=FakeResponse(503, None)), \
            mock.patch.object(parsers.time, "sleep"):
        try:
            parsers.fetch_anilist("x")
            assert False, "expected ApiError"
        except ApiError:
            pass


# ---- MyAnimeList (official API) ----

def mal_item(mal_id, status="completed", score=8, watched=12):
    return {"node": {"id": mal_id, "title": "x"},
            "list_status": {"status": status, "score": score, "num_episodes_watched": watched}}


def test_mal_parser_maps_statuses_and_scores():
    items = [mal_item(1, "completed", 9, 26), mal_item(2, "on_hold", 0, 3), mal_item(3, "plan_to_watch", 0, 0),
             mal_item(4, "dropped", 4, 2), mal_item(5, "watching", 7, 5),
             mal_item(6, "mystery"),                       # unknown status: skipped
             {"list_status": {"status": "completed"}}]     # no node id: skipped
    out = parse_mal_items(items)
    assert [(e["mal_id"], e["status"]) for e in out] == [
        (1, "completed"), (2, "paused"), (3, "planning"), (4, "dropped"), (5, "watching")]
    assert out[0]["score"] == 9.0 and out[1]["score"] is None and out[0]["progress"] == 26
    assert all(e["anilist_id"] is None for e in out)


def test_mal_follows_next_urls_and_sends_the_client_id():
    pages = [FakeResponse(200, {"data": [mal_item(1)], "paging": {"next": "https://api.myanimelist.net/v2/next?offset=1000"}}),
             FakeResponse(200, {"data": [mal_item(2)], "paging": {}})]
    with mock.patch.object(parsers.requests, "get", side_effect=pages) as get, mock.patch.object(parsers.time, "sleep"):
        entries = parsers.fetch_mal("someone", client_id="abc")
    assert [e["mal_id"] for e in entries] == [1, 2]
    assert get.call_args_list[0].kwargs["headers"] == {"X-MAL-CLIENT-ID": "abc"}
    assert get.call_args_list[1].args[0].endswith("offset=1000") and get.call_args_list[1].kwargs["params"] is None


def test_mal_errors_are_clear():
    cases = [(404, UserNotFound), (403, PrivateList), (401, ApiError), (400, ApiError)]
    for status, expected in cases:
        with mock.patch.object(parsers.requests, "get", return_value=FakeResponse(status, {"message": "m"})):
            try:
                parsers.fetch_mal("x", client_id="abc")
                assert False, f"expected an error for {status}"
            except expected:
                pass


def test_mal_retries_rate_limits():
    replies = [FakeResponse(429, None), FakeResponse(200, {"data": [mal_item(1)], "paging": {}})]
    with mock.patch.object(parsers.requests, "get", side_effect=replies) as get, mock.patch.object(parsers.time, "sleep"):
        assert len(parsers.fetch_mal("x", client_id="abc")) == 1
    assert get.call_count == 2


def test_client_id_comes_from_env_then_dotenv_and_is_required():
    with tempfile.TemporaryDirectory() as d:
        env_file = Path(d) / ".env"
        env_file.write_text('OTHER=1\nMAL_CLIENT_ID="from-file"\n', encoding="utf-8")
        with mock.patch.dict(parsers.os.environ, {}, clear=True):
            assert parsers.get_mal_client_id(env_file) == "from-file"
            try:
                parsers.get_mal_client_id(Path(d) / "missing.env")
                assert False
            except parsers.MissingClientId:
                pass
        with mock.patch.dict(parsers.os.environ, {"MAL_CLIENT_ID": "from-env"}):
            assert parsers.get_mal_client_id(env_file) == "from-env"


# ---- cache ----

def test_cache_path_is_safe_and_lowercase():
    p = cache_path("anilist", "Some User/../x", cache_dir="c")
    assert p.name == "anilist_some_user____x.json"


def test_cache_respects_the_ttl():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "p.json"
        save_cache(path, [{"anilist_id": 1}], now=1000)
        assert load_cache(path, ttl=100, now=1050) == [{"anilist_id": 1}]
        assert load_cache(path, ttl=100, now=1200) is None
        assert load_cache(Path(d) / "missing.json") is None


def test_get_profile_uses_the_cache_until_refreshed():
    calls = []

    def fake_fetch(username):
        calls.append(username)
        return [{"anilist_id": 1, "mal_id": None, "status": "completed", "score": 8.0, "progress": 1}]

    with tempfile.TemporaryDirectory() as d, mock.patch.dict(parsers.FETCHERS, {"anilist": fake_fetch}):
        get_profile("anilist", "Bob", cache_dir=d, now=1000)
        get_profile("anilist", "bob", cache_dir=d, now=1001)       # same user, different capitals
        assert calls == ["Bob"]
        get_profile("anilist", "Bob", refresh=True, cache_dir=d, now=1002)
        assert len(calls) == 2


# ---- matching to our index ----

def make_index():
    recs = [
        {"anilist_id": 10, "mal_id": 100, "franchise_id": 10, "title": "A"},
        {"anilist_id": 20, "mal_id": None, "franchise_id": 20, "title": "B"},
        {"anilist_id": 30, "mal_id": 300, "franchise_id": 10, "title": "A2"},
    ]
    return {"by_anilist": {r["anilist_id"]: r for r in recs},
            "by_mal": {r["mal_id"]: r for r in recs if r["mal_id"] is not None}}


def test_match_by_anilist_id_then_by_mal_id():
    entries = [
        {"anilist_id": 10, "mal_id": None, "status": "completed", "score": 9.0, "progress": 1},
        {"anilist_id": None, "mal_id": 300, "status": "dropped", "score": None, "progress": 1},   # MAL list
        {"anilist_id": 999, "mal_id": 999, "status": "completed", "score": 5.0, "progress": 1},   # not indexed
    ]
    matched, unmatched = match_to_index(entries, make_index())
    assert [(m["anilist_id"], m["franchise_id"], m["title"]) for m in matched] == [(10, 10, "A"), (30, 10, "A2")]
    assert [u["anilist_id"] for u in unmatched] == [999]


def test_entries_without_any_id_are_unmatched_not_crashes():
    matched, unmatched = match_to_index(
        [{"anilist_id": None, "mal_id": None, "status": "completed", "score": None, "progress": 0}], make_index())
    assert matched == [] and len(unmatched) == 1


def test_lookup_titles_batches_and_prefers_english_titles():
    page = {"data": {"Page": {"media": [
        {"id": 1, "title": {"english": "Show", "romaji": "Shou"}, "format": "TV_SHORT", "status": "FINISHED",
         "popularity": 10, "seasonYear": 2020},
        {"id": 2, "title": {"english": None, "romaji": "Namae"}, "format": "SPECIAL", "status": "FINISHED",
         "popularity": 5, "seasonYear": None}]}}}
    with mock.patch.object(parsers.requests, "post", return_value=FakeResponse(200, page)) as post, \
            mock.patch.object(parsers.time, "sleep"):
        found = parsers.lookup_titles(range(1, 121))
    assert post.call_count == 3  # 120 ids -> batches of 50
    assert found[1]["title"] == "Show" and found[2]["title"] == "Namae" and found[2]["format"] == "SPECIAL"


def test_mal_parses_a_real_response():
    """Trimmed copy of a real reply from api.myanimelist.net (client ID only, public list)."""
    real = {"data": [
        {"node": {"id": 62524, "title": "#Zombie Sagashitemasu",
                  "main_picture": {"medium": "https://cdn.myanimelist.net/images/anime/1151/160025.jpg"}},
         "list_status": {"status": "watching", "score": 0, "num_episodes_watched": 5, "is_rewatching": False,
                         "updated_at": "2026-11-01T07:56:08+00:00"}},
        {"node": {"id": 54041, "title": "16bit Sensation: Another Layer"},
         "list_status": {"status": "completed", "score": 8, "num_episodes_watched": 13, "is_rewatching": False,
                         "start_date": "2023-10-13", "finish_date": "2023-12-29"}}],
        "paging": {"next": "https://api.myanimelist.net/v2/users/yawoo/animelist?offset=3&limit=3&fields=list_status"}}
    out = parse_mal_items(real["data"])
    assert out == [
        {"anilist_id": None, "mal_id": 62524, "status": "watching", "score": None, "progress": 5},
        {"anilist_id": None, "mal_id": 54041, "status": "completed", "score": 8.0, "progress": 13}]


def test_mal_unknown_user_message_is_a_404():
    with mock.patch.object(parsers.requests, "get",
                           return_value=FakeResponse(404, {"message": "", "error": "not_found"})):
        try:
            parsers.fetch_mal("nobody", client_id="abc")
        except UserNotFound:
            return
    raise AssertionError("expected UserNotFound")
