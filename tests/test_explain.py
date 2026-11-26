import json
import sys
from pathlib import Path

import pytest
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import explain  # noqa: E402
from explain import Facts, check, explain_picks, item_facts, template_sentence  # noqa: E402


def rec(aid, fid, title, genres=(), tags=(), spoilers=(), year=2020, eps=12, pop=100):
    return {"anilist_id": aid, "franchise_id": fid, "title": title, "genres": list(genres), "tags": list(tags),
            "spoiler_tags": list(spoilers), "format": "TV", "year": year, "episodes": eps, "popularity": pop,
            "synopsis": "SECRET PLOT TWIST"}


RECORDS = [
    rec(10, 10, "Pick", ["Drama", "Romance"], ["School", "Music"], spoilers=["Time Loop"]),
    rec(1, 1, "Liked One", ["Drama"], ["School"]),
    rec(2, 2, "Liked Two", ["Action"], []),
    rec(3, 3, "Other Show", ["Drama"], []),
]
SPOILERS = {10: ["Time Loop"]}
FACTS = Facts(RECORDS)
PICK = {"anilist_id": 10, "because": [("Liked One", 0.5), ("Liked Two", 0.4)], "because_franchises": [1, 2]}


def item():
    return item_facts(PICK, FACTS)


def test_item_facts_use_only_visible_attributes_and_shared_ones():
    it = item()
    assert it["attributes"] == ["Drama", "Romance", "School", "Music"]
    assert set(it["liked"][0]["shared"]) == {"Drama", "School"} and it["liked"][1]["shared"] == []


def test_the_prompt_contains_no_synopsis_and_no_spoiler_tag():
    text = json.dumps(explain.build_messages([item()]))
    assert "SECRET PLOT" not in text and "Time Loop" not in text


def test_a_good_sentence_passes():
    assert check("Shares {{School}} with [[Liked One]].", item(), FACTS, SPOILERS) == []


@pytest.mark.parametrize("sentence, fragment", [
    ("Shares {{School}} with [[Liked Two]] and [[Invented Show]].", "title not in the facts"),
    ("Shares {{School}} with [[Liked One]], unlike Other Show.", "another title mentioned"),
    ("Shares {{Mecha}} with [[Liked One]].", "attribute not in the facts"),
    ("Shares {{School}} with [[Liked One]] and a time loop.", "spoiler term"),
    ("Shares {{School}} with [[Liked One]] over 99 episodes.", "number not in the facts"),
    ("Shares themes with [[Liked One]].", "cites no genre or tag"),
    ("Shares {{School}} with something.", "names no liked title"),
    ("", "empty"),
    ("Shares {{School}} with [[Liked One]] " + "and more words " * 12, "too long"),
])
def test_bad_sentences_are_caught(sentence, fragment):
    assert any(fragment in p for p in check(sentence, item(), FACTS, SPOILERS))


def test_the_pick_s_own_number_and_visible_spoiler_like_tag_are_allowed():
    ok = "A {{Drama}} like [[Liked One]], 12 episodes."
    assert check(ok, item(), FACTS, SPOILERS) == []
    visible = {**RECORDS[0], "tags": ["Time Loop"]}
    facts = Facts([visible] + RECORDS[1:])
    it = item_facts(PICK, facts)
    assert check("It has a {{Time Loop}} like [[Liked One]].", it, facts, SPOILERS) == []


def test_the_template_always_passes_the_check():
    assert check(template_sentence(item()), item(), FACTS, SPOILERS) == []
    no_shared = {**item(), "liked": [{"title": "Liked Two", "shared": []}]}
    assert check(template_sentence(no_shared), no_shared, FACTS, SPOILERS) != []   # no attribute cited
    assert template_sentence(no_shared) == "Close to [[Liked Two]] in overall feel."


def test_render_turns_markers_into_markdown():
    assert explain.render("Shares {{School}} with [[Liked One]].") == "Shares School with *Liked One*."


def test_parse_reply_survives_code_fences_and_junk():
    assert explain.parse_reply('```json\n{"10": "x"}\n```') == {"10": "x"}
    assert explain.parse_reply("sorry, no") == {} and explain.parse_reply("{broken") == {}


@pytest.fixture(autouse=True)
def clean_cache():
    explain._cache.clear()


def test_explain_picks_uses_a_checked_model_sentence(monkeypatch):
    monkeypatch.setattr(explain, "call_llm", lambda m, c: '{"10": "Shares {{School}} with [[Liked One]]."}')
    res, err = explain_picks([PICK], FACTS, SPOILERS, config={"x": 1})
    assert err is None and res[0]["source"] == "model" and res[0]["problems"] == []


def test_a_failing_sentence_is_replaced_by_the_template(monkeypatch):
    monkeypatch.setattr(explain, "call_llm", lambda m, c: '{"10": "It is a {{School}} show where [[Invented]] dies."}')
    res, err = explain_picks([PICK], FACTS, SPOILERS, config={"x": 1})
    assert res[0]["source"] == "template" and res[0]["problems"]
    assert check(res[0]["text"], item(), FACTS, SPOILERS) == []


def test_model_errors_fall_back_for_every_pick(monkeypatch):
    def boom(m, c):
        raise explain.LLMError("rate limited")
    monkeypatch.setattr(explain, "call_llm", boom)
    res, err = explain_picks([PICK], FACTS, SPOILERS, config={"x": 1})
    assert err == "rate limited" and res[0]["source"] == "template"
    res, err = explain_picks([PICK], FACTS, SPOILERS, config=None, use_cache=False)
    assert err == "no model configured" and res[0]["source"] == "template"


def test_unusable_reply_and_missing_id_fall_back(monkeypatch):
    monkeypatch.setattr(explain, "call_llm", lambda m, c: "not json")
    res, err = explain_picks([PICK], FACTS, SPOILERS, config={"x": 1})
    assert err and res[0]["source"] == "template"
    monkeypatch.setattr(explain, "call_llm", lambda m, c: '{"99": "x"}')
    res, _ = explain_picks([PICK], FACTS, SPOILERS, config={"x": 1}, use_cache=False)
    assert res[0]["problems"] == ["no sentence returned"]


def test_results_are_cached_and_errors_are_not(monkeypatch):
    calls = []

    def ok(m, c):
        calls.append(1)
        return '{"10": "Shares {{School}} with [[Liked One]]."}'
    monkeypatch.setattr(explain, "call_llm", ok)
    explain_picks([PICK], FACTS, SPOILERS, config={"x": 1})
    explain_picks([PICK], FACTS, SPOILERS, config={"x": 1})
    assert len(calls) == 1

    explain._cache.clear()

    def boom(m, c):
        raise explain.LLMError("HTTP 500")
    monkeypatch.setattr(explain, "call_llm", boom)
    explain_picks([PICK], FACTS, SPOILERS, config={"x": 1})
    assert explain._cache == {}


class FakeResponse:
    def __init__(self, status, body=None, headers=None):
        self.status_code, self._body, self.headers = status, body, headers or {}

    def json(self):
        if self._body is None:
            raise ValueError("no json")
        return self._body


def test_call_llm_maps_failures_to_short_reasons(monkeypatch):
    cfg = {"api_key": "k", "base_url": "https://example.test/v1/", "model": "m"}
    monkeypatch.setattr(requests, "post", lambda *a, **k: FakeResponse(429))
    with pytest.raises(explain.LLMError, match="rate limited"):
        explain.call_llm([], cfg)
    monkeypatch.setattr(requests, "post", lambda *a, **k: FakeResponse(500))
    with pytest.raises(explain.LLMError, match="HTTP 500"):
        explain.call_llm([], cfg)
    monkeypatch.setattr(requests, "post", lambda *a, **k: FakeResponse(200, {"oops": 1}))
    with pytest.raises(explain.LLMError, match="shape"):
        explain.call_llm([], cfg)
    seen = {}

    def good(url, **k):
        seen["url"], seen["auth"] = url, k["headers"]["Authorization"]
        return FakeResponse(200, {"choices": [{"message": {"content": "hi"}}]})
    monkeypatch.setattr(requests, "post", good)
    assert explain.call_llm([], cfg) == "hi"
    assert seen == {"url": "https://example.test/v1/chat/completions", "auth": "Bearer k"}


def test_config_comes_from_the_environment_and_is_none_without_a_key(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    assert explain.get_llm_config() is None
    monkeypatch.setenv("LLM_API_KEY", "abc")
    cfg = explain.get_llm_config()
    assert cfg["api_key"] == "abc" and cfg["base_url"] == explain.DEFAULT_BASE_URL and cfg["model"] == explain.DEFAULT_MODEL


def test_spoiler_terms_missing_file_returns_none(tmp_path):
    assert explain.load_spoiler_terms(tmp_path / "nope.json") is None
    (tmp_path / "s.json").write_text('{"10": ["Time Loop"]}')
    assert explain.load_spoiler_terms(tmp_path / "s.json") == {10: ["Time Loop"]}


def test_tally_separates_call_failures_from_sentence_quality():
    import eval_explanations
    ok = {"source": "model", "problems": []}
    bad = {"source": "template", "problems": ["number not in the facts: 100"]}
    skipped = {"source": "template", "problems": ["no sentence returned"]}
    t = eval_explanations.tally([([ok, bad], None), ([skipped, skipped], "rate limited")])
    assert t["calls"] == 2 and t["calls_ok"] == 1 and t["call_errors"] == {"rate limited": 1}
    assert t["picks_in_ok_calls"] == 2 and t["passed_check"] == 1 and t["pass_rate"] == 0.5
    assert t["failure_reasons"] == {"number not in the facts": 1}


def test_call_llm_waits_once_on_a_short_rate_limit_and_diagnoses_bad_replies(monkeypatch):
    cfg = {"api_key": "k", "base_url": "https://example.test/v1", "model": "m"}
    slept = []
    monkeypatch.setattr(explain.time, "sleep", lambda s: slept.append(s))
    replies = iter([FakeResponse(429, headers={"retry-after": "3"}),
                    FakeResponse(200, {"choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}]})])
    monkeypatch.setattr(requests, "post", lambda *a, **k: next(replies))
    assert explain.call_llm([], cfg) == "hi" and slept == [3.0]

    monkeypatch.setattr(requests, "post", lambda *a, **k: FakeResponse(429, headers={"retry-after": "300"}))
    with pytest.raises(explain.LLMError, match="rate limited"):
        explain.call_llm([], cfg)
    assert slept == [3.0]                                             # a long wait is not slept through

    cut = {"choices": [{"message": {"content": "{\"1\": \"par"}, "finish_reason": "length"}]}
    monkeypatch.setattr(requests, "post", lambda *a, **k: FakeResponse(200, cut))
    with pytest.raises(explain.LLMError, match="token limit"):
        explain.call_llm([], cfg)
    empty = {"choices": [{"message": {"content": ""}, "finish_reason": "stop"}]}
    monkeypatch.setattr(requests, "post", lambda *a, **k: FakeResponse(200, empty))
    with pytest.raises(explain.LLMError, match="empty reply"):
        explain.call_llm([], cfg)


def test_digits_inside_a_verified_title_or_tag_are_not_flagged():
    facts = Facts(RECORDS + [rec(4, 4, "Mob Psycho 100", ["Action"], ["4-koma"])])
    pick = {"anilist_id": 10, "because": [("Mob Psycho 100", 0.5)], "because_franchises": [4]}
    it = item_facts(pick, facts)
    it["attributes"] = it["attributes"] + ["4-koma"]
    assert check("Shares {{Drama}} with [[Mob Psycho 100]].", it, facts, SPOILERS) == []
    assert check("A {{4-koma}} like [[Mob Psycho 100]].", it, facts, SPOILERS) == []
    assert any("number" in p for p in check("Shares {{Drama}} with [[Mob Psycho 100]] over 100 episodes.",
                                            it, facts, SPOILERS))


def test_reasoning_effort_is_sent_only_when_it_applies(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LLM_API_KEY", "k")
    monkeypatch.delenv("LLM_REASONING_EFFORT", raising=False)
    monkeypatch.setenv("LLM_MODEL", "openai/gpt-oss-20b")
    assert explain.get_llm_config()["reasoning_effort"] == "low"
    monkeypatch.setenv("LLM_MODEL", "llama-3.3-70b-versatile")
    assert explain.get_llm_config()["reasoning_effort"] == ""
    monkeypatch.setenv("LLM_REASONING_EFFORT", "medium")
    assert explain.get_llm_config()["reasoning_effort"] == "medium"

    sent = []

    def fake(url, **k):
        sent.append(k["json"])
        return FakeResponse(200, {"choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}]})
    monkeypatch.setattr(requests, "post", fake)
    base = {"api_key": "k", "base_url": "https://example.test/v1", "model": "m"}
    explain.call_llm([], base)
    explain.call_llm([], {**base, "reasoning_effort": "low"})
    assert "reasoning_effort" not in sent[0] and sent[1]["reasoning_effort"] == "low"


def test_a_recommendation_without_explanation_fields_degrades_to_the_template():
    bare = {"anilist_id": 10}                       # e.g. produced by an older recommend.py
    it = item_facts(bare, FACTS)
    assert it["liked"] == []
    assert template_sentence(it) == "Matches the overall shape of your list."
    res, err = explain_picks([bare], FACTS, SPOILERS, config=None, use_cache=False)
    assert res[0]["source"] == "template"
