"""Short, spoiler-safe explanations for each recommendation, written by a language model and checked.

The model sees only: the pick's title, format, year, episode count, its visible genres and tags, and
for each of the liked titles that led to it, the genres and tags they share. It never sees a
synopsis or a spoiler-flagged tag. Every sentence it returns is checked against those facts; a
sentence that fails, or any model error, is replaced by a template built from the same facts, so
nothing unchecked is shown.

Sentence format the model must follow: titles in [[double brackets]], genres and tags in {{braces}}.
That makes the claims machine-checkable. `render` turns the markers into plain markdown.

Configuration (environment or .env): LLM_API_KEY, and optionally LLM_BASE_URL (default Groq's
OpenAI-compatible endpoint) and LLM_MODEL.
"""
import json
import os
import re
import time
from collections import Counter
from pathlib import Path

import requests

DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_MODEL = "openai/gpt-oss-120b"
MAX_WORDS = 40
MAX_SHARED = 4
CACHE_LIMIT = 5000
_cache = {}


class LLMError(Exception):
    pass


# ---------------------------------------------------------------- configuration

def _env(name, env_path=".env"):
    value = os.environ.get(name, "").strip()
    if not value and Path(env_path).exists():
        for line in Path(env_path).read_text(encoding="utf-8").splitlines():
            if line.strip().startswith(f"{name}="):
                value = line.split("=", 1)[1].strip().strip('"').strip("'")
    return value


def get_llm_config():
    """{'api_key', 'base_url', 'model'} or None when no key is set."""
    key = _env("LLM_API_KEY")
    if not key:
        return None
    model = _env("LLM_MODEL") or DEFAULT_MODEL
    # gpt-oss models think before answering and the thinking shares the reply's token budget, so a
    # short structured answer can be cut off. Low effort keeps the thinking short. Other models and
    # providers may reject the setting, so it is only sent when it applies or is asked for.
    effort = _env("LLM_REASONING_EFFORT") or ("low" if model.startswith("openai/gpt-oss") else "")
    return {"api_key": key, "base_url": _env("LLM_BASE_URL") or DEFAULT_BASE_URL, "model": model,
            "reasoning_effort": effort}


def load_spoiler_terms(path):
    """{anilist_id: [spoiler-flagged tags]} for checking only, or None if the file is missing
    (then explanations stay off: without it the spoiler check cannot run)."""
    path = Path(path)
    if not path.exists():
        return None
    return {int(k): v for k, v in json.loads(path.read_text(encoding="utf-8")).items()}


# ---------------------------------------------------------------- facts

class Facts:
    """Lookups over the catalog records that the explanations need."""

    def __init__(self, records):
        self.by_id = {r["anilist_id"]: r for r in records}
        self.by_franchise_title = {(r["franchise_id"], r["title"]): r for r in records}
        best = {}
        for r in records:
            cur = best.get(r["franchise_id"])
            if cur is None or (r.get("popularity") or 0) > (cur.get("popularity") or 0):
                best[r["franchise_id"]] = r
        self.best_in_franchise = best
        self.titles = {r["title"] for r in records}
        self.freq = Counter(a for r in records for a in set(_attrs(r)))

    def liked_record(self, franchise_id, title):
        return self.by_franchise_title.get((franchise_id, title)) or self.best_in_franchise.get(franchise_id)


def _attrs(record):
    """Visible genres and non-spoiler tags, in a stable order, without duplicates."""
    seen, out = set(), []
    for a in list(record.get("genres") or []) + list(record.get("tags") or []):
        if a not in seen:
            seen.add(a)
            out.append(a)
    return out


def item_facts(rec, facts):
    """Everything the model may use about one recommendation (and nothing else)."""
    pick = facts.by_id[rec["anilist_id"]]
    pick_attrs = _attrs(pick)
    liked = []
    for fid, title in zip(rec.get("because_franchises", []), [t for t, _ in rec.get("because", [])]):
        lr = facts.liked_record(fid, title)
        if lr is None:
            continue
        shared = [a for a in pick_attrs if a in set(_attrs(lr))]
        shared.sort(key=lambda a: facts.freq[a])            # rarer attributes say more
        liked.append({"title": lr["title"], "shared": shared[:MAX_SHARED]})
    return {"id": pick["anilist_id"], "title": pick["title"], "format": pick["format"], "year": pick["year"],
            "episodes": pick["episodes"], "attributes": pick_attrs, "liked": liked}


# ---------------------------------------------------------------- template (always safe)

def template_sentence(item):
    """A plain sentence from the same facts. Used whenever the model cannot be used or fails the check."""
    for lk in item["liked"]:
        if lk["shared"]:
            tags = " and ".join("{{" + a + "}}" for a in lk["shared"][:2])
            return f"Shares {tags} with [[{lk['title']}]]."
    if item["liked"]:
        return f"Close to [[{item['liked'][0]['title']}]] in overall feel."
    return "Matches the overall shape of your list."


# ---------------------------------------------------------------- the check

def _ngrams(text, longest=8):
    words = re.findall(r"[^\s]+", text)
    for n in range(1, longest + 1):
        for i in range(len(words) - n + 1):
            yield i, " ".join(w.strip(".,;:!?()\"") for w in words[i:i + n])


def check(sentence, item, facts, spoiler_terms):
    """Problems with a generated sentence (empty list = it passes)."""
    problems = []
    if not sentence or not sentence.strip():
        return ["empty"]
    if len(sentence.split()) > MAX_WORDS:
        problems.append("too long")
    allowed_titles = {lk["title"].lower() for lk in item["liked"]} | {item["title"].lower()}
    allowed_attrs = {a.lower() for a in item["attributes"]}

    titles = re.findall(r"\[\[(.+?)\]\]", sentence)
    attrs = re.findall(r"\{\{(.+?)\}\}", sentence)
    if not any(t.lower() in allowed_titles for t in titles):
        problems.append("names no liked title")
    for t in titles:
        if t.lower() not in allowed_titles:
            problems.append(f"title not in the facts: {t}")
    if not attrs:
        problems.append("cites no genre or tag")
    for a in attrs:
        if a.lower() not in allowed_attrs:
            problems.append(f"attribute not in the facts: {a}")

    plain = re.sub(r"\[\[|\]\]|\{\{|\}\}", "", sentence)
    stripped = re.sub(r"\[\[.+?\]\]|\{\{.+?\}\}", " ", sentence)   # marked titles and attributes are checked above
    for i, gram in _ngrams(stripped):
        if gram in facts.titles and gram.lower() not in allowed_titles and not (i == 0 and " " not in gram):
            problems.append(f"another title mentioned: {gram}")
            break
    visible = {a.lower() for a in item["attributes"]}
    for term in spoiler_terms.get(item["id"], []):
        if term.lower() not in visible and re.search(rf"\b{re.escape(term.lower())}\b", plain.lower()):
            problems.append(f"spoiler term: {term}")
    allowed_numbers = {str(item["year"]), str(item["episodes"])}
    unmarked = re.sub(r"\[\[.+?\]\]|\{\{.+?\}\}", " ", sentence)   # digits inside a verified title or tag are fine
    for num in re.findall(r"\d+", unmarked):
        if num not in allowed_numbers:
            problems.append(f"number not in the facts: {num}")
    return problems


def render(sentence):
    """Markers to markdown: titles in italics, attributes plain."""
    out = re.sub(r"\[\[(.+?)\]\]", lambda m: "*" + m.group(1).replace("*", "") + "*", sentence)
    return re.sub(r"\{\{(.+?)\}\}", r"\1", out)


# ---------------------------------------------------------------- the model call

SYSTEM = (
    "You write one short sentence per anime recommendation, saying why it fits a viewer, using ONLY the "
    "facts given. Rules: (1) at most 25 words; (2) name one or two of the item's liked titles exactly as "
    "given, inside [[double brackets]]; (3) mention at least one genre or tag, chosen only from the item's "
    "'attributes' list and preferably from the 'shared' list of a liked title, inside {{braces}}, spelled "
    "exactly as given; (4) do not describe the plot, characters, setting or ending, and do not use any "
    "number; (5) no other titles. Reply with one JSON object mapping each item id (as a string) to its "
    "sentence, and nothing else."
)


def build_messages(items):
    payload = [{"id": it["id"], "title": it["title"], "attributes": it["attributes"],
                "liked": it["liked"]} for it in items]
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]


def _retry_after(response):
    try:
        return float(getattr(response, "headers", {}).get("retry-after"))
    except (TypeError, ValueError):
        return None


def call_llm(messages, config, timeout=40, max_wait=20):
    """One chat-completion call. On a rate limit, waits once if the provider says it is short."""
    for attempt in range(2):
        try:
            body = {"model": config["model"], "messages": messages, "temperature": 0.2, "max_tokens": 3000}
            if config.get("reasoning_effort"):
                body["reasoning_effort"] = config["reasoning_effort"]
            r = requests.post(config["base_url"].rstrip("/") + "/chat/completions", timeout=timeout,
                              headers={"Authorization": f"Bearer {config['api_key']}"}, json=body)
        except requests.RequestException as e:
            raise LLMError(f"network error: {e.__class__.__name__}")
        wait = _retry_after(r) if r.status_code == 429 else None
        if wait is not None and attempt == 0 and wait <= max_wait:
            time.sleep(wait)
            continue
        break
    if r.status_code == 429:
        raise LLMError("rate limited")
    if r.status_code != 200:
        raise LLMError(f"HTTP {r.status_code}")
    try:
        choice = r.json()["choices"][0]
        content = choice["message"]["content"] or ""
    except (ValueError, KeyError, IndexError):
        raise LLMError("unexpected response shape")
    if choice.get("finish_reason") == "length":
        raise LLMError("reply cut off at the token limit")
    if not content.strip():
        raise LLMError("empty reply")
    return content


def parse_reply(text):
    """{id: sentence} from the model's reply, or {} when it is not usable JSON."""
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return {}
    try:
        data = json.loads(text[start:end + 1])
    except ValueError:
        return {}
    return {str(k): v for k, v in data.items() if isinstance(v, str)} if isinstance(data, dict) else {}


# ---------------------------------------------------------------- putting it together

def explain_picks(recs, facts, spoiler_terms, config=None, use_cache=True):
    """One result per recommendation: {'text', 'source': 'model' | 'template', 'problems': [...]}.

    Returns (results, error). `error` is a short reason when the model could not be used at all.
    """
    items = [item_facts(r, facts) for r in recs]
    results = [None] * len(items)
    keys = [(it["id"], tuple(lk["title"] for lk in it["liked"])) for it in items]
    todo = []
    for i, key in enumerate(keys):
        if use_cache and key in _cache:
            results[i] = _cache[key]
        else:
            todo.append(i)
    error = None
    replies = {}
    if todo and config:
        try:
            replies = parse_reply(call_llm(build_messages([items[i] for i in todo]), config))
            if not replies:
                error = "reply was not valid JSON"
        except LLMError as e:
            error = str(e)
    elif todo:
        error = "no model configured"
    for i in todo:
        item = items[i]
        sentence = replies.get(str(item["id"]))
        problems = check(sentence, item, facts, spoiler_terms) if sentence is not None else ["no sentence returned"]
        if sentence is not None and not problems:
            results[i] = {"text": sentence.strip(), "source": "model", "problems": []}
        else:
            results[i] = {"text": template_sentence(item), "source": "template", "problems": problems}
        if use_cache and error is None:
            if len(_cache) >= CACHE_LIMIT:
                _cache.clear()
            _cache[keys[i]] = results[i]
    return results, error
