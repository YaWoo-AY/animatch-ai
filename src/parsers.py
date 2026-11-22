"""Day 5: read a user's public anime list from AniList or MyAnimeList (through Jikan).

Every platform is turned into the same entry format:
    {"anilist_id": int | None, "mal_id": int | None, "status": str, "score": float | None, "progress": int}
status is one of: completed, watching, paused, dropped, planning.
score is 0-10 (None when the user did not rate the title).

Run from the project root, in the terminal whose prompt starts with (.venv):
    python src/parsers.py anilist <username>
    python src/parsers.py mal <username>
    python src/parsers.py mal-probe <username>     # print the raw Jikan response shape (to check it)
    python src/parsers.py anilist <username> --refresh   # ignore the cache

Profiles are cached for 24 hours in data/cache/ (git-ignored), so repeat runs make no network calls.
"""
import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path
from urllib.parse import quote

import requests

ANILIST_URL = "https://graphql.anilist.co"
JIKAN_URL = "https://api.jikan.moe/v4"
ANIME_PATH = Path("data/anime.json")
CACHE_DIR = Path("data/cache")
CACHE_TTL = 24 * 3600  # seconds


# ---------------------------------------------------------------- errors

class ProfileError(Exception):
    """Base class: anything that stops us reading a list."""


class UserNotFound(ProfileError):
    pass


class PrivateList(ProfileError):
    pass


class ApiError(ProfileError):
    pass


class QueryRejected(ApiError):
    """AniList refused the query itself (HTTP 400), for example an unknown argument."""


# ---------------------------------------------------------------- status and score mapping

ANILIST_STATUS = {
    "COMPLETED": "completed",
    "REPEATING": "completed",  # rewatching: they have seen it
    "CURRENT": "watching",
    "PAUSED": "paused",
    "DROPPED": "dropped",
    "PLANNING": "planning",
}

JIKAN_STATUS = {
    1: "watching", 2: "completed", 3: "paused", 4: "dropped", 6: "planning",
    "watching": "watching", "currentlywatching": "watching", "completed": "completed",
    "onhold": "paused", "dropped": "dropped", "plantowatch": "planning",
}


def _score(raw):
    """0 or missing means "not rated" on both platforms."""
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


# ---------------------------------------------------------------- AniList

# score(format: POINT_10_DECIMAL) returns 0-10 whatever scale the user picked on AniList.
# Custom lists are included on purpose: entries a user hides from their status lists only
# show up there. Entries are de-duplicated by AniList id.
ANILIST_CHUNKED = """
query ($name: String, $chunk: Int, $perChunk: Int) {
  MediaListCollection(userName: $name, type: ANIME, chunk: $chunk, perChunk: $perChunk) {
    hasNextChunk
    lists {
      entries {
        status
        score(format: POINT_10_DECIMAL)
        progress
        media { id idMal }
      }
    }
  }
}
"""

# Fallback if AniList rejects the chunk arguments: a single request, no chunking.
ANILIST_PLAIN = """
query ($name: String) {
  MediaListCollection(userName: $name, type: ANIME) {
    lists {
      entries {
        status
        score(format: POINT_10_DECIMAL)
        progress
        media { id idMal }
      }
    }
  }
}
"""


def classify_anilist_errors(http_status, errors):
    """Turn an AniList error payload into the right exception."""
    first = errors[0] if errors else {}
    status = first.get("status") or http_status
    message = " ".join(str(e.get("message", "")) for e in errors).strip()
    low = message.lower()
    if status == 404 or "not found" in low:
        return UserNotFound(message or "user not found")
    if status == 403 or "private" in low:
        return PrivateList(message or "this list is private")
    if status == 400:
        return QueryRejected(message)
    return ApiError(message or f"AniList error (HTTP {http_status})")


def _retry_after(response, default):
    try:
        return max(1, int(response.headers.get("Retry-After", default)))
    except (TypeError, ValueError):
        return default


def _post_anilist(query, variables, max_retries=5):
    """POST one GraphQL query, retrying rate limits and server errors with backoff."""
    last = "no response"
    for attempt in range(max_retries):
        try:
            r = requests.post(ANILIST_URL, json={"query": query, "variables": variables}, timeout=30)
        except requests.RequestException as e:
            last = str(e)
            time.sleep(2 ** attempt)
            continue
        if r.status_code == 429:
            last = "rate limited"
            time.sleep(_retry_after(r, 5 * 2 ** attempt))
            continue
        if r.status_code >= 500:
            last = f"HTTP {r.status_code}"
            time.sleep(2 ** attempt)
            continue
        try:
            body = r.json()
        except ValueError:
            raise ApiError(f"AniList sent a non-JSON reply (HTTP {r.status_code})")
        if body.get("errors"):
            raise classify_anilist_errors(r.status_code, body["errors"])
        return body["data"]
    raise ApiError(f"AniList failed after {max_retries} attempts: {last}")


def parse_anilist_collection(collection, into):
    """Add entries from one MediaListCollection to the dict `into` (keyed by AniList id)."""
    for group in (collection or {}).get("lists") or []:
        for e in group.get("entries") or []:
            media = e.get("media") or {}
            media_id = media.get("id")
            status = ANILIST_STATUS.get(e.get("status"))
            if media_id is None or status is None or media_id in into:
                continue
            into[media_id] = {
                "anilist_id": media_id,
                "mal_id": media.get("idMal"),
                "status": status,
                "score": _score(e.get("score")),
                "progress": e.get("progress") or 0,
            }


def fetch_anilist(username, per_chunk=500, pause=1.0):
    """Fetch a whole public AniList anime list. Returns a list of entries."""
    entries = {}
    chunk, chunked = 1, True
    while True:
        try:
            if chunked:
                data = _post_anilist(ANILIST_CHUNKED, {"name": username, "chunk": chunk, "perChunk": per_chunk})
            else:
                data = _post_anilist(ANILIST_PLAIN, {"name": username})
        except QueryRejected:
            if chunked and chunk == 1:
                print("note: AniList rejected the chunked query; using a single request instead.")
                chunked = False
                continue
            raise
        collection = data.get("MediaListCollection")
        parse_anilist_collection(collection, entries)
        if not chunked or not (collection or {}).get("hasNextChunk"):
            break
        chunk += 1
        time.sleep(pause)
    return list(entries.values())


# ---------------------------------------------------------------- MyAnimeList through Jikan
# NOTE: the exact shape of Jikan's user-list response could not be checked against its docs
# when this was written, so the parser accepts several field names. Run `mal-probe` once to
# see the real shape, and send it to be checked.

def _jikan_mal_id(item):
    anime = item.get("anime")
    if isinstance(anime, dict) and anime.get("mal_id") is not None:
        return anime["mal_id"]
    for key in ("anime_id", "mal_id"):
        if item.get(key) is not None:
            return item[key]
    return None


def _jikan_status(item):
    value = item.get("watching_status", item.get("status"))
    if isinstance(value, str):
        if value.strip().isdigit():
            value = int(value)
        else:
            value = re.sub(r"[^a-z]", "", value.lower())
    return JIKAN_STATUS.get(value)


def parse_jikan_items(items):
    out = []
    for item in items or []:
        mal_id, status = _jikan_mal_id(item), _jikan_status(item)
        if mal_id is None or status is None:
            continue
        progress = item.get("episodes_watched", item.get("num_watched_episodes")) or 0
        out.append({
            "anilist_id": None,
            "mal_id": mal_id,
            "status": status,
            "score": _score(item.get("score")),
            "progress": progress,
        })
    return out


def _get_jikan(path, params=None, max_retries=5):
    """GET from Jikan with retries. Its limit is 3 requests per second and 60 per minute."""
    last = "no response"
    for attempt in range(max_retries):
        try:
            r = requests.get(JIKAN_URL + path, params=params, timeout=30)
        except requests.RequestException as e:
            last = str(e)
            time.sleep(2 ** attempt)
            continue
        if r.status_code == 429:
            last = "rate limited"
            time.sleep(_retry_after(r, 5 * (attempt + 1)))
            continue
        if r.status_code >= 500:
            last = f"HTTP {r.status_code}"
            time.sleep(2 ** attempt)
            continue
        if r.status_code == 404:
            raise UserNotFound("MyAnimeList user not found (check the spelling)")
        if r.status_code == 403:
            raise PrivateList("this MyAnimeList list is private")
        try:
            body = r.json()
        except ValueError:
            raise ApiError(f"Jikan sent a non-JSON reply (HTTP {r.status_code})")
        if r.status_code >= 400:
            raise ApiError(f"Jikan HTTP {r.status_code}: {body.get('message') or body}")
        return body
    raise ApiError(f"Jikan failed after {max_retries} attempts: {last}")


def fetch_jikan(username, max_pages=60, pause=1.1):
    """Fetch a public MyAnimeList anime list through Jikan. Returns a list of entries."""
    entries, page = [], 1
    while page <= max_pages:
        body = _get_jikan(f"/users/{quote(username)}/animelist", {"page": page})
        entries.extend(parse_jikan_items(body.get("data")))
        if not (body.get("pagination") or {}).get("has_next_page"):
            break
        page += 1
        time.sleep(pause)
    return entries


def probe_jikan(username):
    """Print the raw shape of Jikan's response, so we can confirm the parser matches it."""
    r = requests.get(f"{JIKAN_URL}/users/{quote(username)}/animelist", params={"page": 1}, timeout=30)
    print("HTTP status:", r.status_code)
    try:
        body = r.json()
    except ValueError:
        print("Not JSON:", r.text[:300])
        return
    if isinstance(body, dict):
        print("Top-level keys:", list(body.keys()))
        print("pagination:", body.get("pagination"))
        data = body.get("data")
        if isinstance(data, list) and data:
            print(f"data: list of {len(data)}; first entry:")
            print(json.dumps(data[0], indent=2, ensure_ascii=False)[:1500])
        else:
            print("data:", str(data)[:300], "| message:", body.get("message"))
    else:
        print(str(body)[:500])


# ---------------------------------------------------------------- cache

def cache_path(platform, username, cache_dir=CACHE_DIR):
    safe = re.sub(r"[^a-z0-9_-]", "_", username.lower())
    return Path(cache_dir) / f"{platform}_{safe}.json"


def save_cache(path, entries, now=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"saved_at": now if now is not None else time.time(), "entries": entries}),
                    encoding="utf-8")


def load_cache(path, ttl=CACHE_TTL, now=None):
    """Return cached entries, or None if there is no cache or it is older than ttl seconds."""
    path = Path(path)
    if not path.exists():
        return None
    try:
        blob = json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        return None
    age = (now if now is not None else time.time()) - blob.get("saved_at", 0)
    return blob["entries"] if age <= ttl else None


FETCHERS = {"anilist": fetch_anilist, "mal": fetch_jikan}


def get_profile(platform, username, refresh=False, cache_dir=CACHE_DIR, now=None):
    """Entries for a user, from the cache when fresh, otherwise from the network."""
    path = cache_path(platform, username, cache_dir)
    if not refresh:
        cached = load_cache(path, now=now)
        if cached is not None:
            return cached
    entries = FETCHERS[platform](username)
    save_cache(path, entries, now=now)
    return entries


# ---------------------------------------------------------------- matching to our index

def load_index(path=ANIME_PATH):
    records = json.loads(Path(path).read_text(encoding="utf-8"))
    return {
        "by_anilist": {r["anilist_id"]: r for r in records},
        "by_mal": {r["mal_id"]: r for r in records if r["mal_id"] is not None},
    }


def match_to_index(entries, index):
    """Split entries into (matched, unmatched).

    Matched entries gain franchise_id and title. Unmatched titles are not in our 4,869-title
    index (not in the top 5,000, unreleased, or no synopsis), so they cannot block their
    franchise and cannot be recommended.
    """
    matched, unmatched = [], []
    for e in entries:
        rec = index["by_anilist"].get(e["anilist_id"]) if e["anilist_id"] is not None else None
        if rec is None and e["mal_id"] is not None:
            rec = index["by_mal"].get(e["mal_id"])
        if rec is None:
            unmatched.append(e)
        else:
            matched.append({**e, "anilist_id": rec["anilist_id"],
                            "franchise_id": rec["franchise_id"], "title": rec["title"]})
    return matched, unmatched


# ---------------------------------------------------------------- command line

def summarize(entries, matched, unmatched):
    print(f"\nEntries: {len(entries)}")
    by_status = Counter(e["status"] for e in entries)
    print("By status:", dict(by_status))
    scores = [e["score"] for e in entries if e["score"] is not None]
    if scores:
        print(f"Rated: {len(scores)}  |  mean score {sum(scores) / len(scores):.2f}  |  "
              f"rated 8 or higher: {sum(1 for s in scores if s >= 8)}")
    else:
        print("Rated: 0 (no scores, so the taste profile will have nothing to weight)")
    print(f"In our index: {len(matched)}  |  not in our index: {len(unmatched)}")
    gaps = Counter(e["status"] for e in unmatched)
    if gaps:
        print("Not in index, by status:", dict(gaps))
    top = sorted((e for e in matched if e["score"] is not None and e["score"] >= 9),
                 key=lambda e: -e["score"])[:10]
    if top:
        print("\nTop rated (9+):")
        for e in top:
            print(f"  {e['score']:.1f}  {e['title']}")
    dropped = [e for e in matched if e["status"] == "dropped"][:8]
    if dropped:
        print("\nSome dropped titles:", ", ".join(e["title"] for e in dropped))


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # titles contain accents and symbols
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("platform", choices=["anilist", "mal", "mal-probe"])
    ap.add_argument("username")
    ap.add_argument("--refresh", action="store_true", help="ignore the cache")
    args = ap.parse_args()

    if args.platform == "mal-probe":
        probe_jikan(args.username)
        return
    try:
        entries = get_profile(args.platform, args.username, refresh=args.refresh)
    except UserNotFound as e:
        sys.exit(f"User not found: {e}")
    except PrivateList as e:
        sys.exit(f"Private list: {e}. Ask the user to make the list public, or use another profile.")
    except ProfileError as e:
        sys.exit(f"Could not read the list: {e}")
    if not entries:
        sys.exit("The list is empty (or has no anime entries).")
    matched, unmatched = match_to_index(entries, load_index())
    summarize(entries, matched, unmatched)


if __name__ == "__main__":
    main()
