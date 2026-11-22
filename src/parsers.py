"""Day 5: read a user's public anime list from AniList or MyAnimeList (official API).

Every platform is turned into the same entry format:
    {"anilist_id": int | None, "mal_id": int | None, "status": str, "score": float | None, "progress": int}
status is one of: completed, watching, paused, dropped, planning.
score is 0-10 (None when the user did not rate the title).

Run from the project root, in the terminal whose prompt starts with (.venv):
    python src/parsers.py anilist <username>
    python src/parsers.py mal <username>
    python src/parsers.py mal-probe <username>     # print the raw MAL response shape (to check it)
    python src/parsers.py anilist <username> --refresh   # ignore the cache

MyAnimeList needs a free client ID (register an app at https://myanimelist.net/apiconfig). Put it in the
environment variable MAL_CLIENT_ID, or in a line MAL_CLIENT_ID=... in the git-ignored .env file.

Profiles are cached for 24 hours in data/cache/ (git-ignored), so repeat runs make no network calls.
"""
import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path
from urllib.parse import quote

import requests

ANILIST_URL = "https://graphql.anilist.co"
MAL_URL = "https://api.myanimelist.net/v2"
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

MAL_STATUS = {
    "watching": "watching", "completed": "completed", "on_hold": "paused",
    "dropped": "dropped", "plan_to_watch": "planning",
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


TITLE_LOOKUP = """
query ($ids: [Int], $perPage: Int) {
  Page(perPage: $perPage) {
    media(id_in: $ids, type: ANIME) {
      id title { english romaji } format status popularity episodes seasonYear
    }
  }
}
"""


def lookup_titles(ids, pause=0.8):
    """Look up title, format and popularity for AniList ids (used to explain why titles are unmatched)."""
    found = {}
    ids = list(ids)
    for i in range(0, len(ids), 50):
        part = ids[i:i + 50]
        data = _post_anilist(TITLE_LOOKUP, {"ids": part, "perPage": 50})
        for m in (data.get("Page") or {}).get("media") or []:
            t = m.get("title") or {}
            found[m["id"]] = {"title": t.get("english") or t.get("romaji") or "?", "format": m.get("format"),
                              "status": m.get("status"), "popularity": m.get("popularity"),
                              "year": m.get("seasonYear")}
        if i + 50 < len(ids):
            time.sleep(pause)
    return found


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


# ---------------------------------------------------------------- MyAnimeList (official API v2)
# Verified on 2026-11-22 against a real public list: a client ID alone (header X-MAL-CLIENT-ID) is enough to
# read another user's PUBLIC list, and an unknown user gives HTTP 404 {"error": "not_found"}.
# Real item shape: {"node": {"id": 62524, "title": ..., "main_picture": ...},
#                   "list_status": {"status": "watching", "score": 0, "num_episodes_watched": 5, ...}}
# score 0 means unrated. Not confirmed: the exact reply for a private list (we assume HTTP 403).

MAL_PAGE_SIZE = 100  # verified with limit=3; 100 is the API default page size, so it is safe

class MissingClientId(ProfileError):
    pass


def get_mal_client_id(env_path=".env"):
    value = os.environ.get("MAL_CLIENT_ID", "").strip()
    if not value and Path(env_path).exists():
        for line in Path(env_path).read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("MAL_CLIENT_ID="):
                value = line.split("=", 1)[1].strip().strip('"').strip("'")
    if not value:
        raise MissingClientId("no MAL client ID. Register a free app at https://myanimelist.net/apiconfig, "
                              "then set MAL_CLIENT_ID (PowerShell: $env:MAL_CLIENT_ID=\"...\") "
                              "or add MAL_CLIENT_ID=... to .env")
    return value


def parse_mal_items(items):
    """Turn MAL v2 list items ({"node": {"id"}, "list_status": {...}}) into standard entries."""
    out = []
    for item in items or []:
        node, ls = item.get("node") or {}, item.get("list_status") or {}
        status = MAL_STATUS.get(ls.get("status"))
        if node.get("id") is None or status is None:
            continue
        out.append({"anilist_id": None, "mal_id": int(node["id"]), "status": status,
                    "score": _score(ls.get("score")), "progress": int(ls.get("num_episodes_watched") or 0)})
    return out


def _get_mal(url, client_id, params=None, max_retries=5):
    """GET from the MAL API with retries. Its rate limit is not published; stay near 1 request per second."""
    last = "no response"
    for attempt in range(max_retries):
        try:
            r = requests.get(url, params=params, headers={"X-MAL-CLIENT-ID": client_id}, timeout=(10, 30))
        except requests.RequestException as e:
            last = str(e)
            time.sleep(2 ** attempt)
            continue
        if r.status_code == 429 or r.status_code >= 500:
            last = f"HTTP {r.status_code}"
            time.sleep(_retry_after(r, 2 ** attempt))
            continue
        if r.status_code == 404:
            raise UserNotFound("MyAnimeList user not found (check the spelling)")
        if r.status_code == 403:
            raise PrivateList("MyAnimeList refused access (HTTP 403): the list is probably private")
        try:
            body = r.json()
        except ValueError:
            raise ApiError(f"MyAnimeList sent a non-JSON reply (HTTP {r.status_code})")
        if r.status_code in (400, 401):
            raise ApiError(f"MyAnimeList rejected the request (HTTP {r.status_code}): "
                           f"{body.get('message') or body.get('error') or body}")
        if r.status_code != 200:
            raise ApiError(f"MyAnimeList HTTP {r.status_code}: {body.get('message') or body}")
        return body
    raise ApiError(f"MyAnimeList failed after {max_retries} attempts: {last}")


def fetch_mal(username, client_id=None, limit=MAL_PAGE_SIZE, pause=1.0, max_pages=120):
    """Fetch a public MyAnimeList anime list. Returns a list of standard entries."""
    client_id = client_id or get_mal_client_id()
    url, params = f"{MAL_URL}/users/{quote(username)}/animelist", {"fields": "list_status", "limit": limit}
    entries = []
    for _ in range(max_pages):
        body = _get_mal(url, client_id, params)
        entries.extend(parse_mal_items(body.get("data")))
        url, params = (body.get("paging") or {}).get("next"), None  # "next" is a complete URL
        if not url:
            break
        time.sleep(pause)
    return entries


def probe_mal(username):
    """Print the raw shape of MAL's response, so we can confirm the parser matches it."""
    client_id = get_mal_client_id()
    try:
        r = requests.get(f"{MAL_URL}/users/{quote(username)}/animelist", params={"fields": "list_status", "limit": 2},
                         headers={"X-MAL-CLIENT-ID": client_id}, timeout=(10, 30))
    except requests.RequestException as e:
        print(f"Could not reach MyAnimeList ({type(e).__name__}): {e}")
        return
    print("HTTP status:", r.status_code)
    try:
        body = r.json()
    except ValueError:
        print("Not JSON:", r.text[:300])
        return
    if isinstance(body, dict):
        print("Top-level keys:", list(body.keys()))
        print("paging:", body.get("paging"))
        data = body.get("data")
        if isinstance(data, list) and data:
            print(f"data: list of {len(data)}; first entry:")
            print(json.dumps(data[0], indent=2, ensure_ascii=False)[:1200])
        else:
            print("data:", str(data)[:300], "| error:", body.get("error"), body.get("message"))
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


FETCHERS = {"anilist": fetch_anilist, "mal": fetch_mal}


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
    ap.add_argument("--unmatched", action="store_true",
                    help="list watched titles that are not in our index (with AniList links)")
    args = ap.parse_args()

    if args.platform == "mal-probe":
        try:
            probe_mal(args.username)
        except MissingClientId as e:
            sys.exit(f"Cannot probe: {e}")
        return
    try:
        entries = get_profile(args.platform, args.username, refresh=args.refresh)
    except UserNotFound as e:
        sys.exit(f"User not found on {args.platform}: \"{args.username}\". Check the spelling.")
    except PrivateList as e:
        sys.exit(f"Private list: {e}. Ask the user to make the list public, or use another profile.")
    except ProfileError as e:
        sys.exit(f"Could not read the list: {e}")
    if not entries:
        sys.exit("The list is empty (or has no anime entries).")
    matched, unmatched = match_to_index(entries, load_index())
    summarize(entries, matched, unmatched)
    if args.unmatched:
        shown = [e for e in unmatched if e["status"] in {"completed", "watching", "dropped"}]
        print(f"\nNot in index, seen or dropped ({len(shown)}):")
        ids = [e["anilist_id"] for e in shown if e["anilist_id"]]
        try:
            info = lookup_titles(ids)
        except ProfileError as e:
            info = {}
            print(f"  (could not look up titles: {e})")
        formats = Counter()
        for e in shown[:80]:
            score = "-" if e["score"] is None else f"{e['score']:.1f}"
            meta = info.get(e["anilist_id"])
            if meta:
                formats[meta["format"]] += 1
                print(f"  {e['status']:<9} {score:>4}  {meta['title'][:42]:<42} {str(meta['format']):<9} "
                      f"{str(meta['year']):<5} pop {meta['popularity']}")
            else:
                link = (f"https://anilist.co/anime/{e['anilist_id']}" if e["anilist_id"]
                        else f"https://myanimelist.net/anime/{e['mal_id']}")
                print(f"  {e['status']:<9} {score:>4}  {link}")
        if formats:
            print("By format:", dict(formats))


if __name__ == "__main__":
    main()
