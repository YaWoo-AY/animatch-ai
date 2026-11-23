"""Fetch anime metadata from the AniList GraphQL API and save it to disk.

Fetches pages of titles, cleans them, and crawls with checkpointing, so a crash or rate limit never loses work.

Run from the project root:
    python src/fetch_anilist.py --pages 3     # small test crawl (150 titles)
    python src/fetch_anilist.py --pages 100   # full crawl (about 5,000 titles)
    python src/fetch_anilist.py --pages 100   # run again after a crash: it resumes
    python src/fetch_anilist.py --fresh --pages 100   # throw away progress and start over
"""
import argparse
import html
import json
import re
import time
from collections import Counter
from pathlib import Path

import requests

URL = "https://graphql.anilist.co"
RAW_PATH = Path("data/raw_anime.jsonl")      # one raw AniList record per line
STATE_PATH = Path("data/crawl_state.json")   # remembers the last finished page
CLEAN_PATH = Path("data/clean_anime.json")

# Formats worth recommending. Decision: drop MUSIC, SPECIAL, TV_SHORT for now.
FORMATS = "[TV, MOVIE, ONA, OVA]"

QUERY = """
query ($page: Int) {
  Page(page: $page, perPage: 50) {
    pageInfo { hasNextPage currentPage }
    media(type: ANIME, format_in: %s, isAdult: false, sort: POPULARITY_DESC) {
      id
      idMal
      title { romaji english }
      description(asHtml: false)
      genres
      tags { name rank isMediaSpoiler }
      episodes
      seasonYear
      averageScore
      popularity
      format
      status
      coverImage { large }
      relations { edges { relationType node { id type } } }
    }
  }
}
""" % FORMATS

CREDIT_RE = re.compile(r"\s*[\(\[]\s*(?:Source|Written by)[^\)\]]*[\)\]]", re.I)


def fetch_page(page: int, max_retries: int = 5) -> dict:
    """Fetch one page, retrying with exponential backoff on rate limits or errors."""
    for attempt in range(max_retries):
        try:
            r = requests.post(URL, json={"query": QUERY, "variables": {"page": page}}, timeout=30)
            if r.status_code == 429:
                wait = int(r.headers.get("Retry-After", 2 ** attempt * 5))
                print(f"  rate limited, waiting {wait}s")
                time.sleep(wait)
                continue
            r.raise_for_status()
            body = r.json()
            if "errors" in body:
                raise RuntimeError(body["errors"])
            return body["data"]["Page"]
        except (requests.RequestException, RuntimeError) as e:
            wait = 2 ** attempt
            print(f"  attempt {attempt + 1} failed ({e}); retrying in {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"Page {page} failed after {max_retries} attempts")


def clean_text(raw):
    """Strip HTML tags, entities and source credits, collapse whitespace. Returns '' for None."""
    if not raw:
        return ""
    text = re.sub(r"<br\s*/?>", " ", raw, flags=re.I)
    text = re.sub(r"<[^>]+>", "", text)
    text = html.unescape(text)
    text = CREDIT_RE.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


def clean_record(m: dict) -> dict:
    """Turn a raw AniList media object into the flat record we will index."""
    title = (m["title"].get("english") or m["title"].get("romaji") or "Unknown")
    return {
        "anilist_id": m["id"],
        "mal_id": m["idMal"],  # can be None
        "title": title,
        "synopsis": clean_text(m["description"]),
        "genres": m["genres"] or [],
        # Safe to show: non-spoiler tags only.
        "tags": [t["name"] for t in (m["tags"] or []) if not t["isMediaSpoiler"] and t["rank"] >= 60],
        # Spoiler-flagged tags are often the most informative ones ("Time Loop" is a spoiler tag
        # on Summer Time Rendering). Kept for RETRIEVAL ONLY: never show these to a user or put
        # them in the prompt of the LLM that writes pitches.
        "spoiler_tags": [t["name"] for t in (m["tags"] or []) if t["isMediaSpoiler"] and t["rank"] >= 60],
        "episodes": m["episodes"],
        "year": m["seasonYear"],
        "score": m["averageScore"],
        "popularity": m["popularity"],
        "format": m["format"],
        "status": m.get("status"),
        "cover": (m.get("coverImage") or {}).get("large"),
        "relations": [
            {"type": e["relationType"], "anilist_id": e["node"]["id"], "media_type": e["node"]["type"]}
            for e in (m["relations"]["edges"] if m.get("relations") else [])
        ],
    }


def load_state() -> int:
    """Return the last fully saved page (0 if none)."""
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text(encoding="utf-8")).get("last_page", 0)
    return 0


def save_state(page: int, finished: bool = False) -> None:
    STATE_PATH.write_text(json.dumps({"last_page": page, "finished": finished}), encoding="utf-8")


def crawl(max_pages: int, pause: float = 1.5) -> None:
    """Fetch pages (resuming where we left off) and append each one to the raw file."""
    start = load_state() + 1
    if start > max_pages:
        print(f"Already have {start - 1} pages; nothing to fetch.")
        return
    if start > 1:
        print(f"Resuming from page {start}.")
    with RAW_PATH.open("a", encoding="utf-8") as f:
        for page in range(start, max_pages + 1):
            result = fetch_page(page)
            for m in result["media"]:
                f.write(json.dumps(m, ensure_ascii=False) + "\n")
            f.flush()
            save_state(page, finished=not result["pageInfo"]["hasNextPage"])
            if page % 10 == 0 or page == start:
                print(f"  page {page}/{max_pages} saved")
            if not result["pageInfo"]["hasNextPage"]:
                print("Reached the last page AniList has.")
                return
            time.sleep(pause)  # stay well under the rate limit


def build_clean() -> list:
    """Read the raw file, drop duplicate ids, clean every record, write clean_anime.json."""
    seen, clean = set(), []
    dupes = 0
    with RAW_PATH.open(encoding="utf-8") as f:
        for line in f:
            m = json.loads(line)
            if m["id"] in seen:
                dupes += 1
                continue
            seen.add(m["id"])
            clean.append(clean_record(m))
    CLEAN_PATH.write_text(json.dumps(clean, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Duplicate rows dropped: {dupes}")
    return clean


def report(clean: list) -> None:
    """Print the numbers that decide how the later stages must handle messy data."""
    n = len(clean)
    words = [len(c["synopsis"].split()) for c in clean if c["synopsis"]]
    long_syn = sum(1 for w in words if w > 190)
    print(f"\n=== Data report: {n} titles ===")
    print(f"Null idMal:       {sum(1 for c in clean if c['mal_id'] is None)}")
    print(f"Empty synopsis:   {sum(1 for c in clean if not c['synopsis'])}")
    print(f"Null episodes:    {sum(1 for c in clean if c['episodes'] is None)}")
    print(f"Null score:       {sum(1 for c in clean if c['score'] is None)}")
    print(f"Null year:        {sum(1 for c in clean if c['year'] is None)}")
    if words:
        print(f"Synopses over ~190 words (MiniLM truncation): {long_syn} ({100 * long_syn / len(words):.1f}%)")
    print("Formats:", dict(Counter(c["format"] for c in clean)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", type=int, default=1)
    ap.add_argument("--fresh", action="store_true", help="delete saved progress and start over")
    args = ap.parse_args()

    Path("data").mkdir(exist_ok=True)
    if args.fresh:
        for p in (RAW_PATH, STATE_PATH):
            p.unlink(missing_ok=True)

    crawl(args.pages)
    clean = build_clean()
    report(clean)


if __name__ == "__main__":
    main()
