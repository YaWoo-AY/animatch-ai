"""Day 2: fetch anime metadata from the AniList GraphQL API and save it to disk.

Run from the project root:
    python src/fetch_anilist.py            # fetches 1 page (50 titles)
    python src/fetch_anilist.py --pages 3  # fetches 3 pages (150 titles)
"""
import argparse
import html
import json
import re
import time
from pathlib import Path

import requests

URL = "https://graphql.anilist.co"
RAW_PATH = Path("data/raw_anime.json")
CLEAN_PATH = Path("data/clean_anime.json")

QUERY = """
query ($page: Int) {
  Page(page: $page, perPage: 50) {
    pageInfo { hasNextPage currentPage }
    media(type: ANIME, sort: POPULARITY_DESC) {
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
      coverImage { large }
      relations { edges { relationType node { id type } } }
    }
  }
}
"""


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


CREDIT_RE = re.compile(r"\s*[\(\[]\s*(?:Source|Written by)[^\)\]]*[\)\]]", re.I)
def clean_text(raw):
    """Strip HTML tags and entities, collapse whitespace. Returns '' for None."""
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
        # keep spoiler-flagged tags out of anything shown to the LLM later
        "tags": [t["name"] for t in (m["tags"] or []) if not t["isMediaSpoiler"] and t["rank"] >= 60],
        "episodes": m["episodes"],
        "year": m["seasonYear"],
        "score": m["averageScore"],
        "popularity": m["popularity"],
        "format": m["format"],
        "cover": (m.get("coverImage") or {}).get("large"),
        "relations": [
            {"type": e["relationType"], "anilist_id": e["node"]["id"], "media_type": e["node"]["type"]}
            for e in (m["relations"]["edges"] if m.get("relations") else [])
        ],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", type=int, default=1)
    args = ap.parse_args()

    Path("data").mkdir(exist_ok=True)
    raw, page = [], 1
    while page <= args.pages:
        print(f"Fetching page {page}...")
        result = fetch_page(page)
        raw.extend(result["media"])
        if not result["pageInfo"]["hasNextPage"]:
            break
        page += 1
        time.sleep(1.5)  # stay well under the rate limit

    RAW_PATH.write_text(json.dumps(raw, indent=2, ensure_ascii=False), encoding="utf-8")
    clean = [clean_record(m) for m in raw]
    CLEAN_PATH.write_text(json.dumps(clean, indent=2, ensure_ascii=False), encoding="utf-8")

    missing_mal = sum(1 for c in clean if c["mal_id"] is None)
    empty_syn = sum(1 for c in clean if not c["synopsis"])
    print(f"Saved {len(clean)} titles. Null idMal: {missing_mal}. Empty synopsis: {empty_syn}.")


if __name__ == "__main__":
    main()
