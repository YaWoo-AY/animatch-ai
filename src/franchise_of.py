"""Show which franchise a title is in and how it is linked to other titles.

Run from the project root:
    python src/franchise_of.py "Hori-san to Miyamura-kun"
    python src/franchise_of.py "Edgerunners 2"

Matches titles that contain the text (ignoring case) and prints their franchise members and the
relations AniList gives them, so a missing or unwanted link is easy to see.
"""
import json
import sys
from pathlib import Path

INDEX_PATH = Path("data/anime.json")


def describe(records, text):
    by_id = {r["anilist_id"]: r for r in records}
    members = {}
    for r in records:
        members.setdefault(r["franchise_id"], []).append(r)
    lines = []
    for r in records:
        if text.lower() not in r["title"].lower():
            continue
        lines.append(f"{r['title']}  [{r['anilist_id']}]  {r['format']} {r['year']}  franchise {r['franchise_id']}")
        others = [m for m in members[r["franchise_id"]] if m["anilist_id"] != r["anilist_id"]]
        lines.append("  same franchise: " + (", ".join(f"{m['title']} ({m['format']} {m['year']})" for m in others) or "none"))
        for rel in r["relations"]:
            if rel["media_type"] != "ANIME":
                continue
            t = by_id.get(rel["anilist_id"])
            where = f"{t['title']} ({t['format']} {t['year']})" if t else f"{rel['anilist_id']} (not in the index)"
            lines.append(f"  {rel['type']:<12} {where}")
    return lines


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    if len(sys.argv) < 2:
        sys.exit('Usage: python src/franchise_of.py "title text"')
    records = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
    lines = describe(records, " ".join(sys.argv[1:]))
    print("\n".join(lines) if lines else "No title contains that text.")


if __name__ == "__main__":
    main()
