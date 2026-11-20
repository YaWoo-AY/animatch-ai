"""Group anime into franchises using AniList relation edges, and write the index-ready file.

Why: if a user has finished season 1, we must not recommend season 2, the recap movie or the OVA.
Every title gets a franchise_id; later, one watched title excludes its whole franchise.

Run from the project root, after fetch_anilist.py:
    python src/franchise.py
Reads data/clean_anime.json, writes data/anime.json.
"""
import json
from collections import Counter, defaultdict
from pathlib import Path

CLEAN_PATH = Path("data/clean_anime.json")
OUT_PATH = Path("data/anime.json")

# Two kinds of link, because AniList uses PARENT / SIDE_STORY to hang separate timelines
# under one original (all of Gundam sat under the 1979 show in the first version).
#
# CORE_TYPES: same continuity. Always merged.
# ATTACH_TYPES: a movie / OVA / ONA (or a chain of them) joins its single parent or
#   side-story franchise. Anything containing a TV series, like Gundam Wing, Seed or
#   G Gundam, is NOT pulled into the parent's group.
# SPIN_OFF, ALTERNATIVE, ADAPTATION, OTHER and CHARACTER are never linked.
CORE_TYPES = {"SEQUEL", "PREQUEL", "SUMMARY"}
ATTACH_TYPES = {"PARENT", "SIDE_STORY"}

# A group bigger than this is probably over-merged and worth a manual look.
SUSPICIOUS_SIZE = 25


class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, x):
        self.parent.setdefault(x, x)
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:  # path compression
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)  # smallest id becomes the root


def build_franchises(records: list) -> dict:
    """Return {anilist_id: franchise_id} for every record.

    Edges to titles we did not crawl are still followed, so two crawled titles
    connected only through an uncrawled one end up in the same franchise.
    """
    uf = UnionFind()
    ids = {rec["anilist_id"] for rec in records}
    attach_edges = []
    for rec in records:
        uf.find(rec["anilist_id"])
        for rel in rec["relations"]:
            if rel["media_type"] != "ANIME":
                continue
            if rel["type"] in CORE_TYPES:
                uf.union(rec["anilist_id"], rel["anilist_id"])
            elif rel["type"] in ATTACH_TYPES:
                attach_edges.append((rec["anilist_id"], rel["anilist_id"]))

    # Step 2: standalone titles join their one parent / side-story group.
    # Everything is decided from the step-1 groups, so the order of records never matters.
    # A group may attach to a parent only if it has no TV series in it. A movie / OVA / ONA
    # or a chain of them is an extension of its parent (Gundam 0080, the Pokemon movies).
    # Any group with a TV series (Wing, Seed, G Gundam, Turn A, Witch from Mercury) is its
    # own story and stays separate, even a lone TV show with a PARENT link.
    # Known cost: a TV season linked to its show only by PARENT / SIDE_STORY is not merged.
    size = Counter(uf.find(i) for i in ids)  # crawled titles per group
    has_tv = {uf.find(rec["anilist_id"]) for rec in records if rec.get("format") == "TV"}
    targets = defaultdict(set)  # group root -> set of parent group roots
    for a, b in attach_edges:
        for s, t in ((a, b), (b, a)):
            gs, gt = uf.find(s), uf.find(t)
            if s in ids and gs != gt and size[gt] >= 1 and gs not in has_tv:
                targets[gs].add(gt)
    for gs, roots in targets.items():
        if len(roots) == 1:  # two different parents is ambiguous, so stay standalone
            uf.union(gs, next(iter(roots)))

    return {rec["anilist_id"]: uf.find(rec["anilist_id"]) for rec in records}


def main():
    records = json.loads(CLEAN_PATH.read_text(encoding="utf-8"))
    mapping = build_franchises(records)
    for rec in records:
        rec["franchise_id"] = mapping[rec["anilist_id"]]

    # Kept out of the index (after grouping, so unreleased sequels still link franchises):
    #  - no synopsis: nothing to embed
    #  - NOT_YET_RELEASED / CANCELLED: nothing the user can watch
    indexable = [r for r in records
                 if r["synopsis"] and r.get("status") not in {"NOT_YET_RELEASED", "CANCELLED"}]
    OUT_PATH.write_text(json.dumps(indexable, indent=2, ensure_ascii=False), encoding="utf-8")

    groups = defaultdict(list)
    for rec in records:
        groups[rec["franchise_id"]].append(rec)
    multi = [g for g in groups.values() if len(g) > 1]
    biggest = sorted(groups.values(), key=len, reverse=True)[:8]

    print(f"Titles: {len(records)}  |  written to index file: {len(indexable)}  |  "
          f"dropped (no synopsis or unreleased): {len(records) - len(indexable)}")
    print(f"Franchises: {len(groups)}  |  with 2+ titles: {len(multi)}  |  "
          f"standalone titles: {len(groups) - len(multi)}")
    print("\nLargest franchises (sanity check these by eye):")
    for g in biggest:
        top = max(g, key=lambda r: r["popularity"] or 0)
        flag = "  <-- suspiciously large" if len(g) > SUSPICIOUS_SIZE else ""
        print(f"  {len(g):3d} titles  e.g. {top['title']}{flag}")


if __name__ == "__main__":
    main()
