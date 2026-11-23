"""Recommendations from a real watchlist (first end-to-end version, no reranking yet).

Run from the project root, in the terminal whose prompt starts with (.venv):
    python src/recommend.py anilist <username>
    python src/recommend.py mal <username> --model bge -k 15
    python src/recommend.py anilist <username> --max-episodes 26 --min-year 2015 --min-score 70
    python src/recommend.py anilist <username> --hide-planned      # also hide plan-to-watch titles
    python src/recommend.py anilist <username> --alpha 0           # ignore dislikes (to see their effect)

How it works: the user's watched titles already live in ChromaDB as vectors. We average the ones
they liked (taste.py), subtract a little of the ones they disliked, and ask ChromaDB for the
nearest titles, skipping every franchise they have already seen. No model is loaded here, because
we only reuse vectors that build_index.py already stored. The vector is built from document
vectors, so the query prefix that BGE uses for text queries is not needed.
"""
import argparse
import sys

import numpy as np

import parsers
import taste
from build_index import CHROMA_DIR, MODELS, VARIANTS, collection_name, to_metadata
from search import build_where


def fetch_embeddings(col, anilist_ids, chunk=300):
    """Stored vectors for some titles: {anilist_id: vector}. Titles missing from the collection are skipped."""
    out = {}
    ids = [str(i) for i in anilist_ids]
    for i in range(0, len(ids), chunk):
        res = col.get(ids=ids[i:i + chunk], include=["embeddings"])
        for rid, emb in zip(res["ids"], res["embeddings"]):
            out[int(rid)] = np.asarray(emb, dtype=float)
    return out


FORMAT_RANK = {"TV": 0, "MOVIE": 1, "ONA": 2, "OVA": 3}


def entry_points(records):
    """One title per franchise to actually recommend: where a newcomer should start.

    Retrieval returns whichever title of a franchise is closest to the taste vector, which is often
    season 4 or an OVA. Someone who has not seen the franchise should be sent to its first entry:
    prefer TV over movie over ONA over OVA, then the earliest year, then the most popular.
    Returns {franchise_id: metadata dict in the same shape the database stores}.
    """
    best = {}
    for r in records:
        key = (FORMAT_RANK.get(r["format"], 9), r["year"] if r["year"] is not None else 9999,
               -(r["popularity"] or 0))
        cur = best.get(r["franchise_id"])
        if cur is None or key < cur[0]:
            best[r["franchise_id"]] = (key, r)
    return {fid: to_metadata(r) for fid, (_, r) in best.items()}


def within_limits(meta, max_episodes=None, min_year=None, min_score=None):
    """Python version of build_where, for checking one title. Unknown values are stored as -1."""
    if max_episodes is not None and not 1 <= meta["episodes"] <= max_episodes:
        return False
    if min_year is not None and meta["year"] < min_year:
        return False
    if min_score is not None and meta["score"] < min_score:
        return False
    return True


def retrieve(col, query_vector, hidden_franchises, k=10, where=None, start=None):
    """The k nearest titles whose franchise is not hidden, at most one title per franchise.

    ChromaDB cannot skip hundreds of franchises cheaply, so we ask for more neighbours than we need,
    filter in Python, and ask for twice as many again if too few survive. Each result carries its
    vector so the caller can explain it.
    """
    total = col.count()
    n = start or max(100, k * 10)
    while True:
        n = min(n, total)
        res = col.query(query_embeddings=[list(map(float, query_vector))], n_results=n, where=where,
                        include=["metadatas", "distances", "embeddings"])
        picked, used = [], set()
        for meta, dist, emb in zip(res["metadatas"][0], res["distances"][0], res["embeddings"][0]):
            fid = meta["franchise_id"]
            if fid in hidden_franchises or fid in used:
                continue
            used.add(fid)
            picked.append({**meta, "distance": dist, "vector": np.asarray(emb, dtype=float)})
            if len(picked) == k:
                return picked
        if n >= total or len(res["ids"][0]) < n:   # nothing more to ask for
            return picked
        n *= 2


def recommend(matched, col, episodes_by_id, k=10, where=None, alpha=0.3, unrated_share=0.35,
              hide_planned=False, starts=None, limits=None):
    """Returns (taste, recommendations). taste is None when there is nothing positive to learn from.

    `starts` is the output of entry_points(). When given, each recommendation is shown as its
    franchise's first entry and the title that actually matched is kept in r["via"]. The swap is
    skipped when the first entry breaks the user's limits (`limits` = dict of max_episodes,
    min_year, min_score), because then it is not what they asked for.
    """
    embeddings = fetch_embeddings(col, [e["anilist_id"] for e in matched])
    model = taste.build_taste(matched, embeddings, episodes_by_id, unrated_share, alpha)
    if model is None:
        return None, []
    hidden = taste.seen_franchises(matched)
    planned = taste.planned_franchises(matched, hidden)
    if hide_planned:
        hidden = hidden | planned
    recs = retrieve(col, model.vector, hidden, k, where)
    for r in recs:
        r["because"] = taste.explain(model, r["vector"])
        r["planned"] = r["franchise_id"] in planned
        ep = (starts or {}).get(r["franchise_id"])
        if ep and ep["anilist_id"] != r["anilist_id"] and within_limits(ep, **(limits or {})):
            r["via"] = r["title"]
            r.update({k_: v for k_, v in ep.items()})
    return model, recs


def show(model, recs):
    pos, neg = model.positives, model.negatives
    rated = sum(1 for o in pos if o.rated)
    print(f"\nTaste built from {len(pos)} liked franchises ({rated} rated, {len(pos) - rated} unrated) "
          f"and {len(neg)} disliked.")
    if model.unrated_scale < 1:
        print(f"Unrated titles were scaled to {model.unrated_scale:.2f} of their weight so they do not "
              f"outvote your ratings.")
    print("Strongest likes:", ", ".join(o.title for o in pos[:6]))
    if neg:
        print("Strongest dislikes:", ", ".join(o.title for o in neg[:5]))
    print()
    for i, r in enumerate(recs, 1):
        eps = r["episodes"] if r["episodes"] > 0 else "?"
        score = r["score"] if r["score"] >= 0 else "-"
        year = r["year"] if r["year"] > 0 else "?"
        flag = "  [on your plan-to-watch list]" if r["planned"] else ""
        print(f"{i:2d}. {r['title']}  ({year}, {r['format']}, {eps} eps, score {score})  "
              f"sim {1 - r['distance']:.3f}{flag}")
        if r.get("via"):
            print(f"      (franchise start; the closest title to your taste was: {r['via']})")
        print("      because you liked: " + "; ".join(f"{t} ({s:.2f})" for t, s in r["because"]))


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("platform", choices=["anilist", "mal"])
    ap.add_argument("username")
    ap.add_argument("--model", choices=list(MODELS), default="minilm")
    ap.add_argument("--variant", choices=list(VARIANTS), default="base")
    ap.add_argument("-k", type=int, default=10)
    ap.add_argument("--max-episodes", type=int)
    ap.add_argument("--min-year", type=int)
    ap.add_argument("--min-score", type=int, help="minimum community score, 0-100")
    ap.add_argument("--alpha", type=float, default=0.3, help="weight of the dislike vector (0 = ignore dislikes)")
    ap.add_argument("--unrated-share", type=float, default=0.35,
                    help="largest share of the liked weight that unrated titles may carry")
    ap.add_argument("--hide-planned", action="store_true")
    ap.add_argument("--refresh", action="store_true", help="ignore the cached watchlist")
    args = ap.parse_args()

    try:
        entries = parsers.get_profile(args.platform, args.username, refresh=args.refresh)
    except parsers.UserNotFound:
        sys.exit(f"User not found on {args.platform}: \"{args.username}\". Check the spelling.")
    except parsers.PrivateList as e:
        sys.exit(f"Private list: {e}")
    except parsers.ProfileError as e:
        sys.exit(f"Could not read the list: {e}")

    index = parsers.load_index()
    matched, unmatched = parsers.match_to_index(entries, index)
    print(f"{len(entries)} entries, {len(matched)} in our index, {len(unmatched)} not (ignored).")
    episodes = {a: r["episodes"] for a, r in index["by_anilist"].items()}
    starts = entry_points(index["by_anilist"].values())

    import chromadb  # imported late so --help and the tests need no database
    col = chromadb.PersistentClient(path=CHROMA_DIR).get_collection(collection_name(args.model, args.variant))
    where = build_where(args.max_episodes, args.min_year, args.min_score)
    model, recs = recommend(matched, col, episodes, args.k, where, args.alpha, args.unrated_share,
                            args.hide_planned, starts,
                            dict(max_episodes=args.max_episodes, min_year=args.min_year, min_score=args.min_score))
    if model is None:
        sys.exit("Nothing positive to learn from: rate a few titles 8 or higher, or finish some, and try again.")
    show(model, recs)


if __name__ == "__main__":
    main()
