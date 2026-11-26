"""Recommendations from a real watchlist, using the configuration that was evaluated.

Run from the project root, in the terminal whose prompt starts with (.venv):
    python src/recommend.py anilist <username>
    python src/recommend.py mal <username> -k 15
    python src/recommend.py anilist <username> --max-episodes 26 --min-year 2015 --min-score 70
    python src/recommend.py anilist <username> --pop 0                # taste only, no popularity prior
    python src/recommend.py anilist <username> --skip-popular 300     # deeper cuts: drop the 300 most popular
    python src/recommend.py anilist <username> --hide-planned         # also hide plan-to-watch titles
    python src/recommend.py anilist <username> --keep-retellings      # do not hide retellings of seen titles

Every franchise gets one score from scoring.py (the same code the evaluation ran), franchises the
user has seen are removed, and the best ones are shown at their first entry. Stored vectors are
read from ChromaDB; no embedding model is loaded.
"""
import argparse
import sys

import numpy as np

import parsers
import scoring
import taste
from build_index import MODELS, VARIANTS, to_metadata

FORMAT_RANK = {"TV": 0, "MOVIE": 1, "ONA": 2, "OVA": 3}


def entry_points(records):
    """One title per franchise to actually recommend: where a newcomer should start.

    Prefer TV over movie over ONA over OVA, then the earliest year, then the most popular.
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
    """Checks one title against the user's limits. Unknown values are stored as -1."""
    if max_episodes is not None and not 1 <= meta["episodes"] <= max_episodes:
        return False
    if min_year is not None and meta["year"] < min_year:
        return False
    if min_score is not None and meta["score"] < min_score:
        return False
    return True


def retelling_links(records):
    """{franchise_id: franchise ids of its retellings}, from AniList ALTERNATIVE links (both directions).

    Used only for hiding: someone who has seen Horimiya should not be sent to the 2012 OVA of the
    same story. These links are never used to merge franchises, because ALTERNATIVE is also used
    for loosely related versions.
    """
    fid_of = {r["anilist_id"]: r["franchise_id"] for r in records}
    links = {}
    for r in records:
        for rel in r.get("relations", []):
            other = fid_of.get(rel["anilist_id"])
            if rel["media_type"] == "ANIME" and rel["type"] == "ALTERNATIVE" and other is not None \
                    and other != r["franchise_id"]:
                links.setdefault(r["franchise_id"], set()).add(other)
                links.setdefault(other, set()).add(r["franchise_id"])
    return links


def build_spec(model="minilm", variant="base", alpha=0.3, share=0.35, knn=10, genre=1.0, pop=0.5):
    return f"{model}/{variant}@alpha={alpha}@share={share}@center=1@knn={knn}@genre={genre}@pop={pop}"


def recommend(matched, catalog, matrices, spec=scoring.DEFAULT_SPEC, k=10, limits=None, starts=None,
              hide_planned=False, skip_popular=0, retellings=None):
    """Returns (taste, recommendations). taste is None when there is nothing positive to learn from.

    Filters (`limits` = max_episodes, min_year, min_score) are checked on each franchise's first
    entry, which is the title that gets recommended. `retellings` is the output of retelling_links();
    when given, retellings of anything the user has seen are hidden too.
    """
    starts = starts or entry_points(catalog.records)
    scores = scoring.make_scorer(spec, catalog, matrices)(matched, np.random.default_rng(0))
    if scores is None:
        return None, []

    hidden = taste.seen_franchises(matched)
    planned = taste.planned_franchises(matched, hidden)
    for fid in list(hidden):
        hidden = hidden | (retellings or {}).get(fid, set())
    if hide_planned:
        hidden = hidden | planned
    hidden = hidden | scoring.head_franchises(catalog, skip_popular)
    candidate = np.array([fid not in hidden and fid in starts and within_limits(starts[fid], **(limits or {}))
                          for fid in catalog.franchise_ids.tolist()])
    order = [i for i in np.argsort(-scores, kind="stable") if candidate[i]][:k]

    # The explanation uses the same vectors the score used (centered, if the spec says so).
    mat = scoring.title_matrix(spec, matrices)
    _, _, opts = scoring.parse_method(spec)
    vectors = {e["anilist_id"]: mat[catalog.aid_row[e["anilist_id"]]] for e in matched}
    model = taste.build_taste(matched, vectors, catalog.episodes, opts.get("share", 0.35), opts.get("alpha", 0.3))

    recs = []
    for i in order:
        fid = int(catalog.franchise_ids[i])
        meta = dict(starts[fid])
        near = taste.nearest_liked(model, mat[catalog.aid_row[meta["anilist_id"]]])
        meta["because"] = [(o.title, sim) for o, sim in near]
        meta["because_franchises"] = [o.franchise_id for o, _ in near]
        meta["planned"] = fid in planned
        recs.append(meta)
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
        print(f"{i:2d}. {r['title']}  ({year}, {r['format']}, {eps} eps, score {score}){flag}")
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
    ap.add_argument("--knn", type=int, default=10, help="score by the K nearest liked franchises (0 = average of all)")
    ap.add_argument("--genre", type=float, default=1.0, help="weight of the genre/tag score (0 = embeddings only)")
    ap.add_argument("--pop", type=float, default=0.5, help="weight of the popularity prior (0 = none)")
    ap.add_argument("--skip-popular", type=int, default=0, metavar="N",
                    help="leave out the N most popular franchises (deeper cuts)")
    ap.add_argument("--hide-planned", action="store_true")
    ap.add_argument("--keep-retellings", action="store_true",
                    help="do not hide alternative versions of titles you have seen")
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

    catalog = scoring.Catalog(index["by_anilist"].values())
    spec = build_spec(args.model, args.variant, args.alpha, args.unrated_share, args.knn, args.genre, args.pop)
    limits = dict(max_episodes=args.max_episodes, min_year=args.min_year, min_score=args.min_score)
    model, recs = recommend(matched, catalog, scoring.matrix_loader(catalog), spec, args.k, limits,
                            hide_planned=args.hide_planned, skip_popular=args.skip_popular,
                            retellings=None if args.keep_retellings else retelling_links(catalog.records))
    if model is None:
        sys.exit("Nothing positive to learn from: rate a few titles 8 or higher, or finish some, and try again.")
    show(model, recs)


if __name__ == "__main__":
    main()
