"""Leave-one-out evaluation of the recommenders against baselines.

Run from the project root:
    python src/evaluate.py --methods popularity genre minilm
    python src/evaluate.py --users-dir data/test_users --tail 1000 --methods popularity minilm@center=1@knn=10

For each saved user, hide a whole franchise they rated 8 or higher, build the recommendation from
the rest, and record the rank of the hidden franchise among the unseen ones. Averages are per user,
intervals come from resampling users. Dense methods use exact cosine over the stored vectors.
Method options (@alpha, @share, @center, @knn, @pop, @genre) and the protocol are described in
DECISIONS.md.
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

import parsers
import taste
from scoring import (Catalog, centered, dense_matrix, genre_matrix, head_franchises,  # noqa: F401
                     make_scorer, matrix_loader, parse_method, zscore)

USERS_DIR = Path("data/eval_users")
RESULTS_DIR = Path("results")


# ---------------------------------------------------------------- catalog


# ---------------------------------------------------------------- users

class User:
    def __init__(self, uid, matched):
        self.id = uid
        self.matched = matched   # entries in our index, with franchise_id and title


def load_users(index, users_dir=USERS_DIR):
    users = []
    for path in sorted(Path(users_dir).glob("u[0-9][0-9][0-9].json")):
        data = json.loads(path.read_text())
        matched, _ = parsers.match_to_index(data["entries"], index)
        users.append(User(data["id"], matched))
    return users


def liked_franchises(user, threshold=8.0):
    """Franchises where the user rated some title `threshold` or higher."""
    return sorted({e["franchise_id"] for e in user.matched if e["score"] is not None and e["score"] >= threshold})


def pick_held_out(user, per_user, seed, allowed=None):
    liked = liked_franchises(user)
    if allowed is not None:
        liked = [f for f in liked if f in allowed]
    rng = np.random.default_rng([seed, int(user.id[1:])])
    if len(liked) <= per_user:
        return liked
    return sorted(rng.choice(liked, size=per_user, replace=False).tolist())


# ---------------------------------------------------------------- methods


def rank_of(scores, held_pos, candidate_mask):
    """1-based rank of one franchise among the candidates. Ties are broken by franchise position,
    the same order a stable sort would give, so constant scorers get no free luck."""
    s = scores[candidate_mask]
    idx = np.flatnonzero(candidate_mask)
    h = scores[held_pos]
    return int(1 + np.sum(s > h) + np.sum((s == h) & (idx < held_pos)))


def evaluate(users, catalog, scorers, per_user=20, seed=0, log=print, tail=0):
    """ranks[method][user_id] = list of ranks of the hidden franchises.

    With tail=N the N most popular franchises are neither candidates nor hidden targets.
    """
    ranks = {m: {} for m in scorers}
    fallbacks = {m: 0 for m in scorers}
    head = head_franchises(catalog, tail)
    allowed = None if not head else set(catalog.franchise_ids.tolist()) - head
    for u in users:
        held = pick_held_out(u, per_user, seed, allowed)
        if not held:
            log(f"  {u.id}: no liked franchises outside the head, skipped")
            continue
        for fid in held:
            visible = [e for e in u.matched if e["franchise_id"] != fid]
            seen = taste.seen_franchises(visible)
            mask = ~np.isin(catalog.franchise_ids, list(seen | head))
            held_pos = catalog.franchise_pos[fid]
            for m, scorer in scorers.items():
                rng = np.random.default_rng([seed, int(u.id[1:]), int(fid)])
                s = scorer(visible, rng)
                if s is None:
                    fallbacks[m] += 1
                    s = catalog.popularity
                ranks[m].setdefault(u.id, []).append(rank_of(s, held_pos, mask))
        log(f"  {u.id}: {len(held)} hidden franchises")
    return ranks, fallbacks


# ---------------------------------------------------------------- metrics

def per_user_metric(ranks_by_user, fn):
    return np.array([np.mean([fn(r) for r in rs]) for rs in ranks_by_user.values()])


def recall_at(k):
    return lambda r: 1.0 if r <= k else 0.0


def bootstrap_ci(values, n=2000, seed=0, level=0.95):
    """Mean and confidence interval from resampling users with replacement."""
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(n, len(values)))].mean(axis=1)
    lo, hi = np.percentile(means, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return float(values.mean()), float(lo), float(hi)


def summarize(ranks, ks=(10, 50), seed=0):
    out = {}
    for m, by_user in ranks.items():
        row = {}
        for k in ks:
            row[f"recall@{k}"] = bootstrap_ci(per_user_metric(by_user, recall_at(k)), seed=seed)
        row["mrr"] = bootstrap_ci(per_user_metric(by_user, lambda r: 1.0 / r), seed=seed)
        all_ranks = [r for rs in by_user.values() for r in rs]
        row["median_rank"] = float(np.median(all_ranks))
        row["trials"] = len(all_ranks)
        out[m] = row
    return out


def paired_difference(ranks, a, b, k=10, seed=0):
    """Recall@k of method a minus method b, per user, with a bootstrap interval over users."""
    ua, ub = ranks[a], ranks[b]
    users = sorted(ua)
    diff = np.array([np.mean([r <= k for r in ua[u]]) - np.mean([r <= k for r in ub[u]]) for u in users])
    return bootstrap_ci(diff, seed=seed)


def print_table(summary, ks=(10, 50)):
    w = max(26, max(len(m) for m in summary) + 2)
    head = f"{'method':<{w}}" + "".join(f"{f'Recall@{k} [95% CI]':>26}" for k in ks) + f"{'MRR':>8}{'median rank':>13}"
    print("\n" + head)
    for m, row in summary.items():
        cells = ""
        for k in ks:
            mean, lo, hi = row[f"recall@{k}"]
            cells += f"{f'{mean:.3f} [{lo:.3f}-{hi:.3f}]':>26}"
        print(f"{m:<{w}}{cells}{row['mrr'][0]:>8.4f}{row['median_rank']:>13.0f}")


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", default=["random", "popularity", "genre", "minilm"])
    ap.add_argument("--per-user", type=int, default=20, help="hidden franchises per user (caps heavy users)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--compare-to", nargs="+", default=["popularity"],
                    help="method(s) for the paired comparison; one table per method")
    ap.add_argument("--users-dir", default=str(USERS_DIR),
                    help="users to evaluate (default data/eval_users, the dev set; data/test_users is the final test)")
    ap.add_argument("--tail", type=int, default=0,
                    help="remove the N most popular franchises from the candidates (see the docstring)")
    ap.add_argument("--save", help="write the summary to results/<name>.json")
    args = ap.parse_args()

    index = parsers.load_index()
    catalog = Catalog(index["by_anilist"].values())
    users = load_users(index, args.users_dir)
    if not users:
        sys.exit(f"No users in {args.users_dir}. Run: python src/sample_users.py --target 25 --out {args.users_dir}")
    print(f"{len(users)} users, {catalog.n_franchises} franchises in the index, up to {args.per_user} hidden per user.")

    matrices = matrix_loader(catalog)
    scorers = {spec: make_scorer(spec, catalog, matrices) for spec in args.methods}
    started = time.time()
    if args.tail:
        print(f"TAIL VIEW: the {args.tail} most popular franchises are excluded for every method.")
    ranks, fallbacks = evaluate(users, catalog, scorers, args.per_user, args.seed, tail=args.tail)
    summary = summarize(ranks, seed=args.seed)
    print_table(summary)
    if any(fallbacks.values()):
        print("\nTrials with nothing positive to learn from (fell back to popularity):",
              {m: n for m, n in fallbacks.items() if n})
    width = max(26, max(len(m) for m in ranks) + 2)
    for ref in args.compare_to:
        if ref not in ranks:
            print(f"\n(--compare-to {ref}: not one of the methods run, skipped)")
            continue
        print(f"\nRecall@10 minus {ref}, paired by user (95% CI over users):")
        for m in ranks:
            if m != ref:
                d, lo, hi = paired_difference(ranks, m, ref, seed=args.seed)
                verdict = "clearly better" if lo > 0 else "clearly worse" if hi < 0 else "not clearly different"
                print(f"  {m:<{width}}{d:+.3f} [{lo:+.3f}, {hi:+.3f}]  {verdict}")
    print(f"\n{sum(s['trials'] for s in summary.values()) // len(summary)} hidden franchises per method, "
          f"{time.time() - started:.0f}s.")
    if args.save:
        RESULTS_DIR.mkdir(exist_ok=True)
        path = RESULTS_DIR / f"{args.save}.json"
        path.write_text(json.dumps({"users": len(users), "per_user": args.per_user, "seed": args.seed, "tail": args.tail, "users_dir": args.users_dir,
                                    "summary": summary}, indent=1))
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
