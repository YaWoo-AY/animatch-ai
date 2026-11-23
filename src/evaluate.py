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

USERS_DIR = Path("data/eval_users")
RESULTS_DIR = Path("results")


# ---------------------------------------------------------------- catalog

class Catalog:
    """The searchable index as arrays: one row per title, one franchise id per row."""

    def __init__(self, records):
        self.records = list(records)
        self.aid_row = {r["anilist_id"]: i for i, r in enumerate(self.records)}
        self.franchise_ids = np.array(sorted({r["franchise_id"] for r in self.records}))
        pos = {f: i for i, f in enumerate(self.franchise_ids)}
        self.row_franchise = np.array([pos[r["franchise_id"]] for r in self.records])
        self.franchise_pos = pos
        self.n_franchises = len(self.franchise_ids)
        self.episodes = {r["anilist_id"]: r.get("episodes") for r in self.records}
        pop = np.zeros(self.n_franchises)
        for r, p in zip(self.records, self.row_franchise):
            pop[p] = max(pop[p], r.get("popularity") or 0)
        self.popularity = pop

    def franchise_max(self, title_scores):
        out = np.full(self.n_franchises, -np.inf)
        np.maximum.at(out, self.row_franchise, title_scores)
        return out


def genre_matrix(records):
    """Genres + non-spoiler tags as IDF-weighted, unit-length rows (the 'plain tags' representation)."""
    vocab = {}
    for r in records:
        for t in list(r["genres"]) + list(r["tags"]):
            vocab.setdefault(t, len(vocab))
    m = np.zeros((len(records), max(1, len(vocab))))
    for i, r in enumerate(records):
        for t in set(list(r["genres"]) + list(r["tags"])):
            m[i, vocab[t]] = 1.0
    df = m.sum(axis=0)
    m *= np.log((1 + len(records)) / (1 + df)) + 1.0
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    return m / np.where(norms == 0, 1, norms)


def dense_matrix(catalog, model, variant):
    """Stored embeddings, in catalog row order (needs ChromaDB and a built index)."""
    import chromadb
    from build_index import CHROMA_DIR, collection_name
    col = chromadb.PersistentClient(path=CHROMA_DIR).get_collection(collection_name(model, variant))
    ids = [str(r["anilist_id"]) for r in catalog.records]
    vecs = {}
    for i in range(0, len(ids), 500):
        res = col.get(ids=ids[i:i + 500], include=["embeddings"])
        for rid, emb in zip(res["ids"], res["embeddings"]):
            vecs[rid] = np.asarray(emb, dtype=float)
    missing = [i for i in ids if i not in vecs]
    if missing:
        raise SystemExit(f"{len(missing)} titles are missing from collection '{collection_name(model, variant)}'. "
                         f"Rebuild it: python src/build_index.py --model {model} --variant {variant}")
    return np.vstack([vecs[i] for i in ids])


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

def centered(matrix):
    """Subtract the average vector, then make every row unit length again."""
    m = matrix - matrix.mean(axis=0)
    n = np.linalg.norm(m, axis=1, keepdims=True)
    return m / np.where(n == 0, 1, n)


def zscore(x):
    sd = x.std()
    return (x - x.mean()) / (sd if sd > 0 else 1.0)


def parse_method(spec):
    """'minilm/notitle@alpha=0.0@share=1' -> (kind, variant, options)."""
    head, *opts = spec.split("@")
    options = {}
    for o in opts:
        key, _, val = o.partition("=")
        if key not in ("alpha", "share", "center", "knn", "pop", "genre") or not val:
            raise SystemExit(f"Unknown option '{o}' in '{spec}'. Options: alpha, share, center, knn, pop, genre (e.g. @knn=5).")
        options[key] = float(val)
    name, _, variant = head.partition("/")
    return name, variant or "base", options


def make_scorer(spec, catalog, matrices):
    """Returns f(visible_entries, rng) -> one score per franchise (higher = recommend first)."""
    name, variant, opts = parse_method(spec)
    if name == "random":
        return lambda visible, rng: rng.random(catalog.n_franchises)
    if name == "popularity":
        return lambda visible, rng: catalog.popularity
    alpha, share = opts.get("alpha", 0.3), opts.get("share", 0.35)
    knn, pop_weight, genre_weight = int(opts.get("knn", 0)), opts.get("pop", 0.0), opts.get("genre", 0.0)
    matrix = matrices(name, variant)
    if opts.get("center"):
        matrix = centered(matrix)
    tags = matrices("genre", "base") if genre_weight and name != "genre" else None
    log_pop = zscore(np.log1p(catalog.popularity))

    def liked_scores(mat, visible):
        """One score per franchise from one representation, or None when nothing positive is left."""
        emb = {e["anilist_id"]: mat[catalog.aid_row[e["anilist_id"]]] for e in visible}
        t = taste.build_taste(visible, emb, catalog.episodes, share, alpha)
        if t is None:
            return None
        if knn > 0:
            V, w = taste.liked_vectors(t)
            sims = mat @ V.T                                     # titles x liked franchises
            k = min(knn, V.shape[0])
            near = np.argpartition(-sims, k - 1, axis=1)[:, :k]  # the k most similar liked franchises
            w_near = w[near]
            title_scores = (np.take_along_axis(sims, near, axis=1) * w_near).sum(axis=1) / w_near.sum(axis=1)
        else:
            title_scores = mat @ t.vector
        return catalog.franchise_max(title_scores)

    def score(visible, rng):
        s = liked_scores(matrix, visible)
        if s is None:
            return None            # nothing positive left to learn from
        if tags is not None:
            g = liked_scores(tags, visible)
            if g is not None:
                s = zscore(s) + genre_weight * zscore(g)
        if pop_weight:
            s = zscore(s) + pop_weight * log_pop
        return s
    return score


def rank_of(scores, held_pos, candidate_mask):
    """1-based rank of one franchise among the candidates. Ties are broken by franchise position,
    the same order a stable sort would give, so constant scorers get no free luck."""
    s = scores[candidate_mask]
    idx = np.flatnonzero(candidate_mask)
    h = scores[held_pos]
    return int(1 + np.sum(s > h) + np.sum((s == h) & (idx < held_pos)))


def head_franchises(catalog, n):
    """Ids of the n most popular franchises (empty when n is 0)."""
    if not n:
        return set()
    order = np.argsort(-catalog.popularity, kind="stable")[:n]
    return {int(f) for f in catalog.franchise_ids[order]}


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
    head = f"{'method':<26}" + "".join(f"{f'Recall@{k} [95% CI]':>26}" for k in ks) + f"{'MRR':>8}{'median rank':>13}"
    print("\n" + head)
    for m, row in summary.items():
        cells = ""
        for k in ks:
            mean, lo, hi = row[f"recall@{k}"]
            cells += f"{f'{mean:.3f} [{lo:.3f}-{hi:.3f}]':>26}"
        print(f"{m:<26}{cells}{row['mrr'][0]:>8.4f}{row['median_rank']:>13.0f}")


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", default=["random", "popularity", "genre", "minilm"])
    ap.add_argument("--per-user", type=int, default=20, help="hidden franchises per user (caps heavy users)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--compare-to", default="popularity", help="method for the paired comparison")
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

    cache = {}

    def matrices(name, variant):
        key = (name, variant)
        if key not in cache:
            cache[key] = genre_matrix(catalog.records) if name == "genre" else dense_matrix(catalog, name, variant)
        return cache[key]

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
    if args.compare_to in ranks:
        print(f"\nRecall@10 minus {args.compare_to}, paired by user (95% CI over users):")
        for m in ranks:
            if m != args.compare_to:
                d, lo, hi = paired_difference(ranks, m, args.compare_to, seed=args.seed)
                verdict = "clearly better" if lo > 0 else "clearly worse" if hi < 0 else "not clearly different"
                print(f"  {m:<26}{d:+.3f} [{lo:+.3f}, {hi:+.3f}]  {verdict}")
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
