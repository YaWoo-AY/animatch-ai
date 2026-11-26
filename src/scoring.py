"""Scoring shared by the live recommender and the evaluation, so both run the same code.

A method is a string such as `minilm@center=1@knn=10@genre=1@pop=0.5` (options are described in
DECISIONS.md). `make_scorer` turns it into a function that gives every franchise one score from
a user's list. The chosen configuration is DEFAULT_SPEC.
"""
from pathlib import Path

import numpy as np

import taste

# Chosen on the dev users, then confirmed once on two separate sets of new users (see DECISIONS.md).
DEFAULT_SPEC = "minilm@center=1@knn=10@genre=1@pop=0.5@qual=2"


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
        # community score of the best-scored title in each franchise; unknown scores get the median
        # so a franchise with no score is neither rewarded nor punished
        known = np.full(self.n_franchises, np.nan)
        for r, p in zip(self.records, self.row_franchise):
            sc = r.get("score")
            if sc is not None and sc >= 0:
                known[p] = sc if np.isnan(known[p]) else max(known[p], sc)
        median = np.nanmedian(known) if not np.all(np.isnan(known)) else 0.0
        self.quality = np.where(np.isnan(known), median, known)

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
        if key not in ("alpha", "share", "center", "knn", "pop", "genre", "qual") or not val:
            raise SystemExit(f"Unknown option '{o}' in '{spec}'. Options: alpha, share, center, knn, pop, genre, qual (e.g. @knn=5).")
        options[key] = float(val)
    name, _, variant = head.partition("/")
    return name, variant or "base", options


def title_matrix(spec, matrices):
    """The title-by-dimension matrix a method scores with (centered when the spec asks for it)."""
    name, variant, opts = parse_method(spec)
    matrix = matrices(name, variant)
    return centered(matrix) if opts.get("center") else matrix


def runtime_matrix_loader(catalog, directory):
    """Like matrix_loader, but the stored vectors come from `<name>_<variant>.npz` files written by
    export_runtime.py, so the deployed page needs neither ChromaDB nor an embedding model."""
    cache = {}

    def matrices(name, variant):
        key = (name, variant)
        if key not in cache:
            if name == "genre":
                cache[key] = genre_matrix(catalog.records)
            else:
                path = Path(directory) / f"{name}_{variant}.npz"
                if not path.exists():
                    raise FileNotFoundError(f"{path} is missing. Run: python src/export_runtime.py")
                z = np.load(path)
                row = {int(i): j for j, i in enumerate(z["ids"])}
                missing = [r["anilist_id"] for r in catalog.records if r["anilist_id"] not in row]
                if missing:
                    raise ValueError(f"{len(missing)} titles have no vector in {path}. Run export_runtime.py again.")
                cache[key] = z["vectors"][[row[r["anilist_id"]] for r in catalog.records]].astype(float)
        return cache[key]
    return matrices


def matrix_loader(catalog):
    """matrices(name, variant) with a cache: tag vectors for 'genre', stored embeddings otherwise."""
    cache = {}

    def matrices(name, variant):
        key = (name, variant)
        if key not in cache:
            cache[key] = genre_matrix(catalog.records) if name == "genre" else dense_matrix(catalog, name, variant)
        return cache[key]
    return matrices


def make_scorer(spec, catalog, matrices):
    """Returns f(visible_entries, rng) -> one score per franchise (higher = recommend first)."""
    name, variant, opts = parse_method(spec)
    if name == "random":
        return lambda visible, rng: rng.random(catalog.n_franchises)
    if name == "popularity":
        return lambda visible, rng: catalog.popularity
    if name == "quality":                 # community score alone, ignoring the user: the baseline for @qual
        return lambda visible, rng: catalog.quality
    alpha, share = opts.get("alpha", 0.3), opts.get("share", 0.35)
    knn, pop_weight, genre_weight = int(opts.get("knn", 0)), opts.get("pop", 0.0), opts.get("genre", 0.0)
    qual_weight = opts.get("qual", 0.0)
    matrix = title_matrix(spec, matrices)
    tags = matrices("genre", "base") if genre_weight and name != "genre" else None
    log_pop = zscore(np.log1p(catalog.popularity))
    quality = zscore(catalog.quality)

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
        if pop_weight or qual_weight:
            s = zscore(s) + pop_weight * log_pop + qual_weight * quality
        return s
    return score


def head_franchises(catalog, n):
    """Ids of the n most popular franchises (empty when n is 0)."""
    if not n:
        return set()
    order = np.argsort(-catalog.popularity, kind="stable")[:n]
    return {int(f) for f in catalog.franchise_ids[order]}
