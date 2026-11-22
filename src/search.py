"""Day 4: type a query, see the closest titles. This is your first look at whether retrieval works.

Run from the project root, after build_index.py:
    python src/search.py "slow-burn psychological mystery"
    python src/search.py "cozy slice of life with food" --model bge -k 5
    python src/search.py "time loop thriller" --max-episodes 26 --min-year 2010
    python src/search.py "heist with a found family" --model all     # both models, with overlap
"""
import argparse
import sys

from build_index import CHROMA_DIR, MODELS, collection_name

_models = {}  # loaded models, so --model all does not reload them


def build_where(max_episodes=None, min_year=None, min_score=None):
    """Turn the filter options into a ChromaDB `where` clause (None means no filter).

    An episode cap also requires episodes >= 1: unknown episode counts are stored as -1, and
    the rule in DECISIONS.md is that titles with unknown episodes are excluded under a cap.
    """
    conds = []
    if max_episodes is not None:
        conds.append({"episodes": {"$gte": 1}})
        conds.append({"episodes": {"$lte": max_episodes}})
    if min_year is not None:
        conds.append({"year": {"$gte": min_year}})
    if min_score is not None:
        conds.append({"score": {"$gte": min_score}})
    if not conds:
        return None
    return conds[0] if len(conds) == 1 else {"$and": conds}


def search(query: str, model_key: str, k: int = 10, where=None) -> list:
    """Return up to k results as dicts: title, year, format, episodes, score, distance."""
    import chromadb
    from sentence_transformers import SentenceTransformer

    cfg = MODELS[model_key]
    if model_key not in _models:
        _models[model_key] = SentenceTransformer(cfg["hf_name"])
    col = chromadb.PersistentClient(path=CHROMA_DIR).get_collection(collection_name(model_key))
    emb = _models[model_key].encode([cfg["query_prefix"] + query], normalize_embeddings=True)
    res = col.query(query_embeddings=emb.tolist(), n_results=k, where=where)
    out = []
    for meta, dist in zip(res["metadatas"][0], res["distances"][0]):
        out.append({**meta, "distance": dist})
    return out


def show(results: list) -> None:
    for i, r in enumerate(results, 1):
        eps = r["episodes"] if r["episodes"] > 0 else "?"
        score = r["score"] if r["score"] >= 0 else "-"
        year = r["year"] if r["year"] > 0 else "?"
        print(f"{i:2d}. {r['title']}  ({year}, {r['format']}, {eps} eps, score {score})  "
              f"dist {r['distance']:.3f}")


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # titles contain accents and symbols
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("--model", choices=list(MODELS) + ["all"], default="minilm")
    ap.add_argument("-k", type=int, default=10)
    ap.add_argument("--max-episodes", type=int)
    ap.add_argument("--min-year", type=int)
    ap.add_argument("--min-score", type=int)
    args = ap.parse_args()

    where = build_where(args.max_episodes, args.min_year, args.min_score)
    keys = list(MODELS) if args.model == "all" else [args.model]
    found = {}
    for key in keys:
        print(f"\n--- {key} ---")
        found[key] = search(args.query, key, args.k, where)
        show(found[key])
    if len(keys) == 2:
        a, b = ({r["anilist_id"] for r in found[k]} for k in keys)
        print(f"\nTitles in both top-{args.k}: {len(a & b)}")


if __name__ == "__main__":
    main()
