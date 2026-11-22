"""Embed every title and store it in ChromaDB, one collection per (model, variant).

Run from the project root, after franchise.py has written data/anime.json:
    python src/build_index.py                          # base variant, both models
    python src/build_index.py --model bge              # one model
    python src/build_index.py --variant notitle        # embed without the title
    python src/build_index.py --variant spoilers       # add spoiler tags (retrieval only)
    python src/build_index.py --limit 200              # quick test on the first 200 titles

Why variants: title words leaked into the search results and spoiler-flagged tags (often the best
signal, like "Time Loop") were missing. Each variant is one row in the ablation table later.
"""
import argparse
import json
import time
from pathlib import Path

ANIME_PATH = Path("data/anime.json")
CHROMA_DIR = "data/chroma"

# doc_prefix / query_prefix: some models expect a fixed prefix on queries (BGE) or on both (E5).
MODELS = {
    "minilm": {
        "hf_name": "sentence-transformers/all-MiniLM-L6-v2",
        "doc_prefix": "",
        "query_prefix": "",
    },
    "bge": {
        "hf_name": "BAAI/bge-small-en-v1.5",
        "doc_prefix": "",
        "query_prefix": "Represent this sentence for searching relevant passages: ",
    },
}

# What goes into the text we embed. "base" is the original behaviour.
VARIANTS = {
    "base": {"title": True, "spoilers": False},
    "notitle": {"title": False, "spoilers": False},
    "spoilers": {"title": True, "spoilers": True},
    "notitle_spoilers": {"title": False, "spoilers": True},
}

# Tags that describe the cast or format rather than the story. They add noise to embeddings
# ("Primarily Teen Cast" matches half the catalogue), so they are left out of the embedding
# text only. The tags in anime.json are untouched.
STOP_TAGS = {
    "Primarily Teen Cast", "Primarily Adult Cast", "Primarily Child Cast",
    "Primarily Male Cast", "Primarily Female Cast",
    "Male Protagonist", "Female Protagonist", "Ensemble Cast",
    "Anime Influenced", "Heterosexual", "Cute Boys Doing Cute Things",
}
MAX_TAGS = 10
MAX_SPOILER_TAGS = 8


def collection_name(model_key: str, variant: str = "base") -> str:
    """The base variant keeps the original name, so existing collections still work."""
    return f"anime_{model_key}" if variant == "base" else f"anime_{model_key}_{variant}"


def build_text(rec: dict, title: bool = True, spoilers: bool = False) -> str:
    """The text we embed for one title.

    Title, genres and themes come first and the synopsis last, so if a model truncates a long
    input it cuts the end of the synopsis, never the part that carries the genre and themes.

    With spoilers=True the spoiler-flagged tags are added as "Plot elements". That text is for
    retrieval only: it must never be shown to a user or sent to the LLM that writes pitches.
    """
    tags = [t for t in rec["tags"] if t not in STOP_TAGS][:MAX_TAGS]
    parts = [rec["title"] + "."] if title else []
    if rec["genres"]:
        parts.append("Genres: " + ", ".join(rec["genres"]) + ".")
    if tags:
        parts.append("Themes: " + ", ".join(tags) + ".")
    if spoilers:
        hidden = [t for t in rec.get("spoiler_tags", []) if t not in STOP_TAGS][:MAX_SPOILER_TAGS]
        if hidden:
            parts.append("Plot elements: " + ", ".join(hidden) + ".")
    parts.append(rec["synopsis"])
    return " ".join(parts)


def to_metadata(rec: dict) -> dict:
    """Metadata stored next to each vector.

    ChromaDB accepts only str, int, float and bool, and rejects None, so missing numbers become -1
    and missing text becomes "". Filters rely on this: "episodes >= 1" excludes unknown episode
    counts, which is the rule in DECISIONS.md.
    """
    return {
        "anilist_id": rec["anilist_id"],
        "mal_id": rec["mal_id"] if rec["mal_id"] is not None else -1,
        "title": rec["title"],
        "franchise_id": rec["franchise_id"],
        "format": rec["format"] or "",
        "episodes": rec["episodes"] if rec["episodes"] is not None else -1,
        "year": rec["year"] if rec["year"] is not None else -1,
        "score": rec["score"] if rec["score"] is not None else -1,
        "popularity": rec["popularity"] or 0,
        "genres": "|".join(rec["genres"]),
        "cover": rec["cover"] or "",
    }


def build(model_key: str, records: list, variant: str = "base", batch: int = 64) -> None:
    # Imported here so the helpers above can be tested without the heavy packages installed.
    import chromadb
    from sentence_transformers import SentenceTransformer

    cfg = MODELS[model_key]
    name = collection_name(model_key, variant)
    print(f"\n=== {model_key} / {variant}: {cfg['hf_name']} -> '{name}' ===")
    model = SentenceTransformer(cfg["hf_name"])
    opts = VARIANTS[variant]
    texts = [cfg["doc_prefix"] + build_text(r, **opts) for r in records]

    # Evidence for the embedding comparison: how many texts exceed the model's input limit?
    limit = model.max_seq_length
    lengths = [len(model.tokenizer(t)["input_ids"]) for t in texts]
    over = sum(1 for n in lengths if n > limit)
    print(f"Max input length: {limit} tokens. Texts over the limit: {over} "
          f"({100 * over / len(texts):.1f}%). Longest: {max(lengths)} tokens.")

    client = chromadb.PersistentClient(path=CHROMA_DIR)
    try:
        client.delete_collection(name)  # always rebuild from scratch so it matches anime.json
    except Exception:
        pass  # the collection did not exist yet (the exception type differs between versions)
    col = client.create_collection(name, metadata={"hnsw:space": "cosine"})

    start = time.time()
    for i in range(0, len(records), batch):
        chunk = records[i:i + batch]
        emb = model.encode(texts[i:i + batch], normalize_embeddings=True, show_progress_bar=False)
        col.add(
            ids=[str(r["anilist_id"]) for r in chunk],
            embeddings=emb.tolist(),
            documents=texts[i:i + batch],
            metadatas=[to_metadata(r) for r in chunk],
        )
        done = min(i + batch, len(records))
        if (i // batch) % 10 == 0 or done == len(records):
            print(f"  {done}/{len(records)} embedded ({time.time() - start:.0f}s)")
    print(f"Stored {col.count()} titles in collection '{name}'.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODELS) + ["all"], default="all")
    ap.add_argument("--variant", choices=list(VARIANTS), default="base")
    ap.add_argument("--limit", type=int, default=0, help="only index the first N titles (for testing)")
    args = ap.parse_args()

    records = json.loads(ANIME_PATH.read_text(encoding="utf-8"))
    if args.limit:
        records = records[:args.limit]
    print(f"Loaded {len(records)} titles from {ANIME_PATH}")

    for key in (MODELS if args.model == "all" else [args.model]):
        build(key, records, args.variant)


if __name__ == "__main__":
    main()
