"""Look at one title: the exact text we embedded, and its nearest neighbours by vector.

Run from the project root, in the terminal where the prompt starts with (.venv):
    python src/inspect_title.py "Summer Time Rendering"
    python src/inspect_title.py "Summer Time Rendering" --model minilm -k 10

The title must match exactly as stored (capitals and punctuation included). If it does not,
the script lists titles that contain your text.
"""
import argparse
import sys

import chromadb

from build_index import CHROMA_DIR, MODELS, VARIANTS, collection_name


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # titles contain accents and symbols
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("title")
    ap.add_argument("--model", choices=list(MODELS), default="bge")
    ap.add_argument("--variant", choices=list(VARIANTS), default="base")
    ap.add_argument("-k", type=int, default=10)
    args = ap.parse_args()

    col = chromadb.PersistentClient(path=CHROMA_DIR).get_collection(
        collection_name(args.model, args.variant))
    seed = col.get(where={"title": args.title}, include=["documents", "embeddings"])
    if len(seed["ids"]) == 0:
        close = col.get(where_document={"$contains": args.title}, limit=8, include=["metadatas"])
        print(f"No title named exactly '{args.title}'.")
        if close["metadatas"]:
            print("Titles containing that text:")
            for m in close["metadatas"]:
                print("  ", m["title"])
        return

    print(f"--- Embedded text ({args.model}) ---")
    print(seed["documents"][0][:500])

    vec = [float(x) for x in seed["embeddings"][0]]
    res = col.query(query_embeddings=[vec], n_results=args.k + 1)
    print(f"\n--- Nearest {args.k} to '{args.title}' ({args.model}) ---")
    shown = 0
    for id_, meta, dist in zip(res["ids"][0], res["metadatas"][0], res["distances"][0]):
        if id_ == seed["ids"][0]:
            continue  # the title itself
        eps = meta["episodes"] if meta["episodes"] > 0 else "?"
        year = meta["year"] if meta["year"] > 0 else "?"
        print(f"{shown + 1:2d}. {meta['title']}  ({year}, {meta['format']}, {eps} eps)  dist {dist:.3f}")
        shown += 1
        if shown == args.k:
            break


if __name__ == "__main__":
    main()
