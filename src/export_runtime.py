"""Write the small bundle the deployed page runs on, so it needs neither ChromaDB nor a model.

Run from the project root, after build_index.py:
    python src/export_runtime.py

Writes app_data/anime.json (the index without synopses or spoiler-flagged tags, which the page never
shows), app_data/spoiler_tags.json (the spoiler-flagged tags, used only to CHECK generated sentences,
never shown or sent to a model) and app_data/minilm_base.npz (the stored vectors). Commit app_data/;
keep data/ out of git.
"""
import json
import sys
from pathlib import Path

import numpy as np

import scoring

OUT_DIR = Path("app_data")
DROP = ("synopsis", "spoiler_tags")


def slim(records):
    return [{k: v for k, v in r.items() if k not in DROP} for r in records]


def write_bundle(records, matrix, out_dir, name="minilm", variant="base"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "anime.json").write_text(json.dumps(slim(records), ensure_ascii=False, separators=(",", ":")),
                                        encoding="utf-8")
    spoilers = {str(r["anilist_id"]): r["spoiler_tags"] for r in records if r.get("spoiler_tags")}
    (out_dir / "spoiler_tags.json").write_text(json.dumps(spoilers, ensure_ascii=False, separators=(",", ":")),
                                               encoding="utf-8")
    np.savez_compressed(out_dir / f"{name}_{variant}.npz",
                        ids=np.array([r["anilist_id"] for r in records], dtype=np.int64),
                        vectors=np.asarray(matrix, dtype=np.float32))


def main(model="minilm", variant="base"):
    records = json.loads(Path("data/anime.json").read_text(encoding="utf-8"))
    catalog = scoring.Catalog(records)
    matrix = scoring.dense_matrix(catalog, model, variant)       # exact vectors from ChromaDB
    write_bundle(catalog.records, matrix, OUT_DIR, model, variant)
    for f in sorted(OUT_DIR.iterdir()):
        print(f"{f}  {f.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    sys.exit(main())
