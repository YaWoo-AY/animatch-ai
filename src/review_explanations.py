"""Export model-written sentences to a CSV for hand labelling, then summarize the labels.

    python src/review_explanations.py make --users-dir data/eval_users --users 3 -k 10
    (open data/review/explanations.csv, fill the label column with A, B or C)
    python src/review_explanations.py summarize --save explanation_review

Labels: A = useful and true, B = true but generic, C = misleading or overstated.
The CSV holds titles from sampled users' lists, so it lives under data/, which is never committed.
Only counts are saved to results/.
"""
import argparse
import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path

import evaluate
import explain
import parsers
import recommend
import scoring

RUNTIME = Path("app_data")
DEFAULT_CSV = Path("data/review/explanations.csv")
LABELS = ("A", "B", "C")
WEAK = 0.25
COLUMNS = ["user", "pick", "liked_titles", "shared_attributes", "top_similarity", "source", "sentence", "label", "note"]


def make_rows(uid, recs, results, facts, include_templates=False):
    rows = []
    for rec, res in zip(recs, results):
        if res["source"] != "model" and not include_templates:
            continue
        item = explain.item_facts(rec, facts)
        sims = [s for _, s in rec.get("because", [])]
        rows.append({
            "user": uid,
            "pick": item["title"],
            "liked_titles": "; ".join(lk["title"] for lk in item["liked"]),
            "shared_attributes": "; ".join(sorted({a for lk in item["liked"] for a in lk["shared"]})),
            "top_similarity": f"{max(sims):.2f}" if sims else "",
            "source": res["source"],
            "sentence": explain.render(res["text"]),
            "label": "",
            "note": "",
        })
    return rows


def write_csv(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)


def read_rows(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def summarize(rows, weak=WEAK):
    """Counts of labels overall, by source, and by whether the closest liked title was a weak match."""
    out = {"rows": len(rows), "unlabelled": 0, "invalid_labels": 0, "labelled": 0,
           "overall": Counter(), "by_source": {}, "by_similarity": {}}
    for r in rows:
        label = (r.get("label") or "").strip().upper()
        if not label:
            out["unlabelled"] += 1
            continue
        if label not in LABELS:
            out["invalid_labels"] += 1
            continue
        out["labelled"] += 1
        out["overall"][label] += 1
        out["by_source"].setdefault(r.get("source") or "?", Counter())[label] += 1
        try:
            bucket = f"below {weak}" if float(r["top_similarity"]) < weak else f"{weak} or above"
        except (TypeError, ValueError):
            bucket = "no similarity"
        out["by_similarity"].setdefault(bucket, Counter())[label] += 1
    out["overall"] = dict(out["overall"])
    out["by_source"] = {k: dict(v) for k, v in out["by_source"].items()}
    out["by_similarity"] = {k: dict(v) for k, v in out["by_similarity"].items()}
    return out


def share(counts, label):
    n = sum(counts.values())
    return round(counts.get(label, 0) / n, 3) if n else None


def cmd_make(args):
    config = explain.get_llm_config()
    if not config:
        sys.exit("No LLM_API_KEY set. Put it in the environment or in .env (never commit it).")
    spoilers = explain.load_spoiler_terms(RUNTIME / "spoiler_tags.json")
    if spoilers is None:
        sys.exit("app_data/spoiler_tags.json is missing. Run: python src/export_runtime.py")
    index = parsers.load_index(RUNTIME / "anime.json")
    catalog = scoring.Catalog(index["by_anilist"].values())
    matrices = scoring.runtime_matrix_loader(catalog, RUNTIME)
    facts = explain.Facts(catalog.records)
    starts, links = recommend.entry_points(catalog.records), recommend.retelling_links(catalog.records)

    if args.users_dir:
        lists = [(u.id, u.matched) for u in evaluate.load_users(index, args.users_dir)][:args.users]
    elif args.platform and args.username:
        matched, _ = parsers.match_to_index(parsers.get_profile(args.platform, args.username), index)
        lists = [("you", matched)]
    else:
        sys.exit("Give a platform and username, or --users-dir.")

    rows, made = [], 0
    for uid, matched in lists:
        model, recs = recommend.recommend(matched, catalog, matrices, k=args.k, starts=starts, retellings=links)
        if model is None:
            continue
        if made:
            time.sleep(args.pause)
        results, error = explain.explain_picks(recs, facts, spoilers, config, use_cache=False)
        made += 1
        if error:
            print(f"  {uid}: model not used ({error}); skipped")
            continue
        got = make_rows(uid, recs, results, facts, args.include_templates)
        rows += got
        print(f"  {uid}: {len(got)} sentences exported")
    write_csv(rows, args.out)
    print(f"Wrote {len(rows)} rows to {args.out}. Fill the label column with A, B or C.")


def cmd_summarize(args):
    s = summarize(read_rows(args.csv), args.weak)
    print(f"{s['labelled']} labelled of {s['rows']} rows ({s['unlabelled']} blank, {s['invalid_labels']} invalid)")
    print("overall:", s["overall"])
    for title, groups in (("by source", s["by_source"]), ("by closest-liked similarity", s["by_similarity"])):
        print(title)
        for name, counts in sorted(groups.items()):
            n = sum(counts.values())
            print(f"  {name:<14} n={n:<3} A={counts.get('A', 0)} B={counts.get('B', 0)} C={counts.get('C', 0)}"
                  f"  misleading={share(counts, 'C')}")
    if args.save:
        Path("results").mkdir(exist_ok=True)
        Path("results", f"{args.save}.json").write_text(json.dumps(s, indent=1))
        print(f"Saved results/{args.save}.json")


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    mk = sub.add_parser("make")
    mk.add_argument("platform", nargs="?", choices=["anilist", "mal"])
    mk.add_argument("username", nargs="?")
    mk.add_argument("--users-dir")
    mk.add_argument("--users", type=int, default=3)
    mk.add_argument("-k", type=int, default=10)
    mk.add_argument("--pause", type=float, default=60.0)
    mk.add_argument("--include-templates", action="store_true")
    mk.add_argument("--out", default=str(DEFAULT_CSV))
    sm = sub.add_parser("summarize")
    sm.add_argument("--csv", default=str(DEFAULT_CSV))
    sm.add_argument("--weak", type=float, default=WEAK)
    sm.add_argument("--save", help="write counts to results/<name>.json")
    args = ap.parse_args()
    (cmd_make if args.cmd == "make" else cmd_summarize)(args)


if __name__ == "__main__":
    main()
