"""Measure how often the language model's sentences pass the grounding check.

Run from the project root, with LLM_API_KEY set (environment or .env):
    python src/eval_explanations.py anilist <username> -k 10
    python src/eval_explanations.py --users-dir data/eval_users -k 10 --save explanations

Uses the committed app_data bundle, one model call per user, and no cache. Free tiers limit tokens
per minute, so calls are spaced by --pause seconds. Only counts are saved, never sentences or titles.
"""
import argparse
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


def tally(batches):
    """batches: list of (results, error), one per model call.

    Call-level and sentence-level numbers are kept apart: a provider error or a rate limit says
    nothing about the quality of the sentences, so sentences are counted only for calls that worked.
    """
    ok = [results for results, error in batches if not error]
    errors = Counter(error for _, error in batches if error)
    picks = sum(len(r) for r in ok)
    passed = sum(1 for r in ok for x in r if x["source"] == "model")
    reasons = Counter(p.split(":")[0] for r in ok for x in r for p in x["problems"])
    return {"calls": len(batches), "calls_ok": len(ok), "call_errors": dict(errors),
            "picks_in_ok_calls": picks, "passed_check": passed,
            "pass_rate": round(passed / picks, 3) if picks else None, "failure_reasons": dict(reasons)}


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("platform", nargs="?", choices=["anilist", "mal"])
    ap.add_argument("username", nargs="?")
    ap.add_argument("--users-dir")
    ap.add_argument("-k", type=int, default=10)
    ap.add_argument("--pause", type=float, default=60.0,
                    help="seconds between model calls (free tiers limit tokens per minute)")
    ap.add_argument("--limit", type=int, help="only the first N users")
    ap.add_argument("--save", help="write counts to results/<name>.json")
    args = ap.parse_args()

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
        lists = [(u.id, u.matched) for u in evaluate.load_users(index, args.users_dir)][:args.limit]
    elif args.platform and args.username:
        matched, _ = parsers.match_to_index(parsers.get_profile(args.platform, args.username), index)
        lists = [("you", matched)]
    else:
        sys.exit("Give a platform and username, or --users-dir.")

    batches = []
    for n, (uid, matched) in enumerate(lists):
        model, recs = recommend.recommend(matched, catalog, matrices, k=args.k, starts=starts, retellings=links)
        if model is None:
            continue
        if batches:
            time.sleep(args.pause)
        results, error = explain.explain_picks(recs, facts, spoilers, config, use_cache=False)
        batches.append((results, error))
        got = sum(r["source"] == "model" for r in results)
        print(f"  {uid}: {got}/{len(results)} sentences passed" + (f"  (model not used: {error})" if error else ""))
        if len(lists) == 1:
            for r in results:
                print(f"    [{r['source']}] {explain.render(r['text'])}" + (f"   <- {r['problems']}" if r["problems"] else ""))

    summary = tally(batches)
    print("\n", json.dumps(summary, indent=1))
    if args.save:
        Path("results").mkdir(exist_ok=True)
        out = {**summary, "model": config["model"], "users": len(lists), "k": args.k}
        Path("results", f"{args.save}.json").write_text(json.dumps(out, indent=1))
        print(f"Saved results/{args.save}.json")


if __name__ == "__main__":
    main()
