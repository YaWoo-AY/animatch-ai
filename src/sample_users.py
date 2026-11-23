"""Collect an anonymous evaluation set of public AniList watchlists.

Run from the project root, in the (.venv) terminal:
    python src/sample_users.py --probe                       # 1. check the two AniList queries work
    python src/sample_users.py --target 25 --exclude yawoo   # 2. sample until 25 users qualify
    python src/sample_users.py --report                      # 3. summarize what was collected

What it does: reads the recent-activity feed (people who updated their anime list lately), picks
candidates at random, downloads each candidate's public list, and keeps the user only if the list is
rich enough for a leave-one-out test: enough rated titles and enough titles rated 8 or higher, so
that hiding one still leaves plenty to learn from.

Privacy and terms (please read):
  * AniList's terms prohibit "hoarding or mass collection of data". This script is for ~25-30 users,
    runs slowly (AniList has been limiting to 30 requests/minute) and stops at --target.
  * Only title ids, status, score and progress are stored. No usernames and no profile text. Files
    are numbered u001, u002 and so on. The folder is under data/, which is git-ignored: never commit
    it, and do not republish the lists.
  * The set is a frozen snapshot, so results stay reproducible.
  * It never writes usernames to the profile cache either (it calls the fetcher directly).
The only identifying thing kept is AniList's numeric user id, in data/eval_users/_checked.json, so a
re-run does not download the same people twice.
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import parsers

OUT_DIR = Path("data/eval_users")
CHECKED_NAME = "_checked.json"

# Activity feed: each ListActivity is "user X updated an anime on their list".
ACTIVITY_QUERY = """
query ($page: Int, $perPage: Int) {
  Page(page: $page, perPage: $perPage) {
    pageInfo { hasNextPage }
    activities(type: ANIME_LIST, sort: ID_DESC) {
      ... on ListActivity { user { id name } }
    }
  }
}
"""

# Fallback: the site's heaviest watchers. Biased toward very large lists, but it needs no activity feed.
USERS_QUERY = """
query ($page: Int, $perPage: Int) {
  Page(page: $page, perPage: $perPage) {
    pageInfo { hasNextPage }
    users(sort: WATCHED_TIME_DESC) { id name }
  }
}
"""

STRATEGIES = {"activity": (ACTIVITY_QUERY, "activities"), "users": (USERS_QUERY, "users")}


# ---------------------------------------------------------------- candidates

def extract_users(data, field):
    """Distinct {id, name} pairs from one page of results, in order. Items without a user are skipped."""
    seen, out = set(), []
    for item in (data.get("Page") or {}).get(field) or []:
        user = item.get("user") if field == "activities" else item
        if not user or not user.get("id") or not user.get("name") or user["id"] in seen:
            continue
        seen.add(user["id"])
        out.append({"id": user["id"], "name": user["name"]})
    return out


def gather_candidates(strategy, pages, per_page=50, pause=2.5, post=None, sleep=time.sleep):
    """Walk the first `pages` pages of a strategy and return distinct candidates."""
    post = post or parsers._post_anilist
    query, field = STRATEGIES[strategy]
    found = {}
    for page in range(1, pages + 1):
        data = post(query, {"page": page, "perPage": per_page})
        for u in extract_users(data, field):
            found.setdefault(u["id"], u)
        if not (data.get("Page") or {}).get("pageInfo", {}).get("hasNextPage"):
            break
        sleep(pause)
    return list(found.values())


# ---------------------------------------------------------------- qualification

def assess(entries, index, min_rated=50, min_liked=15, min_matched=100):
    """Is this list good enough for leave-one-out? Returns (ok, stats, reason)."""
    matched, _ = parsers.match_to_index(entries, index)
    rated = [e for e in matched if e["score"] is not None]
    liked = [e for e in rated if e["score"] >= 8]
    stats = {"entries": len(entries), "matched": len(matched), "rated": len(rated), "liked": len(liked),
             "completed": sum(1 for e in matched if e["status"] == "completed")}
    if len(matched) < min_matched:
        return False, stats, f"only {len(matched)} titles in our index (need {min_matched})"
    if len(rated) < min_rated:
        return False, stats, f"only {len(rated)} rated titles (need {min_rated})"
    if len(liked) < min_liked:
        return False, stats, f"only {len(liked)} rated 8+ (need {min_liked})"
    return True, stats, ""


# ---------------------------------------------------------------- state on disk

def load_checked(out_dir):
    path = Path(out_dir) / CHECKED_NAME
    return set(json.loads(path.read_text())) if path.exists() else set()


def save_checked(out_dir, checked):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / CHECKED_NAME).write_text(json.dumps(sorted(checked)))


def saved_users(out_dir):
    return sorted(Path(out_dir).glob("u[0-9][0-9][0-9].json")) if Path(out_dir).exists() else []


def save_user(out_dir, number, entries, stats):
    """Store one anonymous list. No username is written."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    path = Path(out_dir) / f"u{number:03d}.json"
    path.write_text(json.dumps({"id": f"u{number:03d}", "stats": stats, "entries": entries}))
    return path


# ---------------------------------------------------------------- the sampling loop

def sample(candidates, fetch, index, out_dir=OUT_DIR, target=25, exclude=(), seed=0, pause=2.5,
           sleep=time.sleep, log=print, **limits):
    """Download candidates in random order until `target` qualifying lists are saved.

    `fetch(name)` returns a list of entries or raises parsers.ProfileError. Returns the number of
    users saved in total (including ones from earlier runs).
    """
    checked = load_checked(out_dir)
    excluded = {n.lower() for n in exclude}
    pool = [c for c in candidates if c["id"] not in checked and c["name"].lower() not in excluded]
    random.Random(seed).shuffle(pool)
    have = len(saved_users(out_dir))
    tried = 0
    for cand in pool:
        if have >= target:
            break
        tried += 1
        try:
            entries = fetch(cand["name"])
            ok, stats, reason = assess(entries, index, **limits)
        except parsers.ProfileError as e:
            ok, stats, reason = False, {}, f"skipped ({type(e).__name__})"
        checked.add(cand["id"])
        save_checked(out_dir, checked)
        if ok:
            have += 1
            save_user(out_dir, have, entries, stats)
            log(f"  kept as u{have:03d}: {stats['matched']} in index, {stats['rated']} rated, {stats['liked']} rated 8+")
        else:
            log(f"  rejected #{tried}: {reason}")
        sleep(pause)
    return have


def report(out_dir=OUT_DIR):
    files = saved_users(out_dir)
    if not files:
        print("No users saved yet.")
        return
    rows = [json.loads(f.read_text())["stats"] for f in files]
    print(f"{len(rows)} users in {out_dir}")
    print(f"{'id':<6}{'entries':>8}{'in index':>10}{'rated':>7}{'rated 8+':>10}")
    for f, s in zip(files, rows):
        print(f"{f.stem:<6}{s['entries']:>8}{s['matched']:>10}{s['rated']:>7}{s['liked']:>10}")
    for key in ("matched", "rated", "liked"):
        vals = sorted(s[key] for s in rows)
        print(f"median {key}: {vals[len(vals) // 2]}  (min {vals[0]}, max {vals[-1]})")


def probe():
    """Run each query once and show whether it worked (counts only, no names)."""
    for name, (query, field) in STRATEGIES.items():
        print(f"\n--- strategy '{name}' ---")
        try:
            data = parsers._post_anilist(query, {"page": 1, "perPage": 10})
        except parsers.ProfileError as e:
            print(f"FAILED: {e}")
            continue
        users = extract_users(data, field)
        print(f"OK: {len(users)} distinct users on page 1; hasNextPage = "
              f"{(data.get('Page') or {}).get('pageInfo', {}).get('hasNextPage')}")
        if not users:
            print("Raw reply (first 600 chars):", json.dumps(data)[:600])


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true", help="test the AniList queries and exit")
    ap.add_argument("--report", action="store_true", help="summarize saved users and exit")
    ap.add_argument("--target", type=int, default=25, help="stop when this many users are saved")
    ap.add_argument("--strategy", choices=list(STRATEGIES), default="activity")
    ap.add_argument("--pages", type=int, default=6, help="feed pages to read for candidates (50 per page)")
    ap.add_argument("--exclude", default="", help="comma-separated usernames to skip (e.g. your own)")
    ap.add_argument("--min-rated", type=int, default=50)
    ap.add_argument("--min-liked", type=int, default=15, help="minimum titles rated 8 or higher")
    ap.add_argument("--min-matched", type=int, default=100, help="minimum titles that are in our index")
    ap.add_argument("--pause", type=float, default=2.5, help="seconds between requests (AniList may allow 30/min)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.probe:
        return probe()
    if args.report:
        return report()

    index = parsers.load_index()
    print(f"Reading up to {args.pages} pages of the '{args.strategy}' feed...")
    try:
        candidates = gather_candidates(args.strategy, args.pages, pause=args.pause)
    except parsers.ProfileError as e:
        sys.exit(f"Could not read the feed: {e}\nRun with --probe to see which query fails.")
    if not candidates:
        sys.exit("The feed returned no users. Run with --probe, and paste the output.")
    print(f"{len(candidates)} candidates. Sampling until {args.target} qualify "
          f"(rated >= {args.min_rated}, rated 8+ >= {args.min_liked}, in index >= {args.min_matched}).")

    fetch = lambda name: parsers.fetch_anilist(name, pause=args.pause)   # not get_profile: no username in the cache
    try:
        have = sample(candidates, fetch, index, OUT_DIR, args.target, args.exclude.split(","), args.seed,
                      args.pause, min_rated=args.min_rated, min_liked=args.min_liked,
                      min_matched=args.min_matched)
    except KeyboardInterrupt:
        print("\nStopped. Progress is saved; run the same command to continue.")
        return
    print(f"\nDone: {have} users saved in {OUT_DIR}.")
    if have < args.target:
        print("Not enough qualified yet. Run again with a higher --pages, or lower --min-liked.")
    report()


if __name__ == "__main__":
    main()
