"""Day 6: turn a watchlist into (a) a set of franchises to hide and (b) a taste vector.

No network and no ChromaDB in this file, only numpy, so every rule here is easy to test.

The idea
--------
1. Exclusion set: any franchise the user has watched, is watching, paused or dropped is never
   recommended (we do not suggest season 2 of something they already know). Plan-to-watch titles
   are NOT hidden by default, because they are candidates the user already showed interest in.
2. Weights: every list entry gets a number. Positive means "more like this", negative means
   "less like this", zero means "no opinion".
       rated 8-10            +0.6 .. +1.0   (score - 5) / 5
       rated 6-7.9           0              lukewarm, ignored
       rated 5 or lower      -0.25 .. -1.0
       completed, unrated    +0.3           they finished it, which says something
       watching, unrated     +0.15
       dropped, unrated      -0.6, or -0.25 if they got past half of the episodes
       paused, planning      0
   A score always beats the status: a dropped title they rated 9 counts as a 9.
3. One vote per franchise: a five-season show is one observation, not five. The franchise weight
   is the mean of its entries' weights (ignoring zeros) and its vector is the mean of their vectors.
4. Unrated titles cannot drown out rated ones. Some people rate 20% of their list; without a cap,
   600 "completed" titles at +0.3 would outweigh 120 deliberate high ratings and the taste vector
   would just be the average of everything they watched. The unrated share of the positive mass is
   capped (default 35%) whenever the user has any rated positives.
5. Rocchio-style vector: unit(positive centroid) - alpha * unit(negative centroid), normalised.
   alpha is a guess (0.3) to be tuned with the evaluation harness in week 3.

All the numbers above are starting points, not findings.
"""
from dataclasses import dataclass, field

import numpy as np

SEEN_STATUSES = {"completed", "watching", "paused", "dropped"}

UNRATED_COMPLETED = 0.3
UNRATED_WATCHING = 0.15
UNRATED_DROPPED_EARLY = -0.6
UNRATED_DROPPED_LATE = -0.25   # they got through at least half of it, so it is a weaker "no"
LATE_DROP_RATIO = 0.5


# ---------------------------------------------------------------- weights

def entry_weight(entry, episodes=None):
    """Weight of one list entry: positive likes, negative dislikes, 0 no opinion."""
    status, score = entry["status"], entry["score"]
    if status == "planning":
        return 0.0
    if score is not None:
        if score >= 8:
            return (score - 5) / 5
        if score <= 5:
            return -min(1.0, max(0.25, (6 - score) / 4))
        return 0.0
    if status == "completed":
        return UNRATED_COMPLETED
    if status == "watching":
        return UNRATED_WATCHING if entry.get("progress", 0) > 0 else 0.0
    if status == "dropped":
        ratio = entry.get("progress", 0) / episodes if episodes and episodes > 0 else 0.0
        return UNRATED_DROPPED_LATE if ratio >= LATE_DROP_RATIO else UNRATED_DROPPED_EARLY
    return 0.0  # paused


def seen_franchises(matched, seen_statuses=SEEN_STATUSES):
    """Franchise ids that must not be recommended."""
    return {e["franchise_id"] for e in matched if e["status"] in seen_statuses}


def planned_franchises(matched, seen=None):
    """Franchises the user plans to watch (and has not otherwise seen)."""
    seen = seen_franchises(matched) if seen is None else seen
    return {e["franchise_id"] for e in matched if e["status"] == "planning"} - seen


# ---------------------------------------------------------------- taste vector

@dataclass
class Observation:
    """One franchise as seen by the taste model."""
    franchise_id: int
    title: str            # the title the user rated highest (or first), used for explanations
    weight: float
    vector: np.ndarray
    rated: bool           # did the user give any title of this franchise a score?


@dataclass
class Taste:
    vector: np.ndarray
    positives: list = field(default_factory=list)
    negatives: list = field(default_factory=list)
    unrated_scale: float = 1.0   # < 1 when the unrated cap shrank the unrated positives


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def observations(matched, embeddings, episodes_by_id=None):
    """Group entries by franchise into Observations. `embeddings` maps anilist_id -> vector."""
    episodes_by_id = episodes_by_id or {}
    groups = {}
    for e in matched:
        vec = embeddings.get(e["anilist_id"])
        if vec is None:
            continue
        w = entry_weight(e, episodes_by_id.get(e["anilist_id"]))
        if w == 0.0:
            continue
        groups.setdefault(e["franchise_id"], []).append((e, w, np.asarray(vec, dtype=float)))

    out = []
    for fid, items in groups.items():
        weight = float(np.mean([w for _, w, _ in items]))
        vector = _unit(np.mean([v for _, _, v in items], axis=0))
        best = max(items, key=lambda t: t[1])[0]
        out.append(Observation(fid, best["title"], weight, vector,
                               rated=any(e["score"] is not None for e, _, _ in items)))
    return out


def build_taste(matched, embeddings, episodes_by_id=None, unrated_share=0.35, alpha=0.3):
    """Build the taste vector. Returns None when there is nothing positive to learn from."""
    obs = observations(matched, embeddings, episodes_by_id)
    pos = [o for o in obs if o.weight > 0]
    neg = [o for o in obs if o.weight < 0]
    if not pos:
        return None

    # cap the share of positive mass that comes from unrated titles
    rated_mass = sum(o.weight for o in pos if o.rated)
    unrated_mass = sum(o.weight for o in pos if not o.rated)
    scale = 1.0
    if rated_mass > 0 and unrated_mass > 0 and unrated_share < 1:   # share >= 1 means no cap
        allowed = unrated_share / (1 - unrated_share) * rated_mass
        if unrated_mass > allowed:
            scale = allowed / unrated_mass
    weights = [o.weight * (1.0 if o.rated else scale) for o in pos]

    positive = _unit(sum(w * o.vector for w, o in zip(weights, pos)))
    vector = positive
    if neg:
        negative = _unit(sum(-o.weight * o.vector for o in neg))
        vector = positive - alpha * negative
    return Taste(_unit(vector), sorted(pos, key=lambda o: -o.weight),
                 sorted(neg, key=lambda o: o.weight), scale)


def explain(taste, rec_vector, n=3):
    """The liked franchises closest to a recommendation: [(title, similarity), ...].

    Prefers franchises the user rated; falls back to all positives when they rated almost nothing.
    """
    pool = [o for o in taste.positives if o.rated]
    if len(pool) < n:
        pool = taste.positives
    rec = _unit(np.asarray(rec_vector, dtype=float))
    scored = sorted(((float(o.vector @ rec), o.title) for o in pool), reverse=True)
    return [(title, sim) for sim, title in scored[:n]]
