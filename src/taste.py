"""Turn a watchlist into (a) a set of franchises to hide and (b) a taste vector.

No network and no ChromaDB here, only numpy, so every rule is easy to test. The rules and the
reasons for them are in DECISIONS.md (Taste vector and exclusion). In short: seen franchises are
hidden, each entry gets a weight from its score or status, a franchise gets one vote, unrated
titles are capped, and the vector is positive centroid minus alpha times negative centroid.
All numbers are starting points, not findings.
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


def liked_vectors(model):
    """(vectors, weights) of the liked franchises, with the unrated cap already applied.

    Used by scorers that compare a candidate with each liked franchise separately instead of with
    their average.
    """
    V = np.vstack([o.vector for o in model.positives])
    w = np.array([o.weight * (1.0 if o.rated else model.unrated_scale) for o in model.positives])
    return V, w


def nearest_liked(taste, rec_vector, n=3):
    """The liked franchises closest to a recommendation: [(Observation, similarity), ...].

    Prefers franchises the user rated; falls back to all positives when they rated almost nothing.
    """
    pool = [o for o in taste.positives if o.rated]
    if len(pool) < n:
        pool = taste.positives
    rec = _unit(np.asarray(rec_vector, dtype=float))
    scored = sorted(((float(o.vector @ rec), o.franchise_id, o) for o in pool), key=lambda t: (-t[0], t[1]))
    return [(o, sim) for sim, _, o in scored[:n]]


def explain(taste, rec_vector, n=3):
    """The liked franchises closest to a recommendation: [(title, similarity), ...]."""
    return [(o.title, sim) for o, sim in nearest_liked(taste, rec_vector, n)]
