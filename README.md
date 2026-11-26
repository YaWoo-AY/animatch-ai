# AniMatch AI

A spoiler-free anime recommender. Give it your public AniList or MyAnimeList username and it
suggests shows you have not seen, with the titles you liked that led to each pick.

**Live demo:** [ADD YOUR STREAMLIT LINK HERE]

![The page with the main list and a "deeper cuts" list side by side](docs/screenshot.png)

I built it to learn how to evaluate a recommender honestly, not only to make one. The main result
is mixed, and this page says so.

## What it does

- Reads a public watchlist from AniList or MyAnimeList (official API, no login).
- Builds a taste profile from ratings, falling back to completed, watching and dropped status for
  the many titles people never rate. Unrated titles count, but with a cap, so hundreds of them
  cannot outvote a few deliberate high ratings.
- Hides everything you have already seen, counting a whole franchise as seen (all seasons, movies
  and OVAs), plus other versions of the same story.
- Ranks the rest by taste and shows each pick at its first entry, with the liked titles that
  explain it.
- Optionally writes one short sentence per pick with a language model, checked against the facts it was given (below).
- Shows no synopsis and no spoiler-flagged tags. Filters: episode count, release year, community
  score.

## How it works

1. **Data.** The 5,000 most popular anime on AniList (TV, movie, ONA, OVA; adult titles excluded).
   4,869 have a synopsis and a release, and are grouped into 3,013 franchises from AniList's
   relation links.
2. **Vectors.** Each title is embedded (all-MiniLM-L6-v2) from its genres, non-spoiler tags and
   synopsis, and stored in ChromaDB. A second representation is an IDF-weighted genre and tag
   vector.
3. **Taste and scoring.** The method that ships is `minilm@center=1@knn=10@genre=1@pop=0.5`:
   embeddings with the shared "typical anime" component removed, scored against the 10 nearest
   liked franchises, fused with the tag score, plus a popularity prior. The popularity weight is a
   slider on the page.
4. **Deeper cuts.** A toggle shows a second list with the most popular franchises left out.

## How I evaluated it

A recommender cannot be judged by eye on one profile, and my first attempts showed why: the first
version looked reasonable on my own list and scored almost no better than random.

**Test.** For each user, hide one franchise they rated 8 or higher, build the recommendation from
the rest, and see where the hidden franchise ranks among everything they have not seen. Recall@10 is
the share of hidden franchises that land in the top 10, averaged per user, with 95% intervals from
resampling users. Baselines: random, popularity, and a genre/tag vector.

**Users.** 25 public AniList users, sampled from the recent-activity feed, stored as ids and
scores only. A second set of 25 different users was kept untouched. I compared methods on the first
set, wrote down one configuration and the success rule, then ran the second set once.

**Two views.** Popularity alone is a very strong baseline, because the hidden title is something
the user chose to watch and people mostly watch popular things. So I report the full catalog and
two views with the 300 or 1,000 most popular franchises removed, which asks whether the system finds
less obvious titles.

**Result on the untouched test users** (Recall@10, paired difference to popularity):

| View | Popularity | AniMatch | Difference [95% CI] |
|---|---|---|---|
| Top 1,000 removed (217 hidden) | 0.2% | 10.1% | +9.8 pts [+3.0, +19.9] |
| Top 300 removed (370 hidden) | 3.7% | 7.7% | +4.0 pts [-0.3, +8.5] |
| All franchises (487 hidden) | 15.2% | 13.3% | -1.9 pts [-9.1, +5.2] |

By the rule I wrote beforehand this is a **partial success**: clearly better than popularity when
the 1,000 most popular titles are removed, not clearly different in the other two views. It does
not beat plain popularity on the full catalog.

**What I learned**
- Text embeddings compared directly were about as good as random until I centered the vectors, and
  worse than random once popular titles were removed. Centering and a popularity prior mattered more
  than the choice of embedding model.
- Scoring by the nearest few liked titles looked best on one setting of K and poor on others. I did
  not trust the best of a dozen settings, and picked the configuration that held up in both tail
  views instead.
- The genre/tag vector alone was better than popularity when the 1,000 most popular titles were
  removed and clearly worse on the full catalog.
- Retellings slipped through as "unseen": a 2012 OVA of a story I had just watched was recommended
  first, because AniList links them only as `ALTERNATIVE`. They are now hidden, which the evaluation
  above did not include.

## Limits

- The test measures finding things you already watched and loved, not whether a new recommendation
  would be enjoyed. It favors popular titles. It is a proxy for comparing versions.
- 25 users sampled from an activity feed favor people who update often. The tail views rest on a few
  hundred hidden titles, and some users contribute one or two.
- Nothing in the score knows about quality. Titles with low community scores can appear, especially
  with the popularity weight at 0.
- Only the top 5,000 titles, no shorts or specials, and no adult titles. About 8% of my own list's
  entries were not in the index.
- Related titles linked only as `OTHER`, and TV side stories, count as separate franchises, so a
  sequel can be recommended to someone who liked the first series.
- The optional one-sentence explanations are checked for what they name, not for tone or usefulness, and a spoiler that is not in my spoiler-tag list would pass. They go through a free hosted model, so the hosted page will often fall back to a plain template.

## Explanations

Each pick can get one sentence saying why it fits, written by a language model. The model is given
only the pick's genres and tags and the ones it shares with your liked titles, never a synopsis or
a spoiler-flagged tag, and must mark every title and attribute it uses so code can check them.
A sentence that names a title or attribute that was not supplied, mentions another title, includes
a spoiler-flagged tag or uses a stray number is thrown away and replaced by a plain template built
from the same facts. Nothing unchecked is shown.

On 10 development users (100 picks), 98 sentences passed the check on the final run. The two that
failed were a sentence containing a spoiler-flagged term, which the model produced from its own
knowledge of the show, and one with an attribute that was not in the supplied list. The check limits what can be
claimed, not whether the sentence is good, and it only knows the spoiler terms in my list.
`python src/eval_explanations.py --users-dir data/eval_users` reproduces the measurement
(needs an `LLM_API_KEY`).

## Run it

The deployed bundle is committed, so the page runs from a clean checkout:

```
pip install -r requirements.txt
streamlit run app.py
```

Command line, and the whole pipeline (needs `requirements-dev.txt`):

```
python src/recommend.py anilist <username> --skip-popular 300
python src/fetch_anilist.py --pages 100    # crawl
python src/franchise.py                    # group into franchises
python src/build_index.py                  # embed into ChromaDB
python src/export_runtime.py               # write app_data/ for the page
python src/evaluate.py --methods popularity genre minilm@center=1@knn=10@genre=1@pop=0.5
pytest
```

Explanations need `LLM_API_KEY` (any OpenAI-compatible provider; the default is Groq's free tier). MyAnimeList lists need a free client ID in `MAL_CLIENT_ID` (environment or `.env`; a Streamlit
secret when hosted). AniList needs no key.

## Data and privacy

Anime data and cover images come from AniList. A watchlist is cached briefly so the page does not
refetch it, and nothing is logged. The evaluation users are anonymous ids and scores, kept out of
the repository, and the results files contain aggregate numbers only. AniList's terms prohibit mass
collection, so the user sample is small on purpose.

## More

- [NOTES.md](NOTES.md): a dated log of what I did, what failed and what surprised me.
- [DECISIONS.md](DECISIONS.md): each design choice with its reason and cost, including the
  evaluation protocol and the success rule written before the test run.
- `results/`: the evaluation outputs.
