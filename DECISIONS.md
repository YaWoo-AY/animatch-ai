# Decisions

Short list of choices that shape the project, each with the reason and the cost.
When a decision changes, edit it in place and add the date.

## Data scope

**Top 5,000 titles by popularity, formats TV, MOVIE, ONA, OVA, adult titles excluded.**
- Excluded formats: MUSIC, SPECIAL, TV_SHORT (not worth recommending).
- Cost: popularity sorting biases toward mainstream titles. A "hidden gems" mode will down-weight popularity at ranking time instead of changing the crawl.

**Index only released titles with a synopsis.**
- Dropped from the index: NOT_YET_RELEASED, CANCELLED, and anything with an empty synopsis (4,869 of 5,000 kept). RELEASING titles stay.
- Why: a recommender should not suggest something nobody can watch, and a title with no synopsis has nothing to embed.
- The filter runs after franchise grouping, so unreleased sequels still link franchises together.

**Crawl is resumable.**
- Raw records go to `data/raw_anime.jsonl` one per line, progress to `data/crawl_state.json`. Duplicate AniList IDs are dropped when building the clean file.
- Re-running `python src/fetch_anilist.py --pages 100` re-cleans from the saved raw file with no network calls, so cleaning changes never need a recrawl.

## Identifiers

**Key everything on the AniList ID; map to MyAnimeList's `idMal` when present.**
- 16 of 5,000 titles have no `idMal`, so a MAL-only key would break exclusion for them.
- MAL watchlist imports match on `idMal`; AniList imports match on the AniList ID.

## Text cleaning and spoilers

**Strip HTML, entities and source credits from synopses.**
- Credit lines like "(Source: Crunchyroll)" are noise for embeddings and should not be quoted by the LLM.

**Two tag fields: `tags` (safe to show) and `spoiler_tags` (retrieval only).**
- `tags`: non-spoiler tags ranked 60 or higher.
- `spoiler_tags`: spoiler-flagged tags ranked 60 or higher.
- Why two: AniList flags many of the best tags as spoilers (Summer Time Rendering: Survival 100, Time Manipulation 97, Time Loop 95). Dropping them left only weak themes (Coastal, Rural, Shounen) and hurt retrieval.
- **Rule: spoiler tags must never be shown to a user or included in the prompt of the LLM that writes pitches.** Pitch generation reads `tags` and the premise from `anime.json`, never Chroma's stored text. The stored text in the `spoilers` and `notitle_spoilers` collections contains spoiler tags, so those collections are for retrieval only.

## Embeddings and search

**Two models, compared: all-MiniLM-L6-v2 and bge-small-en-v1.5.**
- Each model has its own ChromaDB collection, cosine distance, normalized vectors.
- BGE uses the query prefix "Represent this sentence for searching relevant passages: ". MiniLM uses none.
- Distances are not comparable across models (BGE roughly 0.29-0.32 and MiniLM 0.54-0.61 for similar-quality matches). Never share one distance cutoff between models, and never read a small gap between BGE distances as meaningful (several top results tie at 0.399-0.400).

**Embedding text structure: title (optional), genres, themes, optional plot elements, then synopsis.**
- Why this order: if a model truncates a long input it cuts the end of the synopsis, never the genres and themes. MiniLM truncates 6.5-8.8% of texts (limit 256 tokens); BGE 0.1% (limit 512).
- Themes: up to 10 tags, after removing structural tags with a stoplist ("Primarily Teen Cast", "Male Protagonist", "Ensemble Cast", "Heterosexual" and similar). The stoplist applies to the embedding text only.
- Plot elements: up to 8 spoiler tags, only in the `spoilers` variants.

**Four embedding variants, one collection each.**

| Variant | Title in text | Spoiler tags in text | Collection (per model) |
|---|---|---|---|
| base | yes | no | `anime_<model>` |
| notitle | no | no | `anime_<model>_notitle` |
| spoilers | yes | yes | `anime_<model>_spoilers` |
| notitle_spoilers | no | yes | `anime_<model>_notitle_spoilers` |

- Why variants: title words leaked into results (6 of 10 "Girl" titles for a query about a girl; 7 of 10 "Summer" titles among neighbours of Summer Time Rendering with MiniLM), and spoiler tags carry strong signal.
- **Provisional default for dense retrieval: `notitle_spoilers`.** Evidence is thin (one query judged by eye: roughly 8 of 10 on target for "time loop" vs about 4 for base with BGE). Confirm with Recall@10 in week 3.
- Cost of dropping the title: name lookups and titles that are the best clue (7th Time Loop) fall out. Plan: BM25 over titles alongside the dense search.
- Collections are always rebuilt from scratch so they match `anime.json`.

**Free-text search is a secondary feature; profile-based search is the main path.**
- Small embedding models match topic words, not plot mechanics ("a student keeps waking up on the same morning" returns school shows). Item-to-item neighbours from a real title are much more sensible.
- "Like X" queries use X's own vector as the seed and exclude X's franchise. They are not treated as free text.
- "Shorter", "newer" and similar constraints go through filters (episode cap, year), not the query text.
- Stretch: an LLM rewrites a described plot into tag vocabulary ("relives the same day" becomes "time loop") before searching.

## Stored metadata

**ChromaDB cannot store empty values, so missing numbers are stored as -1 and missing text as "".**
- Covers episodes, year, score, mal_id, format, cover.
- Filters rely on this: an episode cap requires `episodes >= 1`, which excludes titles with unknown episode counts (the rule below).
- Genres are stored as one string joined by "|".

## Franchise grouping

Used to exclude a whole franchise once the user has watched part of it (no recommending season 2 of a show they finished).

**Rule**
- Always merge titles linked by SEQUEL, PREQUEL or SUMMARY (same continuity).
- Attach a group to a parent via PARENT or SIDE_STORY only if the group contains no TV series (a movie, OVA or ONA, or a chain of them) and it has exactly one parent group.
- Never link SPIN_OFF, ALTERNATIVE, ADAPTATION, OTHER or CHARACTER, and ignore anything pointing at manga or novels.
- Links through titles we did not crawl still count, so two crawled titles connected only via a missing one end up together.

**Why this rule**
- AniList uses PARENT / SIDE_STORY both for "movie of this show" and for "separate timeline under the original". Linking them blindly merged all of Gundam (50 titles) into one group.
- Attaching lone TV shows to a parent pulled G Gundam, Turn A, AGE and others back in. Movie-only chains with their own sequels (the Pokemon movies) need to attach, which a "singletons only" rule missed.

**Cost**
- A TV season linked to its show only by PARENT or SIDE_STORY (no sequel or prequel link) stays separate. The recommender could suggest it to someone who has watched the show. I judged that less harmful than blocking an entire franchise because of one parent link.
- A standalone title pointing at two different parents stays alone rather than guessing.

**Result**
- 3,050 franchises, 815 with 2+ titles. Largest: Pokemon 36, Conan 29, Dragon Ball Z 28, Gundam UC 26. Gundam is split into UC plus separate timeline groups.
- Franchise duplicates sit next to each other in search results, so a diversity pass is needed after retrieval.

## Missing values

**Null episodes means ongoing or unknown, not zero.**
- Titles with null episodes are excluded whenever the user sets an episode cap.

**Null score or year.**
- After the status filter, remaining null scores should be rare. Handle them at ranking time (treat as no score signal, do not rank them as zero).

## Profile import (Step 5)

**AniList is the primary source, the official MyAnimeList API is the second.**
- Why AniList first: GraphQL, no key needed, one query returns the whole list including custom lists, and the crawl already uses AniList ids.
- Why the official MAL API: a free client ID is enough to read a public list (checked with a real request). It is a supported product, not a scraper.
- Cost: MAL needs a client ID, kept in `MAL_CLIENT_ID` or the git-ignored `.env`. Private lists cannot be read from either platform.

**Not using Jikan or its replacements.**
- The Jikan public API was discontinued on 1 October 2026.
- Tenrai covers the catalogue only; its docs say to use the official MAL API for user data.
- Jikan-Edge is a single-developer scraper on Cloudflare Workers, and I never confirmed its list endpoint. Self-hosting Jikan needs MongoDB and a server for a feature the official API already provides, and it still scrapes MAL.
- Rule I took from this: a project that depends on a free third-party scraper can break without warning. Prefer official APIs for anything the demo needs.

**One standard entry format, one fetcher per platform.**
- `{anilist_id, mal_id, status, score, progress}`; adding a platform means one new function and one line in `FETCHERS`.

**Scores: 0 means unrated, stored as None.**
- 79% of my own entries are unrated, so unrated completed titles must still count, as a weaker positive signal than a high rating. Designed in Step 6.

**Cache profiles for 24 hours.**
- Protects the rate limits and makes reruns instant. `--refresh` forces a fetch. The cache is in `data/cache/`, which is git-ignored.

**Titles not in the index are ignored for recommendations.**
- They cannot be recommended and cannot block their franchise from being suggested. Measured on a real list: 8% of entries, but 62% of those are shorts and specials excluded on purpose, so the real gap is about 2%.
- Not widening the crawl for now. Revisit if the evaluation shows missing history hurting Recall@10.

## Taste vector and exclusion (Step 6)

- **Weights per entry:** rated 8 to 10 gives (score - 5) / 5; 6 to 7.9 gives 0; 5 or lower is negative. With no score: completed +0.3, watching +0.15, dropped -0.6 (or -0.25 once more than half the episodes were watched), paused and planning 0. A score always beats the status.
- **Vector:** unit(positive centroid) minus alpha times unit(negative centroid), alpha 0.3, then normalised. Alpha and the share cap are guesses to be tuned with the harness.
- **Seen means whole franchises.** Completed, watching, paused and dropped hide the franchise. Plan-to-watch does not, because it is a candidate the user already likes the look of.
- **Unrated titles count, weakly, with a cap.** Many people rate a fifth of their list. Without a cap, hundreds of unrated completed titles would outvote a few dozen deliberate high ratings. Share capped at 35%; this is a starting value, to be tuned.
- **One vote per franchise.** Five seasons are one observation.
- **Recommend the franchise's first entry,** not the season nearest the taste vector, unless that entry breaks the user's filters.
- **Shorts and specials are ignored for seen history.** They are not indexed, so they cannot mark a franchise as seen. Earlier measurement: most unmatched titles were specials or shorts that are excluded by design. Revisit if the evaluation or real users show repeated recommendations of things already watched.

## Evaluation protocol (Step 7)

- **Leave-one-out on whole franchises.** Hide a franchise rated 8 or higher, build from the rest, rank the hidden franchise among all unseen franchises. Averaged per user, with intervals from resampling users, so a user with 1,000 liked titles cannot dominate.
- **Method options in `evaluate.py`** (after `@`, combinable): `alpha` weight of the dislike vector; `share` largest share of liked weight that unrated titles may carry (0 ignores them, 1 removes the cap); `center` 1 subtracts the index's average vector first; `knn` K scores a title by its K nearest liked franchises instead of their average; `pop` w adds w times standardised log popularity to the standardised similarity; `genre` w also scores with the genre/tag representation and adds w times its standardised score. Index variants are written `minilm/notitle` and so on.
- **Two views, both reported.** All franchises, and with the 300 or 1,000 most popular removed. The first rewards recommending what everyone watches (popularity alone gets 19.7%); the second asks whether the system finds less obvious titles. I decided to report both before seeing the second.
- **Dev set and test set.** 25 users to compare methods on, 25 different users reserved for one final run. Tuning on the test set is not allowed. If I change the configuration after looking at the test results, the test set is spent and a new one must be sampled.
- **Users are anonymous and local.** Ids, statuses and scores only, in the git-ignored `data/`. AniList's terms forbid mass collection, so the sets stay small.
- **Chosen after the dev runs, before the test run:** `minilm@center=1@knn=10@genre=1@pop=0.5` (centered MiniLM vectors, scored against the 10 nearest liked franchises, fused with the genre/tag score at equal weight, plus a 0.5 popularity prior). Reason: it is the only configuration that was clearly better than popularity at Recall@10 in both tail views (+6.3 points with the top 1000 removed, +2.7 with the top 300 removed), with similar absolute values (7.5% and 7.3%). Settings that did better in one view (K=5 at 8.1%) did worse in the other and moved a lot with K, which looks like noise. Picking the best of a dozen settings on 25 users also overstates its real strength.
- **Decision rule, written before the test run:**
  - Success: the paired difference to popularity at Recall@10 has a lower bound above zero in both tail views on the test set.
  - Partial: clearly better in one tail view only.
  - Failure: neither. Then the README says so and the configuration is not described as an improvement.
  - The all-franchises view is reported whatever it shows. I expect popularity to stay ahead there.
- **Test commands (run once each):** `--users-dir data/test_users` with `--tail 1000`, `--tail 300` and no tail, and the methods `popularity genre minilm minilm@center=1@knn=10@genre=1@pop=0.5`.

### Test-set result (run once, 2026-11-24)

Configuration `minilm@center=1@knn=10@genre=1@pop=0.5`, 25 test users, Recall@10, paired difference to popularity:

| View | Popularity | Chosen config | Difference [95% CI] | Verdict |
|---|---|---|---|---|
| Top 1000 removed (217 hidden, 22 users) | 0.2% | 10.1% | +9.8 pts [+3.0, +19.9] | clearly better |
| Top 300 removed (370 hidden, 24 users) | 3.7% | 7.7% | +4.0 pts [-0.3, +8.5] | not clearly different |
| All franchises (487 hidden, 25 users) | 15.2% | 13.3% | -1.9 pts [-9.1, +5.2] | not clearly different |

By the rule written beforehand, this is **partial**: clearly better in one tail view, not in the other. I do not describe the configuration as an improvement over popularity in general. The point estimate is positive in both tail views and the top-300 interval only just includes zero, but I do not round that up. The genre/tag score alone was clearly better than popularity with the top 1000 removed (+6.3 pts), not clearly different with the top 300 removed, and clearly worse on all franchises (-8.9 pts).

What I take from it:
- The configuration finds less popular titles better than popularity does, and on this sample it is not clearly worse than popularity overall. It is not shown to be better where most real watching happens.
- Recall@50 and MRR point the same way as Recall@10 in the tail views, but I did not pre-register them and do not rely on them.
- The test set is now spent for this question. Any further tuning needs a new sample of users.

## Open decisions

- **Embedding choices:** MiniLM vs BGE and the index variants (title, spoilers) were compared on the dev set but could not be separated from noise while the base method was near random. Re-run on the chosen configuration once it is in the live recommender.
- **Hybrid retrieval:** BM25 over titles and tags, cross-encoder reranker, diversity pass.
- **Query rewriting by an LLM:** stretch goal.
- **Chroma index:** commit it to the repo or rebuild at startup? Depends on repo size and cold-start time on the demo host. Decide in week 1 of deployment planning.
- **Hidden gems and popularity:** the popularity prior is 0.5 in the chosen configuration. Whether the product shows one list or two (popular picks you have not seen, and deeper cuts for your taste) depends on the test-set results.
- **Private lists:** confirm the error replies from AniList and MAL, then make the demo message friendlier.
