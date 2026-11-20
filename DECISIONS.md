# Decisions

Short list of choices that shape the project, each with the reason and the cost.
Newest decisions go in the relevant section; add a date when one changes.

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
- Why: a crash or rate limit should never lose a 5-minute crawl.

## Identifiers

**Key everything on the AniList ID; map to MyAnimeList's `idMal` when present.**
- 16 of 5,000 titles have no `idMal`, so a MAL-only key would break exclusion for them.
- MAL watchlist imports match on `idMal`; AniList imports match on the AniList ID.

## Text cleaning

**Strip HTML, entities and source credits from synopses.**
- Credit lines like "(Source: Crunchyroll)" are noise for embeddings and should not be quoted by the LLM.

**Tags: drop spoiler-flagged tags and tags ranked below 60.**
- Why: keeps spoilers out of anything shown to the LLM and removes low-confidence crowd tags.
- Still open: a stoplist for structural tags like "Primarily Teen Cast".

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

## Missing values

**Null episodes means ongoing or unknown, not zero.**
- Titles with null episodes are excluded whenever the user sets an episode cap.

**Null score or year.**
- After the status filter, remaining null scores should be rare. Handle them at ranking time (treat as no score signal, do not rank them as zero).

## Open decisions

- **Embedding models:** compare all-MiniLM-L6-v2 against a BGE or E5 model. Only 2.8% of synopses exceed MiniLM's length, so expect a small difference; report it honestly.
- **Chroma index:** commit it to the repo or rebuild at startup? Depends on repo size and cold-start time on the demo host. Decide in week 1.
- **Hidden gems mode:** how strongly to down-weight popularity.
- **Evaluation users:** collect 20-30 public watchlists starting in week 2; anonymize usernames in the repo.
