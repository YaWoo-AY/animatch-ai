# Project notes

A running log of what I tried, what surprised me, and what I changed.
Newest entries go at the top. This becomes the "What failed and what I'd change" section of the README.

## 2026-11-22 (Day 5): reading watchlists from AniList and MyAnimeList

**What I did**
- Wrote `src/parsers.py`. Both platforms become the same entry: `{anilist_id, mal_id, status, score, progress}`, with statuses completed, watching, paused, dropped, planning.
- AniList: one `MediaListCollection` query (custom lists included, because entries hidden from the status lists only appear there), deduped by media id, scores as `POINT_10_DECIMAL` where 0 means unrated (stored as None). Falls back to a plain query if the chunked one is rejected.
- MyAnimeList: the official API v2 with a client ID only (header `X-MAL-CLIENT-ID`), following `paging.next`, 100 entries per page, about 1 request per second.
- Shared: retry with backoff on 429 and 5xx (honours Retry-After), clear errors (user not found, private list, API error, missing client ID), a 24-hour disk cache in `data/cache/` (git-ignored), and matching to the index by AniList id then MAL id.
- Added `--unmatched`, which looks up the title, format and popularity of titles that are not in the index. 60 tests pass, all offline with mocked HTTP.

**Results on my own lists**
- AniList: 1,040 entries (867 completed, 48 watching, 84 planning, 41 dropped); 956 match the index (91.9%).
- MyAnimeList: 964 entries; 887 match (92.0%). Both give the same top-rated titles, so scores and id matching are right. The two lists differ by about 76 entries because I keep them separately.
- Only 209 of 1,040 AniList entries have a score (831 completed titles are unrated). A taste vector built from ratings alone would see 20% of my history.
- Why 52 seen or dropped titles are not in the index: 32 (62%) are SPECIAL (10) or TV_SHORT (22), formats the crawl excludes on purpose. Of the other 20, four are very popular shows (popularity 79k to 151k) that I believe the adult-content filter removed (not checked), and 16 have popularity below roughly 5,000, so they fall outside the top-5,000 crawl. Real gap: about 20 of 1,040 entries (2%), and only 7 of the 52 carry a score.

**What failed**
- I built the MyAnimeList import on Jikan, and it timed out. I first called it a network problem. It was not: the Jikan public API was discontinued on 1 October 2026 (maintainer announcement, jikan-rest issue 613). I had written a parser against a response shape I never saw.
- Lesson: before writing a parser for a third-party API, make one real request and look at the raw response. After that I tested the official MAL API with one curl call, saw the real shape, and only then wrote the parser, with a test built from the real response.

**What surprised me**
- A client ID alone is enough to read another person's public MyAnimeList list. I had read conflicting claims that it needs OAuth, so I tested it instead of assuming.
- Most of the "missing" titles are not a coverage bug. They are shorts and specials I chose not to index.

**Still open**
- The exact reply for a private MAL list (assumed HTTP 403) and for a private AniList list are untested.
- Whether shorts and specials should count as "seen" for the exclusion set (see DECISIONS).
- Collect 20-30 public usernames for the evaluation.

## 2026-11-22 (Day 4): embeddings, ChromaDB and first searches

**What I did**
- Installed chromadb and sentence-transformers. Embedded all 4,869 titles with two models, MiniLM (all-MiniLM-L6-v2) and BGE-small (bge-small-en-v1.5), one ChromaDB collection per model, cosine distance. Build time on CPU: MiniLM about 72-79s, BGE about 176-184s.
- Wrote a search CLI (`search.py`) with filters for episode cap, minimum year and minimum score, and an inspect tool (`inspect_title.py`) that prints the exact embedded text and nearest neighbours of a title. 37 tests pass.
- Built four variants of the embedding text, each in its own collection: `base` (title, genres, themes, synopsis), `notitle`, `spoilers` (adds spoiler-flagged tags as "Plot elements"), `notitle_spoilers`.
- Truncation (share of texts over the model's input limit):

  | Variant | MiniLM (256 tokens) | BGE (512 tokens) |
  |---|---|---|
  | base | 7.7% (375) | 0.1% (7) |
  | notitle | 6.5% (315) | 0.1% (5) |
  | spoilers | 8.8% (427) | 0.1% (7) |
  | notitle_spoilers | 7.2% (350) | 0.1% (6) |

  Longest text: 802 tokens (792 without the title).

**What failed or was wrong**
- My hypothesis that a "Time Loop" theme tag explained why the query "time loop thriller" worked was wrong. Summer Time Rendering's embedded themes were only Urban Fantasy, Super Power, Coastal, Rural, Shounen, Religion. The query had matched the title word "Time" plus the Thriller genre. I overstated how well that query worked.
- Cause: AniList flags Time Loop (rank 95), Survival (100) and Time Manipulation (97) as spoiler tags on that show, and my fetch cleaning removed every spoiler-flagged tag. The most informative tags were thrown away. Fix: a separate `spoiler_tags` field, used for retrieval only.
- I predicted the `spoilers` variant would fix the query "a student keeps waking up on the same morning and tries to prevent a tragedy". It did not: both models still return school shows (Assassination Classroom, Danganronpa, Classroom of the Elite).
- I called Mushi-Shi: The Next Passage 2 an "oddity" in the time-loop results. It was a true match: its synopsis mentions time loops and its plot elements include Time Loop (one episode in an episodic show).
- A Python shell started outside the project terminal gave `ModuleNotFoundError: chromadb`. Use the terminal whose prompt starts with `(.venv)`, from the project folder.

**What surprised me**
- Title leakage is real. For "a girl relives the same day", 6 of MiniLM's top 10 had "Girl" in the title. For neighbours of Summer Time Rendering, 7 of 10 (MiniLM) and 4 of 10 (BGE) had "Summer" in the title. Rewording the query without "girl" removed every "Girl" title from MiniLM's results. The `notitle` variant also removes them, but leaves unrelated slice-of-life shows underneath.
- BGE is clearly better than MiniLM on mood queries. "Slow-burn psychological mystery": BGE had PSYCHO-PASS 3, Dusk Maiden of Amnesia and Monster in its top 10; MiniLM was about 1 of 5 on target (Fire Force first). The two share 0 of 10. These are my own by-eye judgments, not measurements.
- Both models fail on plot-mechanic paraphrases ("a young woman is trapped repeating one day", "student keeps waking up on the same morning"): they latch onto topic words (student leads to school shows).
- "Something like Steins;Gate but shorter" returns the Steins;Gate franchise. "Shorter" is ignored by embeddings. "Like X" needs seed-title vectors, and "shorter" belongs in the episode filter.
- BGE distances are nearly tied at the top (several results at 0.399-0.400), so order within the top 10 is close to arbitrary. Distances are not comparable across models (BGE 0.29-0.32 vs MiniLM 0.54-0.61 for similar quality).
- Query "time loop" across variants (by my read of each top 10):
  - `base`: title junk (91 Days: Shoal of Time, Slow Loop, Golden Time, Time of Eve). BGE about 4 of 10 on target.
  - `spoilers`: BGE improves (Summer Time Rendering, When They Cry Kai, GNOSIA appear), roughly 6 of 10; title junk remains. MiniLM barely changes.
  - `notitle_spoilers`: best for both models, roughly 8 of 10 (ALL YOU NEED IS KILL, GNOSIA, Sagrada Reset, Link Click, Steins;Gate, Steins;Gate 0, The Girl Who Leapt Through Time, Terminator Zero).
  - Cost of dropping the title: 7th Time Loop, a genuine loop show, falls out of MiniLM's top 10.
- Item-to-item neighbours are far better than free text. Mushi-Shi neighbours: its own franchise first, then To Your Eternity (two seasons), good. Summer Time Rendering appears at #6 only because of a shared tag, which is a loose link. The embedding cannot see tag strength: Time Loop is rank 95 on Summer Time Rendering but only one episode of Mushi-Shi.
- The episode cap works: no result shows "?" episodes under a cap. The year filter works too.
- Franchise duplicates sit next to each other in results (Restaurant to Another World 1 and 2, Wagnaria!!2 and !!3, three Mushi-Shi entries). That is what the diversity pass is for.

**What I changed**
- Tag stoplist for the embedding text only ("Primarily Teen Cast", "Male Protagonist", "Ensemble Cast", "Heterosexual" and similar). Resolves the Day 2 open question. Tags in `anime.json` are untouched.
- `spoiler_tags` field (spoiler-flagged tags ranked 60 or higher), re-cleaned from the saved raw file with no recrawl. Used only in the `spoilers` variants.
- Embedding variants with a `--variant` option in `build_index.py`, `search.py` and `inspect_title.py`.
- Embedding text puts title, genres and themes first and the synopsis last, so truncation cuts the synopsis tail, not the genres.

**Open questions (week 3 ablations, decide with Recall@10, not by eye)**
- Title in the embedding text vs not. Provisional default for dense retrieval: `notitle_spoilers`, with BM25 over titles for name lookups.
- Spoiler tags in or out, and a minimum tag rank for them (60 vs 80), or use tag rank as a weight in BM25 or reranking.
- MiniLM vs BGE on Recall@10. BGE looks better by eye on mood queries.
- BM25 over tags and titles, a cross-encoder reranker, and LLM query rewriting ("relives the same day" to "time loop") as a stretch.
- Do any Monogatari TV seasons end up alone (Day 3 check, still pending)?
- Commit the Chroma index to the repo or rebuild at startup?

## 2026-11-21 (Day 3): full crawl and franchise grouping

**What I did**
- Crawled the top 5,000 titles by popularity (TV, MOVIE, ONA, OVA; adult titles excluded). The crawl is resumable and saves each page as it arrives.
- Data report: 16 null idMal, 5 empty synopsis, 126 null episodes, 126 null score, 210 null year. 139 synopses (2.8%) are over ~190 words, so MiniLM truncates only a few.
- Grouped titles into franchises with union-find over AniList relation links: 3,050 franchises, 815 with 2+ titles.
- Status counts: 4,810 FINISHED, 126 NOT_YET_RELEASED, 62 RELEASING, 2 CANCELLED.
- Index file `data/anime.json`: 4,869 titles (dropped 126 unreleased, 2 cancelled, 5 with no synopsis; 2 overlapped).

**What failed**
- v1 linked PARENT and SIDE_STORY along with sequels. It merged all of Gundam (50 titles, 7+ separate timelines) under the 1979 original. Both Wing and Seed point PARENT to "Mobile Suit Gundam", and the original points SIDE_STORY back, so dropping PARENT alone would not have worked.
- v2 let standalone titles attach to a single parent. That pulled lone TV shows (G Gundam, Turn A, AGE, Gundam X, Reconguista in G, GQuuuuuuX) back into the original's group.
- v2 also detached four Pokemon movies (Darkrai, Giratina, Arceus, Zoroark). They form their own movie chain (sequel/prequel to each other) with a PARENT link to Diamond and Pearl, so they were treated as an independent story.
- Final rule: sequel/prequel/summary always merge. Parent/side-story links only attach groups that contain no TV series. Result: Gundam UC is 26 titles, and Seed, 00, Wing, Witch from Mercury, Iron-Blooded Orphans, Build Fighters, Build Divers and the six lone TV shows are separate. Pokemon is back to 36.
- Replacing `franchise.py` with a new version silently removed my status filter. The summary line (still only 5 dropped) gave it away. Check the summary numbers after every rewrite.

**What surprised me**
- The 126 titles with no score and no episodes were all unreleased (Dandadan 3rd Season, Frieren 3rd Season, Infinity Castle Part 2). I did not request `status` in the first crawl, so I had to add it and recrawl.
- AniList uses PARENT for both "movie of this show" and "timeline under the original". Structure alone cannot tell them apart; format (TV vs not) is the best signal I found.
- Franchise sizes are computed before the status filter, so unreleased sequels still link groups and the "largest franchises" numbers do not change when the filter is added.
- Pokemon (36), Conan (29), Dragon Ball Z (28) and Gundam UC (26) are flagged "suspiciously large" but are legitimately long single continuities.

**Open questions**
- Do any Monogatari TV seasons end up alone? (Known cost of the final rule: a TV season linked only by PARENT/SIDE_STORY is not merged.)

## 2026-11-20 (Day 2): first AniList fetch

**What I did**
- Fetched the top 50 anime by popularity from AniList and saved raw and cleaned JSON.
- Null idMal: 0. Empty synopsis: 0 (expected for popular titles; the full crawl showed 16 and 5).
- Added a regex to strip "(Source: ...)" and "[Written by ...]" credit lines from synopses, plus a test. 0 credit lines remain. 4 tests passed.

**What surprised me**
- Tags are noisy. Attack on Titan is tagged "Kaiju". Tags like "Primarily Teen Cast" describe structure, not theme.
- The top 50 are only TV and MOVIE formats. Popularity sorting hides OVA, ONA, SPECIAL, etc.
- Only One Piece has null episodes (ongoing). Episode-limit filters need a rule for null.
- Synopses: min 46, median ~104, max 259 words. MiniLM truncates around 190 words, so only the longest get cut.
- Many synopses ended with "(Source: ...)" credit lines, which are noise for embeddings.
- In the Python shell, a `for` loop needs a blank line after it before the next statement.

**What I changed**
- Credit-line regex in `clean_text`, with a test.
