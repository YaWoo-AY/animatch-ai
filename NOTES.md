# Project notes

A running log of what I tried, what surprised me, and what I changed.
Newest entries go at the top. This becomes the "What failed and what I'd change" section of the README.

## 2026-11-26 (Step 11): README, explanation review kit, repo hygiene

**What I did**
- Reworked the README: result stated up front, not-affiliated line, licence, and the limits split into one point each.
- Added `src/review_explanations.py` to export the model-written sentences to a CSV under `data/` for hand labelling (A useful, B generic but true, C misleading) and to summarize the labels by source and by how close the nearest liked title was.
- Added an MIT licence, `.gitignore`, and a CI workflow that runs the tests and fails if `data/` or `.env` is ever tracked.

**Open:** the hand review is not done yet. Its result decides whether picks with a weak resemblance (nearest liked title below 0.25) need a "loose match" label.

## 2026-11-26 (Step 10): a quality prior, tested on a second set of users

**What I did**
- My own list showed picks with community scores in the 50s and low 60s, so I added a community-score term to the ranking (`@qual=w`), plus a baseline that ranks by community score alone.
- Dev users, first pass: more quality weight kept helping, up to the largest value I tried, which is not a place to stop. Second pass with larger weights: an optimum around 1.5 to 2 in the all-franchises view. I wrote down one configuration (`@qual=2`) and the rule for adopting it, committed that, then ran it once on 25 new users.

**Result on the new users:** passes the rule. Better than the previous default in two of three views (top 300 removed: +3.9 pts; all franchises: +7.4 pts), not clearly different with the top 1000 removed. Compared with popularity: a tie on the full catalog (15.8% each) and clearly better with the top 300 or 1,000 removed. Compared with community score alone: clearly better in two views. Full table in DECISIONS.

**What surprised me**
- The old default was clearly worse than popularity on the full catalog on both sets of users. The quality prior closed the gap exactly (15.8% against 15.8%), which looks too neat; the interval is wide (-4.2 to +3.8 pts).
- Community score alone is a strong baseline: with the top 1000 removed it did as well as the old default.

**Caveats**
- The test rewards a quality prior by construction: the hidden title is one the user rated highly, and community score predicts personal ratings. It does not show the picks are more enjoyable.
- I have not looked at the live list since the change. The point was to remove low-scored picks; I should check that it did.
- The weight was chosen from several values on 25 users, which overstates it. The second set is independent of that choice and is now spent.
- Four users dropped out of the top-1000 view for having no liked titles outside the most popular 1,000.

**Next**
- Check the live list on my own profile and on one or two others.
- Update the README table and the hosted page.

## 2026-11-26 (Step 9): checked language-model explanations

**What I did**
- `explain.py`: one sentence per pick from a hosted model (Groq's free tier, `openai/gpt-oss-120b`). The model sees only the pick's title, format, year, episode count, visible genres and tags, and the attributes it shares with the liked titles that led to it. No synopsis and no spoiler-flagged tags. It must put titles in `[[ ]]` and genres or tags in `{{ }}`, so every claim can be checked by code.
- The checker rejects a sentence that names no liked title, names a title that is not in the facts, mentions any other catalog title, cites an attribute that is not in the pick's list, contains one of the pick's spoiler-flagged tags, uses a number other than the year or episode count, or is over 40 words. A rejected sentence, a bad reply, a rate limit or a missing key all give a template built from the same facts, which passes the same check (tested). The page never shows an unchecked sentence.
- `eval_explanations.py` measures how often sentences pass, with calls and sentences counted separately.

**Results (10 of the 25 dev users, 10 picks each, one model call per user)**
- Final run: 10 of 10 calls worked and 98 of 100 sentences passed the check. The two failures were one sentence containing a spoiler-flagged term and one citing an attribute that was not in the supplied list. Both were replaced by the template.
- Earlier runs, kept because they changed the code: 19 of 25 calls failed on the first full run (12 rate limits, 7 unusable replies), which left 190 of 250 picks with no sentence. After pausing longer, 3 of 10 calls were cut off at the token limit.

**What failed or was wrong**
- My first summary mixed provider failures into the sentence pass rate and reported 22.4%. The sentences were fine; the provider calls were failing. The tally now keeps the two apart.
- My checker flagged digits inside verified titles and tags (a liked title like Mob Psycho 100), so it rejected correct sentences. Fixed, with tests.
- The token cut-offs went away after I set the model's reasoning effort to low, which is the likely cause (the model's thinking shares the reply's token budget). I changed that and the checker in the same step, so I cannot say how much each contributed to the final pass rate.

**What surprised me**
- One sentence contained a spoiler-flagged term even though the model was never shown it. It must have come from the model's own knowledge of the show. The check caught it only because that term is in my spoiler-tag list.

**Caveats**
- The check limits what the model can name. It does not check tone or whether a sentence helps: words like "intense" or "vibrant" pass. A spoiler that is not in my spoiler-tag list would also pass.
- 100 sentences from 10 users is a small sample, and these are the development users, not the held-out ones.
- A free tier is shared by everyone who uses the hosted page, so it will often fall back to the template.
- The liked titles of the person using the page are sent to the model provider.

**Next**
- Read a sample of passing sentences by hand and note how many are misleading or empty, since the pass rate cannot show that.
- Decide whether to turn the checkbox on for the hosted page.

## 2026-11-25 (Step 8): live recommender, demo page and first deployment

**What I did**
- Moved the scoring from `evaluate.py` into `scoring.py` so the live recommender and the evaluation run the same code. The live default is the configuration chosen before the test run (`minilm@center=1@knn=10@genre=1@pop=0.5`), with the popularity weight and a "leave out the N most popular" option as settings.
- Ran it on my own list. The recommendations are plausible, but one franchise problem showed up: the 2012 OVA Hori-san to Miyamura-kun came first in one view, although I had watched Horimiya, the same story. `franchise_of.py` showed AniList links the two only as `ALTERNATIVE`, which grouping never merges. Those links now hide the counterpart of anything I have seen, and are never used to merge franchises.
- Built a Streamlit page over `recommend()`: popularity slider, a "deeper cuts" list shown beside the main one, limits on episodes, year and community score, and "because you liked" lines. No synopsis is shown.
- Deployed it. The first deployment failed with a file-not-found error, because `data/` is git-ignored (it also holds the sampled users) and the host had no index and no vectors. Fixed by exporting a small bundle (`app_data/`: index without synopses or spoiler tags, plus the vectors), so the page needs neither ChromaDB nor an embedding model.

**What I saw on my own list**
- Cyberpunk: Edgerunners 2 is recommended to someone who rated the first series highly. AniList links them only as `OTHER`, so grouping does not merge them. I left this as it is; it looks useful.
- Black Clover leads the default list but its listed reasons are weak. My reading, not tested: the popularity prior is pushing it up.
- With popularity weight 0, titles with community scores in the 50s and low 60s appear. Nothing in the score knows about quality.

**What failed**
- The deployment, as above. Lesson: run the page from a clean checkout before deploying, because anything git-ignored is missing there.
- I guessed that Edgerunners 2 was a sequel the grouping had missed. The relation data said otherwise, which is why I checked before changing anything.

**Next**
- README with the evaluation table, the partial result stated plainly and the caveats.
- Grounded, spoiler-free explanations, with a check that they only use supplied facts.

## 2026-11-24 (Step 7, test run): result of the pre-registered configuration

**What I did**
- Committed the choice and the success rule, then ran the chosen configuration once on the 25 test users against popularity and the genre/tag score, in three views.

**Result:** partial by the rule I set. Clearly better than popularity with the top 1000 removed (10.1% vs 0.2%, difference +9.8 pts, interval +3.0 to +19.9). With the top 300 removed, 7.7% vs 3.7%, difference +4.0 pts but the interval (-0.3 to +8.5) includes zero. On all franchises it scored 13.3% vs 15.2% for popularity, not clearly different (-1.9 pts, interval -9.1 to +5.2). Full table in DECISIONS.

**What I think it means (interpretation, not tested)**
- Taste signal exists beyond popularity: the first tail view shows it, and the top-300 estimate points the same way. It is not yet strong enough to claim, and the first tail view rests on 217 hidden franchises, many from a few users.
- Dev and test agreed in direction in both tail views, but the sizes differ (7.5% vs 10.1% at top 1000), which is the expected spread with this much data.
- The all-franchises result is better than I expected (I expected popularity to stay clearly ahead), but the interval is wide enough to include a 9-point deficit.

**Caveats**
- 22 users contribute to the top-1000 view (3 skipped for having no liked titles outside the head), and some of them contribute one hidden franchise.
- Same proxy limits as before: hiding a title the user already watched is not the same as recommending something new they would enjoy.
- The test set is spent for this question.

**Next**
- Move this configuration into `taste.py` and `recommend.py`, check it on my own list, and keep the popularity weight as a visible setting. Whether the product should show one blended list or separate "popular picks" and "deeper cuts" is still open.

## 2026-11-24 (Step 7): sampling users and a leave-one-out evaluation

**What I did**
- Wrote `src/sample_users.py`: reads AniList's recent-activity feed, picks candidates at random, downloads each public list and keeps the user only if it has at least 100 titles in the index, 50 rated and 15 rated 8 or higher. It stores ids, statuses and scores only, as `u001.json` and so on, never usernames. AniList's terms forbid mass collection and the API was limiting to 30 requests a minute, so it runs slowly and stops at 25. Both queries worked first time under `--probe`.
- Sampled two sets of 25: a dev set (`data/eval_users`) to compare methods on, and a test set (`data/test_users`) that nobody touches until one configuration has been chosen. The test set skips everyone in the dev set.
- Wrote `src/evaluate.py`. For each user it hides franchises they rated 8 or higher (`--per-user` sets how many) (whole franchise, so no sibling season leaks it), builds the recommendation from the rest, hides everything else they have seen, and records the rank of the hidden franchise among all remaining franchises. Recall@10 is averaged per user, then across users; intervals come from resampling users. Baselines: random, popularity, and a genre/tag vector run through the same pipeline.
- Added options to compare ideas without code changes: `@center` (subtract the average vector), `@knn` (score by the nearest liked franchises), `@pop` (popularity prior), `@genre` (fuse tags with embeddings), plus `--tail N` (drop the N most popular franchises from the candidates).

**Results on the dev set (Recall@10)** (exact figures for every method are in `results/*.json`; I am quoting only the ones I have checked)
- All franchises: popularity 19.7%. Nothing I tried beat it on this view.
- Uncentered embedding cosine was about as good as random, and worse than random once the popular titles were removed. Centering the vectors helped.
- Removing the 300 most popular franchises: the genre/tag vector beat the text embeddings.
- Removing the 1000 most popular: centered nearest-liked scoring beat popularity, but the result swung with K (8.1%, 2.0%, 4.6% for three values of K), and the K that did best here scored only 3.2% with the top 300 removed.
- Fusing tags with embeddings and adding a popularity prior of 0.5 (`minilm@center=1@knn=10@genre=1@pop=0.5`) gave 7.5% (95% interval 2.1 to 17.2) with the top 1000 removed, +6.3 points over popularity, and 7.3% with the top 300 removed, +2.7 points (interval +0.6 to +5.0).

**What surprised me**
- Popularity alone puts about 1 in 5 hidden favorites in the top 10. The hidden item is something the user chose to watch, and people mostly watch popular things, so the main view rewards recommending what everyone watches.
- Cosine to the average vector without centering was worse than random once the popular titles were removed. My reading (not tested directly): it favors "typical" titles, and the typical ones are the popular ones.
- Results moved a lot with K (8.1%, 2.0%, 4.6%), which is noise, not a trend. The winner among about a dozen settings on 25 users is partly luck.

**What failed or went wrong**
- My first taste vector scored far below popularity (19.7%). I had spent a day judging recommendations by eye on one profile (mine); the harness is what showed it.
- Two of my own tests were wrong (a popularity formula that was backwards, and a tie-breaking assumption). The code was right both times.

**Caveats**
- 25 users from an activity feed favor people who update often. The test is a proxy: finding things a user already watched and loved is not the same as recommending something new they would enjoy.
- The 1000 view has only 254 hidden franchises, and some users contribute only 1 to 3.

**Next**
- Run the chosen configuration once on the test set (see DECISIONS), then move it into `taste.py` and `recommend.py`.

## 2026-11-23 (Step 6): exclusion set and taste vector

**What I did**
- `src/taste.py`: hides every franchise the user has completed, is watching, paused or dropped (plan-to-watch titles stay eligible and are tagged). Each list entry gets a weight: ratings of 8 to 10 are positive, 6 to 7.9 ignored, 5 or lower negative; with no score, completed is a weak positive, dropped early is a negative and dropped past half the episodes a weaker one. A franchise counts once, however many seasons. Unrated titles may carry at most 35% of the liked weight. The vector is the liked centroid minus 0.3 times the disliked centroid.
- `src/recommend.py`: fetches a watchlist, builds the vector from vectors already stored in ChromaDB (no model needed), queries with the filters, drops seen franchises and prints each result with the liked franchises closest to it.

**What I saw on my own list**
- No seen franchise appeared in the results, and the MiniLM and BGE lists differed.
- It recommended later seasons and OVAs, because retrieval returns whichever title of a franchise is closest to the vector. Fixed by showing each franchise's first entry (TV before movie before ONA before OVA, then earliest year), skipped when that entry breaks the user's filters.
- Low-rated titles can appear because similarity has no quality signal; `--min-score` is a filter for that, not a fix.
- Setting the dislike weight to zero (`--alpha 0`) changed only some of the results.

**Lesson**
- Judging one profile by eye cannot choose between these explanations. That is why Step 7 builds an evaluation.

## 2026-11-22 (Step 5): reading watchlists from AniList and MyAnimeList

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

## 2026-11-22 (Step 4): embeddings, ChromaDB and first searches

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
- Tag stoplist for the embedding text only ("Primarily Teen Cast", "Male Protagonist", "Ensemble Cast", "Heterosexual" and similar). Resolves the Step 2 open question. Tags in `anime.json` are untouched.
- `spoiler_tags` field (spoiler-flagged tags ranked 60 or higher), re-cleaned from the saved raw file with no recrawl. Used only in the `spoilers` variants.
- Embedding variants with a `--variant` option in `build_index.py`, `search.py` and `inspect_title.py`.
- Embedding text puts title, genres and themes first and the synopsis last, so truncation cuts the synopsis tail, not the genres.

**Open questions (week 3 ablations, decide with Recall@10, not by eye)**
- Title in the embedding text vs not. Provisional default for dense retrieval: `notitle_spoilers`, with BM25 over titles for name lookups.
- Spoiler tags in or out, and a minimum tag rank for them (60 vs 80), or use tag rank as a weight in BM25 or reranking.
- MiniLM vs BGE on Recall@10. BGE looks better by eye on mood queries.
- BM25 over tags and titles, a cross-encoder reranker, and LLM query rewriting ("relives the same day" to "time loop") as a stretch.
- Do any Monogatari TV seasons end up alone (Step 3 check, still pending)?
- Commit the Chroma index to the repo or rebuild at startup?

## 2026-11-21 (Step 3): full crawl and franchise grouping

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

## 2026-11-20 (Step 2): first AniList fetch

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
