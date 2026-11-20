# Project notes

A running log of what I tried, what surprised me, and what I changed.
Newest entries go at the top. This becomes the "What failed and what I'd change" section of the README.

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
- Tag stoplist for the embedding text ("Primarily Teen Cast", "Male Protagonist" describe structure, not theme).
- Commit the Chroma index to the repo or rebuild at startup?

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
