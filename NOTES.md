# Project notes

A running log of what I tried, what surprised me, and what I changed.
Newest entries go at the top. This becomes the "What failed and what I'd change" section of the README.

## 2026-11-20 (Day 2): first AniList fetch

**What I did**
- Fetched the top 50 anime by popularity from AniList and saved raw and cleaned JSON.
- Null idMal: 0. Empty synopsis: 0 (expected for popular titles; recheck on the full crawl).

**What surprised me**
- Tags are noisy. Attack on Titan is tagged "Kaiju". Tags like "Primarily Teen Cast" describe structure, not theme.
- The top 50 are only TV and MOVIE formats. Popularity sorting hides OVA, ONA, SPECIAL, etc.
- Only One Piece has null episodes (ongoing). Episode-limit filters need a rule for null.
- Synopses: min 46, median ~104, max 259 words. MiniLM truncates around 190 words, so only the longest get cut.
- Some synopses end with "(Source: ...)" credit lines. These are noise for embeddings.

**What I changed**
- Added a regex to strip "(Source: ...)" and "[Written by ...]" from synopses.

**Open questions**
- Which formats go in the index? (default: TV, MOVIE, ONA, OVA)
- How much of the long tail has null idMal or empty synopses?
- Do I need a tag stoplist for the embedding text?
