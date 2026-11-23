# Evaluation results

Aggregate numbers only: no usernames and no list data. Each file is the output of one
`python src/evaluate.py ... --save <name>` run. The methods compared in a run are listed
inside its file. How to read them: see "Evaluation protocol (Step 7)" in `../DECISIONS.md`.

## Dev set (25 users, `data/eval_users`), used to compare methods

Run in this order, on 2026-11-24. Some of the choices below were made by looking at these
results, so the numbers are optimistic for the method that won.

| File | What it compares |
|---|---|
| `main.json` | First comparison: random, popularity, genre/tag, MiniLM, BGE |
| `ablation_weights.json` | Variations of the taste weights |
| `ablation_index.json` | Variations of the index text (title, spoilers) |
| `fix_dense.json` | Vector centering and nearest-liked scoring |
| `hybrid.json` | Tags and popularity prior fused with embeddings |
| `tail300.json`, `tail1000.json` | Same comparison with the 300 / 1000 most popular franchises removed |
| `dev_tail300.json`, `dev_tail1000.json` | Tail views re-run on the dev users after the test users were sampled |

## Test set (25 different users, `data/test_users`), run once

Configuration chosen and written down in `DECISIONS.md` before these runs:
`minilm@center=1@knn=10@genre=1@pop=0.5`, compared with popularity and the genre/tag score.
Result: partial (clearly better with the top 1000 removed, not clearly different with the
top 300 removed or on all franchises). Do not tune against these users again.

| File | View |
|---|---|
| `test_tail1000.json` | 1000 most popular franchises removed |
| `test_tail300.json` | 300 most popular franchises removed |
| `test_all.json` | All franchises |
