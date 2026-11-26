"""AniMatch AI: a small web page for trying the recommender on a public watchlist.

Run from the project root, in the terminal whose prompt starts with (.venv):
    streamlit run app.py
"""
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
RUNTIME = ROOT / "app_data"                                  # written by src/export_runtime.py
PROFILE_CACHE = Path(tempfile.gettempdir()) / "animatch_profiles"

import streamlit as st  # noqa: E402

import explain  # noqa: E402
import parsers  # noqa: E402
import recommend  # noqa: E402
import scoring  # noqa: E402

st.set_page_config(page_title="AniMatch AI", layout="wide")


@st.cache_resource(show_spinner="Loading the anime index (first time only)...")
def load_catalog():
    index = parsers.load_index(RUNTIME / "anime.json")
    catalog = scoring.Catalog(index["by_anilist"].values())
    matrices = scoring.runtime_matrix_loader(catalog, RUNTIME)
    matrices("minilm", "base")      # read the stored vectors once, not on every click
    matrices("genre", "base")
    records = list(index["by_anilist"].values())
    return (index, catalog, matrices, recommend.entry_points(records), recommend.retelling_links(records),
            explain.Facts(records), explain.load_spoiler_terms(RUNTIME / "spoiler_tags.json"))


@st.cache_data(ttl=3600, show_spinner="Reading the watchlist...")
def load_profile(platform, username):
    return parsers.get_profile(platform, username, cache_dir=PROFILE_CACHE)


def render(recs, notes=None):
    if not recs:
        st.info("Nothing matches these limits. Try loosening them.")
    for i, r in enumerate(recs, 1):
        with st.container(border=True):
            pic, text = st.columns([1, 5])
            if r["cover"]:
                pic.image(r["cover"], width=90)
            eps = f"{r['episodes']} eps" if r["episodes"] > 0 else "? eps"
            year = r["year"] if r["year"] > 0 else "?"
            score = f" · community score {r['score']}" if r["score"] >= 0 else ""
            text.markdown(f"**{i}. {r['title']}**")
            text.caption(f"{year} · {r['format']} · {eps}{score}")
            if r["planned"]:
                text.markdown(":blue[On your plan-to-watch list]")
            if notes:
                n = notes[i - 1]
                text.markdown(explain.render(n["text"]))
                text.caption("Written by a language model from genres and tags, and checked"
                             if n["source"] == "model" else "Plain explanation from genres and tags")
            else:
                text.markdown("Because you liked: " + ", ".join(t for t, _ in r["because"]))


st.title("AniMatch AI")
st.caption("Recommendations from your public AniList or MyAnimeList, with no spoilers. "
           "Nothing you enter is stored beyond a short-lived cache of your list. "
           "Anime data and cover images are from AniList.")

with st.sidebar:
    st.header("Your list")
    platform = st.selectbox("Site", ["anilist", "mal"], format_func=lambda p: {"anilist": "AniList", "mal": "MyAnimeList"}[p])
    username = st.text_input("Username").strip()
    if st.button("Reload my list"):
        load_profile.clear()
        if username:
            try:
                parsers.get_profile(platform, username, refresh=True, cache_dir=PROFILE_CACHE)
            except parsers.ProfileError:
                pass            # the same error is shown below
    st.header("Style")
    pop = st.slider("Popularity weight", 0.0, 2.0, 0.5, 0.1,
                    help="0 = taste only. Higher = lean towards what most people watch. "
                         "On held-out favourites, a blend of 0.5 beat plain popularity on less popular titles.")
    compare = st.toggle("Compare with a 'deeper cuts' list", help="Second list leaves out the most popular titles.")
    skip = st.slider("Deeper cuts: leave out the most popular", 0, 1000, 300, 100) if compare else 0
    st.header("Limits")
    max_eps = st.number_input("Max episodes (0 = no limit)", 0, 2000, 0)
    min_year = st.number_input("Released in or after (0 = any)", 0, 2100, 0)
    min_score = st.slider("Minimum community score", 0, 100, 0)
    hide_planned = st.checkbox("Also hide my plan-to-watch titles")
    keep_retellings = st.checkbox("Keep other versions of things I have seen")
    k = st.slider("How many", 5, 30, 10)
    llm = explain.get_llm_config()
    use_ai = False
    if llm:
        use_ai = st.checkbox("Short explanations", help="Each pick gets one sentence built from genres and tags "
                                                        "it shares with your favourites. A language model writes "
                                                        "it, a checker verifies it, and a plain template is used "
                                                        "if the check fails. Titles you liked are sent to the "
                                                        "model provider to write it.")

if not username:
    st.write("Enter a username in the sidebar to start.")
    st.stop()

try:
    entries = load_profile(platform, username)
except parsers.UserNotFound:
    st.error(f"No user called \"{username}\" on {platform}. Check the spelling.")
    st.stop()
except parsers.PrivateList:
    st.error("This list looks private. Make it public and try again.")
    st.stop()
except parsers.ProfileError as e:
    st.error(f"Could not read the list: {e}")
    st.stop()

index, catalog, matrices, starts, links, facts, spoiler_terms = load_catalog()
matched, unmatched = parsers.match_to_index(entries, index)
st.write(f"{len(entries)} entries, {len(matched)} in the index, {len(unmatched)} not (ignored).")

limits = dict(max_episodes=max_eps or None, min_year=min_year or None, min_score=min_score or None)
spec = recommend.build_spec(pop=pop)


def run(skip_popular):
    return recommend.recommend(matched, catalog, matrices, spec, k, limits, starts, hide_planned,
                               skip_popular, None if keep_retellings else links)


model, recs = run(0)
if model is None:
    st.warning("Nothing positive to learn from yet. Rate a few titles 8 or higher, or finish some, and try again.")
    st.stop()

with st.expander("What the recommender learned about you"):
    pos, neg = model.positives, model.negatives
    rated = sum(1 for o in pos if o.rated)
    st.write(f"Built from {len(pos)} liked franchises ({rated} rated, {len(pos) - rated} unrated) "
             f"and {len(neg)} disliked.")
    st.write("Strongest likes: " + ", ".join(o.title for o in pos[:6]))
    if neg:
        st.write("Strongest dislikes: " + ", ".join(o.title for o in neg[:5]))

def notes_for(recs):
    """Explanations for a list, or None when they are off (or cannot be checked)."""
    if not (use_ai and spoiler_terms is not None and recs):
        return None
    try:
        notes, error = explain.explain_picks(recs, facts, spoiler_terms, llm)
    except Exception as e:             # an optional feature must not take the page down
        st.caption(f"Explanations are unavailable right now ({e.__class__.__name__}); showing the plain list.")
        return None
    if error:
        st.caption(f"The language model was not used this time ({error}); showing plain explanations.")
    return notes


if compare:
    left, right = st.columns(2)
    with left:
        st.subheader("Your list")
        render(recs, notes_for(recs))
    with right:
        st.subheader(f"Deeper cuts (no top {skip})")
        deeper = run(skip)[1]
        render(deeper, notes_for(deeper))
else:
    render(recs, notes_for(recs))
