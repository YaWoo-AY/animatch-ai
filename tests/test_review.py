import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import review_explanations as rv  # noqa: E402
from explain import Facts  # noqa: E402
from test_explain import RECORDS, PICK  # noqa: E402

FACTS = Facts(RECORDS)


def results(source):
    return [{"text": "Shares {{School}} with [[Liked One]].", "source": source, "problems": []}]


def test_rows_hold_the_facts_and_a_blank_label():
    rows = rv.make_rows("u1", [PICK], results("model"), FACTS)
    assert len(rows) == 1
    r = rows[0]
    assert r["pick"] == "Pick" and r["top_similarity"] == "0.50" and r["label"] == ""
    assert "Liked One" in r["liked_titles"] and "School" in r["shared_attributes"]
    assert "[[" not in r["sentence"] and "{{" not in r["sentence"]


def test_template_sentences_are_left_out_unless_asked_for():
    assert rv.make_rows("u1", [PICK], results("template"), FACTS) == []
    assert len(rv.make_rows("u1", [PICK], results("template"), FACTS, include_templates=True)) == 1


def test_csv_round_trip_keeps_commas_quotes_and_accents(tmp_path):
    rows = rv.make_rows("u1", [PICK], results("model"), FACTS)
    rows[0]["sentence"] = 'Like "Kimi, no Na wa" — é'
    rows[0]["label"] = "a"
    path = tmp_path / "x" / "r.csv"
    rv.write_csv(rows, path)
    back = rv.read_rows(path)
    assert back[0]["sentence"] == rows[0]["sentence"] and back[0]["label"] == "a"


def test_summary_counts_labels_by_source_and_similarity_and_flags_bad_rows():
    rows = [
        {"label": "A", "source": "model", "top_similarity": "0.40"},
        {"label": "c", "source": "model", "top_similarity": "0.20"},
        {"label": "B", "source": "model", "top_similarity": "0.25"},
        {"label": "", "source": "model", "top_similarity": "0.30"},
        {"label": "X", "source": "model", "top_similarity": "0.30"},
        {"label": "C", "source": "template", "top_similarity": ""},
    ]
    s = rv.summarize(rows)
    assert s["labelled"] == 4 and s["unlabelled"] == 1 and s["invalid_labels"] == 1
    assert s["overall"] == {"A": 1, "C": 2, "B": 1}
    assert s["by_similarity"]["below 0.25"] == {"C": 1}
    assert s["by_similarity"]["0.25 or above"] == {"A": 1, "B": 1}
    assert s["by_similarity"]["no similarity"] == {"C": 1}
    assert s["by_source"]["template"] == {"C": 1}
    assert rv.share(s["by_source"]["model"], "C") == round(1 / 3, 3)


def test_share_of_nothing_is_none():
    assert rv.share({}, "C") is None
