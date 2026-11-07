"""
Offline tests: no network, no API key. Run with `python -m pytest tests` or
`python tests/test_pipeline.py`.

Each test pins down a defect found while the corpus was being built.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))

import build_corpus  # noqa: E402
import finalize_panel  # noqa: E402
import harvest  # noqa: E402
import repo  # noqa: E402
import resolve_venues  # noqa: E402


def test_abstract_rebuilt_in_order_and_words_counted():
    inv = {"the": [0, 4], "model": [1], "is": [2], "good": [3], "fit": [5]}
    text = harvest.reconstruct_abstract(inv)
    assert text == "the model is good the fit"
    row = harvest.flatten({"id": "https://openalex.org/W1", "abstract_inverted_index": inv})
    assert row["abstract_words"] == 6          # words, not distinct words (5)
    assert row["abstract"] == text             # kept for every journal


def test_page_counts():
    assert harvest.page_count("e00460", "e00460") is None      # article number
    assert harvest.page_count("117115", "117115") is None      # article number
    assert harvest.page_count("1", "6750000") is None          # metadata error
    assert harvest.page_count("101", "112") == 12
    got = build_corpus.pages(pd.Series(["5", "7", "1"]), pd.Series(["5", "20", "900"]))
    assert np.isnan(got[0]) and got[1] == 14 and np.isnan(got[2])


def test_name_matching_is_strict():
    n = resolve_venues.normalise
    assert n("Decision") != n("Decision Support Systems")
    assert n("Engineering, Construction and Architectural Management") == \
        n("Engineering Construction & Architectural Management")
    assert resolve_venues.is_conference("Proceedings of the IEEE/CVF Conference")
    assert not resolve_venues.is_conference("IEEE Transactions on Smart Grid")


def test_panel_is_deterministic_and_correct():
    raw = pd.read_csv(repo.SCHOLAR_RAW)
    a, ea = finalize_panel.build(raw)
    b, eb = finalize_panel.build(raw.sample(frac=1, random_state=3))
    pd.testing.assert_frame_equal(a, b)
    pd.testing.assert_frame_equal(ea, eb)
    ids = set(a["openalex_id"])
    assert "S11479521" not in ids             # Decision Support Systems is not "Decision"
    assert "S4210175686" in ids               # Decision (American Psychological Association)
    assert "S59604973" in ids                 # IEEE Transactions on Smart Grid
    assert "S120049674" in ids                # Engineering, Construction & Architectural Mgmt
    assert a["openalex_id"].is_unique
    assert len(a) == 256
    assert not ea["reason"].str.contains("R2").any()   # every journal venue resolved
    # A cross-listed journal sits in the category where Scholar ranks it highest.
    kept = raw[raw["openalex_id"].notna() & (raw["venue_type"] == "journal")
               & (raw["scholar_category"] != "Theoretical Computer Science")]
    best = kept.sort_values(["scholar_rank", "scholar_category"]).drop_duplicates("openalex_id")
    m = a.merge(best[["openalex_id", "scholar_category"]], on="openalex_id", suffixes=("", "_best"))
    assert (m["scholar_category"] == m["scholar_category_best"]).all()


def test_journal_measures():
    rows = []
    for year in range(2015, 2026):
        rows += [{"source_id": "A", "publication_date": pd.Timestamp(year, 6, 1),
                  "abstract": "x", "abstract_source": "openalex"}] * 20
        if year >= 2018:
            rows += [{"source_id": "B", "publication_date": pd.Timestamp(year, 6, 1),
                      "abstract": None, "abstract_source": None}] * 25
    jm = build_corpus.journal_measures(pd.DataFrame(rows))
    assert jm.loc["A", "cohort"] == "core" and jm.loc["B", "cohort"] == "emergent"
    assert jm.loc["A", "text_tier"] == "A" and jm.loc["B", "text_tier"] == "C"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok ", name)
