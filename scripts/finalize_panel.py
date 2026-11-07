"""
Step 2. Apply the inclusion rules to the resolved venue lists.

No venue is included or excluded by judgement. Each rule is applied in order, and
every excluded venue is written out with the rule that removed it.

  R0  Drop the Theoretical Computer Science category: eight of its twenty venues are
      conferences, and its remaining journals are not engineering journals.
  R1  Drop conferences. Their review models, page limits and revision conventions
      differ from journals'.
  R2  Drop venues that could not be resolved to an OpenAlex journal (step 1), since
      no article data can be collected for them.
  R3  A journal listed in more than one category is kept once. Its h5-index is the
      same in every listing, so it cannot decide; the journal is assigned to the
      category in which Scholar ranks it highest (smallest rank), and a tie on rank
      goes to the category name that sorts first. The other categories are recorded.

The result does not depend on the order of the input rows or on library versions:
rerunning the script on a shuffled input gives an identical panel (tested in
tests/test_pipeline.py).

Cohort (whether a journal published in every year of the window) and abstract
coverage are measured on the harvested data in build_corpus.py, not here.

Outputs: data/journal_panel_final.csv, data/panel_exclusions.csv
"""

import sys

import pandas as pd

import repo
from resolve_venues import is_conference

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DROPPED_CATEGORIES = {
    "Theoretical Computer Science":
        "category dropped (rule R0): 8 of 20 venues are conferences and the "
        "journals are not engineering journals",
}

PANEL_COLUMNS = [
    "venue", "openalex_name", "publisher", "issn_l", "openalex_id",
    "scholar_category", "scholar_rank", "also_listed_in", "n_categories",
    "h5_index", "h5_median", "match", "abstract_coverage_pct", "refs_coverage_pct",
    "works_window_total",
]


def build(raw):
    """Return (panel, exclusions) from the resolved venue table."""
    df = raw.copy()
    df["openalex_id"] = df["openalex_id"].fillna("").astype(str)
    df["scholar_rank"] = pd.to_numeric(df["scholar_rank"])
    df["status"], df["reason"] = "", ""

    for cat, reason in DROPPED_CATEGORIES.items():
        df.loc[df["scholar_category"] == cat, ["status", "reason"]] = ["excluded", reason]

    conference = (df["venue_type"] == "conference") | df["venue"].map(is_conference)
    df.loc[(df["status"] == "") & conference,
           ["status", "reason"]] = ["excluded", "conference venue (rule R1)"]

    df.loc[(df["status"] == "") & (df["openalex_id"] == ""),
           ["status", "reason"]] = ["excluded", "no OpenAlex journal record (rule R2)"]

    kept = df[df["status"] == ""].sort_values(
        ["openalex_id", "scholar_rank", "scholar_category"], kind="mergesort")
    first = ~kept.duplicated("openalex_id", keep="first")

    dupes = kept[~first]
    df.loc[dupes.index, ["status", "reason"]] = [
        "excluded", "listed in another category where it ranks higher (rule R3)"]

    panel = kept[first].copy()
    cats = kept.groupby("openalex_id")["scholar_category"]
    panel["also_listed_in"] = panel["openalex_id"].map(
        cats.apply(lambda s: "; ".join(sorted(set(s)))))
    panel["n_categories"] = panel["openalex_id"].map(cats.nunique())

    panel = (panel[PANEL_COLUMNS]
             .sort_values(["scholar_category", "scholar_rank"], kind="mergesort")
             .reset_index(drop=True))
    excl = (df[df["status"] == "excluded"]
            [["venue", "scholar_category", "scholar_rank", "venue_type", "h5_index", "reason"]]
            .sort_values(["scholar_category", "scholar_rank"], kind="mergesort")
            .reset_index(drop=True))
    return panel, excl


def main():
    raw = pd.read_csv(repo.SCHOLAR_RAW)
    panel, excl = build(raw)
    panel.to_csv(repo.PANEL, index=False, encoding="utf-8")
    excl.to_csv(repo.EXCLUSIONS, index=False, encoding="utf-8")

    print(f"venue pool : {len(raw)}")
    print(f"excluded   : {len(excl)}")
    for reason, n in excl["reason"].value_counts().items():
        print(f"   {n:4d}  {reason[:80]}")
    print(f"journals   : {len(panel)}")
    print(panel.groupby("scholar_category").size().to_string())
    print(f"\nwrote {repo.PANEL}\nwrote {repo.EXCLUSIONS}")


if __name__ == "__main__":
    main()
