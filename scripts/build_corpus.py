"""
Step 4. Assemble the corpus from the harvest shards.

Run once after the harvest, then again after recover_abstracts.py to merge the
recovered abstracts.

What happens, in order:

  1  Load the shard of every journal in the panel (shards of journals no longer in
     the panel are ignored).
  2  Keep articles published 1 January 2015 to 31 December 2025.
  3  Remove machine translations. OpenAlex indexes automated translations of English
     articles as separate records: no DOI, a non-English language label and the
     translation service's name ("Powered by NICT") in the title. They duplicate
     articles already held.
  4  Remove other records that have no DOI and a non-English language label. Records
     with a DOI are kept whatever their language label says: the label is unreliable
     for mathematical text (over a thousand English articles carry a Latvian label).
  5  Derive the fields the analysis uses (month, period, author-country group, title
     form, pages).
  6  Merge abstracts recovered from Semantic Scholar where OpenAlex has none, recording
     the source of every abstract in `abstract_source`.
  7  Measure each journal on the data itself: `cohort` is "core" when the journal has
     at least 20 articles in every year 2015 to 2025, otherwise "emergent"; `text_tier`
     is A, B or C by the share of its articles with an OpenAlex abstract (at least 80
     percent, 30 to 80, below 30).

Outputs:
  data/corpus.parquet       one row per article (not distributed; rebuilt by the pipeline)
  data/article_ids.csv.gz   the identifiers of every article, distributed with the package
  data/journal_summary.csv  per-journal counts, cohort and abstract coverage
"""

import glob
import os
import sys

import numpy as np
import pandas as pd

import repo

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

WINDOW_START = pd.Timestamp("2015-01-01")
WINDOW_END = pd.Timestamp("2025-12-31")

# Public release of ChatGPT; December 2022 to February 2023 is a washout period.
INTERVENTION = pd.Timestamp("2022-11-30")
WASHOUT_START, WASHOUT_END = pd.Timestamp("2022-12-01"), pd.Timestamp("2023-02-28")
COVID_START, COVID_END = pd.Timestamp("2020-03-01"), pd.Timestamp("2021-06-30")

# Countries where English is the main language of higher education and research.
# Used only to stratify aggregate results by corresponding-author country.
ANGLOPHONE = {"US", "GB", "AU", "NZ", "IE", "CA"}

TRANSLATION_MARKER = "Powered by NICT"
MAX_PAGES = 200
CORE_MIN_PER_YEAR = 20


def load_shards(panel_ids):
    frames, missing = [], []
    for sid in panel_ids:
        path = os.path.join(repo.WORKS, f"{sid}.parquet")
        if os.path.exists(path):
            frames.append(pd.read_parquet(path))
        else:
            missing.append(sid)
    if missing:
        raise SystemExit(f"{len(missing)} panel journals have no shard (run harvest.py): "
                         f"{missing[:5]}")
    stray = {os.path.basename(p)[:-8] for p in glob.glob(os.path.join(repo.WORKS, "*.parquet"))}
    stray -= set(panel_ids)
    if stray:
        print(f"ignoring {len(stray)} shard(s) of journals not in the panel: {sorted(stray)}")
    return pd.concat(frames, ignore_index=True)


def pages(first, last):
    first = pd.to_numeric(first, errors="coerce")
    last = pd.to_numeric(last, errors="coerce")
    span = last - first + 1
    return np.where((last > first) & (first > 0) & (span <= MAX_PAGES), span, np.nan)


def merge_recovered(df):
    """Fill missing abstracts from recover_abstracts.py, never overwriting OpenAlex."""
    df["abstract_source"] = np.where(df["abstract"].notna(), "openalex", None)
    if not os.path.exists(repo.RECOVERED):
        print("no recovered abstracts yet (run recover_abstracts.py, then rebuild)")
        return df
    rec = pd.read_parquet(repo.RECOVERED, columns=["doi", "abstract", "abstract_source"])
    rec["doi"] = rec["doi"].astype(str).str.lower()
    rec = rec.drop_duplicates("doi").set_index("doi")
    need = df["abstract"].isna() & df["doi"].notna()
    found = df.loc[need, "doi"].map(rec["abstract"])
    hit = found.dropna().index
    df.loc[hit, "abstract"] = found.loc[hit]
    df.loc[hit, "abstract_source"] = df.loc[hit, "doi"].map(rec["abstract_source"])
    print(f"merged recovered abstracts: {len(hit):,}")
    return df


def journal_measures(df):
    years = range(WINDOW_START.year, WINDOW_END.year + 1)
    counts = (df.groupby(["source_id", df["publication_date"].dt.year]).size()
              .unstack(fill_value=0).reindex(columns=list(years), fill_value=0))
    s = pd.DataFrame(index=counts.index)
    s["n_articles"] = counts.sum(axis=1)
    s["min_year_count"] = counts.min(axis=1)
    s["cohort"] = np.where(s["min_year_count"] >= CORE_MIN_PER_YEAR, "core", "emergent")
    s["openalex_abstract_share"] = df.groupby("source_id")["abstract_source"].apply(
        lambda x: (x == "openalex").mean())
    s["abstract_share"] = df.groupby("source_id")["abstract"].apply(lambda x: x.notna().mean())
    s["text_tier"] = pd.cut(s["openalex_abstract_share"], [-1, 0.3, 0.8, 2],
                            right=False, labels=["C", "B", "A"]).astype(str)
    return s


def main():
    panel = pd.read_csv(repo.PANEL)
    df = load_shards(panel["openalex_id"].tolist())
    print(f"records in shards: {len(df):,}")

    df["publication_date"] = pd.to_datetime(df["publication_date"], errors="coerce")
    df = df[df["publication_date"].between(WINDOW_START, WINDOW_END)]
    print(f"in window {WINDOW_START.date()} to {WINDOW_END.date()}: {len(df):,}")

    df["doi"] = df["doi"].astype("string").str.lower()
    translated = df["title"].fillna("").str.contains(TRANSLATION_MARKER, regex=False)
    df = df[~translated]
    print(f"removed machine translations: {int(translated.sum()):,}")
    suspect = df["language"].notna() & (df["language"] != "en") & df["doi"].isna()
    df = df[~suspect]
    print(f"removed non-English records without a DOI: {int(suspect.sum()):,}")
    df = df.drop_duplicates("openalex_id").reset_index(drop=True)
    print(f"articles: {len(df):,}")

    d = df["publication_date"]
    df["month"] = d.values.astype("datetime64[M]")
    df["period"] = np.select([d <= INTERVENTION, d.between(WASHOUT_START, WASHOUT_END)],
                             ["pre", "washout"], "post")
    df["covid"] = d.between(COVID_START, COVID_END).astype(int)
    corr = df["corresponding_country"].fillna("")
    df["corr_lang_group"] = np.select([corr == "", corr.isin(ANGLOPHONE)],
                                      ["unknown", "anglophone"], "other")
    title = df["title"].fillna("")
    df["title_words"] = title.str.split().str.len().fillna(0).astype(int)
    df["title_has_colon"] = title.str.contains(":", regex=False)
    df["n_pages"] = pages(df["first_page"], df["last_page"])

    df = merge_recovered(df)
    df["abstract_words"] = df["abstract"].fillna("").str.split().str.len()
    df["has_abstract"] = df["abstract"].notna()

    jm = journal_measures(df)
    df = df.drop(columns=[c for c in ("cohort", "text_tier") if c in df.columns])
    df = df.join(jm[["cohort", "text_tier"]], on="source_id")
    meta = panel[["openalex_id", "venue", "publisher", "scholar_category", "h5_index"]]
    df = df.merge(meta.rename(columns={"openalex_id": "source_id"}), on="source_id", how="left")

    df.to_parquet(repo.CORPUS, index=False)
    df[["openalex_id", "doi", "source_id", "publication_date", "abstract_source"]].to_csv(
        repo.ARTICLE_IDS, index=False, compression="gzip", date_format="%Y-%m-%d")
    summary = panel[["openalex_id", "venue", "publisher", "scholar_category"]].join(
        jm, on="openalex_id")
    summary.to_csv(repo.JOURNAL_SUMMARY, index=False)

    print(f"\ncorpus      : {len(df):,} articles, {df['source_id'].nunique()} journals")
    print(f"core cohort : {int((jm['cohort'] == 'core').sum())} journals")
    print("abstracts   : " + ", ".join(
        f"{k} {v:,}" for k, v in df["abstract_source"].fillna("none").value_counts().items()))
    print(f"wrote {repo.CORPUS}\nwrote {repo.ARTICLE_IDS}\nwrote {repo.JOURNAL_SUMMARY}")


if __name__ == "__main__":
    main()
