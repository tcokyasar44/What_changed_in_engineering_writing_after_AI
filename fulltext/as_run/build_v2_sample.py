"""
Draw the full-text sample from the whole corpus, not the open-access part of it.

An earlier version of this drew only open-access articles, on the reasoning that a
script cannot fetch anything else. That confuses two different things: what
automation can reach, and what the sample should contain. Subscription articles are
retrievable by a reader at a subscribing institution, and the study redistributes no
article text, so their access status has no bearing on whether they belong in the
sample.

Excluding them is not neutral. Open access is distributed very unevenly across
publishers: the largest engineering publisher is 41 percent of the corpus but only a
quarter of its output is open, while one fully gold publisher contributes everything
it prints. A sample drawn from open access alone therefore doubles one publisher's
weight and halves another's, and since publishers differ in article length, figure
conventions and section structure, that skew propagates into every level the sample
is used to estimate.

The sample is drawn proportionally by journal within year, across the whole corpus.
Each row is then labelled with how it can be retrieved:

    script   open access, an automated pass may find a file
    manual   subscription, needs a reader at a subscribing institution

Both are part of the same sample. The split governs who fetches what, not what
counts.
"""

import os
import sys

import numpy as np
import pandas as pd

import paths

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
CORPUS = os.path.join(HERE, "data", "corpus.parquet")

# Sized against observed retrieval. An automated pass succeeds on roughly 13
# percent of open-access attempts, so reaching ten thousand parsed articles needs a
# far larger draw than ten thousand.
TARGET_TOTAL = 120000
SEED = 20260927

YEAR_WEIGHTS = {
    2015: 0.6, 2016: 0.6, 2017: 0.6, 2018: 0.7, 2019: 0.8,
    2020: 0.8, 2021: 0.9, 2022: 1.0,
    2023: 2.2, 2024: 2.2, 2025: 2.2,
}

MIN_JOURNAL_YEAR = 20


def main():
    corpus = pd.read_parquet(CORPUS, columns=[
        "doi", "publisher", "venue", "source_id", "publication_year",
        "publication_date", "period", "scholar_category", "is_oa"])
    corpus = corpus[corpus["doi"].notna()]

    print(f"corpus with a DOI : {len(corpus):,}")
    print(f"  open access     : {int(corpus['is_oa'].fillna(False).sum()):,}")
    print(f"  subscription    : "
          f"{int((~corpus['is_oa'].fillna(False)).sum()):,}")

    years = sorted(y for y in corpus["publication_year"].unique()
                   if y in YEAR_WEIGHTS)
    sizes = corpus.groupby("publication_year").size()
    raw = pd.Series({y: sizes.get(y, 0) * YEAR_WEIGHTS[y] for y in years},
                    dtype=float)
    year_quota = (raw / raw.sum() * TARGET_TOTAL).round().astype(int)

    print("\nallocation by year:")
    for year in years:
        print(f"  {year}  corpus {sizes.get(year, 0):7,}  "
              f"sample {year_quota[year]:6,}")

    rng = np.random.default_rng(SEED)
    picks = []

    for year in years:
        quota = int(year_quota[year])
        pool = corpus[corpus["publication_year"] == year]
        counts = pool["source_id"].value_counts()
        counts = counts[counts >= MIN_JOURNAL_YEAR]
        if counts.empty:
            continue

        shares = counts / counts.sum()
        allocation = (shares * quota).round().astype(int)

        for source_id, slots in allocation.items():
            if slots < 1:
                continue
            candidates = pool[pool["source_id"] == source_id]
            take = min(int(slots), len(candidates))
            idx = rng.choice(candidates.index.values, take, replace=False)
            picks.append(pool.loc[idx])

    sample = (pd.concat(picks, ignore_index=True)
              .drop_duplicates("doi")
              .sort_values(["publication_year", "source_id"])
              .reset_index(drop=True))

    # Retrieval route, not a filter. Both routes are part of the sample.
    sample["route"] = np.where(sample["is_oa"].fillna(False), "script", "manual")

    paths.ensure()
    sample.to_csv(paths.SAMPLE_CSV, index=False)
    sample.to_csv(paths.TARGET_SAMPLE_CSV, index=False)

    print(f"\ndrawn: {len(sample):,}")
    print(f"  script route : {int((sample['route'] == 'script').sum()):,}")
    print(f"  manual route : {int((sample['route'] == 'manual').sum()):,}")

    print("\nfidelity, publisher share of sample against corpus:")
    compare = pd.DataFrame({
        "sample_pct": sample["publisher"].value_counts(normalize=True).mul(100),
        "corpus_pct": corpus["publisher"].value_counts(normalize=True).mul(100),
    }).fillna(0)
    print(compare[compare["corpus_pct"] >= 1].round(1).to_string())

    expected = int((sample["route"] == "script").sum() * 0.132)
    print(f"\nexpected from the script route at the observed 13.2 percent: "
          f"{expected:,}")
    print("the remainder must come from manual retrieval")

    print(f"\nwrote {paths.SAMPLE_CSV}")


if __name__ == "__main__":
    main()
