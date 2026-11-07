"""
Draw a second sample, sized per publisher by how often retrieval actually works.

The first draw was proportional to the corpus and treated every article as equally
likely to be obtained. Retrieval does not behave that way. Measured over the whole
first sample, the share of drawn articles that produced a usable file runs from 1.1
percent at one publisher to 8.2 percent at another, a sevenfold spread. A
proportional draw therefore yields a retrieved set whose publisher mix is nothing
like the corpus, however faithful the draw itself was.

This draw corrects for that. Each publisher's deficit against the 10,000 analysis
target is divided by that publisher's observed yield, giving how many articles must
be drawn to expect the deficit to be filled. Within each publisher the draw is
stratified by journal and year exactly as before, so the correction changes how many
articles come from each publisher and nothing else about the sampling structure.

Two things this does not fix, both of which belong in the paper.

It cannot correct selection inside a publisher. The articles that come through at a
subscription publisher are overwhelmingly those the author placed in a repository, and
those authors are better funded and more often at large Western institutions than the
publisher's authorship generally. Only manual retrieval of the specific drawn article
fixes that, which is what manual_by_cell.csv is for.

It cannot help where yield is zero. One publisher returned nothing from 250 attempts;
no amount of oversampling changes that, and its absence is reported rather than
patched.

The draw excludes every article already attempted, so the two samples are disjoint
and can be pooled.

Writes sample_v3.csv. Run resolve_v3.py next.
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

ANALYSIS_TARGET = 10000
SEED = 20260928

SAMPLE_V3_CSV = os.path.join(paths.ROOT, "sample_v3.csv")

# Year weights from the first draw, kept identical so the two samples pool cleanly.
YEAR_WEIGHTS = {
    2015: 0.6, 2016: 0.6, 2017: 0.6, 2018: 0.7, 2019: 0.8,
    2020: 0.8, 2021: 0.9, 2022: 1.0,
    2023: 2.2, 2024: 2.2, 2025: 2.2,
}

# A yield estimated from very few attempts is not a yield. Publishers below this
# are given the overall rate rather than their own noisy one.
MIN_ATTEMPTS_FOR_YIELD = 200

# Drawing a whole publisher's remaining catalogue to chase a handful of articles
# wastes the resolver's time for no measurable gain.
MAX_DRAW_PER_PUBLISHER = 120000

COLUMNS = ["doi", "publisher", "venue", "source_id", "publication_year",
           "publication_date", "period", "scholar_category", "is_oa"]


def observed_yield(sample, retrieved):
    """Share of drawn articles that produced a usable file, by publisher."""
    sample = sample.copy()
    sample["ok"] = sample["doi"].astype(str).isin(retrieved)
    table = sample.groupby("publisher").agg(drawn=("doi", "size"),
                                            got=("ok", "sum"))
    table["yield"] = table["got"] / table["drawn"]
    overall = float(sample["ok"].mean())
    table.loc[table["drawn"] < MIN_ATTEMPTS_FOR_YIELD, "yield"] = overall
    return table, overall


def main():
    if os.path.exists(SAMPLE_V3_CSV):
        print(f"{SAMPLE_V3_CSV} already exists; refusing to overwrite")
        return 1

    sample = pd.read_csv(paths.SAMPLE_CSV)
    status = pd.read_csv(paths.FETCH_STATUS_CSV)

    # A file on disk and a file parsed without being kept both count: both yield
    # measurements, which is what the analysis needs.
    retrieved = set(status.loc[status["bytes"].notna(), "doi"].astype(str))
    attempted = set(status["doi"].astype(str)) | set(sample["doi"].astype(str))

    table, overall = observed_yield(sample, retrieved)
    print(f"first sample : {len(sample):,} drawn, {len(retrieved):,} retrieved "
          f"({100 * overall:.2f}%)")

    corpus = pd.read_parquet(CORPUS, columns=COLUMNS)
    corpus["doi"] = corpus["doi"].astype(str)
    corpus = corpus[corpus["doi"].notna()]

    remaining = corpus[~corpus["doi"].isin(attempted)]
    print(f"corpus       : {len(corpus):,}")
    print(f"already tried: {len(attempted):,}")
    print(f"pool left    : {len(remaining):,}\n")

    corpus_share = corpus["publisher"].value_counts(normalize=True)

    plan = []
    for publisher, share in corpus_share.items():
        want = ANALYSIS_TARGET * share
        have = int(table.loc[publisher, "got"]) if publisher in table.index else 0
        deficit = max(want - have, 0)
        if deficit < 1:
            continue

        rate = (float(table.loc[publisher, "yield"])
                if publisher in table.index else overall)
        if rate <= 0:
            plan.append({"publisher": publisher, "want": want, "have": have,
                         "deficit": deficit, "yield": rate, "draw": 0,
                         "pool": int((remaining["publisher"] == publisher).sum()),
                         "note": "no retrieval observed; cannot be oversampled"})
            continue

        pool = int((remaining["publisher"] == publisher).sum())
        draw = min(int(np.ceil(deficit / rate)), pool, MAX_DRAW_PER_PUBLISHER)
        plan.append({"publisher": publisher, "want": want, "have": have,
                     "deficit": deficit, "yield": rate, "draw": draw,
                     "pool": pool, "note": ""})

    plan = pd.DataFrame(plan)
    plan = plan[plan["draw"] > 0].sort_values("draw", ascending=False)

    print("draw plan:")
    display = plan.copy()
    display["publisher"] = display["publisher"].str.slice(0, 32)
    display["want"] = display["want"].round().astype(int)
    display["deficit"] = display["deficit"].round().astype(int)
    display["yield%"] = (100 * display["yield"]).round(2)
    print(display[["publisher", "want", "have", "deficit", "yield%", "draw",
                   "pool"]].to_string(index=False))

    rng = np.random.default_rng(SEED)
    picks = []

    for row in plan.itertuples():
        pool = remaining[remaining["publisher"] == row.publisher]
        if pool.empty:
            continue

        # Stratify by year on the same weights as the first draw, then allocate
        # each year across journals in proportion to their output that year.
        years = sorted(y for y in pool["publication_year"].unique()
                       if y in YEAR_WEIGHTS)
        sizes = pool.groupby("publication_year").size()
        raw = pd.Series({y: sizes.get(y, 0) * YEAR_WEIGHTS[y] for y in years},
                        dtype=float)
        if raw.sum() <= 0:
            continue
        year_quota = (raw / raw.sum() * row.draw).round().astype(int)

        for year in years:
            quota = int(year_quota.get(year, 0))
            if quota < 1:
                continue
            year_pool = pool[pool["publication_year"] == year]
            counts = year_pool["source_id"].value_counts()
            if counts.empty:
                continue
            allocation = (counts / counts.sum() * quota).round().astype(int)

            for source_id, slots in allocation.items():
                if slots < 1:
                    continue
                candidates = year_pool[year_pool["source_id"] == source_id]
                take = min(int(slots), len(candidates))
                idx = rng.choice(candidates.index.values, take, replace=False)
                picks.append(year_pool.loc[idx])

    if not picks:
        print("\nnothing to draw")
        return 1

    drawn = (pd.concat(picks, ignore_index=True)
             .drop_duplicates("doi")
             .sort_values(["publication_year", "source_id"])
             .reset_index(drop=True))
    drawn["route"] = np.where(drawn["is_oa"].fillna(False), "script", "manual")
    drawn["draw"] = "v3"

    drawn.to_csv(SAMPLE_V3_CSV, index=False)

    print(f"\ndrawn: {len(drawn):,}")
    print(f"  script route : {int((drawn['route'] == 'script').sum()):,}")
    print(f"  manual route : {int((drawn['route'] == 'manual').sum()):,}")
    print(f"\nexpected retrieved, at each publisher's observed rate: "
          f"{int(plan['deficit'].clip(upper=plan['draw'] * plan['yield']).sum()):,}")

    print("\nby year:")
    print(drawn["publication_year"].value_counts().sort_index().to_string())

    overlap = set(drawn["doi"].astype(str)) & attempted
    print(f"\noverlap with anything already attempted: {len(overlap)}")
    print(f"\nwrote {SAMPLE_V3_CSV}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
