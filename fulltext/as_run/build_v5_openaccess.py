"""
Draw open-access substitutes for the cells still short, and record the trade.

The v4 draw was random within each cell, which is the ideal. It also returned 99
files from 4,334 attempts, because roughly three articles in four are closed and a
random substitute is usually unreachable. Continuing that way would need hundreds
of thousands of resolver calls to finish, and would still leave the subscription
cells short.

Restricting substitutes to open-access articles makes them reachable. It also makes
each filled cell an open-access sample of that journal-year rather than a random
one, and that is a real cost which has to be justified rather than assumed away.

The justification is measured, in oa_style_difference.py. Within the same journal
and year, open-access and subscription abstracts differ by 2.5 percent on the
headline marker measure, against a reported effect of 57.9 percent, and the
difference runs negative: open-access articles use slightly fewer marker words. So
the substitution cannot manufacture the finding, and its effect on the estimate is
an order of magnitude smaller than the quantity being estimated.

Two things keep this honest and both are carried in the output. Every substituted
article is flagged, so any estimate can be recomputed on originally drawn articles
alone. And the cell's open-access share is recorded, so the degree of substitution
in each cell is known rather than hidden.

This is not the earlier mistake of sampling only open access. The first draw was
representative and remains the basis of the collection; this fills what could not
be retrieved from it, and the composition change is bounded, measured and reported.

Writes sample_v5.csv. Run resolve_v5.py next.
"""

import os
import sys

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
FT = os.path.join(HERE, "data", "fulltext")
V2 = os.path.join(FT, "v2_target")
CORPUS = os.path.join(HERE, "data", "corpus.parquet")
COVERAGE = os.path.join(FT, "coverage_now.csv")
OUT = os.path.join(V2, "sample_v5.csv")

SEED = 20260930

# Open-access articles that reach the fetcher succeed far more often than the
# mixed draw did, but not always: some sit behind a publisher block even when
# openly licensed. Oversample against the rate v3 achieved on open access.
ASSUMED_YIELD = 0.39
MAX_PER_CELL = 80

COLUMNS = ["doi", "publisher", "venue", "source_id", "publication_year",
           "publication_date", "period", "scholar_category", "is_oa"]


def attempted():
    tried = set()
    for name in ("sample.csv", "sample_v3.csv", "sample_v4.csv"):
        path = os.path.join(V2, name)
        if os.path.exists(path):
            frame = pd.read_csv(path, usecols=["doi"])
            tried |= {str(d).lower().strip() for d in frame["doi"].dropna()}
    return tried


def main():
    rng = np.random.default_rng(SEED)

    cells = pd.read_csv(COVERAGE)
    short = cells[cells["short"] > 0].copy()
    print(f"{len(short):,} cells short by {int(short['short'].sum()):,}\n")

    corpus = pd.read_parquet(CORPUS, columns=COLUMNS)
    corpus = corpus[corpus["doi"].notna()].copy()
    corpus["key"] = corpus["doi"].astype(str).str.lower().str.strip()

    tried = attempted()
    fresh = corpus[~corpus["key"].isin(tried) & corpus["is_oa"]]
    print(f"already attempted        : {len(tried):,}")
    print(f"untried and open access  : {len(fresh):,}\n")

    pools = dict(tuple(fresh.groupby(["source_id", "publication_year"])))

    picked = []
    empty = 0
    for cell in short.itertuples():
        pool = pools.get((cell.source_id, cell.publication_year))
        if pool is None or pool.empty:
            empty += 1
            continue
        want = min(int(np.ceil(cell.short / ASSUMED_YIELD)), MAX_PER_CELL)
        take = min(want, len(pool))
        chosen = pool.iloc[rng.choice(len(pool), take, replace=False)]
        picked.append(chosen.assign(cell_short=cell.short))

    if not picked:
        print("no open-access substitutes available")
        return 1

    draw = pd.concat(picked, ignore_index=True)
    draw["draw"] = "v5"
    draw["route"] = "script"
    draw["substitute"] = True
    draw = draw.drop_duplicates("key").drop(columns=["key"])
    draw.to_csv(OUT, index=False)

    print(f"drawn        : {len(draw):,} open-access articles")
    print(f"cells covered: {len(short) - empty:,} of {len(short):,}")
    if empty:
        print(f"  {empty} cells have no untried open-access article and can "
              f"only be filled by hand")
    print()
    print("by publisher:")
    for name, n in draw["publisher"].value_counts().head(10).items():
        print(f"  {str(name)[:44]:<44} {n:6,}")
    print()
    print(f"wrote {OUT}")
    print("next: python resolve_v5.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
