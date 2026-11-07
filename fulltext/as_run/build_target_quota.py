"""
Set the per-cell target for the full-text sample.

Two different numbers have been called the target and they must not be confused.

The retrieval pool is 119,971 articles. That is not a target. It was sized large
deliberately, because automated retrieval succeeds on a small fraction of what it
attempts, so reaching a few thousand parsed articles required drawing far more than
a few thousand candidates.

The analysis target is 10,000 parsed articles. That is what the full-text measures
need, and it is what each journal-year cell's quota should be measured against.

Reading the retrieval pool as the quota would declare a shortfall of about 116,000
and produce a manual download list longer than any person could work through. This
script allocates the analysis target across cells in the same proportions as the
pool, so the quota is the shape of the drawn sample scaled to the size actually
required.

Largest-remainder allocation is used so the cell quotas sum exactly to the target;
proportional rounding alone would drift by tens of articles.

Writes target_quota.csv. quota_fill.py reads it to decide what is still missing.
"""

import os
import sys

import numpy as np
import pandas as pd

import paths

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ANALYSIS_TARGET = 10000

TARGET_QUOTA_CSV = os.path.join(paths.FT, "target_quota.csv")


def largest_remainder(shares, total):
    """Allocate whole articles so the parts sum exactly to the total."""
    exact = shares * total
    floor = np.floor(exact).astype(int)
    remainder = total - int(floor.sum())
    if remainder > 0:
        order = np.argsort(-(exact - floor))
        floor[order[:remainder]] += 1
    return floor


def main():
    if not os.path.exists(paths.SAMPLE_CSV):
        print("sample.csv missing; run restore_v2_sample.py first")
        return 1

    sample = pd.read_csv(paths.SAMPLE_CSV)
    print(f"retrieval pool : {len(sample):,} articles")
    print(f"analysis target: {ANALYSIS_TARGET:,} articles")

    cells = (sample.groupby(["source_id", "publication_year"])
             .size().rename("pool").reset_index())
    cells["share"] = cells["pool"] / cells["pool"].sum()
    cells["quota"] = largest_remainder(cells["share"].values, ANALYSIS_TARGET)

    # A cell allocated nothing would silently drop that journal-year from the
    # full-text measures. Cells with a real presence in the pool get at least one.
    floor_applied = int(((cells["quota"] == 0) & (cells["pool"] >= 20)).sum())
    cells.loc[(cells["quota"] == 0) & (cells["pool"] >= 20), "quota"] = 1

    venues = (sample.groupby("source_id")["venue"].first().rename("venue"))
    publishers = (sample.groupby("source_id")["publisher"].first()
                  .rename("publisher"))
    cells = cells.join(venues, on="source_id").join(publishers, on="source_id")

    cells.to_csv(TARGET_QUOTA_CSV, index=False)

    print(f"\ncells            : {len(cells):,}")
    print(f"quota total      : {int(cells['quota'].sum()):,}")
    if floor_applied:
        print(f"cells raised to 1: {floor_applied:,} "
              f"(would otherwise have been dropped entirely)")
    print(f"quota per cell   : min {int(cells['quota'].min())}, "
          f"median {int(cells['quota'].median())}, "
          f"max {int(cells['quota'].max())}")

    print("\nquota by year:")
    by_year = cells.groupby("publication_year")["quota"].sum()
    for year, value in by_year.items():
        print(f"  {int(year)}  {int(value):5,}")

    print(f"\nwrote {TARGET_QUOTA_CSV}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
