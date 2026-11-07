"""
Report how much of the 10,000-article target is now held, and where the gaps are.

Three collection routes have run: the original automatic fetch, the v3 tuned
redraw, and manual downloading. Each was tracked separately, so the question the
author is actually asking, how much more manual work is left and can it be
avoided, cannot be answered from any one of them.

This pools all three against the per-cell quota and reports the shortfall by year
and by publisher, because the draw is stratified by journal-year and an even
spread matters more than the total.
"""

import os
import sys

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
FT = os.path.join(HERE, "data", "fulltext")
V2 = os.path.join(FT, "v2_target")
MANUAL = os.path.join(FT, "manual")

QUOTA = os.path.join(FT, "target_quota.csv")
SAMPLE = os.path.join(V2, "sample.csv")
SAMPLE_V3 = os.path.join(V2, "sample_v3.csv")
SAMPLE_V4 = os.path.join(V2, "sample_v4.csv")
SAMPLE_V5 = os.path.join(V2, "sample_v5.csv")
SAMPLE_V6 = os.path.join(V2, "sample_v6.csv")


def held_dois():
    """Every DOI held as a file or measured in memory, by any route."""
    have = set()

    for name in ("fetch_status.csv", "fetch_status_v3.csv",
                 "fetch_status_v4.csv", "fetch_status_v5.csv", "fetch_status_v6.csv"):
        path = os.path.join(V2, name)
        if os.path.exists(path):
            frame = pd.read_csv(path)
            if "bytes" in frame:
                have |= {str(d).lower().strip() for d in
                         frame.loc[frame["bytes"].notna(), "doi"].dropna()}

    for name in ("parsed_transient.parquet", "parsed_transient_v3.parquet",
                 "parsed_transient_v4.parquet", "parsed_transient_v5.parquet", "parsed_transient_v6.parquet"):
        path = os.path.join(V2, name)
        if os.path.exists(path):
            frame = pd.read_parquet(path)
            if "doi" in frame:
                have |= {str(d).lower().strip()
                         for d in frame["doi"].dropna()}

    path = os.path.join(MANUAL, "collected_dois.csv")
    if os.path.exists(path):
        have |= {str(d).lower().strip()
                 for d in pd.read_csv(path)["doi"].dropna()}

    return have


def load_sample():
    """Both draws, with the journal-year key each article belongs to."""
    frames = []
    for path in (SAMPLE, SAMPLE_V3, SAMPLE_V4, SAMPLE_V5, SAMPLE_V6):
        if not os.path.exists(path):
            continue
        frame = pd.read_csv(path, usecols=lambda c: c in (
            "doi", "source_id", "publication_year", "venue", "publisher",
            "oa_status"))
        frames.append(frame)
    both = pd.concat(frames, ignore_index=True)
    both["key"] = both["doi"].astype(str).str.lower().str.strip()
    return both.drop_duplicates("key")


def main():
    quota = pd.read_csv(QUOTA)
    sample = load_sample()
    have = held_dois()

    sample["held"] = sample["key"].isin(have)

    print(f"articles drawn across both samples : {len(sample):,}")
    print(f"articles held by any route         : {int(sample['held'].sum()):,}")
    print(f"target                             : {int(quota['quota'].sum()):,}")
    print()

    got = (sample[sample["held"]]
           .groupby(["source_id", "publication_year"])
           .size().rename("held").reset_index())

    cells = quota.merge(got, on=["source_id", "publication_year"], how="left")
    cells["held"] = cells["held"].fillna(0).astype(int)
    cells["filled"] = cells[["held", "quota"]].min(axis=1)
    cells["short"] = (cells["quota"] - cells["held"]).clip(lower=0)

    filled = int(cells["filled"].sum())
    target = int(cells["quota"].sum())
    print(f"quota filled : {filled:,} of {target:,} "
          f"({100 * filled / target:.0f}%)")
    print(f"still short  : {int(cells['short'].sum()):,} "
          f"across {int((cells['short'] > 0).sum()):,} of {len(cells):,} cells")
    print()

    print("by year:")
    by_year = cells.groupby("publication_year")[
        ["quota", "filled", "short"]].sum()
    for year, row in by_year.iterrows():
        pct = 100 * row["filled"] / row["quota"] if row["quota"] else 0
        bar = "#" * int(24 * pct / 100)
        print(f"  {int(year)}  {int(row['filled']):4d}/{int(row['quota']):4d}"
              f"  {pct:3.0f}%  {bar}")

    print()
    print("largest shortfalls by publisher:")
    by_pub = (cells.groupby("publisher")[["quota", "filled", "short"]]
              .sum().sort_values("short", ascending=False))
    for pub, row in by_pub.head(10).iterrows():
        pct = 100 * row["filled"] / row["quota"] if row["quota"] else 0
        print(f"  {str(pub)[:44]:<44} {int(row['filled']):4d}/"
              f"{int(row['quota']):4d}  {pct:3.0f}%")

    out = os.path.join(FT, "coverage_now.csv")
    cells.to_csv(out, index=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()


