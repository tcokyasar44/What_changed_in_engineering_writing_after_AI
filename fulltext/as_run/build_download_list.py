"""
Produce the manual download list in the order that repairs the sample fastest.

The pooled count of retrieved articles is close to the target, but that number
hides the problem the list exists to fix. Articles are allocated to journal-year
cells, and a surplus in one cell cannot substitute for a gap in another: extra
articles from a megajournal in 2023 say nothing about a subscription journal in
2025. Counting each cell only up to its quota, coverage is near 70 percent rather
than the 99 percent the totals suggest, and 438 cells hold nothing at all.

The gaps are not spread evenly. The worst-covered year is the most recent, which is
also the year carrying most weight in a before-and-after comparison, and the worst
covered publishers are the ones that refuse automated access. So the order of work
matters: the first few hundred articles collected should be the ones that lift the
weakest cells, not whichever happen to come first alphabetically.

Priority is the share of its cell's quota that one article supplies, so an article
from a cell needing two of two counts for more than one from a cell needing two of
forty. Empty cells therefore rise to the top. Ties break towards the recent years.

Titles are joined from the corpus, because searching a library catalogue by title is
easier than by DOI, and a resolvable link is included for each.

Writes three files: everything, the open-access part that needs only a browser, and
the subscription part that needs an entitled campus network.
"""

import os
import sys

import pandas as pd

import paths

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
CORPUS = os.path.join(HERE, "data", "corpus.parquet")

MANUAL_CSV = os.path.join(paths.MANUAL_LISTS, "manual_by_cell.csv")
QUOTA_STATUS = os.path.join(paths.FT, "quota_status.csv")

COLUMNS = ["priority", "doi", "url", "title", "venue", "publisher",
           "publication_year", "cell_needs", "cell_quota", "access"]


def main():
    if not os.path.exists(MANUAL_CSV):
        print("manual_by_cell.csv missing; run quota_fill.py first")
        return 1

    manual = pd.read_csv(MANUAL_CSV)
    manual["is_oa"] = manual["is_oa"].fillna(False).astype(bool)
    print(f"outstanding articles: {len(manual):,}")

    status = pd.read_csv(QUOTA_STATUS)[
        ["source_id", "publication_year", "quota", "filled", "shortfall"]]
    manual = manual.merge(status, on=["source_id", "publication_year"],
                          how="left", suffixes=("", "_cell"))

    titles = pd.read_parquet(CORPUS, columns=["doi", "title"])
    titles["doi"] = titles["doi"].astype(str)
    manual["doi"] = manual["doi"].astype(str)
    manual = manual.merge(titles.drop_duplicates("doi"), on="doi", how="left")
    found = int(manual["title"].notna().sum())
    print(f"titles resolved     : {found:,} of {len(manual):,}")

    # How much of its cell's quota this one article supplies. An article from a
    # cell needing two of two matters more than one from a cell needing two of
    # forty, so empty and nearly empty cells rise to the top.
    quota = manual["quota"].replace(0, pd.NA)
    manual["weight"] = manual["shortfall_cell"].fillna(0) / quota
    manual["weight"] = manual["weight"].fillna(0)

    manual = manual.sort_values(
        ["weight", "publication_year"], ascending=[False, False])
    manual = manual.reset_index(drop=True)
    manual["priority"] = manual.index + 1

    manual["url"] = "https://doi.org/" + manual["doi"]
    manual["access"] = manual["is_oa"].map({True: "open access",
                                            False: "subscription"})
    manual = manual.rename(columns={"shortfall_cell": "cell_needs",
                                    "quota": "cell_quota"})

    out = manual[[c for c in COLUMNS if c in manual.columns]]

    everything = os.path.join(paths.MANUAL_LISTS, "download_list.csv")
    out.to_csv(everything, index=False)
    print(f"\nwrote {os.path.basename(everything)}: {len(out):,} articles")

    for label, subset, name in (
            ("open access, any browser anywhere",
             out[out["access"] == "open access"], "download_open_access.csv"),
            ("subscription, needs campus access",
             out[out["access"] == "subscription"], "download_subscription.csv")):
        path = os.path.join(paths.MANUAL_LISTS, name)
        subset.to_csv(path, index=False)
        post = int((subset["publication_year"] >= 2023).sum())
        print(f"  {label}: {len(subset):,} ({post:,} post-2022) -> {name}")

    print("\nby year:")
    table = out.groupby("publication_year").size()
    for year, count in table.items():
        print(f"  {int(year)}  {count:,}")

    print("\ntop journals by articles needed:")
    top = (out.groupby(["venue", "publisher"]).size()
           .rename("needed").reset_index()
           .sort_values("needed", ascending=False).head(15))
    print(top.to_string(index=False))

    print(f"\nempty cells first: the top {int((out['cell_needs'] >= out['cell_quota']).sum()):,} "
          f"articles come from cells holding nothing at all")
    return 0


if __name__ == "__main__":
    sys.exit(main())
