"""
Rebuild the manual download list from what substitution could not reach.

The original list was drawn before substitution existed, so it names specific
articles that must be obtained by hand. Most of them no longer need to be. A cell
that substitution has filled is complete, whatever happened to the particular
article first drawn from it, because the quota was defined per journal-year and
never per article.

This regenerates the list from the cells that are still short after every
automatic pass, so the remaining manual work is the part that genuinely cannot be
automated. For each short cell it names the articles most likely to succeed in a
browser: ones the resolver located a copy of but could not fetch, since those are
blocked by bot detection rather than by entitlement and a signed-in person can
usually retrieve them.

Writes download_list.csv, replacing the earlier version, and keeps a copy of the
old one.
"""

import os
import shutil
import sys

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
FT = os.path.join(HERE, "data", "fulltext")
V2 = os.path.join(FT, "v2_target")
MANUAL = os.path.join(FT, "manual")

COVERAGE = os.path.join(FT, "coverage_now.csv")
OUT = os.path.join(MANUAL, "download_list.csv")

SAMPLES = ["sample.csv", "sample_v3.csv", "sample_v4.csv", "sample_v5.csv", "sample_v6.csv"]
STATUSES = ["fetch_status.csv", "fetch_status_v3.csv",
            "fetch_status_v4.csv", "fetch_status_v5.csv", "fetch_status_v6.csv"]
RESOLVED = ["resolved.csv", "alternates.csv", "resolved_v3.csv",
            "resolved_v4.csv", "resolved_v5.csv", "resolved_v6.csv"]


def held():
    have = set()
    for name in STATUSES:
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


def located():
    """DOIs the resolver found a copy of, which a browser can often reach."""
    found = set()
    for name in RESOLVED:
        path = os.path.join(V2, name)
        if os.path.exists(path):
            frame = pd.read_csv(path, usecols=lambda c: c in ("doi", "pdf_url"))
            if "pdf_url" in frame:
                frame = frame[frame["pdf_url"].notna()]
                found |= {str(d).lower().strip()
                          for d in frame["doi"].dropna()}
    return found


def main():
    cells = pd.read_csv(COVERAGE)
    short = cells[cells["short"] > 0].copy()
    print(f"{len(short):,} cells short by {int(short['short'].sum()):,} "
          f"articles\n")

    frames = []
    for name in SAMPLES:
        path = os.path.join(V2, name)
        if os.path.exists(path):
            frames.append(pd.read_csv(path, usecols=lambda c: c in (
                "doi", "publisher", "venue", "source_id", "publication_year",
                "is_oa")))
    drawn = pd.concat(frames, ignore_index=True)
    drawn["key"] = drawn["doi"].astype(str).str.lower().str.strip()
    drawn = drawn.drop_duplicates("key")

    have = held()
    findable = located()
    drawn = drawn[~drawn["key"].isin(have)]

    need = short[["source_id", "publication_year", "short", "quota"]]
    candidates = drawn.merge(need, on=["source_id", "publication_year"],
                             how="inner")

    # Prefer articles a copy was found for: those fail to a script because of bot
    # detection, and a signed-in browser usually succeeds.
    candidates["locatable"] = candidates["key"].isin(findable)
    candidates = candidates.sort_values(
        ["locatable", "is_oa"], ascending=[False, False], kind="stable")

    # Ask for only as many per cell as the cell is short. The grouping columns
    # are restored afterwards, since group_keys alone does not reliably keep
    # them out of the index across pandas versions.
    picked = (candidates.groupby(["source_id", "publication_year"],
                                 group_keys=False, sort=False)
              .head(0))
    parts = []
    for _, group in candidates.groupby(["source_id", "publication_year"],
                                       sort=False):
        parts.append(group.head(int(group["short"].iloc[0])))
    picked = (pd.concat(parts, ignore_index=True) if parts
              else candidates.head(0))

    if picked.empty:
        print("nothing left to request by hand")
        return 0

    picked = picked.copy()
    picked["url"] = "https://doi.org/" + picked["doi"].astype(str)
    picked["access"] = picked["is_oa"].map(
        {True: "open access", False: "subscription"})

    # Titles are not carried in the sample files, and the page is far harder to
    # work through without them.
    corpus = pd.read_parquet(os.path.join(HERE, "data", "corpus.parquet"),
                             columns=["doi", "title"])
    corpus["key"] = corpus["doi"].astype(str).str.lower().str.strip()
    titles = dict(zip(corpus["key"], corpus["title"]))
    picked["title"] = picked["key"].map(titles).fillna("")
    picked = picked.sort_values(
        ["locatable", "publisher", "venue", "publication_year"],
        ascending=[False, True, True, True], kind="stable")
    picked.insert(0, "priority", range(1, len(picked) + 1))

    if os.path.exists(OUT):
        shutil.copyfile(OUT, OUT + ".previous")

    keep = ["priority", "doi", "url", "venue", "publisher",
            "publication_year", "access", "locatable", "short", "quota", "title"]
    picked[keep].to_csv(OUT, index=False)

    print(f"manual list rebuilt: {len(picked):,} articles "
          f"(was 2,199 before substitution)")
    print(f"  a copy was located for {int(picked['locatable'].sum()):,}, "
          f"which a browser can usually fetch")
    print()
    print(picked["access"].value_counts().to_string())
    print()
    print("by publisher:")
    for name, n in picked["publisher"].value_counts().head(8).items():
        print(f"  {str(name)[:44]:<44} {n:5,}")
    print()
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())


