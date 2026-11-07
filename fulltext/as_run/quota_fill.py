"""
Fill each journal-year quota by iterative random replacement.

The procedure, per journal and year:

  1  the target is that cell's proportional share of the sample, say 100 articles
  2  draw 100 uniformly at random from everything the cell contains, say 300
  3  attempt retrieval; suppose 30 succeed
  4  draw 70 more uniformly from the 200 not yet drawn
  5  attempt those; repeat until the quota is met, the pool is exhausted, or the
     round limit is reached
  6  whatever is still missing goes on a manual list for that journal and year

Each round draws only what is short, and only from articles not already drawn, so
every attempted article is a uniform random selection from the cell. Proportions
across journals and years are preserved because the target is fixed before any
retrieval is attempted and never adjusted to match what happened to be reachable.

One property of this procedure needs recording rather than hiding. Retrieval failure
is not random within a cell. At a subscription publisher the retrievable articles
are typically those whose authors paid an open-access fee, and those authors hold
larger grants and work at wealthier institutions than the journal's authors
generally. A cell filled after many rounds is therefore complete in count but
skewed in composition, and it is skewed in a direction that correlates with funding.

The script counts rounds per cell for exactly this reason. Cells that filled in one
round are sound. Cells that took many rounds are flagged, and their missing articles
are placed at the top of the manual list, where a person with institutional access
can obtain the originally drawn article rather than a convenient substitute.
"""

import os
import sys

import numpy as np
import pandas as pd

import paths

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

MAX_ROUNDS = 5
SUSPECT_ROUNDS = 2
SEED = 33107


# Retrieval has run as several draws, each recording its outcomes separately so the
# sampling frames stay distinguishable. All of them count towards the same quota
# and all must be excluded from any further draw. The files are discovered rather
# than listed: a hard-coded list silently ignores every later pass, which shows up
# as a shortfall that has in fact already been filled.
def status_paths():
    found = [os.path.join(paths.ROOT, name)
             for name in sorted(os.listdir(paths.ROOT))
             if name.startswith("fetch_status") and name.endswith(".csv")]
    found.append(paths.SORT_LOG_CSV)
    return [p for p in found if os.path.exists(p)]


def sample_paths():
    return [os.path.join(paths.ROOT, name)
            for name in sorted(os.listdir(paths.ROOT))
            if name.startswith("sample") and name.endswith(".csv")]


def retrieved_dois():
    """Articles successfully obtained, by any route or draw.

    A file on disk and an article parsed in memory without being kept both count:
    both yield measurements, and measurements are what the quota is for.
    """
    have = set()
    for path in status_paths():
        frame = pd.read_csv(path)
        if "bytes" in frame.columns:
            frame = frame[frame["bytes"].notna()]
        if "destination" in frame.columns:
            frame = frame[frame["destination"].isin(["manual_pdf", "pdf"])]
        have |= {str(d).lower() for d in frame["doi"].dropna()}
    return have


def attempted_dois():
    """Articles already tried, successfully or not, so they are not drawn twice."""
    tried = set()
    for path in status_paths():
        frame = pd.read_csv(path)
        tried |= {str(d).lower() for d in frame["doi"].dropna()}
    return tried


def main():
    quota_path = os.path.join(paths.FT, "target_quota.csv")
    if not os.path.exists(quota_path):
        print("no target quota yet; run build_target_quota.py first")
        return 1
    if not os.path.exists(paths.SAMPLE_CSV):
        print("sample.csv missing; run restore_v2_sample.py first")
        return 1

    # The quota is the analysis target of 10,000 spread across cells, not the
    # 119,971-article retrieval pool. The pool was sized for a low automated
    # success rate; treating it as the quota would ask for 116,000 manual
    # downloads. build_target_quota.py explains the distinction.
    quota = pd.read_csv(quota_path)[["source_id", "publication_year", "quota"]]
    quota = quota[quota["quota"] > 0]

    target = pd.read_csv(paths.SAMPLE_CSV)
    corpus = pd.read_parquet(os.path.join(DATA, "corpus.parquet"), columns=[
        "doi", "publisher", "venue", "source_id", "publication_year",
        "publication_date", "period", "scholar_category", "is_oa"])

    corpus["doi_lower"] = corpus["doi"].astype(str).str.lower()

    have = retrieved_dois()
    tried = attempted_dois()

    # Everything drawn so far, across all previous rounds.
    drawn_path = os.path.join(paths.FT, "drawn_articles.csv")
    if os.path.exists(drawn_path):
        drawn = pd.read_csv(drawn_path)
    else:
        drawn = target.copy()
        drawn["round"] = 1

    # Later draws are separate sampling frames, recorded separately so they stay
    # distinguishable, but their articles fill the same journal-year cells and must
    # be credited against the same quota. Without this the shortfall counts only
    # the earliest draws and asks for articles already held.
    for path in sample_paths():
        later = pd.read_csv(path)
        later["round"] = 1
        shared = [c for c in drawn.columns if c in later.columns]
        drawn = pd.concat([drawn, later[shared]], ignore_index=True)
    drawn = drawn.drop_duplicates("doi")

    drawn["doi_lower"] = drawn["doi"].astype(str).str.lower()

    rng = np.random.default_rng(SEED)
    new_rows = []

    # Group once rather than filtering the whole corpus per cell. With 2,726 cells
    # against 1.83 million rows the repeated scan does not finish in usable time.
    corpus_cells = dict(list(corpus.groupby(["source_id", "publication_year"])))
    drawn_cells = dict(list(drawn.groupby(["source_id", "publication_year"])))
    empty_drawn = drawn.iloc[0:0]
    empty_corpus = corpus.iloc[0:0]

    summary = []
    for row in quota.itertuples():
        key = (row.source_id, row.publication_year)
        cell_drawn = drawn_cells.get(key, empty_drawn)

        filled = int(cell_drawn["doi_lower"].isin(have).sum())
        shortfall = int(row.quota) - filled
        rounds_used = int(cell_drawn["round"].max()) if len(cell_drawn) else 0

        available = corpus_cells.get(key, empty_corpus)
        remaining = available[~available["doi_lower"].isin(
            set(cell_drawn["doi_lower"]) | tried)]

        # Draw only the shortfall, only from what has not been drawn before.
        if shortfall > 0 and rounds_used < MAX_ROUNDS and not remaining.empty:
            take = min(shortfall, len(remaining))
            idx = rng.choice(remaining.index.values, take, replace=False)
            batch = remaining.loc[idx].copy()
            batch["round"] = rounds_used + 1
            new_rows.append(batch)
            drew = take
        else:
            drew = 0

        summary.append({
            "source_id": row.source_id,
            "publication_year": int(row.publication_year),
            "quota": int(row.quota),
            "filled": filled,
            "shortfall": max(shortfall, 0),
            "rounds_used": rounds_used,
            "drawn_this_round": drew,
            "cell_size": len(available),
            "pool_left": max(len(remaining) - drew, 0),
        })

    status = pd.DataFrame(summary)
    status["exhausted"] = (status["pool_left"] == 0) & (status["shortfall"] > 0)
    status["suspect"] = status["rounds_used"] > SUSPECT_ROUNDS
    status.to_csv(os.path.join(paths.FT, "quota_status.csv"), index=False)

    if new_rows:
        additions = pd.concat(new_rows, ignore_index=True)
        combined = pd.concat([drawn, additions], ignore_index=True)
        combined = combined.drop_duplicates("doi_lower")
        combined.drop(columns=["doi_lower"]).to_csv(drawn_path, index=False)
        print(f"drew {len(additions):,} replacements this round")
    else:
        drawn.drop(columns=["doi_lower"]).to_csv(drawn_path, index=False)
        print("no replacements drawn: quotas met, pools exhausted, or round limit")

    print()
    print(f"cells            : {len(status):,}")
    print(f"quota total      : {int(status['quota'].sum()):,}")
    print(f"filled           : {int(status['filled'].sum()):,}")
    print(f"short            : {int(status['shortfall'].sum()):,}")
    print(f"cells complete   : {int((status['shortfall'] == 0).sum()):,}")
    print(f"cells exhausted  : {int(status['exhausted'].sum()):,} "
          f"(no articles left to draw)")
    print(f"cells suspect    : {int(status['suspect'].sum()):,} "
          f"(filled only after repeated rounds)")

    write_manual_lists(status, drawn, have, corpus)
    return 0


def write_manual_lists(status, drawn, have, corpus):
    """One manual list per journal-year that is still short."""
    short = status[status["shortfall"] > 0]
    if short.empty:
        print("\nno manual retrieval needed")
        return

    outstanding = drawn[~drawn["doi_lower"].isin(have)].copy()

    keyed = short.set_index(["source_id", "publication_year"])
    outstanding["shortfall"] = [
        int(keyed.loc[(r.source_id, r.publication_year), "shortfall"])
        if (r.source_id, r.publication_year) in keyed.index else 0
        for r in outstanding.itertuples()]
    outstanding["rounds_used"] = [
        int(keyed.loc[(r.source_id, r.publication_year), "rounds_used"])
        if (r.source_id, r.publication_year) in keyed.index else 0
        for r in outstanding.itertuples()]

    outstanding = outstanding[outstanding["shortfall"] > 0]

    # Each cell needs its shortfall, not every article it holds. The pool was drawn
    # far larger than the target because automated retrieval mostly fails, so
    # listing every undrawn article would ask for the whole pool rather than the
    # handful each cell is short. Take the shortfall from each cell, at random, so
    # the request stays a uniform draw within the cell.
    rng = np.random.default_rng(SEED)
    wanted = []
    for _, group in outstanding.groupby(["source_id", "publication_year"]):
        need = int(group["shortfall"].iloc[0])
        if need <= 0:
            continue
        take = min(need, len(group))
        idx = rng.choice(group.index.values, take, replace=False)
        wanted.append(group.loc[idx])
    if not wanted:
        print("\nno manual retrieval needed")
        return
    outstanding = pd.concat(wanted)

    outstanding["post"] = outstanding["publication_year"] >= 2023

    # Post-period first, then the cells most distorted by repeated replacement.
    outstanding = outstanding.sort_values(
        ["post", "rounds_used", "publication_year"],
        ascending=[False, False, False])

    os.makedirs(paths.MANUAL_LISTS, exist_ok=True)
    path = os.path.join(paths.MANUAL_LISTS, "manual_by_cell.csv")
    keep = ["doi", "venue", "publisher", "publication_year", "source_id",
            "shortfall", "rounds_used", "is_oa"]
    outstanding[[c for c in keep if c in outstanding.columns]].to_csv(
        path, index=False)

    print(f"\nmanual retrieval needed for {len(short):,} journal-years")
    print(f"  {len(outstanding):,} articles listed in {os.path.basename(path)}")
    print(f"  ordered post-period first, then by how many rounds the cell took")


if __name__ == "__main__":
    sys.exit(main())
