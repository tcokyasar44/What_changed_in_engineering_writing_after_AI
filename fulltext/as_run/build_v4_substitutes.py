"""
Draw replacement articles for the cells the first three passes left short.

The quota is defined per journal-year cell. Which particular article fills a cell
was never part of the design: any article published by that journal in that year is
an equally valid draw. So an article that is paywalled, withdrawn behind a
publisher block, or simply unavailable today does not have to be chased. It can be
replaced by another article from the same cell.

This matters because the alternative is manual downloading, which is slow and, as
the author found when ScienceDirect cut him off mid-session, not always possible at
all. Substitution converts a task that depends on one person's access into one the
resolver can attempt unattended.

The design constraint is where the care goes. Replacements must be drawn at random
from the cell's untried remainder, never chosen for being easy to fetch. Picking
retrievable articles would tilt the sample toward open access, which at a
subscription publisher means author-paid articles from better-funded groups, and
that bias would propagate into every per-cell estimate. So the draw is random
within the cell and the oversampling factor is applied per publisher, exactly as
the v3 draw did: how many to draw is tuned by yield, which article is drawn is not.

Articles already attempted in any earlier pass are excluded, successes and failures
alike. Redrawing the failures would concentrate the sample on precisely the
articles that cannot be obtained.

Writes sample_v4.csv. Run resolve_v4.py next.
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
OUT = os.path.join(V2, "sample_v4.csv")

SEED = 20260930

# Yield measured over the v3 pass, which is the most recent evidence of what the
# resolver and fetcher actually achieve. A cell short by n articles needs roughly
# n / yield drawn for n to come back.
FALLBACK_YIELD = 0.39
MIN_ATTEMPTS_FOR_YIELD = 200

# Oversampling is capped so a single stubborn cell cannot pull tens of thousands
# of articles into the resolver queue for a handful of expected returns.
MAX_PER_CELL = 60

COLUMNS = ["doi", "publisher", "venue", "source_id", "publication_year",
           "publication_date", "period", "scholar_category", "is_oa"]


def attempted():
    """Every DOI drawn in any previous pass, obtained or not."""
    tried = set()
    for name in ("sample.csv", "sample_v3.csv"):
        path = os.path.join(V2, name)
        if os.path.exists(path):
            frame = pd.read_csv(path, usecols=["doi"])
            tried |= {str(d).lower().strip() for d in frame["doi"].dropna()}
    return tried


def publisher_yield():
    """Retained share per publisher, measured on the v3 pass."""
    status = pd.read_csv(os.path.join(V2, "fetch_status_v3.csv"))
    status["key"] = status["doi"].astype(str).str.lower().str.strip()
    status["kept"] = status["bytes"].notna()

    sample = pd.read_csv(os.path.join(V2, "sample_v3.csv"),
                         usecols=["doi", "publisher"])
    sample["key"] = sample["doi"].astype(str).str.lower().str.strip()

    merged = status.merge(sample[["key", "publisher"]], on="key", how="left")
    grouped = merged.groupby("publisher")["kept"].agg(["sum", "count"])

    rates = {}
    for name, row in grouped.iterrows():
        if row["count"] >= MIN_ATTEMPTS_FOR_YIELD and row["sum"] > 0:
            rates[name] = row["sum"] / row["count"]
    return rates


def main():
    rng = np.random.default_rng(SEED)

    cells = pd.read_csv(COVERAGE)
    short = cells[cells["short"] > 0].copy()
    if short.empty:
        print("every cell is filled; no substitution needed")
        return 0

    print(f"{len(short):,} cells short by {int(short['short'].sum()):,} "
          f"articles")

    rates = publisher_yield()
    overall = np.mean(list(rates.values())) if rates else FALLBACK_YIELD
    print(f"measured yield: {len(rates)} publishers, "
          f"mean {overall:.0%}\n")

    corpus = pd.read_parquet(CORPUS, columns=COLUMNS + ["has_abstract"])
    corpus = corpus[corpus["doi"].notna()].copy()
    corpus["key"] = corpus["doi"].astype(str).str.lower().str.strip()

    tried = attempted()
    fresh = corpus[~corpus["key"].isin(tried)]
    print(f"corpus {len(corpus):,}, already attempted {len(tried):,}, "
          f"available to draw {len(fresh):,}\n")

    pools = dict(tuple(fresh.groupby(["source_id", "publication_year"])))

    picked = []
    no_pool = 0
    capped = 0
    expected = 0.0
    for cell in short.itertuples():
        key = (cell.source_id, cell.publication_year)
        pool = pools.get(key)
        if pool is None or pool.empty:
            no_pool += 1
            continue

        rate = rates.get(cell.publisher, overall)
        rate = min(max(rate, 0.05), 1.0)

        want = int(np.ceil(cell.short / rate))
        if want > MAX_PER_CELL:
            want = MAX_PER_CELL
            capped += 1
        take = min(want, len(pool))

        # Random within the cell. This is the line that keeps the substitution
        # unbiased, and it must not be replaced by any preference for articles
        # that look easy to fetch.
        chosen = pool.iloc[rng.choice(len(pool), take, replace=False)]
        picked.append(chosen.assign(cell_short=cell.short,
                                    cell_quota=cell.quota))

        # A cell cannot return more than it is short, however many are drawn.
        expected += min(cell.short, take * rate)

    if not picked:
        print("nothing could be drawn")
        return 1

    draw = pd.concat(picked, ignore_index=True)
    draw["draw"] = "v4"
    draw["route"] = "script"
    draw = draw.drop_duplicates("key").drop(columns=["key"])
    draw.to_csv(OUT, index=False)

    print(f"drawn         : {len(draw):,} articles")
    print(f"cells covered : {len(short) - no_pool:,} of {len(short):,}")
    if no_pool:
        print(f"  {no_pool} cells had nothing left to draw")
    if capped:
        print(f"  {capped} cells hit the per-cell cap of {MAX_PER_CELL}")
    print(f"expected to return roughly {int(expected):,} of the "
          f"{int(short['short'].sum()):,} needed")
    print()
    print("draw by publisher:")
    for name, n in draw["publisher"].value_counts().head(10).items():
        print(f"  {str(name)[:44]:<44} {n:6,}")
    print()
    print(f"wrote {OUT}")
    print("next: python resolve_v4.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
