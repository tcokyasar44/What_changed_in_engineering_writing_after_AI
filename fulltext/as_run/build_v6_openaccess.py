"""
Draw open-access substitutes again, sized by the yield actually observed.

The v5 pass assumed a 39 percent yield, taken from the v3 run, and achieved 10.9
percent. It therefore drew roughly three and a half times too few articles and
closed only part of the gap. The assumption was the error, not the method: v3's
rate came from a draw that included repository-heavy strata, while v5 drew from
the residue those earlier passes had already picked over.

This pass uses the rate v5 measured, per publisher where there were enough
attempts to measure one. The per-cell cap rises accordingly, because with a yield
near 10 percent a cell short by three articles needs roughly thirty drawn, not
eight.

Everything else is unchanged from v5, deliberately. Substitutes are open access so
they can be fetched, random within the journal-year cell so the choice is not made
on retrievability, disjoint from every previous draw, and flagged so any estimate
can be recomputed without them. The justification for open-access substitution is
the 2.5 percent within-cell difference measured in oa_style_difference.py, against
a 57.9 percent effect and running in the opposite direction.

Writes sample_v6.csv. Run resolve_v6.py next.
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
OUT = os.path.join(V2, "sample_v6.csv")

SEED = 20260930_2

# Measured on v5 rather than assumed. A floor keeps a single unlucky publisher
# from demanding an unbounded draw.
MIN_YIELD = 0.05
MIN_ATTEMPTS = 100

# With a yield near a tenth, filling a cell short by five needs about fifty
# drawn. The cap has to allow that or the pass under-draws exactly as v5 did.
MAX_PER_CELL = 400

# Leave headroom so a cell is not left one article short by chance.
SAFETY = 1.3

COLUMNS = ["doi", "publisher", "venue", "source_id", "publication_year",
           "publication_date", "period", "scholar_category", "is_oa"]


def attempted():
    tried = set()
    for name in ("sample.csv", "sample_v3.csv", "sample_v4.csv",
                 "sample_v5.csv"):
        path = os.path.join(V2, name)
        if os.path.exists(path):
            frame = pd.read_csv(path, usecols=["doi"])
            tried |= {str(d).lower().strip() for d in frame["doi"].dropna()}
    return tried


def measured_yield():
    """Open-access retention observed in the v5 pass, by publisher."""
    status = pd.read_csv(os.path.join(V2, "fetch_status_v5.csv"))
    status["key"] = status["doi"].astype(str).str.lower().str.strip()
    status["kept"] = status["bytes"].notna()

    sample = pd.read_csv(os.path.join(V2, "sample_v5.csv"),
                         usecols=["doi", "publisher"])
    sample["key"] = sample["doi"].astype(str).str.lower().str.strip()

    merged = sample.merge(status[["key", "kept"]], on="key", how="left")
    merged["kept"] = merged["kept"].fillna(False)

    overall = max(merged["kept"].mean(), MIN_YIELD)
    grouped = merged.groupby("publisher")["kept"].agg(["mean", "size"])
    rates = {p: max(r["mean"], MIN_YIELD)
             for p, r in grouped.iterrows() if r["size"] >= MIN_ATTEMPTS}
    return overall, rates


def main():
    rng = np.random.default_rng(SEED)

    cells = pd.read_csv(COVERAGE)
    short = cells[cells["short"] > 0].copy()
    print(f"{len(short):,} cells short by {int(short['short'].sum()):,}\n")

    overall, rates = measured_yield()
    print(f"yield measured on v5: {overall:.1%} overall")
    for pub, rate in sorted(rates.items(), key=lambda kv: -kv[1]):
        print(f"  {str(pub)[:44]:<44} {rate:5.1%}")
    print()

    corpus = pd.read_parquet(CORPUS, columns=COLUMNS)
    corpus = corpus[corpus["doi"].notna()].copy()
    corpus["key"] = corpus["doi"].astype(str).str.lower().str.strip()

    tried = attempted()
    fresh = corpus[~corpus["key"].isin(tried) & corpus["is_oa"]]
    print(f"untried open-access articles: {len(fresh):,}\n")

    pools = dict(tuple(fresh.groupby(["source_id", "publication_year"])))

    picked = []
    empty = 0
    limited = 0
    expected = 0.0
    for cell in short.itertuples():
        pool = pools.get((cell.source_id, cell.publication_year))
        if pool is None or pool.empty:
            empty += 1
            continue

        rate = rates.get(cell.publisher, overall)
        want = int(np.ceil(cell.short * SAFETY / rate))
        want = min(want, MAX_PER_CELL)
        take = min(want, len(pool))
        if take < want:
            limited += 1

        chosen = pool.iloc[rng.choice(len(pool), take, replace=False)]
        picked.append(chosen.assign(cell_short=cell.short))
        expected += min(cell.short, take * rate)

    if not picked:
        print("nothing available to draw")
        return 1

    draw = pd.concat(picked, ignore_index=True)
    draw["draw"] = "v6"
    draw["route"] = "script"
    draw["substitute"] = True
    draw = draw.drop_duplicates("key").drop(columns=["key"])
    draw.to_csv(OUT, index=False)

    print(f"drawn         : {len(draw):,} open-access articles")
    print(f"cells covered : {len(short) - empty:,} of {len(short):,}")
    if empty:
        print(f"  {empty} cells have nothing untried and open access left")
    if limited:
        print(f"  {limited} cells could not supply the full draw")
    print(f"expected to fill roughly {int(expected):,} of "
          f"{int(short['short'].sum()):,}")
    print()
    print("by publisher:")
    for name, n in draw["publisher"].value_counts().head(8).items():
        print(f"  {str(name)[:44]:<44} {n:6,}")
    print()
    print(f"wrote {OUT}")
    print("next: python resolve_v6.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
