"""
Rebuild the full-text sample manifest from the DOIs that were actually drawn.

sample.csv was lost during the manuscript restructuring. It must not be redrawn:
build_v2_sample.py takes a fresh random draw, and although the seed is fixed, the
corpus has since been rebuilt with recovered abstracts, so the draw would no longer
select the same articles. The PDFs already retrieved would then belong to a sample
that no longer exists.

The drawn DOIs survive in resolved.csv, which has one row per sampled article and
covers all 119,971. This joins those DOIs back to the corpus to restore the
attributes the manifest carried, so the sampling frame is exactly the one the
retrieval passes ran against.

The route column is recomputed by the original rule: open access is reachable by
script, subscription needs a reader at a subscribing institution. It records who
fetches an article, not whether it belongs in the sample.
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

COLUMNS = ["doi", "publisher", "venue", "source_id", "publication_year",
           "publication_date", "period", "scholar_category", "is_oa"]


def main():
    if os.path.exists(paths.SAMPLE_CSV):
        print(f"{paths.SAMPLE_CSV} already exists; refusing to overwrite")
        return 1
    if not os.path.exists(paths.RESOLVED_CSV):
        print("resolved.csv missing; the drawn DOIs cannot be recovered")
        return 1

    drawn = pd.read_csv(paths.RESOLVED_CSV, usecols=["doi"])
    drawn = drawn["doi"].dropna().astype(str).drop_duplicates()
    print(f"DOIs drawn: {len(drawn):,}")

    corpus = pd.read_parquet(CORPUS, columns=COLUMNS)
    corpus["doi"] = corpus["doi"].astype(str)

    # The corpus holds a small number of exact duplicate records, one DOI appearing
    # twice with identical fields. The original draw deduplicated on DOI and so
    # does this, otherwise the restored manifest would have more rows than the
    # sample that was actually drawn.
    sample = corpus[corpus["doi"].isin(set(drawn))].drop_duplicates("doi").copy()
    missing = len(drawn) - len(sample)
    if missing:
        print(f"warning: {missing:,} drawn DOIs are absent from the corpus")

    sample["route"] = np.where(sample["is_oa"].fillna(False), "script", "manual")
    sample = (sample.sort_values(["publication_year", "source_id"])
              .reset_index(drop=True))

    paths.ensure()
    sample.to_csv(paths.SAMPLE_CSV, index=False)
    sample.to_csv(paths.TARGET_SAMPLE_CSV, index=False)

    print(f"restored: {len(sample):,} articles")
    print(f"  script route : {int((sample['route'] == 'script').sum()):,}")
    print(f"  manual route : {int((sample['route'] == 'manual').sum()):,}")

    print("\nfidelity, publisher share of sample against corpus:")
    compare = pd.DataFrame({
        "sample_pct": sample["publisher"].value_counts(normalize=True).mul(100),
        "corpus_pct": corpus["publisher"].value_counts(normalize=True).mul(100),
    }).fillna(0)
    print(compare[compare["corpus_pct"] >= 1].round(1).to_string())

    print(f"\nwrote {paths.SAMPLE_CSV}")
    print(f"wrote {paths.TARGET_SAMPLE_CSV}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
