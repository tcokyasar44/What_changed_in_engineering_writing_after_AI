"""
Check that the records removed as machine translations are translations.

build_corpus.py removes records whose title carries the translation service's marker,
and records with no DOI and a non-English language label. This script summarises
the non-English records in the harvest shards so that rule can be checked: how many
carry the marker, how many lack a DOI, and a sample of any non-English record that
has a DOI (those are kept in the corpus).
"""

import glob
import os
import sys

import pandas as pd

import repo

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

MARKER = "Powered by NICT"


def main():
    frames = [pd.read_parquet(p, columns=["doi", "title", "language", "publication_year"])
              for p in glob.glob(os.path.join(repo.WORKS, "*.parquet"))]
    if not frames:
        raise SystemExit("no harvest shards; run harvest.py first")
    allr = pd.concat(frames, ignore_index=True)
    non_en = allr[allr["language"].notna() & (allr["language"] != "en")]
    marked = non_en["title"].fillna("").str.contains(MARKER, regex=False)
    no_doi = non_en["doi"].isna()

    print(f"records in shards       : {len(allr):,}")
    print(f"non-English label       : {len(non_en):,}")
    print(f"  with translation mark : {int(marked.sum()):,}")
    print(f"  without a DOI         : {int(no_doi.sum()):,}")
    print(f"  mark and no DOI       : {int((marked & no_doi).sum()):,}")
    print("\nby language:")
    print(non_en["language"].value_counts().head(8).to_string())
    kept = non_en[~marked & ~no_doi]
    print(f"\nnon-English label but a DOI (kept in the corpus): {len(kept):,}")
    for r in kept.head(8).itertuples():
        print(f"  {str(r.language):4s} {str(r.title)[:70]}")


if __name__ == "__main__":
    main()
