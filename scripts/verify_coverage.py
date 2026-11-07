"""
Step 6. Check that the corpus holds every article OpenAlex lists for each panel
journal and year.

For every journal-year the script asks OpenAlex how many works of type "article"
with a DOI it holds, and compares that with the number of distinct DOIs in the
corpus. Records without a DOI are left out of both counts: in this panel they are
almost all machine translations, which the corpus removes on purpose.

Small differences in either direction are expected, because OpenAlex keeps
re-indexing after the harvest. A journal-year below 97 percent is listed.

Usage: python verify_coverage.py [first_n_journals]   (default: all)
Output: data/coverage_by_doi.csv
"""

import sys
import time

import pandas as pd

import config
import repo

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

THRESHOLD = 0.97


def openalex_count(source_id, year):
    data = config.openalex("works", {
        "filter": f"primary_location.source.id:{source_id},type:article,"
                  f"publication_year:{year},has_doi:true",
        "per-page": "1"}, use_cache=False)
    return int(data.get("meta", {}).get("count", 0))


def main():
    corpus = pd.read_parquet(repo.CORPUS, columns=["source_id", "publication_date", "doi"])
    corpus["year"] = pd.to_datetime(corpus["publication_date"]).dt.year
    held = corpus[corpus["doi"].notna()].groupby(["source_id", "year"])["doi"].nunique()
    panel = pd.read_csv(repo.PANEL)
    ids = panel["openalex_id"].tolist()
    if len(sys.argv) > 1:
        ids = ids[:int(sys.argv[1])]

    rows = []
    for i, sid in enumerate(ids, 1):
        for year in range(config.STUDY_START_YEAR, config.STUDY_END_YEAR + 1):
            want = openalex_count(sid, year)
            got = int(held.get((sid, year), 0))
            if want or got:
                rows.append({"source_id": sid, "year": year, "openalex": want, "corpus": got})
            time.sleep(0.05)
        if i % 20 == 0:
            print(f"  {i}/{len(ids)} journals", flush=True)

    c = pd.DataFrame(rows)
    c["ratio"] = c["corpus"] / c["openalex"].where(c["openalex"] > 0)
    c.to_csv(repo.COVERAGE, index=False)
    print(f"\njournals {c['source_id'].nunique()}, journal-years {len(c):,}")
    print(f"OpenAlex articles with a DOI: {c['openalex'].sum():,}; in corpus: {c['corpus'].sum():,} "
          f"({100 * c['corpus'].sum() / c['openalex'].sum():.2f}%)")
    low = c[c["ratio"] < THRESHOLD]
    print(f"journal-years below {THRESHOLD:.0%}: {len(low)}")
    if len(low):
        print(low.sort_values("ratio").head(15).to_string(index=False))
    print(f"wrote {repo.COVERAGE}")


if __name__ == "__main__":
    main()
