"""
Step 1. Resolve the Google Scholar Metrics venue lists to OpenAlex journals.

Input: data/scholar_metrics_lists.csv, the top-20 venue lists of fifteen Google
Scholar Metrics engineering subcategories (rank, venue name, h5-index, h5-median),
as published on scholar.google.com and recorded on 25 September 2026. The lists are
saved in the package rather than fetched by a script: Google Scholar does not permit
automated access, and saving them fixes the sampling frame at a known date.

For each venue this script records whether it is a conference, the OpenAlex source
it corresponds to, the publisher, and two descriptive measures taken from a sample
of 2024 articles (share with an abstract, share with a reference list).

Matching is strict. A venue is matched only when its normalised name equals the
OpenAlex display name or one of its alternate titles. A loose match (one name
contained in the other) once paired the APA journal "Decision" with Elsevier's
"Decision Support Systems"; strictness prevents that. Venues that need a hand
correction are listed, with the reason, in data/venue_corrections.csv and are
resolved by their OpenAlex identifier. Anything else that does not match exactly is
left unmatched and is excluded later by rule R2, so no match is guessed.

Usage:
    python resolve_venues.py               resolve every venue (live OpenAlex)
    python resolve_venues.py --corrections only re-resolve the venues listed in
                                           venue_corrections.csv, keeping every other
                                           row of an existing scholar_panel_raw.csv

Output: data/scholar_panel_raw.csv
"""

import argparse
import csv
import os
import re
import sys

import pandas as pd

import config
import repo

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SAMPLE_SIZE = 50

CONFERENCE_MARKERS = (
    "conference", "symposium", "workshop", "proceedings", "meeting of the",
    "congress", "neural information processing systems", "annual meeting",
    "siggraph", "usenix", "acm sigcomm", "interspeech", "general meeting",
    "society general",
)

FIELDS = [
    "scholar_category", "scholar_slug", "scholar_rank", "venue", "h5_index",
    "h5_median", "venue_type", "openalex_id", "openalex_name", "match",
    "publisher", "issn_l", "abstract_coverage_pct", "refs_coverage_pct",
    "sample_n", "works_window_total", "min_year_count",
]


def normalise(text):
    text = str(text).lower().replace("&", "and")
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def is_conference(venue):
    low = str(venue).lower()
    return any(m in low for m in CONFERENCE_MARKERS)


def exact_match(venue):
    """The OpenAlex journal whose name or alternate title equals the venue name."""
    data = config.openalex("sources", {
        "search": venue, "filter": "type:journal", "per-page": "10"})
    target = normalise(venue)
    for src in data.get("results", []):
        names = [src.get("display_name") or ""] + (src.get("alternate_titles") or [])
        if any(normalise(n) == target for n in names if n):
            return src
    return None


def source_by_id(source_id):
    return config.openalex(f"sources/{source_id}")


def yearly_counts(source_id):
    data = config.openalex("works", {
        "filter": f"primary_location.source.id:{source_id},type:article,"
                  f"from_publication_date:{config.STUDY_START_YEAR}-01-01,"
                  f"to_publication_date:{config.STUDY_END_YEAR}-12-31",
        "group_by": "publication_year"})
    return {int(g["key"]): g["count"] for g in data.get("group_by", [])}


def coverage(source_id):
    """Abstract and reference-list coverage in a sample of 2024 articles."""
    data = config.openalex("works", {
        "filter": f"primary_location.source.id:{source_id},type:article,publication_year:2024",
        "select": "id,abstract_inverted_index,referenced_works",
        "per-page": str(SAMPLE_SIZE)})
    rows = data.get("results", [])
    if not rows:
        return None, None, 0
    n = len(rows)
    return (round(100 * sum(bool(r.get("abstract_inverted_index")) for r in rows) / n, 1),
            round(100 * sum(bool(r.get("referenced_works")) for r in rows) / n, 1), n)


def resolve(row, corrections):
    record = {k: row.get(k, "") for k in FIELDS}
    record["venue_type"] = "conference" if is_conference(row["venue"]) else "journal"
    if record["venue_type"] == "conference":
        return record

    key = (row["venue"], row["scholar_category"])
    if key in corrections:
        src, how = source_by_id(corrections[key]), "correction"
    else:
        src, how = exact_match(row["venue"]), "exact"
    if src is None:
        record["match"] = "none"
        return record

    sid = src["id"].rsplit("/", 1)[-1]
    counts = yearly_counts(sid)
    years = range(config.STUDY_START_YEAR, config.STUDY_END_YEAR + 1)
    abs_pct, ref_pct, n = coverage(sid)
    record.update({
        "openalex_id": sid, "openalex_name": src.get("display_name") or "",
        "match": how, "publisher": src.get("host_organization_name") or "",
        "issn_l": src.get("issn_l") or "",
        "abstract_coverage_pct": "" if abs_pct is None else abs_pct,
        "refs_coverage_pct": "" if ref_pct is None else ref_pct, "sample_n": n,
        "works_window_total": sum(counts.get(y, 0) for y in years),
        "min_year_count": min(counts.get(y, 0) for y in years),
    })
    return record


def load_corrections():
    if not os.path.exists(repo.CORRECTIONS):
        return {}
    c = pd.read_csv(repo.CORRECTIONS)
    return {(r.venue, r.scholar_category): r.openalex_id for r in c.itertuples()}


def write(rows):
    tmp = repo.SCHOLAR_RAW + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, repo.SCHOLAR_RAW)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corrections", action="store_true",
                    help="re-resolve only the venues listed in venue_corrections.csv")
    args = ap.parse_args()

    lists = pd.read_csv(repo.SCHOLAR_LISTS)
    corrections = load_corrections()

    if args.corrections:
        if not os.path.exists(repo.SCHOLAR_RAW):
            raise SystemExit("no scholar_panel_raw.csv to update; run without --corrections")
        old = pd.read_csv(repo.SCHOLAR_RAW, dtype=str, keep_default_na=False)
        rows = old.to_dict("records")
        for i, r in enumerate(rows):
            if (r["venue"], r["scholar_category"]) in corrections:
                rows[i] = resolve(r, corrections)
                print(f"  corrected  {r['venue'][:50]:52s} -> {rows[i]['openalex_id']} "
                      f"{rows[i]['openalex_name']}")
    else:
        rows = []
        for r in lists.to_dict("records"):
            rows.append(resolve(r, corrections))
            rec = rows[-1]
            print(f"  {rec['scholar_category'][:28]:30s} {int(rec['scholar_rank']):2d}. "
                  f"{rec['venue'][:44]:46s} {rec['match'] or rec['venue_type']}")
    write(rows)
    config.save_cache()
    m = pd.DataFrame(rows)
    print(f"\n{len(m)} venues; matched {int((m['openalex_id'].astype(str) != '').sum())}; "
          f"conferences {int((m['venue_type'] == 'conference').sum())}")
    print(f"wrote {repo.SCHOLAR_RAW}")


if __name__ == "__main__":
    main()
