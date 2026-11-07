"""
Step 3. Collect article metadata for every panel journal from OpenAlex.

Pages through every work of type "article" whose primary location is the journal
and whose publication date falls in the study window, and writes one Parquet shard
per journal to data/works/. A manifest records each finished journal with the time
it was collected, so an interrupted run resumes where it stopped.

OpenAlex type "article" covers research articles. Reviews, editorials, letters and
errata have their own types and are not collected.

Per article only the fields the study uses are kept: identifiers and dates, title
and abstract, author and institution counts and countries, reference count, open
access status, page fields and the language label.

Usage:
    python harvest.py                harvest every panel journal not yet in the manifest
    python harvest.py S123 S456      harvest (or re-harvest) the named journals only
"""

import json
import os
import sys
import time
from datetime import datetime, timezone

import pandas as pd

import config
import repo

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PER_PAGE = 200
PAUSE = 0.12

SELECT = ",".join([
    "id", "doi", "title", "publication_date", "publication_year", "type",
    "authorships", "primary_location", "open_access", "referenced_works_count",
    "abstract_inverted_index", "is_retracted", "language", "biblio",
])


def log(msg):
    print(f"[{datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S}] {msg}", flush=True)


def load_manifest():
    if os.path.exists(repo.MANIFEST):
        with open(repo.MANIFEST, encoding="utf-8") as fh:
            return json.load(fh)
    return {"done": {}, "failed": {}}


def save_manifest(man):
    tmp = repo.MANIFEST + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(man, fh, indent=1)
    os.replace(tmp, repo.MANIFEST)


def reconstruct_abstract(inverted):
    """Rebuild abstract text from OpenAlex's inverted index (word -> positions)."""
    if not inverted:
        return None
    positions = sorted((i, w) for w, idxs in inverted.items() for i in idxs)
    return " ".join(w for _, w in positions) or None


def page_count(first, last, cap=200):
    """Pages from the first and last page fields, or None when they give no length.

    Many publishers deposit an article number as both first and last page; that is
    not a one-page article, so equal values give None. Some deposit two unrelated
    identifiers, which can imply millions of pages; spans above `cap` are metadata
    errors and also give None.
    """
    try:
        first, last = int(first), int(last)
    except (TypeError, ValueError):
        return None
    span = last - first + 1
    return span if (last > first > 0 and span <= cap) else None


def flatten(work):
    loc = work.get("primary_location") or {}
    src = loc.get("source") or {}
    oa = work.get("open_access") or {}
    biblio = work.get("biblio") or {}
    authorships = work.get("authorships") or []

    countries, institutions, corresponding = set(), set(), None
    for a in authorships:
        for inst in a.get("institutions") or []:
            if inst.get("country_code"):
                countries.add(inst["country_code"])
            if inst.get("display_name"):
                institutions.add(inst["display_name"])
        if a.get("is_corresponding") and not corresponding:
            ccs = a.get("countries") or []
            corresponding = ccs[0] if ccs else None

    abstract = reconstruct_abstract(work.get("abstract_inverted_index"))
    return {
        "openalex_id": (work.get("id") or "").rsplit("/", 1)[-1],
        "doi": (work.get("doi") or "").replace("https://doi.org/", "").lower() or None,
        "source_id": (src.get("id") or "").rsplit("/", 1)[-1] or None,
        "issn_l": src.get("issn_l"),
        "publication_date": work.get("publication_date"),
        "publication_year": work.get("publication_year"),
        "type": work.get("type"),
        "language": work.get("language"),
        "title": work.get("title"),
        "abstract": abstract,
        "abstract_words": len(abstract.split()) if abstract else 0,
        "n_authors": len(authorships),
        "n_institutions": len(institutions),
        "countries": ";".join(sorted(countries)) or None,
        "n_countries": len(countries),
        "corresponding_country": corresponding,
        "n_references": work.get("referenced_works_count") or 0,
        "is_oa": oa.get("is_oa"),
        "oa_status": oa.get("oa_status"),
        "is_retracted": work.get("is_retracted"),
        "first_page": biblio.get("first_page"),
        "last_page": biblio.get("last_page"),
        "n_pages": page_count(biblio.get("first_page"), biblio.get("last_page")),
    }


def harvest_journal(source_id):
    rows, cursor = [], "*"
    while cursor:
        data = config.openalex("works", {
            "filter": f"primary_location.source.id:{source_id},type:article,"
                      f"from_publication_date:{config.STUDY_START_YEAR}-01-01,"
                      f"to_publication_date:{config.STUDY_END_YEAR}-12-31",
            "select": SELECT, "per-page": str(PER_PAGE), "cursor": cursor,
        }, use_cache=False)
        results = data.get("results", [])
        rows.extend(flatten(w) for w in results)
        cursor = (data.get("meta") or {}).get("next_cursor")
        if not results:
            break
        time.sleep(PAUSE)
    return rows


def main():
    repo.ensure()
    panel = pd.read_csv(repo.PANEL)
    man = load_manifest()
    names = dict(zip(panel["openalex_id"], panel["venue"]))

    wanted = sys.argv[1:]
    if wanted:
        todo = [s for s in wanted if s in names]
    else:
        todo = [s for s in panel["openalex_id"] if s not in man["done"]]
    log(f"panel={len(panel)} done={len(man['done'])} to harvest={len(todo)}")

    for i, sid in enumerate(todo, start=1):
        try:
            rows = harvest_journal(sid)
        except Exception as exc:
            man["failed"][sid] = f"{type(exc).__name__}: {exc}"
            save_manifest(man)
            log(f"  {i}/{len(todo)} FAILED {names[sid][:44]} ({type(exc).__name__})")
            continue
        pd.DataFrame(rows).to_parquet(os.path.join(repo.WORKS, f"{sid}.parquet"), index=False)
        man["done"][sid] = {
            "venue": names[sid], "n_works": len(rows),
            "harvested_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        man["failed"].pop(sid, None)
        save_manifest(man)
        log(f"  {i}/{len(todo)} {names[sid][:44]:46s} {len(rows):>7,} works")

    log(f"finished: {len(man['done'])} journals in manifest, {len(man['failed'])} failed")


if __name__ == "__main__":
    main()
