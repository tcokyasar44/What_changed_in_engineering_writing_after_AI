"""
Locate open-access copies for the substitution sample, keeping every location.

The first pass kept only the first location Unpaywall reported. That is almost
always the publisher copy, and publisher copies were retrieved 0.3 percent of the
time against 50.5 percent for repository copies. Recovering the discarded
alternatives afterwards added 1,483 articles to a collection of 2,616, so the loss
was larger than the original result.

This records every location from the start, repositories first, and writes them as
one row per location rather than one per article. fetch_v6.py works through the
candidates for an article until one succeeds.

Resumable: progress is written every batch and rerunning skips articles already
probed. Expect roughly 20 hours for 175,000 articles at Unpaywall's pace.
"""

import json
import os
import sys
import time
import urllib.parse
import urllib.request

import pandas as pd

import config
import paths

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

EMAIL = config.CONTACT_EMAIL
HEADERS = {"User-Agent": f"engineering-publishing-study/1.0 (mailto:{EMAIL})"}

SAMPLE_v6_CSV = os.path.join(paths.ROOT, "sample_v6.csv")
RESOLVED_v6_CSV = os.path.join(paths.ROOT, "resolved_v6.csv")

SAVE_EVERY = 200
PAUSE = 0.12

BLOCKED_LICENCES = {"cc-by-nd", "cc-by-nc-nd"}


def unpaywall(doi):
    url = f"https://api.unpaywall.org/v2/{urllib.parse.quote(doi)}?email={EMAIL}"
    try:
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.load(resp)
    except Exception as exc:
        return {"_error": type(exc).__name__}


def locations_of(record):
    """Every usable PDF location, repositories first."""
    if "_error" in record:
        return []

    seen = set()
    found = []
    candidates = []
    best = record.get("best_oa_location")
    if best:
        candidates.append(best)
    candidates.extend(record.get("oa_locations") or [])

    for loc in candidates:
        if not loc:
            continue
        url = loc.get("url_for_pdf")
        if not url or url in seen:
            continue
        licence = (loc.get("license") or "").lower() or None
        if licence in BLOCKED_LICENCES:
            continue
        seen.add(url)
        found.append({
            "pdf_url": url,
            "host_type": loc.get("host_type"),
            "license": licence,
            "version": loc.get("version"),
        })

    found.sort(key=lambda d: 0 if d["host_type"] == "repository" else 1)
    return found


def main():
    if not os.path.exists(SAMPLE_v6_CSV):
        print("sample_v6.csv missing; run build_v6_openaccess.py first")
        return 1

    sample = pd.read_csv(SAMPLE_v6_CSV, usecols=["doi", "publisher"])
    todo = sample["doi"].dropna().astype(str).tolist()

    done = set()
    if os.path.exists(RESOLVED_v6_CSV):
        prior = pd.read_csv(RESOLVED_v6_CSV, usecols=["doi"])
        done = set(prior["doi"].astype(str))
        print(f"resuming: {len(done):,} already probed")
    todo = [d for d in todo if d not in done]
    print(f"to resolve: {len(todo):,}", flush=True)

    rows = []
    with_pdf = 0
    with_repo = 0

    for i, doi in enumerate(todo, start=1):
        record = unpaywall(doi)
        found = locations_of(record)
        if found:
            with_pdf += 1
            if any(loc["host_type"] == "repository" for loc in found):
                with_repo += 1
            for rank, loc in enumerate(found):
                rows.append({"doi": doi, "rank": rank, **loc,
                             "oa_status": record.get("oa_status"),
                             "publisher": record.get("publisher")})
        else:
            rows.append({"doi": doi, "rank": None, "pdf_url": None,
                         "host_type": None, "license": None, "version": None,
                         "oa_status": record.get("oa_status"),
                         "publisher": record.get("publisher")})

        if i % SAVE_EVERY == 0 or i == len(todo):
            frame = pd.DataFrame(rows)
            if os.path.exists(RESOLVED_v6_CSV):
                frame = pd.concat([pd.read_csv(RESOLVED_v6_CSV), frame],
                                  ignore_index=True)
            frame.drop_duplicates(["doi", "pdf_url"], keep="last").to_csv(
                RESOLVED_v6_CSV, index=False)
            rows = []
            print(f"  {i:,}/{len(todo):,}  pdf {with_pdf:,} "
                  f"({100 * with_pdf / i:.0f}%)  repository copy "
                  f"{with_repo:,} ({100 * with_repo / i:.0f}%)", flush=True)

        time.sleep(PAUSE)

    print("\nNext: python fetch_v6.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())




