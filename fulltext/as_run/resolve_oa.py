"""
Locate a downloadable PDF for each sampled article.

Unpaywall indexes legal open-access copies and reports the best one per DOI. It is
used here rather than scraping publisher pages, because it respects licensing and
because publisher sites block automated access quickly.

The script is resumable: results are written after every batch, and rerunning skips
DOIs already resolved. Not every open-access article has a retrievable PDF, so the
resolution rate is reported and carried into the analysis as a coverage figure.

Stage 2 of the pipeline. Run fetch_fulltext.py next.
"""

import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

import pandas as pd

import config
import paths

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
FT = paths.FT

SAMPLE = paths.SAMPLE_CSV
RESOLVED = paths.RESOLVED_CSV

EMAIL = config.CONTACT_EMAIL
HEADERS = {"User-Agent": f"engineering-publishing-study/1.0 (mailto:{EMAIL})"}

SAVE_EVERY = 50
PAUSE = 0.12


def unpaywall(doi):
    url = f"https://api.unpaywall.org/v2/{urllib.parse.quote(doi)}?email={EMAIL}"
    try:
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.load(resp)
    except urllib.error.HTTPError as exc:
        return {"_error": f"http{exc.code}"}
    except Exception as exc:
        return {"_error": type(exc).__name__}


# Licences that explicitly forbid derivative use. Counting words is unlikely to
# constitute a derivative work, but excluding these costs almost nothing and removes
# any argument about it.
BLOCKED_LICENCES = {"cc-by-nd", "cc-by-nc-nd"}


def best_pdf(record):
    """Return the first retrievable PDF, together with its licence and host."""
    if "_error" in record:
        return None, None, None, record["_error"]

    locations = []
    best = record.get("best_oa_location")
    if best:
        locations.append(best)
    locations.extend(record.get("oa_locations") or [])

    for loc in locations:
        if not loc:
            continue
        url = loc.get("url_for_pdf")
        if not url:
            continue
        licence = (loc.get("license") or "").lower() or None
        if licence in BLOCKED_LICENCES:
            continue
        return url, loc.get("host_type"), licence, None

    return None, None, None, "no_pdf_url"


def main():
    sample = pd.read_csv(SAMPLE)

    # Prior results are reloaded as plain dictionaries. Mixing pandas Series with
    # dictionaries in the same list makes DataFrame construction fail.
    done = {}
    if os.path.exists(RESOLVED):
        prior = pd.read_csv(RESOLVED)
        done = {r["doi"]: r for r in prior.to_dict("records")}
        print(f"resuming: {len(done):,} already resolved")

    rows = list(done.values())
    todo = [d for d in sample["doi"].dropna().tolist() if d not in done]
    print(f"to resolve: {len(todo):,}")

    for i, doi in enumerate(todo, start=1):
        record = unpaywall(doi)
        url, host, licence, error = best_pdf(record)
        rows.append({
            "doi": doi,
            "pdf_url": url,
            "host_type": host,
            "license": licence,
            "error": error,
            "oa_status": record.get("oa_status"),
            "publisher": record.get("publisher"),
        })

        if i % SAVE_EVERY == 0 or i == len(todo):
            pd.DataFrame(rows).to_csv(RESOLVED, index=False)
            found = sum(1 for r in rows if r.get("pdf_url"))
            print(f"  {i:,}/{len(todo):,}  pdf found for {found:,} "
                  f"({100 * found / max(len(rows), 1):.0f}%)", flush=True)

        time.sleep(PAUSE)

    frame = pd.DataFrame(rows)
    frame.to_csv(RESOLVED, index=False)

    found = frame["pdf_url"].notna().sum()
    print(f"\nresolved {len(frame):,} DOIs, {found:,} with a PDF "
          f"({100 * found / max(len(frame), 1):.0f}%)")
    if "host_type" in frame:
        print("\nby host type:")
        print(frame["host_type"].value_counts().to_string())
    if "oa_status" in frame:
        print("\nby open-access status:")
        print(frame.loc[frame["pdf_url"].notna(), "oa_status"]
              .value_counts(dropna=False).to_string())
    if "license" in frame:
        print("\nby licence (none means free to read without an explicit licence):")
        print(frame.loc[frame["pdf_url"].notna(), "license"]
              .value_counts(dropna=False).head(12).to_string())
    print("\nNext: python fetch_fulltext.py")


if __name__ == "__main__":
    main()
