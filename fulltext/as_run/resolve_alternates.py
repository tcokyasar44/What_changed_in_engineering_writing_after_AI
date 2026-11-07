"""
Recover the open-access locations the first resolution pass discarded.

resolve_oa.py returns the first PDF URL Unpaywall reports and stops. That is almost
always the publisher copy, because Unpaywall lists best_oa_location first. The
fetcher then skipped most publisher hosts, so those articles were recorded as
unreachable even when a repository held the same paper.

The gap is large. Publisher-hosted copies converted at 0.3 percent, repository copies
at 50.5 percent, and a stratified probe of the skipped set found that 24 percent by
volume have a repository copy that was never attempted.

This pass rereads Unpaywall for articles whose only attempted location was a host the
fetcher skips, and records every location rather than one. It writes a separate file
and leaves resolved.csv untouched, so the original pass stays reproducible and the
two can be compared.

Nothing here evades a refusal. A publisher that declines automated access still
declines; this looks elsewhere for a copy the author or their institution posted
legitimately.

Run resolve_alternates.py, then fetch_alternates.py.
"""

import json
import os
import re
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

ALTERNATES_CSV = os.path.join(paths.ROOT, "alternates.csv")

SAVE_EVERY = 100
PAUSE = 0.12

# Licences forbidding derivative use, excluded exactly as in the first pass.
BLOCKED_LICENCES = {"cc-by-nd", "cc-by-nc-nd"}

# Hosts the fetcher will not request. Kept identical to fetch_loop.py so that this
# pass targets precisely the articles the fetcher would otherwise pass over. The two
# redirectors the first pass wrongly listed as publishers are absent: doi.org and
# hdl.handle.net forward to repositories and are followed rather than skipped.
SKIPPED_HOSTS = {
    "onlinelibrary.wiley.com", "www.onlinelibrary.wiley.com",
    "www.mdpi.com", "pubs.rsc.org", "pubs.acs.org",
    "ieeexplore.ieee.org", "www.sciencedirect.com", "ars.els-cdn.com",
    "link.springer.com",
}


def host_of(url):
    return re.sub(r"^https?://", "", str(url or "")).split("/")[0].lower()


def unpaywall(doi):
    url = f"https://api.unpaywall.org/v2/{urllib.parse.quote(doi)}?email={EMAIL}"
    try:
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.load(resp)
    except Exception as exc:
        return {"_error": type(exc).__name__}


def usable_locations(record):
    """Every PDF location, repositories first, worst-licensed removed."""
    if "_error" in record:
        return []

    seen = set()
    out = []
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
        out.append({
            "pdf_url": url,
            "host_type": loc.get("host_type"),
            "license": licence,
            "version": loc.get("version"),
            "repository": loc.get("repository_institution"),
        })

    # Repository copies are tried first: they convert at 50.5 percent against 0.3
    # for publisher copies, and they are the locations this pass exists to find.
    out.sort(key=lambda d: 0 if d["host_type"] == "repository" else 1)
    return out


def main():
    resolved = pd.read_csv(paths.RESOLVED_CSV,
                           usecols=["doi", "pdf_url", "oa_status", "publisher"])
    status = pd.read_csv(paths.FETCH_STATUS_CSV, usecols=["doi", "retained",
                                                          "error"])
    merged = resolved.merge(status, on="doi", how="left")

    # Articles worth revisiting: never retained, and either skipped outright or
    # located only at a host the fetcher will not request.
    unretained = merged[merged["retained"] != True]  # noqa: E712
    targets = unretained[
        (unretained["error"] == "publisher_blocks_automated_access")
        | (unretained["pdf_url"].map(host_of).isin(SKIPPED_HOSTS))
    ]
    todo = sorted(set(targets["doi"].dropna().astype(str)))

    done = set()
    if os.path.exists(ALTERNATES_CSV):
        prior = pd.read_csv(ALTERNATES_CSV, usecols=["doi"])
        done = set(prior["doi"].astype(str))
        print(f"resuming: {len(done):,} already probed")
    todo = [d for d in todo if d not in done]

    print(f"articles to reprobe: {len(todo):,}", flush=True)
    if not todo:
        print("nothing to do")
        return 0

    rows = []
    found = 0
    for i, doi in enumerate(todo, start=1):
        record = unpaywall(doi)
        locations = usable_locations(record)
        repository = [loc for loc in locations
                      if loc["host_type"] == "repository"]
        if repository:
            found += 1
        for rank, loc in enumerate(locations):
            rows.append({"doi": doi, "rank": rank, **loc,
                         "oa_status": record.get("oa_status"),
                         "publisher": record.get("publisher")})
        if not locations:
            rows.append({"doi": doi, "rank": None, "pdf_url": None,
                         "host_type": None, "license": None, "version": None,
                         "repository": None,
                         "oa_status": record.get("oa_status"),
                         "publisher": record.get("publisher")})

        if i % SAVE_EVERY == 0 or i == len(todo):
            frame = pd.DataFrame(rows)
            if os.path.exists(ALTERNATES_CSV):
                frame = pd.concat([pd.read_csv(ALTERNATES_CSV), frame],
                                  ignore_index=True)
            frame.drop_duplicates(["doi", "pdf_url"], keep="last").to_csv(
                ALTERNATES_CSV, index=False)
            rows = []
            print(f"  {i:,}/{len(todo):,}  repository copy for {found:,} "
                  f"({100 * found / i:.0f}%)", flush=True)

        time.sleep(PAUSE)

    frame = pd.read_csv(ALTERNATES_CSV)
    repo = frame[frame["host_type"] == "repository"]
    print(f"\n{frame['doi'].nunique():,} articles probed")
    print(f"{repo['doi'].nunique():,} have a repository copy never attempted")
    print("\nby repository host:")
    print(repo["pdf_url"].map(host_of).value_counts().head(15).to_string())
    print("\nNext: python fetch_alternates.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
