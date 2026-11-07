"""
Measure how much was lost by keeping only one open-access location per article.

resolve_oa.py returns the first PDF URL Unpaywall reports and discards the rest.
That is almost always the publisher copy, and publisher copies converted at 0.3
percent while repository copies converted at 50.5 percent. If the same articles
also have a repository copy, they were never unreachable; the alternative was
recorded and thrown away.

This probes a stratified sample of the articles that were skipped as publisher
blocks, and reports how many have a repository copy that was never attempted.
Read-only: it writes nothing and changes nothing.
"""

import json
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
PER_HOST = 12


def host_of(url):
    return re.sub(r"^https?://", "", str(url)).split("/")[0].lower()


def unpaywall(doi):
    url = f"https://api.unpaywall.org/v2/{urllib.parse.quote(doi)}?email={EMAIL}"
    try:
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.load(resp)
    except Exception:
        return None


def main():
    resolved = pd.read_csv(paths.RESOLVED_CSV,
                           usecols=["doi", "pdf_url", "oa_status", "host_type"])
    status = pd.read_csv(paths.FETCH_STATUS_CSV,
                         usecols=["doi", "retained", "error"])
    merged = resolved[resolved["pdf_url"].notna()].merge(status, on="doi",
                                                         how="left")
    blocked = merged[merged["error"] == "publisher_blocks_automated_access"].copy()
    blocked["host"] = blocked["pdf_url"].map(host_of)

    print(f"skipped as publisher blocks: {len(blocked):,}")
    print(blocked["host"].value_counts().head(10).to_string())
    print()

    # Recent pandas drops the grouping column inside apply, so sample per host
    # explicitly and concatenate.
    sample = pd.concat([group.sample(min(PER_HOST, len(group)), random_state=1)
                        for _, group in blocked.groupby("host")])
    print(f"probing {len(sample)} articles for alternate locations\n", flush=True)

    tally = {}
    for record in sample.to_dict("records"):
        data = unpaywall(record["doi"])
        time.sleep(0.12)
        if not data:
            continue
        locations = [loc for loc in (data.get("oa_locations") or [])
                     if loc and loc.get("url_for_pdf")]
        repository = [loc for loc in locations
                      if loc.get("host_type") == "repository"]
        entry = tally.setdefault(record["host"], {"n": 0, "locs": 0, "repo": 0})
        entry["n"] += 1
        entry["locs"] += len(locations)
        entry["repo"] += 1 if repository else 0

    print(f"{'host':<32}{'n':>5}{'avg locs':>10}{'has repo copy':>16}")
    for host, entry in sorted(tally.items(), key=lambda kv: -kv[1]["n"]):
        share = 100 * entry["repo"] / entry["n"]
        print(f"{host:<32}{entry['n']:>5}{entry['locs'] / entry['n']:>10.1f}"
              f"{entry['repo']:>10} ({share:.0f}%)")

    total = sum(e["n"] for e in tally.values())
    repo = sum(e["repo"] for e in tally.values())
    if total:
        print(f"\noverall {repo}/{total} ({100 * repo / total:.0f}%) have a "
              f"repository copy that was never attempted")
        print(f"projected recoverable: "
              f"{int(len(blocked) * repo / total):,} articles")
    return 0


if __name__ == "__main__":
    sys.exit(main())
