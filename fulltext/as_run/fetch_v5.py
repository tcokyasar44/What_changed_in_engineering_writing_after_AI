"""
Download the substitution sample, trying every located copy of an article.

Retrieval policy is unchanged and deliberately so. Hosts that refuse automated
access are skipped rather than argued with, robots directives are honoured, requests
are paced per host, the identity sent is the study's own, and articles free to read
without an explicit licence are measured in memory and never stored. A publisher
that declines a script still declines it; the refusal is recorded and reported as a
finding about what the literature permits.

What differs from the first pass is only persistence. Each article may have several
candidate locations and they are tried in turn, repositories first, because a
repository copy is fifty times more likely to be retrievable than a publisher copy.
Momentary failures, timeouts and reset connections and 5xx responses, are retried
rather than recorded as permanent.

Runs alongside resolve_v5.py: it works through what has been located so far, sleeps,
and rereads for whatever arrived meanwhile, so retrieval and resolution proceed
together rather than in sequence.
"""

import hashlib
import os
import re
import shutil
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import urllib.robotparser
from collections import defaultdict

import pandas as pd
import pymupdf

import config
import paths

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

EMAIL = config.CONTACT_EMAIL
HEADERS = {
    "User-Agent": (f"engineering-publishing-study/1.0 (mailto:{EMAIL}) "
                   "academic non-commercial text mining"),
    "Accept": "application/pdf,*/*",
}

RESOLVED_v5_CSV = os.path.join(paths.ROOT, "resolved_v5.csv")
STATUS_v5_CSV = os.path.join(paths.ROOT, "fetch_status_v5.csv")
TRANSIENT_v5 = os.path.join(paths.ROOT, "parsed_transient_v5.parquet")

# Hosts that refuse automated requests, established by direct testing. Redirectors
# are deliberately absent: doi.org and hdl.handle.net forward, usually to
# repositories, and following a redirect defeats no refusal.
BLOCKED_HOSTS = {
    "onlinelibrary.wiley.com", "www.onlinelibrary.wiley.com",
    "www.mdpi.com", "pubs.rsc.org", "pubs.acs.org",
    "ieeexplore.ieee.org", "www.sciencedirect.com", "ars.els-cdn.com",
    "link.springer.com",
}

TRANSIENT = {"URLError", "TimeoutError", "ConnectionResetError",
             "IncompleteRead", "RemoteDisconnected", "http500", "http502",
             "http503", "http504"}

PER_HOST_INTERVAL = 1.5
REQUEST_PAUSE = 0.4
TIMEOUT = 45
MAX_BYTES = 40 * 1024 * 1024
MAX_TRIES = 3
RETRY_BACKOFF = 4.0
SAVE_EVERY = 25
RESCAN_SECONDS = 120
IDLE_ROUNDS_BEFORE_EXIT = 60

_robots = {}
_last_request = defaultdict(float)


def host_of(url):
    return re.sub(r"^https?://", "", str(url or "")).split("/")[0].lower()


def robots_allows(url):
    host = host_of(url)
    if not host:
        return False
    base = f"https://{host}"
    if base not in _robots:
        parser = urllib.robotparser.RobotFileParser()
        parser.set_url(base + "/robots.txt")
        try:
            parser.read()
        except Exception:
            parser = None
        _robots[base] = parser
    parser = _robots[base]
    if parser is None:
        return True
    try:
        return parser.can_fetch(HEADERS["User-Agent"], url)
    except Exception:
        return True


def throttle(url):
    host = host_of(url)
    wait = PER_HOST_INTERVAL - (time.time() - _last_request[host])
    if wait > 0:
        time.sleep(wait)
    _last_request[host] = time.time()


def filename_for(doi):
    return hashlib.sha1(str(doi).encode("utf-8")).hexdigest()[:20] + ".pdf"


def is_transient_licence(record):
    """Free to read but with no explicit licence: parse without keeping a copy."""
    status = str(record.get("oa_status") or "").lower().strip()
    licence = record.get("license")
    licence = None if pd.isna(licence) else str(licence).lower().strip()
    if licence in {"cc-by-nd", "cc-by-nc-nd"}:
        return True
    return status in {"bronze", "closed"}


def measure_in_memory(body):
    try:
        import parse_fulltext as pf
        doc = pymupdf.open(stream=body, filetype="pdf")
        pages = len(doc)
        text = "\n".join(page.get_text("text") for page in doc)
        doc.close()
        return pf.measure(text, pages)
    except Exception:
        return None


def attempt(url, path, retain):
    last = None
    for tries in range(1, MAX_TRIES + 1):
        throttle(url)
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
                ctype = (resp.headers.get("Content-Type") or "").lower()
                body = resp.read(MAX_BYTES)
            if not body.startswith(b"%PDF"):
                return None, f"not_pdf:{ctype[:20]}", None
            if retain:
                with open(path, "wb") as fh:
                    fh.write(body)
                return len(body), None, None
            stats = measure_in_memory(body)
            return (0, None, stats) if stats else (None, "parse_failed", None)
        except urllib.error.HTTPError as exc:
            last = f"http{exc.code}"
        except Exception as exc:
            last = type(exc).__name__
        if last not in TRANSIENT:
            return None, last, None
        if tries < MAX_TRIES:
            time.sleep(RETRY_BACKOFF * tries)
    return None, last, None


def flush(rows, transient_rows):
    if rows:
        frame = pd.DataFrame(rows)
        if os.path.exists(STATUS_v5_CSV):
            frame = pd.concat([pd.read_csv(STATUS_v5_CSV), frame],
                              ignore_index=True)
        frame.drop_duplicates("doi", keep="last").to_csv(STATUS_v5_CSV,
                                                         index=False)
    if transient_rows:
        pd.DataFrame(transient_rows).to_parquet(TRANSIENT_v5, index=False)


def read_resolved():
    """Snapshot resolved_v5.csv, which the resolver may still be rewriting."""
    for _ in range(5):
        try:
            snapshot = RESOLVED_v5_CSV + ".snapshot"
            shutil.copyfile(RESOLVED_v5_CSV, snapshot)
            return pd.read_csv(snapshot)
        except Exception:
            time.sleep(3)
    return None


def pending_work(attempted):
    frame = read_resolved()
    if frame is None:
        return None
    frame = frame[frame["pdf_url"].notna()].copy()
    frame["host"] = frame["pdf_url"].map(host_of)
    frame = frame[~frame["host"].isin(BLOCKED_HOSTS)]
    frame = frame[~frame["doi"].astype(str).isin(attempted)]
    return list(frame.sort_values(["doi", "rank"]).groupby("doi"))


def main():
    paths.ensure()
    print(f"collection: {paths.COLLECTION}")
    print(f"writing to: {paths.PDF_DIR}\n", flush=True)

    transient_rows = []
    if os.path.exists(TRANSIENT_v5):
        transient_rows = pd.read_parquet(TRANSIENT_v5).to_dict("records")

    attempted = set()
    if os.path.exists(STATUS_v5_CSV):
        prior = pd.read_csv(STATUS_v5_CSV, usecols=["doi"])
        attempted = set(prior["doi"].astype(str))
        print(f"resuming: {len(attempted):,} already attempted", flush=True)

    kept = 0
    idle_rounds = 0

    while True:
        if not os.path.exists(RESOLVED_v5_CSV):
            time.sleep(RESCAN_SECONDS)
            continue

        grouped = pending_work(attempted)
        if grouped is None:
            time.sleep(RESCAN_SECONDS)
            continue

        if not grouped:
            idle_rounds += 1
            if idle_rounds >= IDLE_ROUNDS_BEFORE_EXIT:
                held = len(os.listdir(paths.PDF_DIR))
                print(f"\nno new locations for a long while; stopping with "
                      f"{kept:,} new files, {held:,} held in total", flush=True)
                return 0
            time.sleep(RESCAN_SECONDS)
            continue

        idle_rounds = 0
        print(f"{len(grouped):,} articles with an unattempted location",
              flush=True)

        rows = []
        for i, (doi, group) in enumerate(grouped, start=1):
            doi = str(doi)
            attempted.add(doi)
            path = os.path.join(paths.PDF_DIR, filename_for(doi))
            error = "no_usable_location"
            size = None

            for record in group.to_dict("records"):
                url = record["pdf_url"]
                if not robots_allows(url):
                    error = "robots_disallowed"
                    continue
                retain = not is_transient_licence(record)
                size, error, stats = attempt(url, path, retain)
                if stats is not None:
                    stats["doi"] = doi
                    transient_rows.append(stats)
                if size is not None:
                    kept += 1
                    break
                time.sleep(REQUEST_PAUSE)

            rows.append({
                "doi": doi,
                "file": os.path.basename(path) if (size and size > 0) else None,
                "bytes": size,
                "error": error,
                "retained": bool(size and size > 0),
            })

            if len(rows) >= SAVE_EVERY:
                flush(rows, transient_rows)
                rows = []
                held = len(os.listdir(paths.PDF_DIR))
                print(f"  {i:,}/{len(grouped):,}  files held {held:,}  "
                      f"new this pass {kept:,}", flush=True)

        flush(rows, transient_rows)
        time.sleep(RESCAN_SECONDS)


if __name__ == "__main__":
    sys.exit(main())



