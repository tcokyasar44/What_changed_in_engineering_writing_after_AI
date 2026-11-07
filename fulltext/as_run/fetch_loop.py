"""
Download located PDFs continuously, in one process.

The download pass has repeatedly stopped early. Its own logic is sound: it fetches
everything listed in resolved.csv and exits. The problem is that the resolver keeps
appending to that file, so a clean exit leaves thousands of located files
unattempted, with nothing in the log to suggest anything went wrong.

The obvious fix was a supervisor that restarts the downloader. That failed for a
different reason: a process whose only job is to spawn children does not survive in
this environment, while long-running single processes do.

So this does the work itself rather than delegating. It loops over resolved.csv,
fetches what has not been attempted, sleeps, and rereads the file to pick up
whatever the resolver added meanwhile. One process, no children, no supervision.

Retrieval policy is unchanged from the original pass: hosts known to refuse
automated access are skipped rather than retried, robots directives are honoured,
requests are paced per host, and articles that are free to read without an explicit
licence are parsed in memory and never stored.
"""

import hashlib
import os
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

HERE = os.path.dirname(os.path.abspath(__file__))

EMAIL = config.CONTACT_EMAIL
HEADERS = {
    "User-Agent": (f"engineering-publishing-study/1.0 (mailto:{EMAIL}) "
                   "academic non-commercial text mining"),
    "Accept": "application/pdf,*/*",
}

# Hosts that refuse automated requests outright, established by direct testing.
# Retrying them wastes time and achieves nothing; the gap is reported instead.
BLOCKED_HOSTS = {
    "onlinelibrary.wiley.com", "www.onlinelibrary.wiley.com",
    "www.mdpi.com", "pubs.rsc.org", "pubs.acs.org",
    "ieeexplore.ieee.org", "www.sciencedirect.com", "ars.els-cdn.com",
    "link.springer.com", "hdl.handle.net", "dx.doi.org", "doi.org",
}

PER_HOST_INTERVAL = 1.5
REQUEST_PAUSE = 0.5
TIMEOUT = 45
MAX_BYTES = 40 * 1024 * 1024
SAVE_EVERY = 20
RESCAN_SECONDS = 90
IDLE_ROUNDS_BEFORE_EXIT = 400

_robots = {}
_last_request = defaultdict(float)


def host_of(url):
    try:
        return urllib.parse.urlparse(url or "").netloc.lower()
    except Exception:
        return ""


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


def is_transient(record):
    """Free to read but with no explicit licence: parse without keeping a copy."""
    status = str(record.get("oa_status") or "").lower().strip()
    licence = record.get("license")
    licence = None if pd.isna(licence) else str(licence).lower().strip()
    if licence in {"cc-by-nd", "cc-by-nc-nd"}:
        return True
    return status in {"bronze", "closed"}


def measure_in_memory(body):
    """Extract measurements from bytes, then discard them."""
    try:
        import parse_fulltext as pf
        doc = pymupdf.open(stream=body, filetype="pdf")
        pages = len(doc)
        text = "\n".join(page.get_text("text") for page in doc)
        doc.close()
        return pf.measure(text, pages)
    except Exception:
        return None


def fetch_one(url, path, retain):
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


def load_attempted():
    if not os.path.exists(paths.FETCH_STATUS_CSV):
        return set()
    try:
        return set(pd.read_csv(paths.FETCH_STATUS_CSV,
                               usecols=["doi"])["doi"].astype(str))
    except Exception:
        return set()


def main():
    paths.ensure()
    print(f"collection: {paths.COLLECTION}")
    print(f"writing to: {paths.PDF_DIR}\n", flush=True)

    attempted = load_attempted()
    rows = []
    transient_rows = []
    downloaded = 0
    idle_rounds = 0

    if os.path.exists(paths.TRANSIENT_PARQUET):
        transient_rows = pd.read_parquet(paths.TRANSIENT_PARQUET).to_dict("records")

    while True:
        if not os.path.exists(paths.RESOLVED_CSV):
            time.sleep(RESCAN_SECONDS)
            continue

        resolved = pd.read_csv(paths.RESOLVED_CSV)
        targets = resolved[resolved["pdf_url"].notna()]
        pending = targets[~targets["doi"].astype(str).isin(attempted)]

        if pending.empty:
            idle_rounds += 1
            if idle_rounds >= IDLE_ROUNDS_BEFORE_EXIT:
                print(f"\nno new work for a long while; stopping with "
                      f"{downloaded:,} files", flush=True)
                return 0
            time.sleep(RESCAN_SECONDS)
            continue

        idle_rounds = 0
        print(f"{len(pending):,} newly located files to attempt", flush=True)

        for i, record in enumerate(pending.to_dict("records"), start=1):
            doi = str(record["doi"])
            url = record["pdf_url"]
            attempted.add(doi)

            host = host_of(url)
            if host in BLOCKED_HOSTS:
                rows.append({"doi": doi, "file": None, "bytes": None,
                             "error": "publisher_blocks_automated_access",
                             "retained": False})
            elif not robots_allows(url):
                rows.append({"doi": doi, "file": None, "bytes": None,
                             "error": "robots_disallowed", "retained": False})
            else:
                retain = not is_transient(record)
                path = os.path.join(paths.PDF_DIR, filename_for(doi))
                throttle(url)
                try:
                    size, error, stats = fetch_one(url, path, retain)
                except urllib.error.HTTPError as exc:
                    size, error, stats = None, f"http{exc.code}", None
                except Exception as exc:
                    size, error, stats = None, type(exc).__name__, None

                rows.append({
                    "doi": doi,
                    "file": os.path.basename(path) if (size and retain) else None,
                    "bytes": size,
                    "error": error,
                    "retained": bool(retain and size),
                })
                if stats is not None:
                    stats["doi"] = doi
                    transient_rows.append(stats)
                if size is not None:
                    downloaded += 1
                time.sleep(REQUEST_PAUSE)

            if len(rows) >= SAVE_EVERY:
                frame = pd.DataFrame(rows)
                if os.path.exists(paths.FETCH_STATUS_CSV):
                    frame = pd.concat([pd.read_csv(paths.FETCH_STATUS_CSV), frame],
                                      ignore_index=True)
                frame.drop_duplicates("doi", keep="last").to_csv(
                    paths.FETCH_STATUS_CSV, index=False)
                rows = []
                if transient_rows:
                    pd.DataFrame(transient_rows).to_parquet(
                        paths.TRANSIENT_PARQUET, index=False)
                held = len(os.listdir(paths.PDF_DIR))
                print(f"  {i:,}/{len(pending):,}  files held {held:,}  "
                      f"parsed without keeping {len(transient_rows):,}",
                      flush=True)

        if rows:
            frame = pd.DataFrame(rows)
            if os.path.exists(paths.FETCH_STATUS_CSV):
                frame = pd.concat([pd.read_csv(paths.FETCH_STATUS_CSV), frame],
                                  ignore_index=True)
            frame.drop_duplicates("doi", keep="last").to_csv(
                paths.FETCH_STATUS_CSV, index=False)
            rows = []

        time.sleep(RESCAN_SECONDS)


if __name__ == "__main__":
    sys.exit(main())

