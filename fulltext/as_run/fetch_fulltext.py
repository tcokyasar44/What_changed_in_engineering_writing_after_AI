"""
Download the located open-access PDFs.

Downloading is deliberately conservative. Requests are paced, failures are recorded
rather than retried aggressively, and each file is written once and skipped on
subsequent runs. The point is to collect a sample without placing meaningful load on
any publisher or repository.

Only articles whose PDF Unpaywall reported as openly available are fetched. Nothing
is retrieved from behind a paywall, and no publisher text is redistributed: the PDFs
stay local and only derived counts are released.

Stage 3 of the pipeline. Run parse_fulltext.py next.
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
FT = paths.FT
PDFS = paths.PDF_DIR

RESOLVED = paths.RESOLVED_CSV
STATUS = paths.FETCH_STATUS_CSV
TRANSIENT = paths.TRANSIENT_PARQUET

os.makedirs(PDFS, exist_ok=True)

EMAIL = config.CONTACT_EMAIL
HEADERS = {
    "User-Agent": (f"engineering-publishing-study/1.0 (mailto:{EMAIL}) "
                   "academic non-commercial text mining"),
    "Accept": "application/pdf,*/*",
}

PAUSE = 0.7
TIMEOUT = 45
MAX_BYTES = 40 * 1024 * 1024
SAVE_EVERY = 25


_ROBOTS = {}
_LAST_REQUEST = defaultdict(float)
PER_HOST_INTERVAL = 1.5

# Hosts that refuse automated requests outright. Tested directly: the refusal is
# unconditional and does not depend on the request headers, so it is a deliberate
# block rather than a configuration problem. They are skipped rather than retried,
# and the resulting gap is reported in the paper as a coverage limitation. Presenting
# a different user agent in order to evade the block would be both ineffective and
# improper.
BLOCKED_HOSTS = {
    "onlinelibrary.wiley.com",
    "www.onlinelibrary.wiley.com",
}

# Hosts that return a landing page rather than the file itself. Following those
# would require scraping the page for a download link, which these publishers also
# discourage, so they are skipped for the same reason.
LANDING_PAGE_HOSTS = {
    "hdl.handle.net",
}


def host_of(url):
    try:
        return urllib.parse.urlparse(url).netloc.lower()
    except Exception:
        return ""


def skip_reason(url):
    """Return why a URL should not be requested, or None to proceed."""
    host = host_of(url)
    if host in BLOCKED_HOSTS:
        return "publisher_blocks_automated_access"
    if host in LANDING_PAGE_HOSTS:
        return "landing_page_not_pdf"
    return None


def robots_allows(url):
    """Respect each host's robots.txt. Hosts that do not publish one are allowed."""
    try:
        parts = urllib.parse.urlparse(url)
        host = f"{parts.scheme}://{parts.netloc}"
    except Exception:
        return True

    if host not in _ROBOTS:
        parser = urllib.robotparser.RobotFileParser()
        parser.set_url(host + "/robots.txt")
        try:
            parser.read()
        except Exception:
            parser = None
        _ROBOTS[host] = parser

    parser = _ROBOTS[host]
    if parser is None:
        return True
    try:
        return parser.can_fetch(HEADERS["User-Agent"], url)
    except Exception:
        return True


def throttle(url):
    """Keep at least PER_HOST_INTERVAL seconds between requests to one host."""
    try:
        host = urllib.parse.urlparse(url).netloc
    except Exception:
        return
    wait = PER_HOST_INTERVAL - (time.time() - _LAST_REQUEST[host])
    if wait > 0:
        time.sleep(wait)
    _LAST_REQUEST[host] = time.time()


def filename_for(doi):
    return hashlib.sha1(doi.encode("utf-8")).hexdigest()[:20] + ".pdf"


# Retention policy.
#
# Bronze articles are free to read at the publisher's discretion with no reuse
# licence attached. Those are parsed in memory and the bytes discarded, so the study
# never holds a copy: the same thing a human reader does, and consistent with the
# deletion requirement in most publisher text-mining terms.
#
# Gold, hybrid and green articles are open access by licence. A local copy is kept
# so the parser can be rerun without refetching, which matters because the caption
# and equation patterns may need refinement. Note that Unpaywall leaves the licence
# field empty for most records even when the article is gold open access, so an
# absent licence is treated as unknown rather than as a restriction.
TRANSIENT_OA_STATUS = {"bronze", "closed"}


def is_transient(record):
    """True when the article must be parsed without keeping a copy."""
    status = str(record.get("oa_status") or "").lower().strip()
    licence = record.get("license")
    licence = None if pd.isna(licence) else str(licence).lower().strip()

    # An explicit no-derivatives licence is always treated as transient.
    if licence in {"cc-by-nd", "cc-by-nc-nd"}:
        return True

    return status in TRANSIENT_OA_STATUS


def fetch(url, path, retain=True):
    """Fetch a PDF. When retain is False the bytes are parsed and never stored."""
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
        ctype = (resp.headers.get("Content-Type") or "").lower()
        body = resp.read(MAX_BYTES)

    if not body.startswith(b"%PDF"):
        return None, f"not_pdf:{ctype[:24]}", None

    if retain:
        with open(path, "wb") as fh:
            fh.write(body)
        return len(body), None, None

    # Parse immediately, keep only the measurements, then drop the bytes.
    try:
        import parse_fulltext as pf
        doc = pymupdf.open(stream=body, filetype="pdf")
        pages = len(doc)
        text = "\n".join(page.get_text("text") for page in doc)
        doc.close()
        stats = pf.measure(text, pages)
    except Exception as exc:
        return None, f"parse_failed:{type(exc).__name__}", None
    finally:
        body = None

    if stats is None:
        return None, "too_short", None
    return 0, None, stats


def main():
    resolved = pd.read_csv(RESOLVED)
    targets = resolved[resolved["pdf_url"].notna()].copy()
    print(f"PDFs to fetch: {len(targets):,}")

    done = {}
    if os.path.exists(STATUS):
        prior = pd.read_csv(STATUS)
        done = {r["doi"]: r.to_dict() for _, r in prior.iterrows()}
        print(f"resuming: {len(done):,} already attempted")

    rows = list(done.values())
    pending = targets[~targets["doi"].isin(done.keys())]
    print(f"remaining: {len(pending):,}\n")

    transient_rows = []
    if os.path.exists(TRANSIENT):
        transient_rows = pd.read_parquet(TRANSIENT).to_dict("records")

    ok = sum(1 for r in rows if r.get("bytes") is not None
             and not pd.isna(r.get("bytes")))
    for i, (_, rec) in enumerate(pending.iterrows(), start=1):
        doi = rec["doi"]
        path = os.path.join(PDFS, filename_for(doi))
        retain = not is_transient(rec)

        if retain and os.path.exists(path) and os.path.getsize(path) > 1000:
            rows.append({"doi": doi, "file": os.path.basename(path),
                         "bytes": os.path.getsize(path), "error": None,
                         "retained": True})
            ok += 1
        elif skip_reason(rec["pdf_url"]):
            rows.append({"doi": doi, "file": None, "bytes": None,
                         "error": skip_reason(rec["pdf_url"]), "retained": False})
        elif not robots_allows(rec["pdf_url"]):
            rows.append({"doi": doi, "file": None, "bytes": None,
                         "error": "robots_disallowed", "retained": False})
        else:
            throttle(rec["pdf_url"])
            try:
                size, error, stats = fetch(rec["pdf_url"], path, retain=retain)
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
                ok += 1
            time.sleep(PAUSE)

        if i % SAVE_EVERY == 0 or i == len(pending):
            pd.DataFrame(rows).to_csv(STATUS, index=False)
            if transient_rows:
                pd.DataFrame(transient_rows).to_parquet(TRANSIENT, index=False)
            print(f"  {i:,}/{len(pending):,}  processed {ok:,} "
                  f"(parsed without retaining: {len(transient_rows):,})", flush=True)

    frame = pd.DataFrame(rows)
    frame.to_csv(STATUS, index=False)
    if transient_rows:
        pd.DataFrame(transient_rows).to_parquet(TRANSIENT, index=False)

    good = frame["bytes"].notna().sum()
    retained = int(frame.get("retained", pd.Series(dtype=bool)).sum())
    print(f"\nprocessed {good:,} of {len(frame):,} attempted")
    print(f"  retained as files      : {retained:,} (openly licensed)")
    print(f"  parsed without storing : {len(transient_rows):,} "
          f"(free to read, no explicit licence)")
    if frame["error"].notna().any():
        print("\nfailure reasons:")
        print(frame["error"].value_counts().head(10).to_string())
    print("\nNext: python parse_fulltext.py")


if __name__ == "__main__":
    main()
