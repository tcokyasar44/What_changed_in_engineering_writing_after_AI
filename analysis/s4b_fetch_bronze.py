"""
Step 4b. Re-measure the free-to-read ("bronze") articles, which are never stored.

During collection, articles that are free to read but carry no reuse licence were
downloaded, measured in memory and discarded, so no copy exists to re-parse. Their
September measurements used the retired word list and disclosure rule. This script
fetches each of them again under the same policy and measures it with the clean
measure_text() from s4_fulltext_parse.py:

  - the request identifies the study and a contact address, never a browser;
  - robots.txt is obeyed, and requests to one host are at least 1.5 seconds apart;
  - hosts that refuse automated access are not requested (listed in BLOCKED_HOSTS);
  - the PDF bytes are parsed in memory and discarded; only the measures are written.

The URL tried first is the one that succeeded in September; other recorded copies of
the same article (alternates.csv) are tried after it.

Output: out/fulltext_bronze.parquet (same columns as fulltext_articles.parquet,
folder = "memory"). Resumable: DOIs already in the output are skipped.
Run with --remeasure after a change of measure_text(): every article measured with an
earlier PARSER_VERSION is fetched again and re-measured (no copy is kept); when a fetch
fails, the earlier measurement is kept and the column recheck_failed says why.
"""
import glob, os, sys, time, urllib.error, urllib.parse, urllib.request, urllib.robotparser
import settings as P
from collections import defaultdict
import pandas as pd
import pymupdf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s4_fulltext_parse as S

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
OUTFILE = os.path.join(S.OUT, "fulltext_bronze.parquet")
SECRETS = P.SECRETS

BLOCKED_HOSTS = {
    "onlinelibrary.wiley.com", "www.onlinelibrary.wiley.com", "www.mdpi.com", "pubs.rsc.org",
    "pubs.acs.org", "ieeexplore.ieee.org", "www.sciencedirect.com", "ars.els-cdn.com",
    "link.springer.com", "hdl.handle.net", "dx.doi.org", "doi.org"}
PER_HOST = 1.5
TIMEOUT = 45
MAX_BYTES = 40 * 1024 * 1024
_robots, _last = {}, defaultdict(float)


def contact():
    for line in open(SECRETS, encoding="utf-8"):
        if line.startswith("CONTACT_EMAIL="):
            return line.split("=", 1)[1].strip()
    raise SystemExit("CONTACT_EMAIL missing from secrets.env")


UA = f"engineering-publishing-study/1.0 (mailto:{contact()}) academic non-commercial text mining"


def host(url):
    return urllib.parse.urlparse(url).netloc.lower()


def allowed(url):
    h = host(url)
    if h not in _robots:
        rp = urllib.robotparser.RobotFileParser()
        rp.set_url(f"https://{h}/robots.txt")
        try:
            rp.read()
        except Exception:
            rp = None
        _robots[h] = rp
    rp = _robots[h]
    try:
        return True if rp is None else rp.can_fetch(UA, url)
    except Exception:
        return True


def fetch(url):
    h = host(url)
    wait = PER_HOST - (time.time() - _last[h])
    if wait > 0:
        time.sleep(wait)
    _last[h] = time.time()
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/pdf,*/*"})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
        body = resp.read(MAX_BYTES)
    return body if body.startswith(b"%PDF") else None


def main():
    st = pd.concat([pd.read_csv(p, low_memory=False) for p in glob.glob(os.path.join(S.V2, "fetch_status*.csv"))])
    st["bytes"] = pd.to_numeric(st["bytes"], errors="coerce")
    bronze = st[st["bytes"] == 0].drop_duplicates("doi")
    urls = pd.concat([pd.read_csv(p, low_memory=False, usecols=lambda c: c in ("doi", "pdf_url", "rank"))
                      for p in glob.glob(os.path.join(S.V2, "resolved*.csv")) + [os.path.join(S.V2, "alternates.csv")]
                      if not p.endswith(".snapshot")])
    urls = urls[urls.pdf_url.notna()]
    known, titles, english = S.lookups()
    S.init(known, titles, english)

    done = pd.read_parquet(OUTFILE) if os.path.exists(OUTFILE) else pd.DataFrame(columns=["doi"])
    recheck = "--remeasure" in sys.argv
    if recheck:
        # After a change of measure_text() (a new PARSER_VERSION), every article not yet
        # measured with the current version is fetched again, since no copy is kept.
        # If a fetch now fails, the earlier measurement is kept and recheck_failed
        # records why.
        redo = done[done.parser != S.PARSER_VERSION]
        keep_old = {r["doi"]: r for r in redo.to_dict("records")}
        todo = [d for d in bronze.doi if str(d).lower() in keep_old]
    else:
        todo = [d for d in bronze.doi if d not in set(done.doi)]
    # Keyed by DOI: a re-measured article replaces its earlier row, and rows not yet
    # reached stay in the file if the run is interrupted.
    rows = {r["doi"]: r for r in done.to_dict("records")}
    print(f"bronze articles {len(bronze):,}; in output {len(done):,}; to do {len(todo):,}", flush=True)
    for i, doi in enumerate(todo, 1):
        cands = urls[urls.doi == doi].sort_values("rank", na_position="first").pdf_url.drop_duplicates().tolist()
        rec = {"doi": str(doi).lower(), "folder": "memory", "file": None, "parser": S.PARSER_VERSION}
        status = "no_url"
        for url in cands:
            if host(url) in BLOCKED_HOSTS:
                status = "host_blocks_scripts"; continue
            if not allowed(url):
                status = "robots_disallowed"; continue
            try:
                body = fetch(url)
            except urllib.error.HTTPError as e:
                status = f"http{e.code}"; continue
            except Exception as e:
                status = type(e).__name__; continue
            if body is None:
                status = "not_pdf"; continue
            try:
                with pymupdf.open(stream=body, filetype="pdf") as d:
                    pages = [p.get_text("text") for p in d]
            except Exception:
                status = "unreadable"; continue
            finally:
                body = None
            rec.update(pages=len(pages), url_host=host(url),
                       **S.measure_text("\n".join(pages), "\n".join(pages[:2]), titles.get(rec["doi"])))
            status = rec["status"]
            break
        rec["status"] = status
        if recheck and status != "ok":
            rec = dict(keep_old[rec["doi"]], recheck_failed=status)
        rows[rec["doi"]] = rec
        if i % 25 == 0 or i == len(todo):
            pd.DataFrame(list(rows.values())).to_parquet(OUTFILE, index=False)
            print(f"  {i:,}/{len(todo):,}", pd.DataFrame(list(rows.values())).status.value_counts().head(4).to_dict(), flush=True)
    print(pd.DataFrame(list(rows.values())).status.value_counts().to_string())


if __name__ == "__main__":
    main()
