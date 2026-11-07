"""
Step 5. Recover abstracts that OpenAlex does not hold, from Semantic Scholar.

OpenAlex has no abstract for a large share of the articles of some publishers,
because those publishers do not deposit abstracts openly. Semantic Scholar holds
some of them. This script asks its batch endpoint (up to 500 DOIs per request) for
every article in the corpus that has a DOI and no abstract, and saves what it finds
to data/recovered_abstracts.parquet. Rebuild the corpus afterwards to merge them.

Recovered abstracts never replace an OpenAlex abstract, and each keeps its source,
so any result can be recomputed on OpenAlex abstracts alone. The recovered text is
for local analysis only and is not distributed: Semantic Scholar's API licence
governs its use.

The run is resumable: every attempted DOI is logged and skipped on a rerun.
Usage: python recover_abstracts.py [max_articles]
"""

import json
import os
import re
import sys
import time
import urllib.error
import urllib.request

import pandas as pd

import config
import repo

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

URL = "https://api.semanticscholar.org/graph/v1/paper/batch?fields=abstract"
BATCH = 500
PAUSE = 1.2
MIN_CHARS = 120
TAG, SPACE = re.compile(r"<[^>]+>"), re.compile(r"\s+")


def clean(text):
    """Strip markup and a leading 'Abstract' label; reject fragments."""
    if not text:
        return None
    text = SPACE.sub(" ", TAG.sub(" ", str(text))).strip()
    for prefix in ("Abstract", "ABSTRACT", "Summary", "SUMMARY"):
        if text.startswith(prefix):
            text = text[len(prefix):].lstrip(" :.-")
    return text if len(text) >= MIN_CHARS else None


def batch(dois):
    payload = json.dumps({"ids": [f"DOI:{d}" for d in dois]}).encode()
    headers = {"User-Agent": config.user_agent(), "Content-Type": "application/json",
               "Accept": "application/json"}
    for attempt in range(4):
        try:
            req = urllib.request.Request(URL, data=payload, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=120) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as exc:
            time.sleep((10 if exc.code == 429 else 3) * (attempt + 1))
        except Exception:
            time.sleep(3 * (attempt + 1))
    return None


def main():
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    corpus = pd.read_parquet(repo.CORPUS, columns=["doi", "abstract", "period"])
    todo = corpus[corpus["abstract"].isna() & corpus["doi"].notna()].drop_duplicates("doi")

    tried = set()
    if os.path.exists(repo.RECOVERY_LOG):
        tried = set(pd.read_csv(repo.RECOVERY_LOG, usecols=["doi"])["doi"].astype(str).str.lower())
    todo = todo[~todo["doi"].str.lower().isin(tried)]
    # Post-release articles first: the comparison is limited by the smaller side.
    todo = todo.sort_values("period")
    if limit:
        todo = todo.head(limit)
    print(f"without an abstract: {int(corpus['abstract'].isna().sum()):,}; "
          f"already tried: {len(tried):,}; to try: {len(todo):,}")

    found = pd.read_parquet(repo.RECOVERED).to_dict("records") if os.path.exists(repo.RECOVERED) else []
    dois = todo["doi"].str.lower().tolist()
    for start in range(0, len(dois), BATCH):
        chunk = dois[start:start + BATCH]
        res = batch(chunk)
        log = []
        for i, doi in enumerate(chunk):
            text = clean((res[i] or {}).get("abstract")) if res else None
            if text:
                found.append({"doi": doi, "abstract": text, "abstract_source": "semantic_scholar"})
            # A failed request is not logged, so its DOIs are tried again next run.
            if res is not None:
                log.append({"doi": doi, "found": bool(text)})
        pd.DataFrame(found).to_parquet(repo.RECOVERED, index=False)
        if log:
            pd.DataFrame(log).to_csv(repo.RECOVERY_LOG, mode="a", index=False,
                                     header=not os.path.exists(repo.RECOVERY_LOG))
        print(f"  {min(start + BATCH, len(dois)):,}/{len(dois):,} tried, "
              f"{len(found):,} recovered in total", flush=True)
        time.sleep(PAUSE)
    print("done; rerun build_corpus.py to merge")


if __name__ == "__main__":
    main()
