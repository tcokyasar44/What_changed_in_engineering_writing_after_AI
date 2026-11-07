"""
Shared configuration and HTTP helpers for the corpus-building scripts.

API keys are read from analysis/secrets.env, which is gitignored. Nothing in this
module hard-codes a credential, so the scripts can be published as-is alongside the
paper.
"""

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
SECRETS_PATH = os.path.join(HERE, "secrets.env")


def _load_secrets():
    values = {}
    if os.path.exists(SECRETS_PATH):
        with open(SECRETS_PATH, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, val = line.partition("=")
                values[key.strip()] = val.strip()
    for key in ("OPENALEX_API_KEY", "ELSEVIER_API_KEY", "CONTACT_EMAIL"):
        if os.environ.get(key):
            values[key] = os.environ[key]
    return values


SECRETS = _load_secrets()
OPENALEX_KEY = SECRETS.get("OPENALEX_API_KEY", "")
ELSEVIER_KEY = SECRETS.get("ELSEVIER_API_KEY", "")
CONTACT_EMAIL = SECRETS.get("CONTACT_EMAIL", "")

USER_AGENT = f"engineering-publishing-study/1.0 (mailto:{CONTACT_EMAIL})"

STUDY_START_YEAR = 2015

# The harvest reaches into the current year. Recent months are deliberately
# collected and then trimmed during corpus construction rather than excluded here,
# because how far back the indexing lag extends can only be judged from the data.
STUDY_END_YEAR = 2026

CACHE_PATH = os.path.join(HERE, ".api_cache.json")
_CACHE = None


def _cache():
    global _CACHE
    if _CACHE is None:
        _CACHE = {}
        if os.path.exists(CACHE_PATH):
            try:
                with open(CACHE_PATH, encoding="utf-8") as fh:
                    _CACHE = json.load(fh)
            except Exception:
                _CACHE = {}
    return _CACHE


def save_cache():
    cache = _cache()
    tmp = CACHE_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(cache, fh)
    os.replace(tmp, CACHE_PATH)


def openalex(path, params=None, use_cache=True, retries=4):
    """GET an OpenAlex endpoint with the API key attached."""
    params = dict(params or {})
    if OPENALEX_KEY:
        params["api_key"] = OPENALEX_KEY
    if CONTACT_EMAIL:
        params.setdefault("mailto", CONTACT_EMAIL)

    url = f"https://api.openalex.org/{path.lstrip('/')}"
    if params:
        url = f"{url}?{urllib.parse.urlencode(params)}"

    cache = _cache()
    cache_key = url.replace(OPENALEX_KEY, "KEY") if OPENALEX_KEY else url
    if use_cache and cache_key in cache:
        return cache[cache_key]

    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.load(resp)
            if use_cache:
                cache[cache_key] = data
            return data
        except urllib.error.HTTPError as exc:
            last = exc
            if exc.code == 429:
                time.sleep(10 * (attempt + 1))
            elif exc.code in (400, 404):
                raise
            else:
                time.sleep(2 * (attempt + 1))
        except Exception as exc:
            last = exc
            time.sleep(2 * (attempt + 1))
    raise last


def elsevier(path, params=None, accept="application/json", retries=3):
    """GET an Elsevier API endpoint using the institutional key."""
    if not ELSEVIER_KEY:
        raise RuntimeError("ELSEVIER_API_KEY is not configured")

    params = dict(params or {})
    url = f"https://api.elsevier.com/{path.lstrip('/')}"
    if params:
        url = f"{url}?{urllib.parse.urlencode(params)}"

    headers = {
        "User-Agent": USER_AGENT,
        "Accept": accept,
        "X-ELS-APIKey": ELSEVIER_KEY,
    }

    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=60) as resp:
                body = resp.read()
                quota = {
                    "limit": resp.headers.get("X-RateLimit-Limit"),
                    "remaining": resp.headers.get("X-RateLimit-Remaining"),
                }
                if accept == "application/json":
                    return json.loads(body), quota
                return body.decode("utf-8", "replace"), quota
        except urllib.error.HTTPError as exc:
            last = exc
            if exc.code in (401, 403, 404):
                raise
            time.sleep(3 * (attempt + 1))
        except Exception as exc:
            last = exc
            time.sleep(3 * (attempt + 1))
    raise last
