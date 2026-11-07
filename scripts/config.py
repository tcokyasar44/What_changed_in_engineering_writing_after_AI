"""
Settings and the OpenAlex client shared by the scripts.

Credentials are read from secrets.env in the package root (never distributed) or
from environment variables of the same name, which take precedence:

    OPENALEX_API_KEY   free key from https://openalex.org
    CONTACT_EMAIL      your address; sent to the APIs so their operators can reach you

Every request identifies the study and its contact address. No request pretends to
come from a browser.
"""

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request

import repo

KEYS = ("OPENALEX_API_KEY", "CONTACT_EMAIL")


def _load_secrets():
    values = {}
    if os.path.exists(repo.SECRETS):
        with open(repo.SECRETS, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, val = line.partition("=")
                values[key.strip()] = val.strip()
    for key in KEYS:
        if os.environ.get(key):
            values[key] = os.environ[key]
    return values


SECRETS = _load_secrets()
OPENALEX_KEY = SECRETS.get("OPENALEX_API_KEY", "")
CONTACT_EMAIL = SECRETS.get("CONTACT_EMAIL", "")

# The study window. Articles published in 2026 are not part of the study: their
# records were still being completed when the corpus was collected.
STUDY_START_YEAR = 2015
STUDY_END_YEAR = 2025


def user_agent():
    """The identity sent with every request. A contact address is required."""
    if not CONTACT_EMAIL:
        raise SystemExit("Set CONTACT_EMAIL in secrets.env (see secrets.env.example).")
    return f"corpus-pipeline/1.0 (research use; mailto:{CONTACT_EMAIL})"


_CACHE = None


def _cache():
    global _CACHE
    if _CACHE is None:
        _CACHE = {}
        if os.path.exists(repo.API_CACHE):
            try:
                with open(repo.API_CACHE, encoding="utf-8") as fh:
                    _CACHE = json.load(fh)
            except Exception:
                _CACHE = {}
    return _CACHE


def save_cache():
    if _CACHE is None:
        return
    tmp = repo.API_CACHE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(_CACHE, fh)
    os.replace(tmp, repo.API_CACHE)


def openalex(path, params=None, use_cache=True, retries=5):
    """GET an OpenAlex endpoint, with retry and backoff on rate limits and errors."""
    params = dict(params or {})
    if OPENALEX_KEY:
        params["api_key"] = OPENALEX_KEY
    params.setdefault("mailto", CONTACT_EMAIL)
    url = f"https://api.openalex.org/{path.lstrip('/')}?{urllib.parse.urlencode(params)}"

    cache = _cache()
    cache_key = url.replace(OPENALEX_KEY, "KEY") if OPENALEX_KEY else url
    if use_cache and cache_key in cache:
        return cache[cache_key]

    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": user_agent()})
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.load(resp)
            if use_cache:
                cache[cache_key] = data
            return data
        except urllib.error.HTTPError as exc:
            last = exc
            if exc.code in (400, 401, 403, 404):
                raise
            time.sleep((10 if exc.code == 429 else 2) * (attempt + 1))
        except Exception as exc:
            last = exc
            time.sleep(2 * (attempt + 1))
    raise last
