"""
Paths used by every script, resolved from where the package is unpacked.

No script builds a path from the current working directory, so the package runs
from any location. Set the environment variable CORPUS_PIPELINE_DATA to keep the
data folder somewhere else (for example on a larger disk).

Layout:

    corpus_pipeline/
        scripts/     the pipeline
        data/        inputs, outputs and the harvest shards
        tests/       offline tests
        secrets.env  API key and contact address, never distributed
"""

import os

SCRIPTS = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPTS)
DATA = os.environ.get("CORPUS_PIPELINE_DATA") or os.path.join(ROOT, "data")
WORKS = os.path.join(DATA, "works")
SECRETS = os.path.join(ROOT, "secrets.env")

# Step 1: the sampling frame and its resolution to OpenAlex
SCHOLAR_LISTS = os.path.join(DATA, "scholar_metrics_lists.csv")
CORRECTIONS = os.path.join(DATA, "venue_corrections.csv")
SCHOLAR_RAW = os.path.join(DATA, "scholar_panel_raw.csv")

# Step 2: the panel
PANEL = os.path.join(DATA, "journal_panel_final.csv")
EXCLUSIONS = os.path.join(DATA, "panel_exclusions.csv")

# Step 3: the harvest
MANIFEST = os.path.join(DATA, "harvest_manifest.json")
WATCHDOG_STATUS = os.path.join(DATA, "watchdog_status.json")
API_CACHE = os.path.join(DATA, "api_cache.json")

# Steps 4 and 5: the corpus and recovered abstracts
CORPUS = os.path.join(DATA, "corpus.parquet")
ARTICLE_IDS = os.path.join(DATA, "article_ids.csv.gz")
JOURNAL_SUMMARY = os.path.join(DATA, "journal_summary.csv")
RECOVERED = os.path.join(DATA, "recovered_abstracts.parquet")
RECOVERY_LOG = os.path.join(DATA, "recovery_attempts.csv")

# Step 6: verification
COMPLETENESS = os.path.join(DATA, "completeness_check.csv")
COVERAGE = os.path.join(DATA, "coverage_by_doi.csv")


def ensure():
    """Create the folders the pipeline writes into."""
    for path in (DATA, WORKS):
        os.makedirs(path, exist_ok=True)


if __name__ == "__main__":
    ensure()
    print(f"package root : {ROOT}")
    print(f"data         : {DATA}")
    n = len([f for f in os.listdir(WORKS) if f.endswith(".parquet")])
    print(f"harvest shards present: {n}")
