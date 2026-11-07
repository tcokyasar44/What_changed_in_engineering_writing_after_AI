"""
Run harvest.py until every panel journal is in the manifest.

A full harvest makes several hundred thousand requests and takes hours. OpenAlex's
daily request budget for a free key can run out partway, and connections drop. This
script restarts harvest.py whenever it exits or makes no progress for STALL_MINUTES,
with growing pauses between restarts, and stops once the manifest is complete. The
harvest itself is resumable, so a restart loses at most the journal in progress.

Usage: python watchdog.py   (runs on Windows, macOS and Linux)
"""

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone

import pandas as pd

import repo

POLL_SECONDS = 20
STALL_MINUTES = 15
MAX_RESTARTS = 500


def log(msg):
    print(f"[{datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S}] watchdog: {msg}", flush=True)


def done_count(panel_ids):
    try:
        with open(repo.MANIFEST, encoding="utf-8") as fh:
            done = json.load(fh).get("done", {})
    except Exception:
        return 0
    return sum(1 for s in panel_ids if s in done)


def main():
    panel_ids = set(pd.read_csv(repo.PANEL)["openalex_id"])
    harvest_log = open(os.path.join(repo.DATA, "harvest.log"), "a", encoding="utf-8")
    proc, restarts, backoff = None, 0, 0
    last_done, last_progress = done_count(panel_ids), time.time()

    while True:
        done = done_count(panel_ids)
        if done >= len(panel_ids):
            log(f"complete: {done}/{len(panel_ids)} journals")
            if proc and proc.poll() is None:
                proc.terminate()
            return 0
        if done > last_done:
            last_done, last_progress, backoff = done, time.time(), 0

        running = proc is not None and proc.poll() is None
        if running and time.time() - last_progress > STALL_MINUTES * 60:
            log(f"no progress for {STALL_MINUTES} min; restarting")
            proc.terminate()
            running, last_progress = False, time.time()
        if not running:
            if restarts >= MAX_RESTARTS:
                log("restart limit reached; stopping")
                return 1
            if backoff:
                log(f"harvest not running; waiting {backoff}s")
                time.sleep(backoff)
            proc = subprocess.Popen([sys.executable, "-u", "harvest.py"], cwd=repo.SCRIPTS,
                                    stdout=harvest_log, stderr=subprocess.STDOUT)
            restarts += 1
            backoff = min(max(2 * backoff, 15), 600)
            log(f"started harvest (pid {proc.pid}, start #{restarts}); {done}/{len(panel_ids)} done")
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    sys.exit(main())
