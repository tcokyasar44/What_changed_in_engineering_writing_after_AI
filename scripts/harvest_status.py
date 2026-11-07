"""Report harvest progress from the manifest (safe to run while harvest.py runs)."""

import json
import os
import sys

import pandas as pd

import repo

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def main():
    if not os.path.exists(repo.MANIFEST):
        print("no manifest yet")
        return
    with open(repo.MANIFEST, encoding="utf-8") as fh:
        man = json.load(fh)
    panel = pd.read_csv(repo.PANEL)
    done = {k: v for k, v in man.get("done", {}).items() if k in set(panel["openalex_id"])}
    print(f"journals : {len(done)} of {len(panel)}")
    print(f"articles : {sum(v.get('n_works', 0) for v in done.values()):,}")
    failed = man.get("failed", {})
    print(f"failed   : {len(failed)}")
    for sid, why in list(failed.items())[:5]:
        print(f"  {sid}: {str(why)[:70]}")


if __name__ == "__main__":
    main()
