"""
Route manually downloaded PDFs into the right folder by reading the DOI inside each.

Files arriving in the Downloads folder come from two different activities. Some are
articles from the random draw, which belong in the main sample. Others are whole
issues downloaded in bulk, which must be kept separate: every article in an issue
shares a journal, an editor, a year and one set of formatting conventions, so
pooling them with a random draw would silently turn a random sample into a clustered
one and overstate the effective sample size.

Sorting by hand is unnecessary. The DOI is printed inside essentially every
published PDF, so each file can be matched against the sample and routed
automatically:

  manual_pdf  the DOI appears in the random draw
  bulk_pdf    a research article whose DOI is not in the draw
  rejected    front matter, editorials, indexes and similar non-research items

Files are moved out of Downloads once they have been read and routed, so the folder
does not accumulate hundreds of articles. A file is only removed after the copy at
the destination has been verified to exist and match in size.
"""

import os
import re
import shutil
import sys
from collections import Counter

import pandas as pd
import pymupdf

import parse_fulltext as pf
import paths

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))

# Destinations come from paths.py. This script previously wrote to data/fulltext
# directly, which is the parent of both collections and belongs to neither, so
# manually collected articles would have landed where no parser looks for them.
FT = paths.FT

DOWNLOADS = os.path.join(os.path.expanduser("~"), "Downloads")
MANUAL_DIR = paths.MANUAL_PDF_DIR
BULK_DIR = paths.BULK_PDF_DIR
REJECT_DIR = paths.REJECT_PDF_DIR
LEDGER = paths.SORT_LOG_CSV

for folder in (MANUAL_DIR, BULK_DIR, REJECT_DIR):
    os.makedirs(folder, exist_ok=True)

# The Downloads folder contains unrelated personal documents. Only files saved on or
# after the day manual collection began are examined, so nothing else is opened.
CUTOFF = "2026-09-26"

DOI_RE = re.compile(r"\b10\.\d{4,9}/[^\s\"'<>)\]},;]+", re.I)


def relocate(src, dest_dir, name):
    """Move a file into place, verifying the copy before removing the original.

    A plain move would be simpler, but the destination may sit on a different
    volume and a partial write would then destroy the only copy. Copying first,
    confirming the size matches, and only then deleting makes the operation safe to
    interrupt at any point.
    """
    dest = os.path.join(dest_dir, name)

    if os.path.exists(dest) and os.path.getsize(dest) == os.path.getsize(src):
        os.remove(src)
        return dest

    shutil.copy2(src, dest)

    if os.path.exists(dest) and os.path.getsize(dest) == os.path.getsize(src):
        os.remove(src)
    return dest

# Publisher filename conventions that encode the DOI or a resolvable identifier.
PII_RE = re.compile(r"1-s2\.0-(S\d{16,17})-main", re.I)


def extract_text(path, pages=4):
    try:
        doc = pymupdf.open(path)
    except Exception:
        return None, 0
    try:
        total = len(doc)
        text = "\n".join(doc[i].get_text("text") for i in range(min(pages, total)))
        full = "\n".join(page.get_text("text") for page in doc)
    except Exception:
        doc.close()
        return None, 0
    doc.close()
    return (text, full, total)


def doi_from_filename(name, known):
    """Recover a DOI encoded in the filename.

    Accepted manuscripts and author copies frequently do not print their own DOI
    anywhere in the document, so a text scan finds only the DOIs of cited works.
    Publisher download filenames often encode the identifier instead, either
    directly with separators substituted, or as an Elsevier PII.
    """
    stem = os.path.splitext(os.path.basename(name))[0]

    # Form: 10.1016_j.cemconres.2024.107524
    guess = re.sub(r"^\d{4}[_-]", "", stem).replace("_", "/").lower()
    if guess in known:
        return guess

    # Some tools substitute other characters for the slash.
    for sep in ("-", "%2f", "%2F"):
        alt = stem.replace(sep, "/", 1).lower()
        if alt in known:
            return alt

    return None


def find_doi(head, known, filename=None):
    """Return the sampled DOI this document belongs to, if any.

    The DOI printed inside the document is the primary key, but it is not always
    present. When the text yields nothing that matches, the filename is tried, and
    only then is the document treated as unidentified.
    """
    candidates = []
    for match in DOI_RE.finditer(head):
        candidates.append(match.group(0).rstrip(".,;)]").lower())

    for candidate in candidates:
        if candidate in known:
            return candidate, True

    # Trailing punctuation and hyphenation sometimes corrupt the printed DOI, so a
    # prefix match against the known set is attempted before giving up.
    for candidate in candidates:
        for known_doi in known:
            if candidate.startswith(known_doi) or known_doi.startswith(candidate):
                return known_doi, True

    if filename:
        from_name = doi_from_filename(filename, known)
        if from_name:
            return from_name, True

    return (candidates[0] if candidates else None), False


def main():
    # Every draw forms one sampling design, and a manually collected article may
    # belong to any of them. The files are discovered rather than listed, because
    # omitting a draw routes genuine sample members to the rejected folder, where
    # the loss looks like a bad download rather than a bug.
    frames = [pd.read_csv(path, usecols=["doi"]) for path in pf.SAMPLE_FILES]
    sample = pd.concat(frames, ignore_index=True).drop_duplicates("doi")
    known = {str(d).lower() for d in sample["doi"].dropna()}
    print(f"sampling frame: {len(known):,} DOIs")

    already = set()
    if os.path.exists(LEDGER):
        prior = pd.read_csv(LEDGER)
        already = set(prior["file"])
        print(f"previously sorted: {len(already)}")

    # The Downloads folder holds unrelated personal documents. Only files created
    # after the manual collection began are considered, which keeps the scan
    # confined to the article downloads and avoids opening anything else.
    cutoff = pd.Timestamp(CUTOFF).timestamp()

    files = []
    for name in os.listdir(DOWNLOADS):
        if not name.lower().endswith(".pdf"):
            continue
        path = os.path.join(DOWNLOADS, name)
        try:
            if os.path.getmtime(path) >= cutoff:
                files.append(name)
        except OSError:
            continue

    todo = [f for f in files if f not in already]
    print(f"PDFs modified since {CUTOFF}: {len(files)}   new: {len(todo)}")

    rows = []
    counts = Counter()

    for name in todo:
        src = os.path.join(DOWNLOADS, name)
        extracted = extract_text(src)
        if not extracted or not extracted[0]:
            counts["unreadable"] += 1
            rows.append({"file": name, "doi": None, "destination": None,
                         "reason": "unreadable"})
            continue

        head, full, page_count = extracted

        body, refs = pf.split_body(full)
        is_article, why = pf.looks_like_research_article(full, body, refs)
        if not is_article:
            relocate(src, REJECT_DIR, name)
            counts[f"rejected: {why}"] += 1
            rows.append({"file": name, "doi": None, "destination": "rejected_pdf",
                         "reason": why})
            continue

        doi, in_sample = find_doi(head, known, filename=name)

        if in_sample:
            relocate(src, MANUAL_DIR, name)
            counts["matched the random draw"] += 1
            rows.append({"file": name, "doi": doi, "destination": "manual_pdf",
                         "reason": "in sample"})
        else:
            relocate(src, BULK_DIR, name)
            counts["research article, not in the draw"] += 1
            rows.append({"file": name, "doi": doi, "destination": "bulk_pdf",
                         "reason": "not in sample"})

    if rows:
        ledger = pd.DataFrame(rows)
        if os.path.exists(LEDGER):
            ledger = pd.concat([pd.read_csv(LEDGER), ledger], ignore_index=True)
        ledger.to_csv(LEDGER, index=False)

    print()
    for label, count in counts.most_common():
        print(f"  {count:4d}  {label}")

    print()
    print(f"  manual_pdf   : {len(os.listdir(MANUAL_DIR))} files")
    print(f"  bulk_pdf     : {len(os.listdir(BULK_DIR))} files")
    print(f"  rejected_pdf : {len(os.listdir(REJECT_DIR))} files")
    print()
    print("Files were moved out of Downloads after the destination copy was verified.")


if __name__ == "__main__":
    main()



