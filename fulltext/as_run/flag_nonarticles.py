"""
Find files that are not research articles and should not be measured.

The author noticed that some downloads are journal cover pages. They are a real
contaminant: a cover, an editorial board listing or a graphical-abstract blurb
carries the article's DOI, so it satisfies the DOI check and is counted as
collected, but it contains almost none of the article's text. Measuring word
frequencies over such a file dilutes every text metric, and because covers cluster
at particular publishers the dilution is not evenly spread.

Detection uses three signals, all deliberate rather than a single length cutoff:

  very little text          a cover has a few dozen words across one or two pages
  front-matter phrasing     "editorial board", "table of contents", "in article
                            number", "inside back cover"
  no article structure      a research article names an abstract, introduction,
                            method, result or conclusion somewhere early

A file is flagged only when it is short *and* shows one of the other two signs, so
a genuinely brief article is not discarded for being brief.

Nothing is deleted. Flagged files are listed for review and written to a CSV the
measurement step can exclude, because throwing away a real article is worse than
keeping a cover.
"""

import os
import re
import sys

import pandas as pd

try:
    import pymupdf as fitz
except ImportError:
    try:
        import fitz
    except ImportError:
        fitz = None

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
FT = os.path.join(HERE, "data", "fulltext")
V2 = os.path.join(FT, "v2_target")
FOLDERS = [os.path.join(V2, "manual_pdf"), os.path.join(V2, "pdf")]
OUT = os.path.join(FT, "nonarticle_files.csv")

MIN_WORDS = 600

FRONT_MATTER = [
    "editorial board", "table of contents", "contents lists available",
    "in article number", "inside back cover", "inside front cover",
    "front cover", "back cover", "cover picture", "cover image",
    "issue information", "masthead", "call for papers",
    "instructions for authors", "author index", "subject index",
    "this page intentionally", "advertisement",
]

STRUCTURE = ["abstract", "introduction", "keywords", "methodology",
             "materials and methods", "results", "conclusion", "references",
             "acknowledg"]


def long_path(path):
    full = os.path.abspath(path)
    if os.name == "nt" and len(full) > 255 and not full.startswith("\\\\?\\"):
        return "\\\\?\\" + full
    return full


def inspect(path):
    """Word count, page count and the text of the opening pages."""
    try:
        with fitz.open(long_path(path)) as doc:
            pages = doc.page_count
            head = "".join(doc[i].get_text()
                           for i in range(min(3, pages))).lower()
            total = sum(len(doc[i].get_text().split())
                        for i in range(min(25, pages)))
        return total, pages, head
    except Exception:
        return None, 0, ""


def main():
    if fitz is None:
        print("PyMuPDF unavailable")
        return 1

    files = []
    for folder in FOLDERS:
        if os.path.isdir(folder):
            files += [os.path.join(folder, n) for n in sorted(os.listdir(folder))
                      if n.lower().endswith(".pdf")]

    print(f"examining {len(files):,} files\n")

    rows = []
    for i, path in enumerate(files, start=1):
        if i % 1000 == 0:
            print(f"  {i:,}/{len(files):,}")
        words, pages, head = inspect(path)

        if words is None:
            rows.append({"file": path, "reason": "unreadable",
                         "words": 0, "pages": 0})
            continue

        if words >= MIN_WORDS:
            continue

        phrase = next((p for p in FRONT_MATTER if p in head), None)
        structured = any(s in head for s in STRUCTURE)

        if phrase:
            reason = f"front matter: {phrase}"
        elif not structured:
            reason = "short with no article structure"
        else:
            # Short but looks like a real article; keep it.
            continue

        rows.append({"file": path, "reason": reason,
                     "words": words, "pages": pages})

    report = pd.DataFrame(rows)
    if report.empty:
        print("\nno non-article files found")
        return 0

    report["name"] = report["file"].map(os.path.basename)
    report["folder"] = report["file"].map(
        lambda p: os.path.basename(os.path.dirname(p)))
    report.to_csv(OUT, index=False)

    print(f"\n{len(report)} files flagged of {len(files):,} "
          f"({100 * len(report) / len(files):.1f}%)\n")
    for reason, group in report.groupby("reason"):
        print(f"  {len(group):4d}  {reason}")

    print("\n  by folder:")
    for folder, group in report.groupby("folder"):
        print(f"  {len(group):4d}  {folder}")

    print("\nsmallest examples:")
    for row in report.nsmallest(12, "words").itertuples():
        print(f"  {row.words:5d} words  {row.name[:66]}")

    print(f"\nwrote {OUT}")
    print("Nothing was deleted. The measurement step should exclude these.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
