"""
Directory layout for the full-text collections, and the rule that keeps them apart.

Two collections exist and must never be pooled.

v1_exploratory was gathered while the retrieval machinery was being built. It was
drawn from the open-access subset only, so it is 100 percent open access against 48
percent in the corpus, and it is dominated by the years automated sources happen to
hold: 2016 and 2017 are over-represented roughly fivefold while 2023 onward is
under-represented tenfold. It is useful for validating the parser, for measuring
what automated retrieval can and cannot reach, and as an independent robustness
check. It is not a basis for estimating any level.

v2_target is drawn from the completed corpus, stratified by year and allocated to
journals in proportion to their share of each year, and it includes subscription
articles. Estimates reported in the paper come from this collection.

Mixing them would be unrecoverable. Once a file sits in the wrong folder there is no
reliable way to tell which sampling frame produced it, and every level computed
afterwards would silently blend a representative draw with a convenience sample.
Every script therefore takes its directory from here, and COLLECTION selects which.
"""

import os

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
FT = os.path.join(DATA, "fulltext")

# Switch deliberately. Nothing else in the pipeline chooses a collection.
COLLECTION = "v2_target"

V1 = os.path.join(FT, "v1_exploratory")
V2 = os.path.join(FT, "v2_target")

ROOT = V2 if COLLECTION == "v2_target" else V1

PDF_DIR = os.path.join(ROOT, "pdf")
MANUAL_PDF_DIR = os.path.join(ROOT, "manual_pdf")
BULK_PDF_DIR = os.path.join(ROOT, "bulk_pdf")
REJECT_PDF_DIR = os.path.join(ROOT, "rejected_pdf")
XML_DIR = os.path.join(ROOT, "xml")

SAMPLE_CSV = os.path.join(ROOT, "sample.csv")
RESOLVED_CSV = os.path.join(ROOT, "resolved.csv")
FETCH_STATUS_CSV = os.path.join(ROOT, "fetch_status.csv")
SORT_LOG_CSV = os.path.join(ROOT, "manual_sort_log.csv")
PARSED_PARQUET = os.path.join(ROOT, "parsed.parquet")
TRANSIENT_PARQUET = os.path.join(ROOT, "parsed_transient.parquet")

# The target draw lives outside either collection, since it is a plan rather than a
# result and is regenerated whenever the corpus changes.
TARGET_SAMPLE_CSV = os.path.join(FT, "target_sample.csv")

MANUAL_LISTS = os.path.join(FT, "manual")


def ensure():
    """Create the directories for the selected collection."""
    for path in (ROOT, PDF_DIR, MANUAL_PDF_DIR, BULK_PDF_DIR,
                 REJECT_PDF_DIR, XML_DIR, MANUAL_LISTS):
        os.makedirs(path, exist_ok=True)
    return ROOT


def describe():
    lines = [f"collection: {COLLECTION}", f"root      : {ROOT}"]
    for label, path in (("pdf", PDF_DIR), ("manual", MANUAL_PDF_DIR),
                        ("bulk", BULK_PDF_DIR), ("xml", XML_DIR)):
        n = len(os.listdir(path)) if os.path.isdir(path) else 0
        lines.append(f"  {label:8s} {n:6,} files")
    return "\n".join(lines)


if __name__ == "__main__":
    ensure()
    print(describe())
