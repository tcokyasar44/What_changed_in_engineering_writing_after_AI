"""
Measure the downloaded PDFs.

These are the quantities abstracts cannot provide:

  figures and tables   counted from caption lines, which follow strong conventions
                       in engineering journals ("Figure 3.", "Table 2:")
  equations            counted from display-equation numbering of the form (12),
                       which is the dominant convention in the field
  body length          words in the article body, excluding references
  body vocabulary      the same marker, hedging and boosting measures used on
                       abstracts, so the two can be compared directly
  section structure    presence of the sections that disclosure policies affect

Caption counting is a heuristic and will miss captions typeset unconventionally, so
it is validated against a manual check and reported with that caveat. The figures
here are sample estimates, not a census, and the paper must say so.

Only derived counts leave this stage. No article text is written to the release.

Stage 4 of the pipeline.
"""

import os
import re
import sys
from collections import Counter

import pandas as pd
import pymupdf

import paths
import text_metrics as tm

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))

# Locations come from paths.py rather than being rebuilt here. This script
# previously pointed at data/fulltext directly, which is the parent of both
# collections and belongs to neither. Reading from there would have silently mixed
# the retired exploratory draw with the current one, or found nothing at all.
FT = paths.FT
PDFS = paths.PDF_DIR

SAMPLE = paths.SAMPLE_CSV
OUT = paths.PARSED_PARQUET

# Retrieval has run as several draws against the same sampling design, each
# recording its outcomes separately so the frames stay distinguishable. They are
# discovered rather than listed, because a hard-coded list silently drops any
# later pass and the omission looks like a smaller sample rather than a bug.
def _collection_files(prefix, suffix):
    found = sorted(name for name in os.listdir(paths.ROOT)
                   if name.startswith(prefix) and name.endswith(suffix))
    return [os.path.join(paths.ROOT, name) for name in found]


STATUS_FILES = _collection_files("fetch_status", ".csv")
TRANSIENT_FILES = _collection_files("parsed_transient", ".parquet")
SAMPLE_FILES = _collection_files("sample", ".csv")

# Caption conventions. Anchored to a line start so that in-text cross references
# such as "as shown in Figure 4" are not counted.
#
# Trailing punctuation is optional. Several publishers, Elsevier among them, place a
# table caption on a line of its own with no punctuation at all, and requiring a
# full stop or colon silently undercounts tables in those journals. The caption may
# instead be followed by end of line, or by a space and the caption text.
FIG_CAPTION = re.compile(
    r"(?im)^\s*(?:fig(?:ure)?\.?|abb\.)\s*(\d{1,3})\s*(?:[.:）)]|$|\s+\S)",
    re.UNICODE)
TAB_CAPTION = re.compile(
    r"(?im)^\s*(?:table|tab\.)\s*(\d{1,3})\s*(?:[.:）)]|$|\s+\S)",
    re.UNICODE)

# Display equations numbered at the right margin.
EQ_NUMBER = re.compile(r"(?m)\(\s*(\d{1,3}[a-z]?)\s*\)\s*$")

SECTION_MARKERS = {
    # The reference heading appears in many forms: all capitals, numbered, with a
    # trailing colon, or spaced out by the typesetter. A heading-anchored match that
    # is too strict silently reports zero references for perfectly ordinary papers.
    "references": re.compile(
        r"(?im)^\s*(?:\d+\.?\s*|[ivxlc]+\.?\s*)?"
        r"(references?|bibliography|literature\s+cited|works\s+cited|"
        r"r\s?e\s?f\s?e\s?r\s?e\s?n\s?c\s?e\s?s)"
        r"\s*:?\s*$"),
    "acknowledg": re.compile(r"(?im)^\s*acknowledge?ments?\b"),
    "data_availability": re.compile(
        r"(?im)\bdata\s+availability\b|\bdata\s+access(ibility)?\s+statement\b"),
    "declaration": re.compile(r"(?im)\bdeclaration of (competing|interest)\b"),
    "ai_disclosure": re.compile(
        r"(?i)\b(chatgpt|gpt-4|gpt-3|large language model|generative ai|"
        r"copilot|claude|gemini)\b"),
}

MIN_BODY_WORDS = 500

# Front matter, editorials, indexes and similar items are typeset like articles but
# are not research contributions. They appear in bulk issue downloads and would
# distort every per-article measure if counted, so they are identified and dropped.
NON_ARTICLE_TITLES = re.compile(
    r"(?im)^\s*("
    r"editorial\s+board|editorial\s+advisory|advisory\s+board|"
    r"table\s+of\s+contents|contents\s+list|"
    r"author\s+index|subject\s+index|keyword\s+index|cumulative\s+index|"
    r"calendar|forthcoming\s+(events|papers|articles)|meetings?\s+calendar|"
    r"call\s+for\s+papers|announcements?|"
    r"instructions?\s+(for|to)\s+authors|guide\s+for\s+authors|"
    r"list\s+of\s+reviewers|acknowledgement\s+(to|of)\s+reviewers|"
    r"masthead|front\s+matter|back\s+matter|copyright\s+notice|"
    r"erratum|corrigendum|retraction\s+notice|publisher'?s\s+note|"
    r"book\s+review|obituary|in\s+memoriam|letter\s+to\s+the\s+editor"
    r")\b")

# A research article cites prior work and is structured. Section headings are
# numbered in several conventions, including Roman numerals in IEEE journals, so the
# numbering prefix is optional and the match is not anchored to a line start, since
# two-column extraction frequently merges a heading into the preceding line.
STRUCTURE_SIGNALS = re.compile(
    r"(?im)(?:^|\n|\s)(?:\d+\.?\s*|[ivxlc]{1,4}\.\s*)?"
    r"(abstract|introduction|related\s+work|background|"
    r"methodology|methods?|materials\s+and\s+methods|experimental|"
    r"results?(?:\s+and\s+discussion)?|discussion|"
    r"conclusions?(?:\s+and\s+future\s+work)?|acknowledge?ments?)\b")

# Counting citation markers is a more robust signal than the length of a reference
# section, because extraction sometimes loses the heading entirely while the
# in-text markers survive.
CITATION_MARKERS = re.compile(r"\[\s*\d{1,3}(?:\s*[,-]\s*\d{1,3})*\s*\]")

MIN_REFERENCE_WORDS = 150
MIN_STRUCTURE_HITS = 2
MIN_CITATION_MARKERS = 8


def read_pdf(path):
    try:
        doc = pymupdf.open(path)
    except Exception:
        return None, 0
    try:
        pages = len(doc)
        text = "\n".join(page.get_text("text") for page in doc)
    except Exception:
        doc.close()
        return None, 0
    doc.close()
    return text, pages


def split_body(text):
    """Return the article body, excluding the reference list."""
    match = SECTION_MARKERS["references"].search(text)
    if match and match.start() > len(text) * 0.3:
        return text[:match.start()], text[match.start():]
    return text, ""


def distinct_captions(pattern, text):
    """Count distinct caption numbers, so a repeated running header counts once."""
    return len({m.group(1) for m in pattern.finditer(text)})


def looks_like_research_article(text, body, refs):
    """Reject front matter, editorials, indexes and other non-article items.

    Bulk issue downloads contain calendars, editorial boards, author indexes and
    similar material typeset like articles. Counting them would depress every
    per-article measure. Three signals are used together, since any one alone is
    unreliable: the opening text does not announce itself as front matter, the
    document carries a substantive reference list, and it shows the section
    structure of a research paper.
    """
    opening = text[:1500]
    if NON_ARTICLE_TITLES.search(opening):
        return False, "front_matter_or_editorial"

    # Evidence that the document cites prior work. Either a reference section that
    # was successfully separated, or enough in-text citation markers to make a
    # reference list certain even when the heading was lost in extraction.
    cites = (len(refs.split()) >= MIN_REFERENCE_WORDS
             or len(CITATION_MARKERS.findall(text)) >= MIN_CITATION_MARKERS)

    structured = len(set(m.lower() for m in STRUCTURE_SIGNALS.findall(text))) \
        >= MIN_STRUCTURE_HITS

    # Both signals are imperfect on their own: some journals drop numbered headings,
    # and author-year citation styles produce no bracketed markers. Requiring either
    # one, rather than both, keeps the false positive rate low while still excluding
    # calendars, indexes and editorial boards, which satisfy neither.
    if not cites and not structured:
        return False, "neither_citations_nor_article_structure"

    return True, None


def measure(text, pages, strict=False):
    """Measure one article. With strict set, non-research items are rejected."""
    body, refs = split_body(text)

    if strict:
        ok, _ = looks_like_research_article(text, body, refs)
        if not ok:
            return None

    tokens = tm.tokenize(body)
    if len(tokens) < MIN_BODY_WORDS:
        return None

    counts = Counter(tokens)
    n = len(tokens)
    marker = set(tm.MARKER_WORDS)
    hedge = set(tm.HEDGES)
    boost = set(tm.BOOSTERS)

    return {
        "pdf_pages": pages,
        "body_words": n,
        "n_figures": distinct_captions(FIG_CAPTION, text),
        "n_tables": distinct_captions(TAB_CAPTION, text),
        "n_equations": len({m.group(1) for m in EQ_NUMBER.finditer(body)}),
        "marker_per_10k": 1e4 * sum(counts[w] for w in marker) / n,
        "hedge_per_10k": 1e4 * sum(counts[w] for w in hedge) / n,
        "booster_per_10k": 1e4 * sum(counts[w] for w in boost) / n,
        "long_word_share": sum(1 for t in tokens if len(t) >= 9) / n,
        "ttr100": tm.ttr_window(tokens),
        "has_data_availability": bool(SECTION_MARKERS["data_availability"].search(text)),
        "has_declaration": bool(SECTION_MARKERS["declaration"].search(text)),
        "mentions_ai_tool": bool(SECTION_MARKERS["ai_disclosure"].search(text)),
        "ref_section_words": len(refs.split()),
    }


DOI_IN_TEXT = re.compile(r"\b10\.\d{4,9}/[^\s\"'<>)\]}]+", re.I)


def doi_from_pdf(text, filename, known):
    """Identify which sampled article a manually saved PDF corresponds to.

    The DOI printed inside the document is the primary key, but accepted
    manuscripts and author copies often do not print one, and a text scan then finds
    only the DOIs of cited works. Matching is therefore attempted against the whole
    document rather than the first few pages, and the filename is used as a
    fallback, since publisher downloads usually encode the identifier there.
    """
    for match in DOI_IN_TEXT.finditer(text):
        candidate = match.group(0).rstrip(".,;)]").lower()
        if candidate in known:
            return candidate

    stem = os.path.splitext(os.path.basename(filename))[0]
    stem = re.sub(r"^\d{4}[_-]", "", stem)

    for guess in (stem.replace("_", "/").lower(),
                  stem.replace("-", "/", 1).lower()):
        if guess in known:
            return guess

    return None


def parse_manual(sample, known):
    """Parse PDFs a person saved by hand into data/fulltext/manual_pdf."""
    folder = paths.MANUAL_PDF_DIR
    if not os.path.isdir(folder):
        return []

    files = [f for f in os.listdir(folder) if f.lower().endswith(".pdf")]
    if not files:
        return []

    print(f"\nmanually saved PDFs found: {len(files)}")
    meta = sample.set_index("doi")
    rows, unmatched, rejected = [], 0, Counter()

    for name in files:
        path = os.path.join(folder, name)
        text, pages = read_pdf(path)
        if not text:
            rejected["unreadable"] += 1
            continue

        # Manual folders may contain whole-issue downloads, so non-research items
        # are screened out before anything is measured.
        body, refs = split_body(text)
        ok, reason = looks_like_research_article(text, body, refs)
        if not ok:
            rejected[reason] += 1
            continue

        doi = doi_from_pdf(text, name, known)
        if doi is None or doi not in meta.index:
            unmatched += 1
            continue
        stats = measure(text, pages)
        if stats is None:
            rejected["too_short"] += 1
            continue
        info = meta.loc[doi]
        stats.update({
            "doi": doi,
            "publication_year": info["publication_year"],
            "publication_date": info["publication_date"],
            "period": info["period"],
            "publisher": info["publisher"],
            "scholar_category": info["scholar_category"],
            "source_id": info["source_id"],
            "source": "manual",
        })
        rows.append(stats)

    print(f"  matched to the sample: {len(rows)}")
    if unmatched:
        print(f"  could not be matched : {unmatched} "
              f"(no recognisable DOI inside the file)")
    if rejected:
        print("  screened out as non-research items:")
        for reason, count in rejected.most_common():
            print(f"    {count:4d}  {reason}")
    return rows


def load_sample():
    """The sampling frame, every draw, as one table keyed by DOI."""
    frames = [pd.read_csv(path) for path in SAMPLE_FILES]
    sample = pd.concat(frames, ignore_index=True)
    return sample.drop_duplicates("doi")


def main():
    status = pd.concat([pd.read_csv(p) for p in STATUS_FILES
                        if os.path.exists(p)], ignore_index=True)
    have = status[status["bytes"].notna() & status["file"].notna()]
    have = have.drop_duplicates("doi")
    sample = load_sample()
    meta = sample.set_index("doi")

    print(f"PDFs to parse: {len(have):,}")

    rows = []
    failed = 0
    for i, (_, rec) in enumerate(have.iterrows(), start=1):
        path = os.path.join(PDFS, rec["file"])
        if not os.path.exists(path):
            continue
        text, pages = read_pdf(path)
        if not text:
            failed += 1
            continue
        stats = measure(text, pages)
        if stats is None:
            failed += 1
            continue

        doi = rec["doi"]
        if doi in meta.index:
            info = meta.loc[doi]
            stats.update({
                "doi": doi,
                "publication_year": info["publication_year"],
                "publication_date": info["publication_date"],
                "period": info["period"],
                "publisher": info["publisher"],
                "scholar_category": info["scholar_category"],
                "source_id": info["source_id"],
            })
            rows.append(stats)

        if i % 250 == 0:
            print(f"  parsed {i:,}/{len(have):,}", flush=True)

    known_dois = {str(d).lower() for d in sample["doi"].dropna()}
    manual_rows = parse_manual(sample, known_dois)
    if manual_rows:
        rows.extend(manual_rows)

    # Articles that were free to read but carried no explicit reuse licence were
    # parsed during the fetch and never stored. Their measurements are merged here
    # so the analysis sees one sample, while no publisher file was retained.
    existing = [p for p in TRANSIENT_FILES if os.path.exists(p)]
    if existing:
        transient = pd.concat([pd.read_parquet(p) for p in existing],
                              ignore_index=True).drop_duplicates("doi")
        if not transient.empty:
            transient = transient.merge(
                sample[["doi", "publication_year", "publication_date", "period",
                        "publisher", "scholar_category", "source_id"]],
                on="doi", how="left")
            transient["retained_copy"] = False
            rows_frame = pd.DataFrame(rows)
            if not rows_frame.empty:
                rows_frame["retained_copy"] = True
            frame = pd.concat([rows_frame, transient], ignore_index=True)
            frame = frame.drop_duplicates(subset="doi", keep="first")
            print(f"  merged {len(transient):,} articles parsed without retention")
        else:
            frame = pd.DataFrame(rows)
            frame["retained_copy"] = True
    else:
        frame = pd.DataFrame(rows)
        if not frame.empty:
            frame["retained_copy"] = True

    frame.to_parquet(OUT, index=False)

    print(f"\nparsed {len(frame):,} articles, {failed:,} unusable")
    if frame.empty:
        return

    print("\nmedians by period:")
    cols = ["n_figures", "n_tables", "n_equations", "body_words",
            "pdf_pages", "marker_per_10k", "hedge_per_10k"]
    print(frame.groupby("period")[cols].median().round(2).to_string())
    print("\nAI tool mentioned in text, by period:")
    print((100 * frame.groupby("period")["mentions_ai_tool"].mean()).round(2).to_string())
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
