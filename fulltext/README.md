# The full-text sample

The article compares the full text of a sample of articles published before December
2022 with articles published from March 2023 onward. This folder documents how the
sample was drawn and obtained, and holds its record.

## Files

| File | Contents |
|---|---|
| `fulltext_sample.csv.gz` | One row per drawn article: DOI, draw, whether it is an open-access replacement, how the copy was obtained, measurement status, whether the copy is the publisher's version, and every measure (counts only, no text) |
| `as_run/` | The sampling and retrieval scripts exactly as they were run in September 2026 (only the author's address was removed as a fallback contact). They read and write paths of the original project and are kept as a record, not as a pipeline to rerun |

`analysis/s5_fulltext_results.py` reads `fulltext_sample.csv.gz`, so the full-text
results can be reproduced without any PDF.

## How the sample was drawn

The target was 10,000 analysed articles, allocated to journal-year cells. Articles
published from 2023 were given 2.2 times their share, so the after-release side is
estimated precisely; the analysis weights every article by (corpus articles in its
journal-year) / (analysed articles in that journal-year), which removes that tilt.
Within a cell, articles were always drawn at random, never chosen for being easy to
obtain. Five draws were made:

| Draw | Script | What it drew |
|---|---|---|
| v2 | `build_v2_sample.py` | a pool of about 120,000 articles, proportional to each journal's output within each year |
| v3 | `build_v3_sample.py` | more articles from publishers whose articles were retrieved less often, so the retrieved set kept the corpus's publisher mix |
| v4 | `build_v4_substitutes.py` | random replacements within cells that were still short |
| v5, v6 | `build_v5_openaccess.py`, `build_v6_openaccess.py` | open-access replacements within cells still short; flagged as `substitute` so every estimate is also reported without them |

Every draw excluded all articles drawn before.

## How copies were obtained

1. **Locating.** Unpaywall (`resolve_*.py`) lists the legal open copies of each DOI:
   publisher copies and repository copies (institutional repositories, arXiv, OSTI, HAL).
2. **Automated download** (`fetch_*.py`). Requests identify the study and a contact
   address, never a browser; robots.txt is obeyed; requests to one host are at least
   1.5 seconds apart; hosts that refuse automated access (Wiley, MDPI, RSC, ACS, IEEE,
   ScienceDirect, Springer) are not requested. Openly licensed copies were stored.
   Free-to-read copies without a licence ("bronze") were measured in memory and never
   stored; `analysis/s4b_fetch_bronze.py` fetches and measures them again in the same way.
3. **Browser downloads.** For cells automation could not fill, the author downloaded
   specific drawn articles in a web browser (`make_download_page.py` lists them), and `sort_downloads.py` filed each PDF by the DOI printed
   in it.

## Quality checks before measurement

`analysis/s4_fulltext_parse.py` measures a document only if it passes four checks:
the corpus title of the DOI appears in its first two pages (otherwise `title_mismatch`);
it is not a supporting-information file (`supplementary`); its text extracts as English
words rather than scrambled characters (`unreadable_text`); and its body has at least
500 words and is not front matter (`not_article_or_short`).

## The disclosure measure

An article counts as disclosing the use of an AI writing tool when, outside its
reference list, it states that such a tool was used on its text. Three forms count:
the standard declaration ("During the preparation of this work the author(s) used
[tool] in order to [purpose]"); a declaration heading ("Declaration of generative AI
and AI-assisted technologies in the writing process") followed by a statement of use;
and any sentence stating use as done ("used", "was employed", "acknowledge the use of"),
naming a tool and naming a purpose that concerns the text (language, grammar,
readability, proofreading, translation). Statements of non-use and quoted policy
("should only be used") do not count. The rule was checked by reading all 127 documents
it flags among the stored copies; `s4_fulltext_parse.py` documents it in full.
