# The full-text sample

The article compares the full text of a sample of articles published before December
2022 with articles published from March 2023 onward. This folder documents how the
sample was drawn and obtained, and holds its record.

## Files

| File | Contents |
|---|---|
| `fulltext_sample.csv.gz` | One row per drawn article: DOI, draw, whether it is an open-access replacement, how the copy was obtained, measurement status, whether the copy is the publisher's version, and every measure (counts only, no text) |
| `as_run/` | The sampling and retrieval scripts exactly as they were run in September 2026 (the author's address was removed as a fallback contact, and a one-off diagnostic that probed alternative download locations was left out). They read and write paths of the original project and are kept as a record, not as a pipeline to rerun |

`analysis/s5_fulltext_results.py` reads `fulltext_sample.csv.gz`, so the full-text
results can be reproduced without any PDF.

## How the sample was drawn

The sample is a random subsample of the same corpus the abstract analysis uses.

- **Within each year**, every journal's share of the sample matches its share of that
  year's articles (journals with fewer than 20 articles in a year get no allocation).
- **Across years**, each year's allocation is proportional to its corpus size times a
  weight: 0.6 (2015 to 2017), 0.7 (2018), 0.8 (2019, 2020), 0.9 (2021), 1.0 (2022),
  and 2.2 (2023 to 2025). The after-release side covers three years against eight, so
  it is sampled at a higher rate to be estimated as precisely as the before side.
  The resulting design rates are about 0.21% of 2015 articles, rising to 0.41% of
  2022 articles, and 0.90% of each year from 2023 to 2025.
- **The target** was 10,000 analyzed articles (3,617 before 2023 and 6,383 after),
  allocated to 2,726 journal-year cells by largest remainder
  (`build_target_quota.py`). Within a cell, articles were always drawn at random,
  never chosen for being easy to obtain.
- **All measured articles are kept.** Refill draws were retrieved in full rather than
  stopped at each cell's quota, so the analyzed sample is larger than 10,000 and some
  cells exceed their quota while others fall short. The analysis weights every article
  by (corpus articles in its journal-year) / (analyzed articles in that journal-year),
  which removes both the higher rate after 2022 and the uneven filling.
- Seven documents downloaded in a browser matched no draw file (`draw` =
  `browser_only`); they stay in the record but are not analyzed.

Five draws were made:

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
