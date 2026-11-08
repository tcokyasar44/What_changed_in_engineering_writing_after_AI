# Changes

## 1.0.2 (7 November 2026): version used for the submitted article

- Figures: the public-release date is drawn as a dash-dot line and the months left out
  of the fit (December 2022 to February 2023) as a hatched band, and both appear in the
  legend. No result changed. [`analysis/s6_figures.py`]

## 1.0.1 (7 November 2026)

- `analysis/out/coverage_by_publisher.csv` now counts the 24,654 articles that have no
  publisher recorded in OpenAlex, as "(no publisher recorded)". They were left out of
  that table before, so its rows summed to less than the corpus. No other result
  changed. [`analysis/s2_series.py`]
- The README and citation file give the Zenodo DOI.

## 1.0.0 (7 November 2026)

An audit of the September 2026 pipeline found the defects below. Each was fixed in the
code, the corpus was rebuilt, and every result was recomputed. The machine-readable
records of each change are named in brackets.

### Journal panel
- The Scholar venue "Decision" (American Psychological Association) had been matched to
  *Decision Support Systems* (Elsevier) because one name contains the other. Matching is
  now by exact name; the 1,297 *Decision Support Systems* articles were removed and
  *Decision* added. [`data/venue_corrections.csv`]
- *IEEE Transactions on Smart Grid* and *Engineering, Construction and Architectural
  Management* had been excluded as "no OpenAlex record" because a table of name
  corrections was never applied. Both are now included. [`data/venue_corrections.csv`]
- A journal listed in two categories was kept in whichever category the sort order put
  first (its h5-index is the same in both listings). It is now kept where Scholar ranks
  it highest. [`scripts/finalize_panel.py`; tested for order independence]
- Panel: 254 journals before, 256 after.

### Corpus
- The harvester stored abstracts only for journals whose 2024 sample showed at least 30%
  abstract coverage. OpenAlex abstracts of the other 81 journals (49,245 articles) were
  restored on 7 November 2026 by fetching the abstract field alone, by OpenAlex id, into
  the September records. [`data/harvest_manifest.json`: `abstracts_backfilled_at`]
  The headline estimate is also reported without these journals.
- Abstract word counts had used the number of distinct words; they now count words.
- Page counts reject spans over 200 pages (one record implied 6.75 million pages).
- The study window (2015–2025) is set explicitly rather than inferred.
- Cohort (journals with at least 20 articles in every year) and abstract tier are measured
  on the harvested corpus instead of a 50-article sample from 2024.
- Corpus: 1,833,993 articles before, 1,839,962 after.
- The coverage check now covers every journal (it had stopped after the first 20).
  [`data/coverage_by_doi.csv`]

### Full text
- Four quality checks were added before measurement: title match, supporting-information
  file, readable text, article length. Each excluded document keeps its status.
  [`fulltext/fulltext_sample.csv.gz`, column `status`]
- A front-matter rule had rejected real articles beginning "Calendar aging…" or carrying a
  "Table of Contents" graphic; it now applies only to short documents.
- 1,511 free-to-read articles, measured in memory in September with a retired word list,
  were fetched again under the same policy and measured with the current code.
  [`fulltext/fulltext_sample.csv.gz`, `route == "memory"`]
- The disclosure measure was audited by reading every flagged document and every
  sentence containing "During the preparation of this work". It had counted that phrase
  in acknowledgements, sentences about language models as a research subject ("natural
  language", "improve the model"), quoted journal policy and statements of non-use, and
  it had missed declarations naming writing assistants (Grammarly, DeepL, QuillBot) or
  broken across lines ("Chat- GPT"). The rule now requires use stated as done, a tool,
  and a purpose that names the text; it excludes negations and prescriptive wording.
  All stored documents were measured again (parser 4), and the free-to-read articles
  were fetched again: 1,458 of the 1,460 measured ones were re-measured; two could not
  be fetched and keep their earlier measurement (`recheck_failed` in `out/fulltext_bronze.parquet`).
- A robustness estimate restricted to publisher versions was added.

### Package
- Google Scholar Metrics lists are no longer fetched by a script (Google does not permit
  automated access, and the earlier script presented itself as a web browser). The lists
  are saved with their date. [`data/scholar_metrics_lists.csv`]
- Fixed: a script that failed on start (`check_translations.py`), a Windows-only
  supervisor, and an unused Springer step whose results were never merged (removed).
  A progress monitor that read the wrong folder and a one-off probe of alternative
  download locations were removed; neither affected any result.
- No personal contact address is used as a fallback; every request carries the
  `CONTACT_EMAIL` the user sets.
- Added offline tests, pinned versions, a licence, a citation file and the list of article
  identifiers. [`data/article_ids.csv.gz`]
