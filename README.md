# Replication package: What Changed in Engineering Writing After Generative AI Chat Assistants

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24028663.svg)](https://doi.org/10.5281/zenodo.24028663)

Code and data for the article *What Changed in Engineering Writing After Generative AI
Chat Assistants: Evidence from 1.8 Million Articles* (Taner Cokyasar, Texas A&M
University). The package rebuilds the article corpus from open sources, measures it,
and reproduces every number, table and figure in the article.

No language model reads, summarizes or judges any article: every measurement is made
by the deterministic code in this repository.

## What is here

| Folder | Contents |
|---|---|
| `scripts/` | Builds the corpus: 1,839,962 research articles published 2015–2025 in 256 engineering journals |
| `analysis/` | Measures abstracts, records and full texts; fits the time-series models; draws the figures. `analysis/out/` holds the computed result tables (CSV and JSON) that the article reports; a rerun overwrites them |
| `data/` | The journal panel and every exclusion, the identifiers of all articles, verification results |
| `fulltext/` | The record of the full-text sample (one row per drawn article) and the retrieval code as run |
| `tests/` | Offline tests (no network, no key) |

Article text is not distributed. Abstracts come from OpenAlex and Semantic Scholar and
full texts from publishers and repositories; the pipeline fetches them again.

## Quick start

```bash
pip install -r requirements.txt
cp secrets.env.example secrets.env      # add a free OpenAlex key and your e-mail address
python -m pytest tests                  # offline checks, seconds

cd scripts
python resolve_venues.py      # 1 match the Scholar Metrics lists to OpenAlex      ~15 min
python finalize_panel.py      # 2 apply the inclusion rules                         seconds
python watchdog.py            # 3 harvest every journal (runs harvest.py)           hours
python build_corpus.py        # 4 assemble the corpus                               ~2 min
python recover_abstracts.py   # 5 fill missing abstracts from Semantic Scholar      hours
python build_corpus.py        # 4 again, to merge the recovered abstracts
python verify_coverage.py     # 6 compare with OpenAlex, journal by journal          ~15 min

cd ../analysis
python run_all.py --no-pdf    # everything except re-reading the PDFs               ~1 hour
```

`run_all.py --no-pdf` uses the shipped full-text record (`fulltext/fulltext_sample.csv.gz`)
in place of the PDFs. Running `run_all.py` without the flag re-reads the PDFs, which
requires obtaining them first (see `fulltext/README.md`).

## The corpus, step by step

**1. Sampling frame.** The journals are the top-20 venues by h5-index in fifteen Google
Scholar Metrics subcategories of Engineering & Computer Science, as published on
25 September 2026 (`data/scholar_metrics_lists.csv`). Google Scholar does not allow
automated access, so the lists are saved here rather than fetched.
`resolve_venues.py` matches each venue to an OpenAlex journal by exact name. Three
venues need a hand correction, each recorded with its reason in
`data/venue_corrections.csv`: one name differs by a letter, one by punctuation, and
one ("Decision") would otherwise be confused with *Decision Support Systems*.

**2. Inclusion rules** (`finalize_panel.py`, exclusions in `data/panel_exclusions.csv`):
drop the Theoretical Computer Science category (mostly conferences); drop conferences;
drop venues with no OpenAlex journal record (none remain after the corrections); keep a
journal listed in several categories once, in the category where it ranks highest.
Result: 256 journals (`data/journal_panel_final.csv`). The panel does not depend on the
order of the input or on library versions (tested).

**3. Harvest** (`harvest.py`, supervised by `watchdog.py`). Every OpenAlex work of type
"article" whose primary location is a panel journal, published 2015–2025; one Parquet
shard per journal; resumable. The corpus used in the article was harvested on 25–26
September 2026 (three journals added after the corrections on 7 November 2026); the
dates are in `data/harvest_manifest.json`.

**4. Assembly** (`build_corpus.py`). Removes 20,424 machine translations (OpenAlex lists
automated translations of English articles as separate records) and 4,095 other
non-English records without a DOI; keeps records with a DOI whatever their language
label (the label is unreliable for mathematical text). Derives months, periods,
author-country groups and page counts (article numbers and implausible spans are not
page counts). Measures each journal on the data: *core* journals have at least 20
articles in every year (229 of 256).

**5. Abstract recovery** (`recover_abstracts.py`). OpenAlex has an abstract for 60.2% of
the articles; Semantic Scholar supplies 180,238 more (9.8%). Every abstract records its
source, so any result can be recomputed on OpenAlex abstracts alone.

**6. Verification** (`verify_coverage.py`, `check_translations.py`). Compares the corpus
with OpenAlex's current count of articles with a DOI for every journal and year
(`data/coverage_by_doi.csv`). Run on 7 November 2026, the corpus held 102.1% of that
count: since the September harvest OpenAlex has re-typed many works from "article" to
"review", above all in *Critical Reviews in Food Science and Nutrition* and several MDPI
journals. The corpus therefore contains the works OpenAlex classified as articles in
September 2026, including some reviews. 69 of 2,776 journal-years hold under 97% of
the current count, mostly small journal-years.

`data/article_ids.csv.gz` lists the OpenAlex identifier, DOI, journal and date of every
article, so the corpus can be rebuilt exactly as used even after OpenAlex changes.

## The analysis

See `analysis/run_all.py` for the order. In brief: per-article counts of 31 marker
words and 4 control words (inflections counted as one word), qualifying and emphatic
words, long words, word variety and readability (`s1`); monthly pooled rates and a
segmented regression of each measure against the projection of its pre-2023 trend,
with robustness checks (`s2`); excess ratios for the whole vocabulary and word lists
chosen from the data (`s3`); the full-text sample (`s4`, `s4b`, `s4c`, `s5`); figures
(`s6`).

## Reproducibility

OpenAlex and Semantic Scholar change continuously, so a fresh harvest will differ
slightly from the September 2026 snapshot. To reproduce the article exactly, restrict a
fresh harvest to the identifiers in `data/article_ids.csv.gz`. The analysis is
deterministic given the corpus.

## Licence and citation

Code: MIT. Data files: CC0. See `LICENSE` and `CITATION.cff`.
