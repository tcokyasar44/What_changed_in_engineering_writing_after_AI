"""
Paths for the analysis steps.

Defaults assume the repository layout:

    data/corpus.parquet          built by scripts/build_corpus.py
    fulltext/                    the full-text sample record (fulltext_sample.csv.gz);
                                 the PDFs themselves are not distributed
    analysis/out/                every result file written by the analysis
    analysis/out/figures/        the figures

To point the analysis at PDFs or a figure folder elsewhere, put any of these lines in
a file named paths.env next to this one (it is not distributed):

    FULLTEXT_DIR=C:/path/to/folder/with/pdf/manual_pdf/and/the/retrieval/records
    FIGURES_DIR=C:/path/to/manuscript
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def _local():
    values, path = {}, os.path.join(HERE, "paths.env")
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            if "=" in line and not line.lstrip().startswith("#"):
                k, _, v = line.strip().partition("=")
                values[k.strip()] = os.path.normpath(os.path.join(HERE, v.strip()))
    return values


_L = _local()
CORPUS = os.path.join(ROOT, "data", "corpus.parquet")
OUT = os.path.join(HERE, "out")
FULLTEXT = _L.get("FULLTEXT_DIR", os.path.join(ROOT, "fulltext"))
FIGURES = _L.get("FIGURES_DIR", os.path.join(OUT, "figures"))
SECRETS = os.path.join(ROOT, "secrets.env")
os.makedirs(OUT, exist_ok=True)
os.makedirs(FIGURES, exist_ok=True)
