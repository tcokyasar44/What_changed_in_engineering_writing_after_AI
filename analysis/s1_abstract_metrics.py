"""
Step 1. Per-article measures from abstracts and the bibliographic record.

One row per article in the corpus. Text measures are filled for articles with an
abstract of at least 40 words. Counts are stored, not rates, so that every later
aggregate is a pooled rate (total occurrences / total words) and every family is
counted from its listed forms.
"""
import os, sys, time
import settings as P
from collections import Counter
from multiprocessing import Pool
import numpy as np, pandas as pd
import lexicon as L

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = os.path.dirname(os.path.abspath(__file__))
CORPUS = P.CORPUS
OUT = P.OUT
MIN_WORDS = 40
FAMS = L.ALL_FAMILIES


def measure(text):
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return None
    toks = L.tokenize(text)
    n = len(toks)
    if n < MIN_WORDS:
        return None
    c = Counter(toks)
    fam = Counter()
    for w, k in c.items():
        f = L.FORM_TO_FAMILY.get(w)
        if f:
            fam[f] += k
    return (n,
            sum(fam[f] for f in L.MARKER_FAMILIES),
            sum(fam[f] for f in L.CONTROL_FAMILIES),
            sum(k for w, k in c.items() if w in L.HEDGES),
            sum(k for w, k in c.items() if w in L.BOOSTERS),
            sum(k for w, k in c.items() if len(w) >= 9),
            sum(L.syllables(w) * k for w, k in c.items()),
            L.ttr_window(toks),
            [fam[f] for f in FAMS])


def work(chunk):
    return [measure(t) for t in chunk]


def main():
    t = time.time()
    cols = ["openalex_id", "doi", "source_id", "publisher", "scholar_category", "cohort",
            "text_tier", "publication_date", "abstract", "abstract_source",
            "corr_lang_group", "n_authors", "n_institutions", "n_references", "n_pages",
            "title_words", "title_has_colon", "is_oa"]
    df = pd.read_parquet(CORPUS, columns=cols)
    print("corpus", len(df), flush=True)
    texts = df["abstract"].tolist()
    chunks = [texts[i:i + 5000] for i in range(0, len(texts), 5000)]
    with Pool(12) as p:
        res = [r for part in p.imap(work, chunks, chunksize=1) for r in part]
    print("measured", time.time() - t, flush=True)
    df = df.drop(columns=["abstract"])
    names = ["n_words", "marker", "control", "hedge", "booster", "long_words",
             "syllables", "ttr100"]
    arr = {k: np.full(len(df), np.nan) for k in names}
    famarr = np.zeros((len(df), len(FAMS)), dtype=np.int32)
    has = np.zeros(len(df), dtype=bool)
    for i, r in enumerate(res):
        if r is None:
            continue
        has[i] = True
        for k, v in zip(names, r[:8]):
            arr[k][i] = v
        famarr[i] = r[8]
    for k in names:
        df[k] = arr[k]
    df["has_text"] = has
    for j, f in enumerate(FAMS):
        df["fam_" + f] = famarr[:, j]
    d = df["publication_date"]
    df["jan1"] = (d.dt.month == 1) & (d.dt.day == 1)
    df["month"] = d.values.astype("datetime64[M]")
    df["year"] = d.dt.year
    df.to_parquet(os.path.join(OUT, "articles.parquet"), index=False)
    print("articles with text", has.sum(), "of", len(df), "time", round(time.time() - t), flush=True)


if __name__ == "__main__":
    main()
