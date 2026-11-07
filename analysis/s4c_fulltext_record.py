"""
Step 4c. Write the record of the full-text sample: one row per drawn article.

Columns:
  doi, draw            the article and the draw that selected it (v2 proportional pool,
                       v3 yield-sized, v4 random replacements, v5/v6 open-access
                       replacements)
  substitute           True for the open-access replacements (v5, v6)
  route                how the measured copy was obtained: file (automated download,
                       kept), browser (the author's download), memory (free to read,
                       measured without keeping a copy), or none
  retrieval_error      the last automated-retrieval outcome when no copy was obtained
  status               measurement status (ok, title_mismatch, supplementary,
                       unreadable_text, not_article_or_short, parse_timeout, ...)
  publisher_copy       the measured copy is known to be the publisher's version
  title_overlap, english_share, pages, body_words and every measure of step 4

Only derived counts are recorded, never text. s5_fulltext_results.py reads this file,
so the analysis can be repeated without the PDFs.

Output: out/fulltext_sample.csv.gz
"""
import glob, os, sys
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s4_fulltext_parse as S

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
V2, OUT = S.V2, S.OUT
DRAWS = {"sample.csv": "v2", "sample_v3.csv": "v3", "sample_v4.csv": "v4",
         "sample_v5.csv": "v5", "sample_v6.csv": "v6"}


def low(s):
    return s.astype(str).str.lower().str.strip()


def main():
    drawn = []
    for f, label in DRAWS.items():
        d = pd.read_csv(os.path.join(V2, f), usecols=["doi"], low_memory=False)
        drawn.append(pd.DataFrame({"doi": low(d.doi), "draw": label}))
    drawn = pd.concat(drawn).drop_duplicates("doi")          # earliest draw wins
    drawn["substitute"] = drawn.draw.isin(["v5", "v6"])

    st = pd.concat([pd.read_csv(p, low_memory=False) for p in glob.glob(os.path.join(V2, "fetch_status*.csv"))])
    st["doi"] = low(st.doi)
    st["bytes"] = pd.to_numeric(st["bytes"], errors="coerce")
    st = st.sort_values("bytes", na_position="first").drop_duplicates("doi", keep="last")
    err = dict(zip(st.doi, st.error))

    docs = [pd.read_parquet(os.path.join(OUT, "fulltext_articles.parquet"))]
    if os.path.exists(os.path.join(OUT, "fulltext_bronze.parquet")):
        docs.append(pd.read_parquet(os.path.join(OUT, "fulltext_bronze.parquet")))
    docs = pd.concat(docs, ignore_index=True)
    docs = docs[docs.doi.notna()].copy()
    docs["doi"] = low(docs.doi)
    docs["route"] = docs.folder.map({"pdf": "file", "manual_pdf": "browser", "memory": "memory"})
    docs["ok"] = docs.status == "ok"
    docs["pref"] = docs.route.map({"browser": 0, "file": 1, "memory": 2})
    docs = docs.sort_values(["ok", "pref"], ascending=[False, True]).drop_duplicates("doi")

    host = {}
    for f in ("resolved.csv", "resolved_v3.csv"):           # one recorded location per article
        r = pd.read_csv(os.path.join(V2, f), usecols=["doi", "host_type"], low_memory=False)
        host.update(dict(zip(low(r.doi), r.host_type)))
    alt = set(low(pd.read_csv(os.path.join(V2, "alternates.csv"), usecols=["doi"]).doi))
    aam = docs.file.fillna("").str.contains(r"(?i)aam|accepted|manuscript|preprint|clean")
    docs["publisher_copy"] = (((docs.route == "browser") & ~aam)
                              | ((docs.route == "file") & (docs.doi.map(host) == "publisher")
                                 & ~docs.doi.isin(alt)))

    keep = ["doi", "route", "status", "publisher_copy", "title_overlap", "english_share", "pages",
            "body_words", "marker", "control", "hedge", "booster", "long_words", "ttr100",
            "n_figures", "n_tables", "n_equations", "ai_mention", "ai_disclosure"] + \
           [c for c in docs.columns if c.startswith("fam_")]
    rec = drawn.merge(docs[keep], on="doi", how="outer")
    rec["draw"] = rec.draw.fillna("browser_only")            # downloaded but not in a draw file
    rec["substitute"] = rec.substitute.fillna(False)
    rec["route"] = rec.route.fillna("none")
    rec["retrieval_error"] = np.where(rec.route == "none", rec.doi.map(err), None)
    rec["publisher_copy"] = rec.publisher_copy.fillna(False)
    rec.to_csv(os.path.join(OUT, "fulltext_sample.csv.gz"), index=False, compression="gzip")
    print(f"drawn articles {len(drawn):,}; record rows {len(rec):,}")
    print(rec.groupby(["route", "status"], dropna=False).size().to_string())


if __name__ == "__main__":
    main()
