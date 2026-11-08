"""
Step 5. Full-text results.

Before = published before 1 December 2022; after = from 1 March 2023 (the washout
months are dropped, as in the abstract analysis). Three estimates for every measure:

  sample      plain comparison of the parsed articles
  weighted    each article weighted by (corpus articles in its journal-year) /
              (parsed articles in that journal-year), so the comparison describes the
              literature rather than the allocation of the sample
  within      within-journal change: for journals with at least 5 articles on each
              side, the corpus-size-weighted average of journal-level differences,
              divided by the same average of journal-level 'before' values

Frequencies are pooled within the group (occurrences / words). 95% intervals come from
a bootstrap that resamples journals (1,000 draws, seed 71). Open-access substitutes
(draws v5 and v6) are flagged so every estimate can be recomputed without them.

Input: the full-text record written by s4c_fulltext_record.py (one row per drawn
article, with route, quality status, flags and measures). Only rows with status "ok"
enter; articles no longer in the corpus drop out at the merge. A third party can
rerun this step from the shipped record without any PDF.
"""
import os, sys, json
import settings as P
import numpy as np, pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = P.OUT
B = 1000

ft = pd.read_csv(os.path.join(OUT, "fulltext_sample.csv.gz"), low_memory=False)
ft = ft[ft.status == "ok"].copy()
# Only articles from one of the random draws enter; a few documents downloaded in a
# browser matched no draw file ("browser_only") and are kept in the record but not analyzed.
ft = ft[ft.draw != "browser_only"].copy()
for c in ("ai_mention", "ai_disclosure", "substitute", "publisher_copy"):
    ft[c] = ft[c].astype(str).str.lower().eq("true")
art = pd.read_parquet(os.path.join(OUT, "articles.parquet"),
                      columns=["doi", "source_id", "publisher", "year", "publication_date", "scholar_category"])
art = art[art.doi.notna()].assign(doi=lambda d: d.doi.str.lower()).drop_duplicates("doi")
ft = ft.merge(art, on="doi", how="inner")
ft["side"] = np.where(ft.publication_date < pd.Timestamp("2022-12-01"), "before",
              np.where(ft.publication_date >= pd.Timestamp("2023-03-01"), "after", "washout"))
ft = ft[ft.side != "washout"].copy()
cell_corpus = art.groupby(["source_id", "year"]).size()
cell_sample = ft.groupby(["source_id", "year"]).size()
ft["w"] = [cell_corpus[(s, y)] / cell_sample[(s, y)] for s, y in zip(ft.source_id, ft.year)]
ft["one"] = 1.0
ft.to_parquet(os.path.join(OUT, "fulltext_analysis.parquet"), index=False)

RATES = {"marker": ("Marker words per 10,000 words", "marker", "body_words", 1e4),
         "control": ("Control words per 10,000 words", "control", "body_words", 1e4),
         "hedge": ("Qualifying words per 10,000 words", "hedge", "body_words", 1e4),
         "booster": ("Emphatic words per 10,000 words", "booster", "body_words", 1e4),
         "long": ("Share of words with nine or more letters", "long_words", "body_words", 1)}
MEANS = {"ai_disclosure": "Discloses use of an AI writing tool (%)",
         "ai_mention": "Mentions a language model in the body (%)",
         "n_figures": "Figures per article", "n_tables": "Tables per article",
         "n_equations": "Numbered equations per article", "body_words": "Body length in words",
         "ttr100": "Distinct words per 100 words"}


def stat(d, key, weighted):
    w = d.w if weighted else d.one
    if key in RATES:
        _, num, den, sc = RATES[key]
        return sc * (w * d[num]).sum() / (w * d[den]).sum()
    v = d[key].astype(float)
    return (w * v).sum() / w.sum() * (100 if key.startswith("ai_") else 1)


def within(d, key):
    rows = []
    for sid, g in d.groupby("source_id"):
        a, b = g[g.side == "before"], g[g.side == "after"]
        if len(a) >= 5 and len(b) >= 5:
            rows.append((cell_corpus[sid].sum(), stat(a, key, False), stat(b, key, False)))
    if not rows:
        return np.nan, 0
    r = np.array(rows)
    diff = np.sum(r[:, 0] * (r[:, 2] - r[:, 1])) / np.sum(r[:, 0])
    if key.startswith("ai_"):                     # shares in percent: change in points
        return diff, len(rows)
    return 100 * diff / (np.sum(r[:, 0] * r[:, 1]) / np.sum(r[:, 0])), len(rows)


def estimates(d):
    out = {}
    for key in list(RATES) + list(MEANS):
        a, b = d[d.side == "before"], d[d.side == "after"]
        res = {}
        for wt in (False, True):
            x, y = stat(a, key, wt), stat(b, key, wt)
            res["w" if wt else "s"] = (x, y, (y - x) if key.startswith("ai_") else 100 * (y / x - 1))
        res["within"] = within(d, key)
        out[key] = res
    return out


main = estimates(ft)
rng = np.random.default_rng(71)
jids = ft.source_id.unique(); groups = {j: g for j, g in ft.groupby("source_id")}
boot = {k: [] for k in main}
for _ in range(B):
    pick = rng.choice(jids, len(jids), replace=True)
    d = pd.concat([groups[j] for j in pick])
    for k in main:
        a, b = d[d.side == "before"], d[d.side == "after"]
        x, y = stat(a, k, True), stat(b, k, True)
        boot[k].append((y - x) if k.startswith("ai_") else 100 * (y / x - 1))
rows = []
for k, r in main.items():
    lo, hi = np.percentile(boot[k], [2.5, 97.5])
    rows.append(dict(key=k, label=(RATES.get(k, (MEANS.get(k),))[0]),
                     before=r["w"][0], after=r["w"][1], change=r["w"][2], ci_lo=lo, ci_hi=hi,
                     sample_change=r["s"][2], within_change=r["within"][0], within_journals=r["within"][1]))
res = pd.DataFrame(rows)
def reweighted(d):
    """A subset re-weighted to the corpus by its own journal-year counts."""
    d = d.copy()
    n = d.groupby(["source_id", "year"]).size()
    d["w"] = [cell_corpus[(s, y)] / n[(s, y)] for s, y in zip(d.source_id, d.year)]
    return d


nos = estimates(reweighted(ft[~ft.substitute]))
res["no_substitutes_change"] = [nos[k]["w"][2] for k in res.key]
pub = estimates(reweighted(ft[ft.publisher_copy]))
res["publisher_copy_change"] = [pub[k]["w"][2] for k in res.key]
res.to_csv(os.path.join(OUT, "fulltext_results.csv"), index=False)
pd.set_option("display.width", 250)
print(res.round(3).to_string())
yr = ft.groupby("year").apply(lambda g: pd.Series({"n": len(g), "disclosure_pct": 100 * np.average(g.ai_disclosure, weights=g.w),
                                                    "mention_pct": 100 * np.average(g.ai_mention, weights=g.w),
                                                    "marker": 1e4 * (g.w * g.marker).sum() / (g.w * g.body_words).sum()}))
yr.to_csv(os.path.join(OUT, "fulltext_by_year.csv")); print(yr.round(2).to_string())
# Disclosure in 2025 by publisher (unweighted within publisher: each publisher's own articles).
pubd = (ft[ft.year == 2025].groupby("publisher")
        .agg(n=("doi", "size"), disclosing=("ai_disclosure", "sum"), disclosure_pct=("ai_disclosure", "mean"))
        .assign(disclosure_pct=lambda d: 100 * d.disclosure_pct).sort_values("n", ascending=False))
pubd.to_csv(os.path.join(OUT, "fulltext_disclosure_by_publisher_2025.csv")); print(pubd.head(10).round(2).to_string())
nonels = ft[(ft.year == 2025) & (ft.publisher != "Elsevier BV")]
print("2025 weighted disclosure without Elsevier:", round(100 * np.average(nonels.ai_disclosure, weights=nonels.w), 2))
info = dict(n=len(ft), before=int((ft.side == "before").sum()), after=int((ft.side == "after").sum()),
            journals=int(ft.source_id.nunique()), publishers=int(ft.publisher.nunique()),
            substitute_share=float(ft.substitute.mean()),
            publisher_copies=int(ft.publisher_copy.sum()),
            publisher_copies_before=int((ft.publisher_copy & (ft.side == "before")).sum()),
            from_memory=int((ft.route == "memory").sum()),
            from_browser=int((ft.route == "browser").sum()),
            excluded_by_quality=int(pd.read_csv(os.path.join(OUT, "fulltext_sample.csv.gz"), usecols=["status"])
                                    .status.isin(["title_mismatch", "supplementary", "unreadable_text"]).sum()),
            disclosure_before_2022_count=int(ft[(ft.year <= 2021)].ai_disclosure.sum()))
json.dump(info, open(os.path.join(OUT, "fulltext_info.json"), "w"), indent=1); print(info)

# ---------------- diagnostics: do the exclusions differ before and after? ----------------
rec = pd.read_csv(os.path.join(OUT, "fulltext_sample.csv.gz"), low_memory=False,
                  usecols=["doi", "route", "status", "title_overlap", "english_share"])
rec = rec.merge(art[["doi", "publication_date"]], on="doi", how="inner")
rec["side"] = np.where(rec.publication_date < pd.Timestamp("2022-12-01"), "before",
               np.where(rec.publication_date >= pd.Timestamp("2023-03-01"), "after", "washout"))
rec = rec[rec.side != "washout"]
got = rec[rec.route != "none"]
diag = (got.assign(excluded=got.status.isin(["title_mismatch", "supplementary", "unreadable_text",
                                              "not_article_or_short", "parse_timeout"]))
        .groupby("side").agg(obtained=("doi", "size"), excluded_share=("excluded", "mean")))
br = pd.read_parquet(os.path.join(OUT, "fulltext_bronze.parquet"), columns=["doi", "status"]) \
    if os.path.exists(os.path.join(OUT, "fulltext_bronze.parquet")) else pd.DataFrame(columns=["doi", "status"])
br = br.assign(doi=br.doi.str.lower()).merge(rec[["doi", "side"]], on="doi", how="inner")
diag["bronze_attempted"] = br.groupby("side").size()
diag["bronze_measured_share"] = br.assign(ok=br.status == "ok").groupby("side").ok.mean()
diag.to_csv(os.path.join(OUT, "fulltext_exclusions_by_side.csv"))
print(diag.round(4).to_string())

# Stricter quality thresholds (looser ones would need documents that were not measured).
sens = []
for t_title, t_eng in [(0.3, 0.5), (0.5, 0.5), (0.3, 0.7), (0.5, 0.7)]:
    keep = ft[(ft.title_overlap.isna() | (ft.title_overlap >= t_title)) & (ft.english_share >= t_eng)]
    e = estimates(reweighted(keep))
    sens.append({"title_min": t_title, "english_min": t_eng, "n": len(keep),
                 "marker_change": e["marker"]["w"][2], "control_change": e["control"]["w"][2]})
sens = pd.DataFrame(sens); sens.to_csv(os.path.join(OUT, "fulltext_threshold_sensitivity.csv"), index=False)
print(sens.round(2).to_string())
