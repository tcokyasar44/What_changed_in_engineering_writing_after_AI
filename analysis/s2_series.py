"""
Step 2. Monthly series, segmented regressions, robustness, distribution.

Sample rules (applied everywhere unless a robustness variant says otherwise):
  * journals of the core cohort (published in every year 2015-2025)
  * articles dated 1 January are excluded from monthly series: for most of them
    the month is unknown (the record carries a year only), and they pile into
    January. They are kept in annual totals.
  * text series use articles with an abstract of at least 40 words.
Frequencies are pooled: 10,000 x total occurrences / total words in the month.
"""
import os, sys, json
import settings as P
import numpy as np, pandas as pd
import its_model as M
import lexicon as L

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = P.OUT
A = pd.read_parquet(os.path.join(OUT, "articles.parquet"))
A = A[A.year.between(2015, 2025)]
core = A[A.cohort == "core"]
R = {}


def pooled(df, num, den="n_words", scale=1e4, by="month"):
    g = df.groupby(by)[[num, den]].sum()
    return (scale * g[num] / g[den]).rename("value").reset_index()


def mean_series(df, col, by="month"):
    return df.groupby(by)[col].mean().rename("value").reset_index()


def fixed_weight(df, num, den="n_words"):
    """Journal-balanced rate: each journal keeps the same weight every month."""
    g = df.groupby(["source_id", "month"])[[num, den]].sum().reset_index()
    g["rate"] = 1e4 * g[num] / g[den]
    w = df.groupby("source_id")[den].sum(); w = w / w.sum()
    g["w"] = g.source_id.map(w)
    s = g.groupby("month").apply(lambda x: np.average(x.rate, weights=x.w))
    return s.rename("value").reset_index()


text = core[core.has_text & ~core.jan1]
text_all = core[core.has_text]
meta = core[~core.jan1]

# ---------------- text measures ----------------
TEXT = {
    "marker": ("Marker words per 10,000 words", lambda d: pooled(d, "marker")),
    "control": ("Control words per 10,000 words", lambda d: pooled(d, "control")),
    "hedge": ("Qualifying words per 10,000 words", lambda d: pooled(d, "hedge")),
    "booster": ("Emphatic words per 10,000 words", lambda d: pooled(d, "booster")),
    "long": ("Share of words with nine or more letters", lambda d: pooled(d, "long_words", scale=1)),
    "ttr": ("Distinct words per 100 words", lambda d: mean_series(d, "ttr100")),
    "fk": ("Flesch-Kincaid grade", None),
    "len": ("Abstract length in words", lambda d: mean_series(d, "n_words")),
}


def fk_series(d):
    g = d.groupby("month")[["syllables", "n_words"]].sum()
    v = 0.39 * 22 + 11.8 * g.syllables / g.n_words - 15.59
    return v.rename("value").reset_index()


TEXT["fk"] = (TEXT["fk"][0], fk_series)
rows = []
for k, (lab, fn) in TEXT.items():
    res, fr = M.fit(fn(text))
    fr.to_csv(os.path.join(OUT, f"cf_text_{k}.csv"), index=False)
    rows.append({"key": k, "label": lab, **res})
its_text = pd.DataFrame(rows); its_text.to_csv(os.path.join(OUT, "its_text.csv"), index=False)
print(its_text[["key", "pre_mean", "departure_pct", "level_p", "slope_p", "seasonal_range"]].round(4).to_string())

# ---------------- bibliographic measures ----------------
meta = meta.assign(one=1.0, n_pages=meta.n_pages.where(meta.n_pages.between(1, 200)))
META = {
    "n_articles": ("Articles published per month", lambda d: d.groupby("month").size().rename("value").reset_index()),
    "n_authors": ("Authors per article", lambda d: mean_series(d, "n_authors")),
    "n_institutions": ("Institutions per article", lambda d: mean_series(d, "n_institutions")),
    "n_references": ("References per article", lambda d: mean_series(d, "n_references")),
    "n_pages": ("Pages per article", lambda d: mean_series(d, "n_pages")),
    "title_words": ("Title length in words", lambda d: mean_series(d, "title_words")),
    "colon": ("Share of titles with a colon", lambda d: mean_series(d.assign(c=d.title_has_colon.astype(float)), "c")),
    "oa": ("Open access share", lambda d: mean_series(d.assign(c=d.is_oa.astype(float)), "c")),
}
rows = []
for k, (lab, fn) in META.items():
    res, fr = M.fit(fn(meta))
    fr.to_csv(os.path.join(OUT, f"cf_meta_{k}.csv"), index=False)
    rows.append({"key": k, "label": lab, **res})
its_meta = pd.DataFrame(rows); its_meta.to_csv(os.path.join(OUT, "its_meta.csv"), index=False)
print(its_meta[["key", "pre_mean", "departure_pct", "level_p", "slope_p"]].round(4).to_string())

# ---------------- annual table for the marker measure ----------------
mk_res, mk = M.fit(pooled(text, "marker"))
mk["year"] = mk.month.dt.year
annual = mk.groupby("year")[["value", "cf"]].mean()
annual["ratio"] = annual.value / annual.cf
annual.round(3).to_csv(os.path.join(OUT, "annual_marker.csv"))
print(annual.round(2).to_string())

# ---------------- robustness ----------------
rob = []
def add(name, variant, res):
    rob.append({"check": name, "variant": variant, **res})

for y in [2018, 2019, 2020]:
    t0 = pd.Timestamp(f"{y}-11-01")
    res, _ = M.fit(pooled(text, "marker"), t0=t0, cutoff=M.T0,
                   washout=(t0 + pd.DateOffset(months=1), t0 + pd.DateOffset(months=3)))
    add("placebo", f"November {y}", res)
add("placebo", "November 2022 (actual)", mk_res)
add("sample", "Journal-balanced weights", M.fit(fixed_weight(text, "marker"))[0])
add("sample", "Including records dated 1 January", M.fit(pooled(text_all, "marker"))[0])
add("sample", "Journals with near-complete abstracts (tier A)", M.fit(pooled(text[text.text_tier == "A"], "marker"))[0])
add("sample", "OpenAlex abstracts only", M.fit(pooled(text[text.abstract_source == "openalex"], "marker"))[0])
restored = {k for k, v in json.load(open(os.path.join(P.ROOT, "data", "harvest_manifest.json"),
                                       encoding="utf-8"))["done"].items() if v.get("abstracts_backfilled_at")}
add("sample", f"Excluding the {len(restored & set(text.source_id))} journals whose abstracts were restored",
    M.fit(pooled(text[~text.source_id.isin(restored)], "marker"))[0])
add("sample", f"All {A.source_id.nunique()} journals", M.fit(pooled(A[A.has_text & ~A.jan1], "marker"))[0])
add("model", "No level-change term", M.fit(pooled(text, "marker"), level=False)[0])
add("words", "Control words only", M.fit(pooled(text, "control"))[0])
robust = pd.DataFrame(rob); robust.to_csv(os.path.join(OUT, "robustness.csv"), index=False)
print(robust[["check", "variant", "departure_pct", "slope_p", "level_p"]].round(4).to_string())

loo = []
for sid, g in text.groupby("source_id"):
    if g.n_words.sum() < 1:
        continue
    res, _ = M.fit(pooled(text[text.source_id != sid], "marker"))
    loo.append({"source_id": sid, **res})
loo = pd.DataFrame(loo); loo.to_csv(os.path.join(OUT, "loo.csv"), index=False)
print("LOO journals", len(loo), loo.departure_pct.min(), loo.departure_pct.max(), (loo.slope_p < 0.05).all())

cat = []
for c, g in text.groupby("scholar_category"):
    res, _ = M.fit(pooled(g, "marker"))
    cat.append({"category": c, "journals": g.source_id.nunique(), "articles": len(g), **res})
cat = pd.DataFrame(cat).sort_values("departure_pct"); cat.to_csv(os.path.join(OUT, "by_category.csv"), index=False)
print(cat[["category", "journals", "departure_pct", "slope_p"]].round(4).to_string())

lang = []
for c, g in text.groupby("corr_lang_group"):
    res, fr = M.fit(pooled(g, "marker"))
    fr.to_csv(os.path.join(OUT, f"cf_lang_{c}.csv"), index=False)
    lang.append({"group": c, "articles": len(g), **res})
pd.DataFrame(lang).to_csv(os.path.join(OUT, "by_langgroup.csv"), index=False)
print(pd.DataFrame(lang)[["group", "articles", "departure_pct"]].to_string())

# ---------------- per-article distribution, all core journals, all years ----------------
t2 = text_all.assign(rate=1e4 * text_all.marker / text_all.n_words)
pre = t2[t2.publication_date < pd.Timestamp("2022-12-01")].rate
post = t2[t2.publication_date >= pd.Timestamp("2023-03-01")].rate
dist = pd.DataFrame({
    "stat": ["p25", "median", "p75", "p90", "p99", "mean", "share_zero", "n"],
    "before": [pre.quantile(.25), pre.median(), pre.quantile(.75), pre.quantile(.9), pre.quantile(.99), pre.mean(), (pre == 0).mean(), len(pre)],
    "after": [post.quantile(.25), post.median(), post.quantile(.75), post.quantile(.9), post.quantile(.99), post.mean(), (post == 0).mean(), len(post)],
})
dist.to_csv(os.path.join(OUT, "distribution.csv"), index=False)
print(dist.round(3).to_string())

# ---------------- descriptives ----------------
desc = {
    "articles_all": int(len(A)), "journals_all": int(A.source_id.nunique()),
    "articles_core": int(len(core)), "journals_core": int(core.source_id.nunique()),
    "with_text_core": int(core.has_text.sum()), "text_share_all": float(A.has_text.mean()),
    "jan1_share_core": float(core.jan1.mean()),
    "text_months_articles": int(len(text)),
}
by_pub = A.groupby("publisher").agg(articles=("openalex_id", "size"), with_text=("has_text", "mean")).sort_values("articles", ascending=False)
by_pub.to_csv(os.path.join(OUT, "coverage_by_publisher.csv"))
cov_year = A.groupby("year").has_text.mean(); cov_year.to_csv(os.path.join(OUT, "coverage_by_year.csv"))
json.dump(desc, open(os.path.join(OUT, "descriptives.json"), "w"), indent=1)
print(desc); print(by_pub.head(10).round(3).to_string()); print(cov_year.round(3).to_string())
