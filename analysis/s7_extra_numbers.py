"""
Step 7. Descriptive numbers quoted in the text, written to out/extra_numbers.json.

Every number in the article that is not in a results table comes from here or from the
files written by steps 2 to 5.
"""
import os, json
import settings as P
import numpy as np, pandas as pd, statsmodels.api as sm
import its_model as M

OUT = P.OUT
A = pd.read_parquet(os.path.join(OUT, "articles.parquet"))
A = A[A.year.between(2015, 2025)]
t = A[A.has_text]
core = A[A.cohort == "core"]
text = core[core.has_text & ~core.jan1]
out = {
    "s2_share_of_usable_abstracts": float((t.abstract_source == "semantic_scholar").mean()),
    "usable_abstracts": int(len(t)),
    "usable_share": float(A.has_text.mean()),
    "mean_abstract_words": float(t.n_words.mean()),
    "jan1_share_core": float(core.jan1.mean()),
    "elsevier_springer_rsc_share": float(A.publisher.isin(
        ["Elsevier BV", "Springer Science+Business Media", "Royal Society of Chemistry"]).mean()),
}
ft_path = os.path.join(OUT, "fulltext_analysis.parquet")
if os.path.exists(ft_path):
    out["mean_body_words"] = float(pd.read_parquet(ft_path, columns=["body_words"]).body_words.mean())


def pooled(d, num="marker"):
    g = d.groupby("month")[[num, "n_words"]].sum()
    return (1e4 * g[num] / g.n_words).rename("value").reset_index()


# Calendar-month profile before the release, without records dated only by year.
s = pooled(text)
pre = s[s.month <= M.T0].assign(mon=lambda d: d.month.dt.month).groupby("mon").value.mean()
out["calendar_profile_min"], out["calendar_profile_max"] = float(pre.min()), float(pre.max())
s_all = pooled(core[core.has_text])
jan = s_all[s_all.month <= M.T0]
out["january_excess_if_kept_pct"] = float(100 * (jan[jan.month.dt.month == 1].value.mean()
                                                 / jan[jan.month.dt.month != 1].value.mean() - 1))

# Projection fitted on pre-release months only.
p = s[s.month <= M.T0]
d = M.design(p.month); cols = ["t", "covid"] + M.SEAS
fit = sm.OLS(p.value.values, sm.add_constant(d[cols])).fit()
post = s[s.month >= pd.Timestamp("2023-03-01")]
cf = fit.predict(sm.add_constant(M.design(post.month)[cols], has_constant="add"))
out["pre_only_departure_pct"] = float(100 * (post.value.mean() / np.mean(cf) - 1))

# The second half of 2025.
m = s.set_index("month").value
out["peak_month"] = str(m.idxmax().date()); out["peak_value"] = float(m.max())
out["dec_2025_value"] = float(m.loc[pd.Timestamp("2025-12-01")])
h = text[text.year == 2025].assign(half=lambda d: np.where(d.month.dt.month <= 6, "H1", "H2"))
for half, g in h.groupby("half"):
    out[f"abstract_words_2025{half}"] = float(g.n_words.mean())
    out[f"s2_share_2025{half}"] = float((g.abstract_source == "semantic_scholar").mean())
    out[f"marker_rate_2025{half}"] = float(1e4 * g.marker.sum() / g.n_words.sum())
fams = [c for c in h.columns if c.startswith("fam_")]
g = h.groupby("half")[fams + ["n_words"]].sum()
rate = g[fams].div(g.n_words, axis=0)
chg = (100 * (rate.loc["H2"] / rate.loc["H1"] - 1)).round(1)
out["family_change_2025H1_H2_pct"] = {k[4:]: float(v) for k, v in chg.items()}
pj = h.groupby(["source_id", "half"]).apply(lambda d: d.marker.sum() / d.n_words.sum()).unstack().dropna()
out["journals_falling_2025H1_H2_share"] = float((pj["H2"] < pj["H1"]).mean())

# Publication volume, 2023-2025 against 2015-2022.
v = pd.read_csv(os.path.join(OUT, "cf_meta_n_articles.csv"), parse_dates=["month"])
out["volume_growth_pct"] = float(100 * (v[v.month >= pd.Timestamp("2023-01-01")].value.mean()
                                        / v[v.month < pd.Timestamp("2023-01-01")].value.mean() - 1))

json.dump(out, open(os.path.join(OUT, "extra_numbers.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
