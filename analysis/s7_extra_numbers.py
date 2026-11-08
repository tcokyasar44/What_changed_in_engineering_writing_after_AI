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

# References: the newest records are incomplete. Share with no reference list, and the
# references result with the series ended in August 2025.
cm = core[~core.jan1]
r25 = cm[cm.year == 2025].assign(mon=lambda d: d.month.dt.month)
zero = r25.groupby("mon").n_references.apply(lambda s: float((s.fillna(0) == 0).mean()))
out["refs_zero_share_2025_jan_aug_range"] = [float(zero.loc[1:8].min()), float(zero.loc[1:8].max())]
out["refs_zero_share_2025_dec"] = float(zero.loc[12])
withref = r25[r25.n_references.fillna(0) > 0].groupby("mon").n_references.mean()
out["refs_mean_with_list_2025_jan_aug_range"] = [float(withref.loc[1:8].min()), float(withref.loc[1:8].max())]
out["refs_mean_with_list_2025_dec"] = float(withref.loc[12])
rs = pd.read_csv(os.path.join(OUT, "cf_meta_n_references.csv"), parse_dates=["month"])[["month", "value"]]
full, _ = M.fit(rs)
cut, _ = M.fit(rs, cutoff=pd.Timestamp("2025-09-01"))
out["refs_departure_full_pct"], out["refs_trend_change_p_full"] = full["departure_pct"], full["slope_p"]
out["refs_departure_to_aug2025_pct"], out["refs_trend_change_p_to_aug2025"] = cut["departure_pct"], cut["slope_p"]

# Open access: change in the publisher mix and in recorded open-access status.
cm = cm.assign(publisher=cm.publisher.fillna("(no publisher recorded)"))
pub = pd.crosstab(cm.year, cm.publisher, normalize="index")
for name, key in [("Multidisciplinary Digital Publishing Institute", "mdpi"), ("Elsevier BV", "elsevier")]:
    out[f"{key}_share_2022"], out[f"{key}_share_2025"] = float(pub.loc[2022, name]), float(pub.loc[2025, name])
oa_pub = cm.pivot_table(index="year", columns="publisher", values="is_oa", aggfunc="mean")
out["mdpi_oa_share_2025"] = float(oa_pub.loc[2025, "Multidisciplinary Digital Publishing Institute"])
out["elsevier_oa_share_2025"] = float(oa_pub.loc[2025, "Elsevier BV"])
oa_y = cm.groupby("year").is_oa.mean()
out["oa_share_2022"], out["oa_share_2023"], out["oa_share_2025"] = (float(oa_y.loc[2022]), float(oa_y.loc[2023]),
                                                                  float(oa_y.loc[2025]))
# 2025 open-access share if each publisher kept its 2022 share of articles
w22 = pub.loc[2022]; oa25 = oa_pub.loc[2025].reindex(w22.index)
ok = oa25.notna()
out["oa_share_2025_at_2022_mix"] = float((w22[ok] * oa25[ok]).sum() / w22[ok].sum())

json.dump(out, open(os.path.join(OUT, "extra_numbers.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
