"""
The one segmented-regression estimator used for every series in the paper.

  y_t = b0 + b1 t + b2 D_t + b3 (t - T0) D_t + b4 C_t + sum_m g_m S_mt + e_t

t counts months from January 2015; T0 = November 2022; D_t = 1 after T0; C_t = 1 from
March 2020 to June 2021; S_mt calendar-month indicators (February..December).
December 2022 to February 2023 are excluded from estimation (washout).
HAC (Newey-West) standard errors, 12 lags.

The counterfactual is the fitted model with b2 = b3 = 0. The reported departure is
mean(observed, post) / mean(counterfactual, post) - 1, in percent, over March 2023
to December 2025.
"""
import numpy as np, pandas as pd, statsmodels.api as sm

T0 = pd.Timestamp("2022-11-01")
WASH = (pd.Timestamp("2022-12-01"), pd.Timestamp("2023-02-01"))
START = pd.Timestamp("2015-01-01")
COVID = (pd.Timestamp("2020-03-01"), pd.Timestamp("2021-06-01"))
SEAS = [f"m{m}" for m in range(2, 13)]


def design(months, t0=T0, start=START):
    d = pd.DataFrame({"month": pd.to_datetime(months)})
    d["t"] = (d.month.dt.year - start.year) * 12 + d.month.dt.month - start.month
    it = (t0.year - start.year) * 12 + t0.month - start.month
    d["post"] = (d.month > t0).astype(float)
    d["t_post"] = np.where(d.post == 1, d.t - it, 0.0)
    d["covid"] = d.month.between(*COVID).astype(float)
    for m in range(2, 13):
        d[f"m{m}"] = (d.month.dt.month == m).astype(float)
    return d


COLS = ["t", "post", "t_post", "covid"] + SEAS


def fit(series, t0=T0, cutoff=None, washout=WASH, level=True):
    """series: DataFrame with columns month, value. Returns dict + frame."""
    s = series.dropna(subset=["value"]).copy()
    s["month"] = pd.to_datetime(s["month"])
    if cutoff is not None:
        s = s[s.month < cutoff]
    if washout is not None:
        s = s[~s.month.between(*washout)]
    s = s.sort_values("month").reset_index(drop=True)
    d = design(s.month, t0)
    cols = COLS if level else [c for c in COLS if c != "post"]
    X = sm.add_constant(d[cols], has_constant="add")
    y = s["value"].astype(float).values
    m = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 12})
    dcf = d.copy(); dcf["post"] = 0.0; dcf["t_post"] = 0.0
    Xcf = sm.add_constant(dcf[cols], has_constant="add")
    pr = m.get_prediction(Xcf).summary_frame(alpha=0.05)
    out = s[["month", "value"]].copy()
    out["cf"] = pr["mean"].values
    out["cf_lo"] = pr["obs_ci_lower"].values
    out["cf_hi"] = pr["obs_ci_upper"].values
    out["fitted"] = m.predict(X)
    # seasonal component, so lines can be drawn without the month-to-month zigzag
    seas = (d[SEAS].values * m.params[SEAS].values).sum(axis=1)
    out["seasonal"] = seas - seas.mean()
    post = out[out.month > t0]
    obs, cf = post.value.mean(), post.cf.mean()
    return dict(
        n_months=len(out), pre_mean=out[out.month <= t0].value.mean(),
        post_obs=obs, post_cf=cf, departure_pct=100 * (obs / cf - 1),
        level=m.params.get("post", np.nan), level_p=m.pvalues.get("post", np.nan),
        slope=m.params["t_post"], slope_p=m.pvalues["t_post"],
        pre_slope=m.params["t"], seasonal_range=float(np.ptp(m.params[SEAS].tolist() + [0.0])),
    ), out
