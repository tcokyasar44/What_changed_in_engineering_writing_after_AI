"""
Step 6. Every figure in the paper, from the clean outputs only.

Conventions requested by the author: no plot titles; every text element the same
size (16 pt) except word labels on the vocabulary scatter (11 pt); 'words', never
'tokens'; x axis 'Time of publication'; Times New Roman. Monthly series exclude
records dated 1 January. Lines are drawn without the calendar-month component, so
the projection is a smooth trend rather than a zigzag; there is no fitted
post-release line, only the observed series and the projection.
"""
import os, json
import settings as P
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from adjustText import adjust_text

HERE = os.path.dirname(os.path.abspath(__file__)); OUT = P.OUT
FIG = P.FIGURES
FS = 16
plt.rcParams.update({"font.family": "Times New Roman", "font.size": FS, "axes.labelsize": FS,
                     "xtick.labelsize": FS, "ytick.labelsize": FS, "legend.fontsize": FS,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42})  # vector PDF with embedded fonts (Journal of Informetrics)
T0 = pd.Timestamp("2022-11-01")


def read_cf(name):
    d = pd.read_csv(os.path.join(OUT, name), parse_dates=["month"])
    for c in ("cf", "cf_lo", "cf_hi"):
        d[c + "_s"] = d[c] - d["seasonal"]
    return d


def marker_axes(ax, d, ylabel, legend=True, obs_label="Observed"):
    ax.plot(d.month, d.value, color="black", lw=1.6, label=obs_label)
    ax.plot(d.month, d.cf_s, color="0.35", ls=":", lw=2.2, label="Projection of pre-2023 trend")
    ax.fill_between(d.month, d.cf_lo_s, d.cf_hi_s, color="0.8", alpha=.6, lw=0, label="95% prediction interval")
    ax.axvline(pd.Timestamp("2022-11-30"), color="black", ls="-.", lw=1.3,
               label="Public release (November 30, 2022)")
    ax.axvspan(pd.Timestamp("2022-12-01"), pd.Timestamp("2023-02-28"), facecolor="none", edgecolor="0.45",
               hatch="////", lw=0, label="Months left out of the fit (Dec. 2022 to Feb. 2023)")
    ax.set_xlabel("Time of publication"); ax.set_ylabel(ylabel)
    if legend:
        ax.legend(frameon=False, loc="upper left")


# Figure: monthly marker series
d = read_cf("cf_text_marker.csv")
fig, ax = plt.subplots(figsize=(10, 5.6))
marker_axes(ax, d, "Marker words per 10,000 words")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_marker_words.pdf"), dpi=300); plt.close(fig)

# Figure: full vocabulary
ex = pd.read_csv(os.path.join(OUT, "excess_vocabulary.csv"))
ex = ex[ex.ratio.notna()]
bg = ex[ex.kind == "other"]; mk = ex[ex.kind == "marker"]; ct = ex[ex.kind == "control"]
fig, ax = plt.subplots(figsize=(12, 8))
ax.scatter(bg.cf, bg.ratio, s=4, color="0.75", lw=0, label="All other words", rasterized=True)
ax.scatter(mk.cf, mk.ratio, s=50, facecolor="none", edgecolor="black", marker="^", lw=1.2, label="Marker words")
ax.scatter(ct.cf, ct.ratio, s=50, color="black", marker="s", label="Control words")
ax.axhline(1, color="0.3", ls="--", lw=1)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("Projected frequency, 2023 to 2025 (per million words)")
ax.set_ylabel("Observed frequency / projection")
texts = [ax.text(r.cf, r.ratio, r.label, fontsize=11) for r in pd.concat([mk, ct]).itertuples()]
adjust_text(texts, ax=ax, expand=(1.3, 1.6), arrowprops=dict(arrowstyle="-", color="0.4", lw=0.6))
ax.legend(frameon=False, loc="upper right")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_excess_vocabulary.pdf"), dpi=300); plt.close(fig)

# Figure: categories
cat = pd.read_csv(os.path.join(OUT, "by_category.csv")).sort_values("departure_pct")
pooled = pd.read_csv(os.path.join(OUT, "its_text.csv")).set_index("key").loc["marker", "departure_pct"]
fig, ax = plt.subplots(figsize=(10, 7))
ax.barh(cat.category.str.replace("&", "and"), cat.departure_pct, color="0.55")
ax.axvline(pooled, color="black", ls="--", lw=1.2)
ax.set_xlabel("Departure from projection (%)")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_by_category.pdf"), dpi=300); plt.close(fig)

# Figure: style panels
panels = [("hedge", "Qualifying words\nper 10,000 words"), ("booster", "Emphatic words\nper 10,000 words"),
          ("long", "Share of words with\nnine or more letters"), ("ttr", "Distinct words\nper 100 words"),
          ("fk", "Flesch-Kincaid grade"), ("len", "Abstract length\n(words)")]
fig, axs = plt.subplots(3, 2, figsize=(12, 12), sharex=True)
for ax, (k, lab) in zip(axs.flat, panels):
    marker_axes(ax, read_cf(f"cf_text_{k}.csv"), lab, legend=False)
    ax.set_xlabel("")
for ax in axs[-1]:
    ax.set_xlabel("Time of publication")
h, l = axs[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=2, frameon=False)
fig.tight_layout(rect=(0, 0.07, 1, 1)); fig.savefig(os.path.join(FIG, "fig_style_panels.pdf"), dpi=300); plt.close(fig)

# Figure: bibliographic panels
panels = [("n_articles", "Articles per month"), ("n_authors", "Authors per article"),
          ("n_references", "References per article"), ("n_pages", "Pages per article"),
          ("colon", "Share of titles\nwith a colon"), ("oa", "Open access share")]
fig, axs = plt.subplots(3, 2, figsize=(12, 12), sharex=True)
for ax, (k, lab) in zip(axs.flat, panels):
    marker_axes(ax, read_cf(f"cf_meta_{k}.csv"), lab, legend=False)
    ax.set_xlabel("")
for ax in axs[-1]:
    ax.set_xlabel("Time of publication")
h, l = axs[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=2, frameon=False)
fig.tight_layout(rect=(0, 0.07, 1, 1)); fig.savefig(os.path.join(FIG, "fig_metadata_panels.pdf"), dpi=300); plt.close(fig)

# Figure: corresponding-author country group (diagnostic only, written to out/)
fig, ax = plt.subplots(figsize=(10, 5.6))
for g, ls, lab in [("anglophone", "-", "English-dominant country"), ("other", "--", "Other country")]:
    dd = pd.read_csv(os.path.join(OUT, f"cf_lang_{g}.csv"), parse_dates=["month"])
    ax.plot(dd.month, dd.value, color="black", ls=ls, lw=1.6, label=lab)
ax.axvline(pd.Timestamp("2022-11-30"), color="0.3", ls="--", lw=1)
ax.set_xlabel("Time of publication"); ax.set_ylabel("Marker words per 10,000 words")
ax.legend(frameon=False, loc="upper left")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_langgroup.pdf"), dpi=300); plt.close(fig)
print("figures written to", os.path.abspath(FIG))
