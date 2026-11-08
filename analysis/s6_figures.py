"""
Step 6. Every figure in the paper, from the clean outputs only.

Conventions requested by the author: no plot titles; every text element the same
size (16 pt), except the vocabulary scatter, which is printed at about half size and so uses 20 pt
(17 pt for its word labels); 'words', never
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
FS2 = 20                         # printed at about half size: axis text about 10 pt, labels about 9 pt
fig, ax = plt.subplots(figsize=(12, 10))   # printed at 0.95 of the text width
ax.scatter(bg.cf, bg.ratio, s=6, color="0.75", lw=0, label="All other words", rasterized=True)
sm = ax.scatter(mk.cf, mk.ratio, s=140, facecolor="none", edgecolor="black", marker="^", lw=1.8,
                label="Marker words")
sc = ax.scatter(ct.cf, ct.ratio, s=140, color="black", marker="s", label="Control words")
ax.axhline(1, color="0.3", ls="--", lw=1.5)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("Projected frequency, 2023 to 2025 (per million words)", fontsize=FS2)
ax.set_ylabel("Observed frequency / projection", fontsize=FS2)
ax.tick_params(labelsize=FS2)
ax.set_ylim(max(ex.ratio.min() * 0.9, 0.2), ex.ratio.max() * 1.6)
# legend below the axes, so it cannot collide with the word labels; lay out the figure
# before placing labels, because the label solver works in screen coordinates
leg = fig.legend(loc="lower center", ncol=3, frameon=False, fontsize=FS2, handletextpad=0.2,
                 columnspacing=1.0)
leg.legend_handles[0].set_sizes([80])            # make the grey dot visible in the legend
fig.tight_layout(rect=(0, 0.07, 1, 1))
# Word labels are placed by hand (log10 x, log10 y of the label), because at this text
# size an automatic solver leaves overlaps. Words on the right are listed in a column in
# the empty region past the data, in the order of their points.
right = ["seamless", "additionally", "leverage", "crucial", "innovative", "landscape",
         "comprehensive", "significant", "robust", "furthermore", "consequently",
         "state-of-the-art", "moreover"]
POS = {w: (3.15, 0.86 - 0.112 * i) for i, w in enumerate(right)}
POS.update({
    "underscore": (0.78, 1.40), "intricate": (0.80, 1.06), "pivotal": (1.45, 0.94),
    "align": (1.22, 0.81), "showcase": (1.55, 0.69), "foster": (1.33, 0.58),
    "notably": (1.85, 0.56), "meticulous": (-0.12, 0.96), "realm": (-0.12, 0.84),
    "nuanced": (-0.12, 0.72), "multifaceted": (-0.12, 0.33), "elevate": (-0.12, 0.06),
    "noteworthy": (-0.12, -0.24), "cutting-edge": (-0.12, -0.36), "unveil": (-0.12, -0.48),
    "harness": (-0.12, -0.60), "holistic": (0.95, -0.52), "paradigm": (1.62, -0.52),
    "vital": (2.38, -0.52)})
lab = pd.concat([mk, ct])
for r in lab.itertuples():
    lx, ly = POS.get(r.label, (np.log10(r.cf) + 0.08, np.log10(r.ratio)))
    px, py = np.log10(r.cf), np.log10(r.ratio)
    # attach the leader to the side of the label that faces the point, so it never
    # runs through the text
    if ly < py - 0.3 and lx < px:
        rel = (0.5, 1.0) if px - lx < 0.6 else (1.0, 0.5)
    elif lx + 0.3 < px:
        rel = (1.0, 0.5)
    else:
        rel = (0.0, 0.5)
    ax.annotate(r.label, xy=(r.cf, r.ratio), xytext=(10 ** lx, 10 ** ly), fontsize=17,
                ha="left", va="center",
                arrowprops=dict(arrowstyle="-", color="0.45", lw=0.8, shrinkA=0, shrinkB=6,
                                relpos=rel))
ax.set_xlim(10 ** -0.15, 10 ** 5)
fig.savefig(os.path.join(FIG, "fig_excess_vocabulary.pdf"), dpi=300); plt.close(fig)

# Figure: categories
cat = pd.read_csv(os.path.join(OUT, "by_category.csv")).sort_values("departure_pct")
pooled = pd.read_csv(os.path.join(OUT, "its_text.csv")).set_index("key").loc["marker", "departure_pct"]
fig, ax = plt.subplots(figsize=(10, 7))
ax.barh(cat.category.str.replace("&", "and"), cat.departure_pct, color="0.55")
ax.axvline(pooled, color="black", ls="--", lw=1.2)
ax.set_xlabel("Departure from projection (%)")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_by_category.pdf"), dpi=300); plt.close(fig)

# Figures: style and bibliographic measures, one file per panel so the article can
# place them as subfigures (a) to (f) and refer to each; one shared legend file.
FSP = 18                          # panels print at about half the text width
PANEL_RC = {"font.size": FSP, "axes.labelsize": FSP, "xtick.labelsize": FSP,
            "ytick.labelsize": FSP, "legend.fontsize": FSP}


def panel(prefix, key, ylabel):
    # Every panel gets the same page size and the same plotting box, and its y-axis
    # title sits at the same distance from the axis, so panels placed side by side or
    # one above the other line up whatever the width of their tick labels.
    with plt.rc_context(PANEL_RC):
        fig = plt.figure(figsize=(6, 4.4))
        ax = fig.add_axes([0.26, 0.18, 0.71, 0.78])
        marker_axes(ax, read_cf(f"cf_{prefix}_{key}.csv"), ylabel, legend=False)
        ax.yaxis.set_label_coords(-0.25, 0.5)
        fig.savefig(os.path.join(FIG, f"fig_{prefix}_{key}.pdf"), dpi=300)
        plt.close(fig)


for key, lab in [("hedge", "Qualifying words\nper 10,000 words"), ("booster", "Emphatic words\nper 10,000 words"),
                 ("long", "Share of words with\nnine or more letters"), ("ttr", "Distinct words\nper 100 words"),
                 ("fk", "Flesch-Kincaid grade"), ("len", "Abstract length\n(words)")]:
    panel("text", key, lab)
for key, lab in [("n_articles", "Articles per month"), ("n_authors", "Authors per article"),
                 ("n_references", "References per article"), ("n_pages", "Pages per article"),
                 ("colon", "Share of titles\nwith a colon"), ("oa", "Open access share")]:
    panel("meta", key, lab)

# Shared legend for the panel figures
with plt.rc_context(PANEL_RC):
    fig, ax = plt.subplots(figsize=(12, 1.2))
    marker_axes(ax, read_cf("cf_text_hedge.csv"), "", legend=False)
    h, l = ax.get_legend_handles_labels()
    fig.clf()
    fig.legend(h, l, loc="center", ncol=3, frameon=False)
    fig.savefig(os.path.join(FIG, "fig_panels_legend.pdf"), dpi=300, bbox_inches="tight")
    plt.close(fig)

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
