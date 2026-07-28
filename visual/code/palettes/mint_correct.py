"""Mint Mambo — CORRECTED to the green+blue family (from the reference figure).

  mm-swatches.jpg     the corrected palette (qualitative + green/blue sequential + diverging)
  mm-versatility.jpg  green+blue Mint Mambo across UMAP / heatmap / box / volcano
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Ellipse, Rectangle

OUT = Path(__file__).resolve().parents[2] / "figures"
RNG = np.random.default_rng(7)

plt.rcParams.update({
    "font.size": 9.5, "axes.edgecolor": "#a9a9a9", "axes.linewidth": 0.7,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": "#5a5a5a", "ytick.color": "#5a5a5a",
    "axes.labelcolor": "#333", "axes.titlecolor": "#222", "figure.facecolor": "white",
})

# ---- CORRECT Mint Mambo: green + blue ----
MM_QUAL  = ["#3D9F3C", "#9ED17B", "#367DB0", "#9DC7DD"]                       # UMAP 4-class
MM_QUAL6 = ["#2E6F40", "#3D9F3C", "#9ED17B", "#367DB0", "#5385BD", "#9DC7DD"]  # extended 6
MM_GREEN = ["#F3FBF2", "#DDF3DE", "#C4E9CA", "#AADCA9", "#8BCF8B", "#519D78"]  # green sequential
MM_BLUE  = ["#DBF1FA", "#D8E5F7", "#BAD2E1", "#96C2D4", "#6CBAD8", "#367DB0"]  # blue sequential
MM_DIV   = ["#04579B", "#6CBAD8", "#F5FBF3", "#8BCF8B", "#2E6F40"]             # blue ↔ pale ↔ green


def cmap(cs):
    return LinearSegmentedColormap.from_list("c", cs, N=256)


def save(fig, name):
    fig.savefig(OUT / name, dpi=160, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {name}")


def conf_ellipse(ax, x, y, color, n_std=2.0):
    cov = np.cov(x, y); mean = (x.mean(), y.mean())
    vals, vecs = np.linalg.eigh(cov); order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    w, h = 2 * n_std * np.sqrt(vals)
    ax.add_patch(Ellipse(mean, w, h, angle=theta, facecolor=color, alpha=0.16,
                         edgecolor=color, lw=1.3, zorder=1))


# ---------------------------------------------------------------- swatches sheet
def strip(ax, cols, title):
    w = 1.0 / len(cols)
    for i, c in enumerate(cols):
        ax.add_patch(Rectangle((i * w, 0.28), w * 0.94, 0.5, facecolor=c, edgecolor="#ccc", lw=0.5))
        ax.text(i * w + w * 0.47, 0.16, c.lstrip("#"), ha="center", va="top", fontsize=6.5,
                family="monospace", color="#666")
    ax.text(0, 0.96, title, ha="left", va="top", fontsize=10, fontweight="bold")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")


fig = plt.figure(figsize=(10.5, 5.2), facecolor="white")
gs = fig.add_gridspec(4, 1, hspace=0.9, left=0.04, right=0.97, top=0.9, bottom=0.05)
strip(fig.add_subplot(gs[0]), MM_QUAL6, "Mint Mambo — QUALITATIVE (green + blue)")
strip(fig.add_subplot(gs[1]), MM_GREEN, "Sequential · green")
strip(fig.add_subplot(gs[2]), MM_BLUE, "Sequential · blue")
strip(fig.add_subplot(gs[3]), MM_DIV, "Diverging · blue ↔ pale ↔ green")
fig.suptitle("Mint Mambo (corrected) — the green + blue family", x=0.04, ha="left",
             fontsize=13, fontweight="bold")
save(fig, "mm-swatches.jpg")

# ---------------------------------------------------------------- versatility
fig = plt.figure(figsize=(12, 9.4))
gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.26, left=0.06, right=0.97, top=0.92, bottom=0.07)

# (a) gradient heatmap — values change REGULARLY along a gradient axis (meta-ROI style)
axa = fig.add_subplot(gs[0, 0])
rlabels = ["Reading", "Language", "Emotion", "Learning", "Object", "Social", "Reasoning", "Salience"]
nr, nc = len(rlabels), 6
grad = np.linspace(0.12, 0.92, nc)                       # transmodal → unimodal
row_mod = np.linspace(-0.10, 0.14, nr)                   # gentle per-row shift
gmat = np.clip(grad[None, :] + row_mod[:, None] + RNG.normal(0, 0.015, (nr, nc)), 0, 1)
MM_SEQ = ["#FFFFCC", "#C7E9B4", "#7FCDBB", "#41B6C4", "#1D91C0", "#225EA8", "#0C2C84"]
sns.heatmap(gmat, annot=True, fmt=".2f", annot_kws={"size": 5.5},
            cmap=cmap(MM_SEQ), vmin=0, vmax=1, linewidths=0.5, linecolor="white",
            cbar_kws=dict(shrink=0.7), yticklabels=rlabels,
            xticklabels=["Transmodal", "", "", "", "", "Unimodal"], ax=axa)
axa.tick_params(labelsize=6.5, length=0)
axa.set_title("(a) gradient heatmap (seaborn)", loc="left", fontsize=10.5)

# (b) UMAP scatter — green+blue qualitative (cell types, like the reference)
axb = fig.add_subplot(gs[0, 1])
cents = [(-3, 2), (3, 2.4), (-1.6, -2.6), (3.2, -1.6)]
for c, col, nm in zip(cents, MM_QUAL, ["Cardiomyocytes", "Fibroblasts", "Myeloid", "Endothelial"]):
    n = RNG.integers(420, 700)
    pts = RNG.normal(c, (1.25, 1.0), (n, 2))
    axb.scatter(pts[:, 0], pts[:, 1], s=6, color=col, alpha=0.6, edgecolor="none", label=nm)
axb.set_xlabel("UMAP_1"); axb.set_ylabel("UMAP_2"); axb.set_xticks([]); axb.set_yticks([])
axb.legend(frameon=False, fontsize=7.5, loc="upper left", markerscale=2.2,
           handletextpad=0.3, labelspacing=0.4)
axb.set_title("(b) cell-type UMAP", loc="left", fontsize=10.5)

# (c) grouped box — green+blue
axc = fig.add_subplot(gs[1, 0])
data = [RNG.normal(m, s, 120) for m, s in zip([6, 9, 7, 11, 8, 10], [1.4, 1.8, 1.2, 2.0, 1.5, 1.7])]
bp = axc.boxplot(data, patch_artist=True, widths=0.6, showfliers=False,
                 medianprops=dict(color="white", lw=1.3),
                 whiskerprops=dict(color="#888"), capprops=dict(color="#888"))
for patch, col in zip(bp["boxes"], MM_QUAL6):
    patch.set_facecolor(col); patch.set_alpha(0.8); patch.set_edgecolor(col)
axc.set_xticks(range(1, 7), [f"g{i}" for i in range(1, 7)])
axc.set_ylabel("Value"); axc.grid(axis="y", color="#eee"); axc.set_axisbelow(True)
axc.set_title("(c) grouped box", loc="left", fontsize=10.5)

# (d) volcano — original version: colour continuously by log2FC (green↔blue diverging)
axd = fig.add_subplot(gs[1, 1])
lfc = RNG.normal(0, 1.6, 1600)
p = 10 ** (-np.abs(RNG.normal(0, 1.4, 1600)) * (1 + np.abs(lfc) / 3))
sc = axd.scatter(lfc, -np.log10(p), c=lfc, cmap=cmap(MM_DIV), vmin=-4, vmax=4,
                 s=9, alpha=0.75, edgecolor="none")
axd.axvline(-1, ls="--", color="#ccc", lw=0.8); axd.axvline(1, ls="--", color="#ccc", lw=0.8)
axd.axhline(-np.log10(0.05), ls="--", color="#ccc", lw=0.8)
axd.set_xlabel("log$_2$FC"); axd.set_ylabel("-log$_{10}$ p")
axd.set_title("(d) volcano", loc="left", fontsize=10.5)

fig.suptitle("Mint Mambo (green + blue) — one palette across sophisticated charts",
             x=0.06, ha="left", fontsize=14, fontweight="bold")
save(fig, "mm-versatility.jpg")
print("done")
