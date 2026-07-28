"""Piece 10 rework — sophisticated charts matched to each palette's character.

  mm-versatility.jpg        Mint Mambo across 4 high-end charts (seaborn heatmap,
                            PCA + confidence ellipses, split box, volcano)
  demo-morandi-corr.jpg     Morandi -> mixed correlation matrix (Piece-3 style)
  demo-slateclay-pca.jpg    Slate & Clay -> PCA biplot with 95% ellipses
  demo-teal-ridge.jpg       Teal Tonal -> ridgeline (joyplot)
  demo-sage-hex.jpg         Sage -> hexbin density
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Ellipse
from scipy.stats import gaussian_kde

OUT = Path(__file__).resolve().parents[2] / "figures"
RNG = np.random.default_rng(7)

plt.rcParams.update({
    "font.size": 9.5, "axes.edgecolor": "#a9a9a9", "axes.linewidth": 0.7,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": "#5a5a5a", "ytick.color": "#5a5a5a",
    "axes.labelcolor": "#333", "axes.titlecolor": "#222", "figure.facecolor": "white",
})

MINT     = ["#AD8632", "#1E619C", "#63B6BF", "#77211D", "#A8D2C2", "#CBA5AE"]
MINT_DIV = ["#1E619C", "#8FB8CC", "#F3EEE0", "#D2B48C", "#AD8632"]
MORANDI  = ["#B4A79A", "#8C9AA0", "#A88B7D", "#9DA88B", "#B0929B", "#C7B7A3"]
MOR_DIV  = ["#8C9AA0", "#C2C4BC", "#F2ECE4", "#D8C3B0", "#A88B7D"]
SLATE    = ["#576B8E", "#7EA03C", "#B27276", "#7BA887", "#E5B327", "#741B2D"]
TEAL     = ["#EAF0F1", "#9BC0C7", "#6BA3AD", "#417D89", "#2A5A64"]
SAGE     = ["#F3FBF2", "#C4E9CA", "#8BCF8B", "#519D78", "#2E6F40"]


def cmap(cs):
    return LinearSegmentedColormap.from_list("c", cs, N=256)


def save(fig, name):
    fig.savefig(OUT / name, dpi=160, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {name}")


def conf_ellipse(ax, x, y, color, n_std=2.0):
    cov = np.cov(x, y); mean = (x.mean(), y.mean())
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]; vals, vecs = vals[order], vecs[:, order]
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    w, h = 2 * n_std * np.sqrt(vals)
    ax.add_patch(Ellipse(mean, w, h, angle=theta, facecolor=color, alpha=0.16,
                         edgecolor=color, lw=1.3, zorder=1))


# ---------------------------------------------------------------- 1. Mint Mambo versatility
fig = plt.figure(figsize=(12, 9.4))
gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.26, left=0.06, right=0.97, top=0.92, bottom=0.07)

# (a) seaborn heatmap — a gene x condition expression matrix, mint diverging
axa = fig.add_subplot(gs[0, 0])
mat = np.clip(RNG.normal(0, 1, (10, 8)) + np.sin(np.linspace(0, 3, 10))[:, None], -2.5, 2.5)
sns.heatmap(mat, cmap=cmap(MINT_DIV), center=0, ax=axa, cbar_kws=dict(shrink=0.7),
            xticklabels=False, yticklabels=False, linewidths=0.4, linecolor="white")
axa.set_title("(a) expression heatmap (seaborn)", loc="left", fontsize=10.5)

# (b) PCA scatter + 95% confidence ellipses
axb = fig.add_subplot(gs[0, 1])
cents = [(-3, 2), (3, 2.4), (-1.6, -2.6), (3.2, -1.6)]
for c, col, nm in zip(cents, MINT, ["Cardio", "Fibro", "Myeloid", "Endo"]):
    x = RNG.normal(c[0], 1.0, 150); y = RNG.normal(c[1], 0.8, 150)
    axb.scatter(x, y, s=10, color=col, alpha=0.6, edgecolor="none", label=nm, zorder=2)
    conf_ellipse(axb, x, y, col)
axb.set_xlabel("PC1 (34%)"); axb.set_ylabel("PC2 (19%)")
axb.legend(frameon=False, fontsize=8, loc="upper left")
axb.set_title("(b) PCA + 95% ellipses", loc="left", fontsize=10.5)

# (c) grouped box
axc = fig.add_subplot(gs[1, 0])
data = [RNG.normal(m, s, 120) for m, s in zip([6, 9, 7, 11, 8, 10], [1.4, 1.8, 1.2, 2.0, 1.5, 1.7])]
bp = axc.boxplot(data, patch_artist=True, widths=0.6, showfliers=False,
                 medianprops=dict(color="white", lw=1.3),
                 whiskerprops=dict(color="#888"), capprops=dict(color="#888"))
for patch, col in zip(bp["boxes"], MINT):
    patch.set_facecolor(col); patch.set_alpha(0.75); patch.set_edgecolor(col)
axc.set_xticks(range(1, 7), [f"g{i}" for i in range(1, 7)])
axc.set_ylabel("Value"); axc.grid(axis="y", color="#eee"); axc.set_axisbelow(True)
axc.set_title("(c) grouped box", loc="left", fontsize=10.5)

# (d) volcano
axd = fig.add_subplot(gs[1, 1])
lfc = RNG.normal(0, 1.6, 1600)
p = 10 ** (-np.abs(RNG.normal(0, 1.4, 1600)) * (1 + np.abs(lfc) / 3))
sc = axd.scatter(lfc, -np.log10(p), c=lfc, cmap=cmap(MINT_DIV), vmin=-4, vmax=4,
                 s=9, alpha=0.7, edgecolor="none")
axd.axvline(-1, ls="--", color="#ccc", lw=0.8); axd.axvline(1, ls="--", color="#ccc", lw=0.8)
axd.axhline(-np.log10(0.05), ls="--", color="#ccc", lw=0.8)
axd.set_xlabel("log$_2$FC"); axd.set_ylabel("-log$_{10}$ p")
axd.set_title("(d) volcano", loc="left", fontsize=10.5)

fig.suptitle("Mint Mambo — one palette across sophisticated chart types",
             x=0.06, ha="left", fontsize=14, fontweight="bold")
save(fig, "mm-versatility.jpg")

# ---------------------------------------------------------------- 2. Morandi mixed corr matrix
labels = ["NDVI", "Temp", "Build", "PM2.5", "Income", "Green"]
n = len(labels)
L = RNG.normal(size=(240, 3))
W = RNG.normal(size=(n, 3))
X = L @ W.T + RNG.normal(0, 0.6, (240, n))
corr = np.corrcoef(X.T)
fig = plt.figure(figsize=(7.6, 7.2))
gs = fig.add_gridspec(n, n, wspace=0.06, hspace=0.06, left=0.08, right=0.9, top=0.9, bottom=0.08)
norm = Normalize(-1, 1); dc = cmap(MOR_DIV)
for i in range(n):
    for j in range(n):
        ax = fig.add_subplot(gs[i, j])
        if i == j:
            xs = np.linspace(X[:, i].min(), X[:, i].max(), 60)
            ax.fill_between(xs, 0, gaussian_kde(X[:, i])(xs), color="#A88B7D", alpha=0.5)
        elif i > j:
            ax.scatter(X[:, j], X[:, i], s=5, color="#8C9AA0", alpha=0.5, edgecolor="none")
        else:
            ax.set_facecolor(dc(norm(corr[i, j])))
            ax.text(0.5, 0.5, f"{corr[i, j]:.2f}", ha="center", va="center",
                    transform=ax.transAxes, fontsize=8, color="#333")
        ax.set_xticks([]); ax.set_yticks([])
        if i == n - 1:
            ax.set_xlabel(labels[j], fontsize=8, rotation=40, ha="right")
        if j == 0:
            ax.set_ylabel(labels[i], fontsize=8, rotation=0, ha="right", va="center")
        for s in ax.spines.values():
            s.set_color("#d5d5d5"); s.set_linewidth(0.6)
fig.suptitle("Morandi — mixed correlation matrix", x=0.08, ha="left", fontsize=13, fontweight="bold")
save(fig, "demo-morandi-corr.jpg")

# ---------------------------------------------------------------- 3. Slate & Clay PCA biplot
fig, ax = plt.subplots(figsize=(7.0, 6.0))
cents = [(-3, 2), (3, 2.4), (-1.6, -2.6), (3.2, -1.6), (0, 4), (-4, 0)]
names = ["A", "B", "C", "D", "E", "F"]
for c, col, nm in zip(cents, SLATE, names):
    x = RNG.normal(c[0], 1.0, 120); y = RNG.normal(c[1], 0.8, 120)
    ax.scatter(x, y, s=12, color=col, alpha=0.6, edgecolor="none", label=nm, zorder=2)
    conf_ellipse(ax, x, y, col)
for (dx, dy, nm) in [(3.5, 1.5, "V1"), (-2.5, 3, "V2"), (1, -3.5, "V3")]:
    ax.annotate("", (dx, dy), (0, 0), arrowprops=dict(arrowstyle="->", color="#555", lw=1.2))
    ax.text(dx * 1.08, dy * 1.08, nm, fontsize=9, color="#555")
ax.set_xlabel("PC1 (41%)"); ax.set_ylabel("PC2 (23%)")
ax.legend(frameon=False, fontsize=8, ncol=6, loc="upper center", bbox_to_anchor=(0.5, 1.08))
ax.set_title("Slate & Clay — PCA biplot + 95% ellipses", loc="left", fontsize=11)
save(fig, "demo-slateclay-pca.jpg")

# ---------------------------------------------------------------- 4. Teal Tonal ridgeline
fig, ax = plt.subplots(figsize=(7.4, 5.2))
cats = ["2019", "2020", "2021", "2022", "2023"]
xs = np.linspace(-3, 9, 300)
for i, (col, nm) in enumerate(zip(TEAL, cats)):
    samp = RNG.normal(i * 0.85 + RNG.normal(0, 0.2), 0.9, 400)
    dens = gaussian_kde(samp)(xs); dens = dens / dens.max() * 1.7
    ax.fill_between(xs, i, i + dens, color=col, alpha=0.9, lw=1.0,
                    edgecolor="white", zorder=len(TEAL) - i)
ax.set_yticks(range(len(cats)), cats); ax.set_xlabel("Land-surface temperature (°C)")
ax.spines["left"].set_visible(False)
ax.set_title("Teal Tonal — ridgeline (year-on-year distributions)", loc="left", fontsize=11)
save(fig, "demo-teal-ridge.jpg")

# ---------------------------------------------------------------- 5. Sage hexbin
fig, ax = plt.subplots(figsize=(6.8, 5.6))
x = RNG.normal(0, 1, 6000); y = 0.6 * x + RNG.normal(0, 0.8, 6000)
hb = ax.hexbin(x, y, gridsize=32, cmap=cmap(SAGE), mincnt=1, linewidths=0.2)
cb = fig.colorbar(hb, ax=ax, fraction=0.045, pad=0.02); cb.set_label("count", fontsize=8)
ax.set_xlabel("Observed"); ax.set_ylabel("Predicted")
ax.set_title("Sage — hexbin density", loc="left", fontsize=11)
save(fig, "demo-sage-hex.jpg")
print("done")
