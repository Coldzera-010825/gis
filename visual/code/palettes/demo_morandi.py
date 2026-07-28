"""Morandi palette — its signature chart: a mixed correlation matrix (Piece-3 style).

Structured data (latent factors) so correlations show real patterns. Lower triangle
scatter + diagonal density are coloured per-variable with the full Morandi palette;
upper triangle r-cells use a muted Morandi diverging ramp.

  demo-morandi.jpg
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize
from scipy.stats import gaussian_kde

OUT = Path(__file__).resolve().parents[2] / "figures"
RNG = np.random.default_rng(20260714)

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": "#c9c9c9", "axes.linewidth": 0.6,
    "figure.facecolor": "white",
})

MORANDI = ["#B4A79A", "#8C9AA0", "#A88B7D", "#9DA88B", "#B0929B", "#C7B7A3"]
MOR_DIV = ["#8C9AA0", "#C2C4BC", "#F2ECE4", "#D8C3B0", "#A88B7D"]   # blue-grey ↔ cream ↔ clay
labels = ["NDVI", "Temp", "Build", "PM2.5", "Income", "Green"]
n = len(labels)


def cmap(cs):
    return LinearSegmentedColormap.from_list("c", cs, N=256)


# ---- structured data: 3 latent factors -> real, patterned correlations
Lf = RNG.normal(size=(260, 3))
Wv = np.array([
    [ 0.9, -0.1,  0.2],   # NDVI
    [-0.8,  0.3,  0.1],   # Temp   (anti-correlated with NDVI)
    [-0.5,  0.7,  0.0],   # Build
    [-0.3,  0.8,  0.1],   # PM2.5  (tracks Build)
    [ 0.2,  0.1,  0.9],   # Income
    [ 0.8, -0.2,  0.3],   # Green  (tracks NDVI)
])
X = Lf @ Wv.T + RNG.normal(0, 0.45, (260, n))
corr = np.corrcoef(X.T)

fig = plt.figure(figsize=(8.2, 7.8))
gs = fig.add_gridspec(n, n, wspace=0.08, hspace=0.08,
                      left=0.09, right=0.9, top=0.9, bottom=0.09)
norm = Normalize(-1, 1); dc = cmap(MOR_DIV)

for i in range(n):
    for j in range(n):
        ax = fig.add_subplot(gs[i, j])
        if i == j:
            xs = np.linspace(X[:, i].min(), X[:, i].max(), 80)
            ax.fill_between(xs, 0, gaussian_kde(X[:, i])(xs), color=MORANDI[i], alpha=0.75, lw=0)
        elif i > j:
            ax.scatter(X[:, j], X[:, i], s=6, color=MORANDI[j], alpha=0.5, edgecolor="none")
        else:
            ax.set_facecolor(dc(norm(corr[i, j])))
            ax.text(0.5, 0.5, f"{corr[i, j]:.2f}", ha="center", va="center",
                    transform=ax.transAxes, fontsize=9,
                    color="#2b2b2b" if abs(corr[i, j]) < 0.6 else "white")
        ax.set_xticks([]); ax.set_yticks([])
        if i == n - 1:
            ax.set_xlabel(labels[j], fontsize=8.5, rotation=40, ha="right")
        if j == 0:
            ax.set_ylabel(labels[i], fontsize=8.5, rotation=0, ha="right", va="center")
        for s in ax.spines.values():
            s.set_color("#dcdcdc")

# a small colourbar for the upper-triangle r scale
cax = fig.add_axes([0.915, 0.5, 0.016, 0.32])
import matplotlib as mpl
mpl.colorbar.ColorbarBase(cax, cmap=dc, norm=norm).set_label("Pearson r", fontsize=8)
cax.tick_params(labelsize=7)

fig.suptitle("Morandi — mixed correlation matrix (scatter · density · r)",
             x=0.09, ha="left", fontsize=13, fontweight="bold")
fig.savefig(OUT / "demo-morandi.jpg", dpi=160, facecolor="white", bbox_inches="tight")
plt.close(fig)
print("saved demo-morandi.jpg")
