"""Correlation-heatmap suite (Python track) for Piece 09.

Simulates an environmental-monitoring table (soil / climate / vegetation
variables with a realistic correlation structure), then renders:

  corr-sub-basic.jpg      annotated full correlation heatmap
  corr-sub-triangle.jpg   masked lower triangle + significance stars
  corr-sub-clustermap.jpg correlation clustermap (hierarchically reordered)
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr

OUT = Path(__file__).resolve().parents[2] / "figures"
RNG = np.random.default_rng(20260713)
N = 220

VARS = ["Temp", "Rainfall", "Humidity", "Soil moisture", "pH",
        "Organic C", "Total N", "NDVI", "Elevation", "Slope"]


def simulate_env():
    """Latent-factor environmental data → realistic correlations."""
    climate = RNG.normal(size=N)        # warm/wet gradient
    soil = RNG.normal(size=N)           # fertility gradient
    terrain = RNG.normal(size=N)        # elevation gradient

    def mk(a, b, c, noise=0.6):
        return a * climate + b * soil + c * terrain + RNG.normal(0, noise, N)

    data = {
        "Temp":          mk(1.0, 0.0, -0.6),
        "Rainfall":      mk(0.8, 0.1, 0.2),
        "Humidity":      mk(0.7, 0.2, 0.1),
        "Soil moisture": mk(0.5, 0.6, 0.1),
        "pH":            mk(-0.2, -0.7, 0.1),
        "Organic C":     mk(0.1, 0.9, 0.0),
        "Total N":       mk(0.1, 0.85, 0.0),
        "NDVI":          mk(0.4, 0.6, -0.1),
        "Elevation":     mk(-0.5, 0.0, 1.0),
        "Slope":         mk(-0.1, 0.0, 0.7),
    }
    return pd.DataFrame(data)[VARS]


df = simulate_env()
corr = df.corr()


def pval_matrix(frame):
    cols = frame.columns
    p = pd.DataFrame(np.ones((len(cols), len(cols))), index=cols, columns=cols)
    for i, a in enumerate(cols):
        for j, b in enumerate(cols):
            if i != j:
                p.iloc[i, j] = pearsonr(frame[a], frame[b])[1]
    return p


def stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


pvals = pval_matrix(df)


def save(fig, name):
    fig.savefig(OUT / name, dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {name}")


# 1) basic full heatmap, annotated
fig, ax = plt.subplots(figsize=(7.6, 6.4), facecolor="white")
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1,
            center=0, square=True, linewidths=0.5, annot_kws={"size": 7},
            cbar_kws={"shrink": 0.8, "label": "Pearson r"}, ax=ax)
ax.set_title("Correlation matrix (full, annotated)", loc="left", fontsize=11)
save(fig, "corr-sub-basic.jpg")

# 2) masked lower triangle + significance stars
mask = np.triu(np.ones_like(corr, dtype=bool))          # hide upper triangle
labels = corr.round(2).astype(str) + pvals.map(stars)
fig, ax = plt.subplots(figsize=(7.6, 6.4), facecolor="white")
sns.heatmap(corr, mask=mask, annot=labels, fmt="", cmap="RdBu_r",
            vmin=-1, vmax=1, center=0, square=True, linewidths=0.5,
            annot_kws={"size": 6.5}, cbar_kws={"shrink": 0.8, "label": "Pearson r"},
            ax=ax)
ax.set_title("Lower triangle + significance (* p<.05  ** p<.01  *** p<.001)",
             loc="left", fontsize=10)
save(fig, "corr-sub-triangle.jpg")

# 3) correlation clustermap (hierarchically reordered)
cg = sns.clustermap(corr, cmap="RdBu_r", vmin=-1, vmax=1, center=0,
                    annot=True, fmt=".2f", annot_kws={"size": 6.5},
                    figsize=(8.2, 8.0), linewidths=0.5,
                    dendrogram_ratio=(0.12, 0.12),
                    cbar_pos=(0.02, 0.83, 0.03, 0.13))
cg.fig.savefig(OUT / "corr-sub-clustermap.jpg", dpi=150, facecolor="white",
               bbox_inches="tight")
plt.close(cg.fig)
print("saved corr-sub-clustermap.jpg")
print("done")
