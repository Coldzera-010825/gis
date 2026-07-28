"""Dusty Blue palette — its signature chart: an annotated correlation heatmap.

Structured (latent-factor) data so correlations show real patterns.

  demo-dustyblue.jpg
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

OUT = Path(__file__).resolve().parents[2] / "figures"
RNG = np.random.default_rng(20260714)

plt.rcParams.update({
    "font.size": 9.5, "axes.edgecolor": "#a9a9a9", "axes.linewidth": 0.7,
    "figure.facecolor": "white",
})

DUSTYBLUE = ["#EAF3FA", "#BAD2E1", "#96C2D4", "#6CBAD8", "#367DB0", "#04579B"]
labels = ["Temp", "Rain", "Humid", "SoilM", "pH", "OrgC", "TotN", "Elev"]
n = len(labels)


def cmap(cs):
    return LinearSegmentedColormap.from_list("c", cs, N=256)


# structured data: 3 latent factors -> real, patterned correlations
Lf = RNG.normal(size=(240, 3))
Wv = RNG.normal(size=(n, 3))
X = Lf @ Wv.T + RNG.normal(0, 0.5, (240, n))
corr = np.corrcoef(X.T)

# order variables by first-factor loading so the block structure is legible
order = np.argsort(Wv[:, 0])
corr = corr[np.ix_(order, order)]
labels = [labels[i] for i in order]

mask = np.triu(np.ones_like(corr, dtype=bool), k=1)   # show lower triangle + diagonal

fig, ax = plt.subplots(figsize=(7.2, 6.2))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", annot_kws={"size": 7},
            cmap=cmap(DUSTYBLUE), vmin=corr.min(), vmax=1, square=True,
            linewidths=0.6, linecolor="white",
            cbar_kws=dict(shrink=0.72, label="Pearson r"),
            xticklabels=labels, yticklabels=labels, ax=ax)
ax.tick_params(labelsize=8, length=0)
ax.set_title("Dusty Blue — correlation heatmap", loc="left", fontsize=11, pad=10)
fig.savefig(OUT / "demo-dustyblue.jpg", dpi=160, facecolor="white", bbox_inches="tight")
plt.close(fig)
print("saved demo-dustyblue.jpg")
