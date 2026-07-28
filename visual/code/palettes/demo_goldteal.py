"""Gold & Teal palette — its signature chart: density-contour scatter (kde + points).

  demo-goldteal.jpg
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

OUT = Path(__file__).resolve().parents[2] / "figures"
RNG = np.random.default_rng(20260714)

plt.rcParams.update({
    "font.size": 9.5, "axes.edgecolor": "#a9a9a9", "axes.linewidth": 0.7,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": "#5a5a5a", "ytick.color": "#5a5a5a",
    "axes.labelcolor": "#333", "figure.facecolor": "white",
})

GOLDTEAL = ["#AD8632", "#1E619C", "#63B6BF", "#77211D"]
names = ["Gold", "Blue", "Teal", "Wine"]
cents = [(-3.0, 1.6), (3.2, 2.4), (-1.4, -3.0), (3.4, -1.8)]
spreads = [(1.2, 0.9), (1.0, 1.2), (1.3, 0.9), (0.9, 1.1)]

fig, ax = plt.subplots(figsize=(7.2, 6.0))
for c, sp, col, nm in zip(cents, spreads, GOLDTEAL, names):
    nk = RNG.integers(300, 460)
    pts = RNG.normal(c, sp, (nk, 2))
    sns.kdeplot(x=pts[:, 0], y=pts[:, 1], color=col, fill=True, alpha=0.22,
                levels=4, thresh=0.15, ax=ax, zorder=1)
    sns.kdeplot(x=pts[:, 0], y=pts[:, 1], color=col, levels=4, thresh=0.15,
                linewidths=0.8, alpha=0.8, ax=ax, zorder=2)
    ax.scatter(pts[:, 0], pts[:, 1], s=6, color=col, alpha=0.5, edgecolor="none",
               label=nm, zorder=3)
ax.set_xlabel("Dim 1"); ax.set_ylabel("Dim 2"); ax.set_xticks([]); ax.set_yticks([])
ax.legend(frameon=False, fontsize=8.5, loc="upper left", markerscale=2.2,
          handletextpad=0.3, labelspacing=0.5)
ax.set_title("Gold & Teal — density-contour scatter (kde + points)", loc="left", fontsize=11)
fig.savefig(OUT / "demo-goldteal.jpg", dpi=160, facecolor="white", bbox_inches="tight")
plt.close(fig)
print("saved demo-goldteal.jpg")
