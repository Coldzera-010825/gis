"""Sage palette — its signature chart: hexbin density (observed vs predicted).

  demo-sage.jpg
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

OUT = Path(__file__).resolve().parents[2] / "figures"
RNG = np.random.default_rng(20260714)

plt.rcParams.update({
    "font.size": 9.5, "axes.edgecolor": "#a9a9a9", "axes.linewidth": 0.7,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": "#5a5a5a", "ytick.color": "#5a5a5a",
    "axes.labelcolor": "#333", "figure.facecolor": "white",
})

SAGE = ["#F3FBF2", "#C4E9CA", "#8BCF8B", "#519D78", "#2E6F40"]


def cmap(cs):
    return LinearSegmentedColormap.from_list("c", cs, N=256)


# observed vs predicted: strong agreement + scatter (dense near the 1:1 line)
obs = RNG.normal(0, 1, 9000)
pred = 0.9 * obs + RNG.normal(0, 0.45, 9000)

fig, ax = plt.subplots(figsize=(6.8, 5.8))
hb = ax.hexbin(obs, pred, gridsize=34, cmap=cmap(SAGE), mincnt=1, linewidths=0.2)
lim = [-3.6, 3.6]
ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel("Observed"); ax.set_ylabel("Predicted")
cb = fig.colorbar(hb, ax=ax, fraction=0.045, pad=0.02)
cb.set_label("count", fontsize=8); cb.ax.tick_params(labelsize=7)
ax.set_title("Sage — hexbin density (observed vs predicted)", loc="left", fontsize=11)
fig.savefig(OUT / "demo-sage.jpg", dpi=160, facecolor="white", bbox_inches="tight")
plt.close(fig)
print("saved demo-sage.jpg")
