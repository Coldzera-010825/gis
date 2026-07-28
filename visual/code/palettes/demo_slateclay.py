"""Slate & Clay palette — its signature chart: a labelled UMAP with 95% ellipses.

Structured clusters (real blob structure) so groups separate meaningfully.

  demo-slateclay.jpg
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse

OUT = Path(__file__).resolve().parents[2] / "figures"
RNG = np.random.default_rng(20260714)

plt.rcParams.update({
    "font.size": 9.5, "axes.edgecolor": "#a9a9a9", "axes.linewidth": 0.7,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": "#5a5a5a", "ytick.color": "#5a5a5a",
    "axes.labelcolor": "#333", "figure.facecolor": "white",
})

SLATE = ["#576B8E", "#7EA03C", "#B27276", "#7BA887", "#E5B327", "#741B2D"]
names = ["Cluster 1", "Cluster 2", "Cluster 3", "Cluster 4", "Cluster 5", "Cluster 6"]
cents = [(-4.2, 2.0), (3.6, 3.2), (-2.0, -3.4), (4.4, -1.6), (0.4, 5.2), (-5.0, -0.6)]
spreads = [(1.1, 0.9), (1.0, 1.2), (1.3, 0.8), (0.9, 1.1), (1.2, 0.9), (1.0, 1.0)]


def conf_ellipse(ax, x, y, color, n_std=2.0):
    cov = np.cov(x, y); mean = (x.mean(), y.mean())
    vals, vecs = np.linalg.eigh(cov); order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    w, h = 2 * n_std * np.sqrt(vals)
    ax.add_patch(Ellipse(mean, w, h, angle=theta, facecolor=color, alpha=0.14,
                         edgecolor=color, lw=1.4, zorder=1))


fig, ax = plt.subplots(figsize=(7.4, 6.2))
for c, sp, col, nm in zip(cents, spreads, SLATE, names):
    nk = RNG.integers(320, 520)
    pts = RNG.normal(c, sp, (nk, 2))
    ax.scatter(pts[:, 0], pts[:, 1], s=7, color=col, alpha=0.6, edgecolor="none",
               label=nm, zorder=2)
    conf_ellipse(ax, pts[:, 0], pts[:, 1], col)
ax.set_xlabel("UMAP_1"); ax.set_ylabel("UMAP_2")
ax.set_xticks([]); ax.set_yticks([])
ax.legend(frameon=False, fontsize=8.5, loc="upper left", markerscale=2.2,
          handletextpad=0.3, labelspacing=0.5)
ax.set_title("Slate & Clay — UMAP with 95% confidence ellipses", loc="left", fontsize=11)
fig.savefig(OUT / "demo-slateclay.jpg", dpi=160, facecolor="white", bbox_inches="tight")
plt.close(fig)
print("saved demo-slateclay.jpg")
