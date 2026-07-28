"""Dusty Bloom palette — its signature chart: a ridgeline (joyplot).

Structured distributions (each group shifts + reshapes regularly).

  demo-dustybloom.jpg
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

OUT = Path(__file__).resolve().parents[2] / "figures"
RNG = np.random.default_rng(20260714)

plt.rcParams.update({
    "font.size": 9.5, "axes.edgecolor": "#a9a9a9", "axes.linewidth": 0.7,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": "#5a5a5a", "ytick.color": "#5a5a5a",
    "axes.labelcolor": "#333", "figure.facecolor": "white",
})

BLOOM = ["#C45863", "#F1B8CC", "#8EBFE3", "#B8ABC5", "#9E9DA0"]
cats = ["Site A", "Site B", "Site C", "Site D", "Site E"]

xs = np.linspace(-3, 11, 320)
fig, ax = plt.subplots(figsize=(7.6, 5.2))
for i, (col, nm) in enumerate(zip(BLOOM, cats)):
    # regular structure: mean shifts right and spread grows a little per group
    samp = RNG.normal(i * 1.4, 0.85 + 0.06 * i, 500)
    dens = gaussian_kde(samp)(xs); dens = dens / dens.max() * 1.7
    ax.fill_between(xs, i, i + dens, color=col, alpha=0.85, lw=1.1,
                    edgecolor="white", zorder=len(BLOOM) - i)
    ax.plot(xs, i + dens, color="white", lw=0.8, zorder=len(BLOOM) - i)
ax.set_yticks(range(len(cats)), cats)
ax.set_xlabel("Land-surface temperature (°C)"); ax.set_ylabel("")
ax.spines["left"].set_visible(False)
ax.set_title("Dusty Bloom — ridgeline (per-site distributions)", loc="left", fontsize=11)
fig.savefig(OUT / "demo-dustybloom.jpg", dpi=160, facecolor="white", bbox_inches="tight")
plt.close(fig)
print("saved demo-dustybloom.jpg")
