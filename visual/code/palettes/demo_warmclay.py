"""Warm Clay palette — its signature chart: tonal multi-line curves + CI bands.

Structured data: a monotone family of dose-response curves whose depth of colour
tracks the treatment level, each with a shaded confidence band.

  demo-warmclay.jpg
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

WARMCLAY = ["#F3EAE2", "#E0C7B5", "#C99E86", "#A9755A", "#7C4A34"]


def cmap(cs):
    return LinearSegmentedColormap.from_list("c", cs, N=256)


# structured data: 5 dose levels -> logistic curves, higher dose = steeper/higher
x = np.linspace(0, 10, 120)
levels = [0.5, 1.0, 1.5, 2.0, 2.5]
ramp = cmap(WARMCLAY)(np.linspace(0.18, 0.95, len(levels)))

fig, ax = plt.subplots(figsize=(7.2, 5.4))
for i, (lv, col) in enumerate(zip(levels, ramp)):
    ceiling = 0.45 + 0.14 * i          # higher dose -> higher plateau
    k = 0.7 + 0.28 * i                 # higher dose -> steeper slope
    x0 = 6.2 - 0.55 * i                # higher dose -> earlier onset
    y = ceiling / (1 + np.exp(-k * (x - x0)))
    ci = 0.02 + 0.015 * y / y.max()    # heteroscedastic band
    ax.fill_between(x, y - ci, y + ci, color=col, alpha=0.22, lw=0)
    ax.plot(x, y, color=col, lw=2.2, label=f"Dose {lv:.1f}", solid_capstyle="round")

ax.set_xlabel("Time (h)"); ax.set_ylabel("Response")
ax.set_xlim(0, 10); ax.set_ylim(0, None)
ax.set_title("Warm Clay — tonal dose-response curves", loc="left", fontsize=11)
ax.legend(title="Treatment", frameon=False, fontsize=8.5, title_fontsize=9,
          loc="upper left")
fig.tight_layout()
fig.savefig(OUT / "demo-warmclay.jpg", dpi=160, facecolor="white", bbox_inches="tight")
plt.close(fig)
print("saved demo-warmclay.jpg")
