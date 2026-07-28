"""Recommended Morandi / muted palettes for Piece 10 — curated & shown on real charts.

Outputs into ../../figures:
  rec-qualitative.jpg   curated qualitative palettes on scatter demos
  rec-sequential.jpg    curated sequential (tonal) palettes on ramp/field demos
  rec-diverging.jpg     curated diverging palettes on mini dot-plot / heatmap demos
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

OUT = Path(__file__).resolve().parents[2] / "figures"
RNG = np.random.default_rng(11)

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": "#a6a6a6", "axes.linewidth": 0.7,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": "#5a5a5a", "ytick.color": "#5a5a5a",
    "figure.facecolor": "white",
})


def H(s):
    return ["#" + c.strip() for c in s.split(",")]


def cmap(cs):
    return LinearSegmentedColormap.from_list("c", cs, N=256)


# ---- curated recommendations (name, hex, one-line rationale) ----
QUALITATIVE = [
    ("Morandi", "B4A79A,8C9AA0,A88B7D,9DA88B,B0929B,C7B7A3",
     "classic Morandi: greyed earth tones, nothing shouts"),
    ("Slate & Clay", "576B8E,7EA03C,B27276,7BA887,E5B327,741B2D",
     "earthy journal set: cool slate anchors warm clays"),
    ("Mint Mambo", "AD8632,1E619C,63B6BF,77211D,A8D2C2,CBA5AE",
     "gold + teal + wine — lively yet muted"),
    ("Meadow & Sky", "3D9F3C,9ED17B,367DB0,9DC7DD",
     "two analogous pairs (green / blue) — great for 4 groups"),
    ("Dusty Bloom", "C45863,F1B8CC,8EBFE3,B8ABC5,9E9DA0",
     "soft roses + powder blue + neutral grey"),
]

SEQUENTIAL = [
    ("Sage", "F3FBF2,C4E9CA,8BCF8B,519D78,2E6F40",
     "vegetation / NDVI; near-white low end"),
    ("Dusty Blue", "EAF3FA,BAD2E1,96C2D4,6CBAD8,367DB0",
     "water / density; calm and print-safe"),
    ("Warm Clay", "F3EAE2,E0C7B5,C99E86,A9755A,7C4A34",
     "warm magnitude; soil, temperature"),
    ("Teal Tonal", "EAF0F1,9BC0C7,6BA3AD,417D89,2A5A64",
     "single-hue elegance; monotone luminance"),
]

DIVERGING = [
    ("Mint Mambo Div", "1E619C,8FB8CC,F3EEE0,D2B48C,AD8632",
     "blue↔cream↔gold; the dot-plot classic"),
    ("Rose-Sage", "B27276,DDBFC1,F2ECE4,B6CFC7,5A8A82",
     "muted rose↔sage; gentle, biology-friendly"),
    ("Slate-Terracotta", "4E6E8E,A9C0D0,F0ECE4,D6A98C,A65E42",
     "cool↔warm earth; anomaly / change maps"),
]

_CENT = RNG.normal(0, 3.6, (8, 2))


def demo_qual(ax, cols):
    for i, c in enumerate(cols):
        pts = RNG.normal(_CENT[i % len(_CENT)], 0.62, (30, 2))
        ax.scatter(pts[:, 0], pts[:, 1], s=8, color=c, alpha=0.7, edgecolor="none")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def demo_seq(ax, cols):
    yy, xx = np.mgrid[0:1:50j, 0:1:70j]
    f = np.exp(-((xx-.35)**2+(yy-.6)**2)/.06) + .6*np.exp(-((xx-.72)**2+(yy-.32)**2)/.04)
    ax.imshow(f, cmap=cmap(cols), aspect="auto", origin="lower")
    ax.contour(f, levels=5, colors="w", linewidths=0.4, alpha=0.5)
    ax.set_xticks([]); ax.set_yticks([])


def demo_div(ax, cols):
    xs, ys, ss, cs = [], [], [], []
    for i in range(6):
        for j in range(7):
            xs.append(j); ys.append(i)
            ss.append(12 + RNG.uniform(0, 9)**1.4*9); cs.append(RNG.uniform(-.05, .05))
    ax.scatter(xs, ys, s=ss, c=cs, cmap=cmap(cols), vmin=-.05, vmax=.05,
               edgecolor="#7a7a7a", linewidth=0.35)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def render(fname, title, palettes, demo_fn):
    n = len(palettes)
    fig = plt.figure(figsize=(13, 1.5*n + 0.6))
    grid = fig.add_gridspec(n, 3, width_ratios=[2.7, 2.9, 2.2], hspace=0.75, wspace=0.14,
                            left=0.02, right=0.98, top=0.91, bottom=0.04)
    fig.suptitle(title, x=0.02, ha="left", fontsize=14, fontweight="bold")
    for r, (name, hx, why) in enumerate(palettes):
        cols = H(hx)
        axs = fig.add_subplot(grid[r, 0]); axs.axis("off"); axs.set_xlim(0, 1); axs.set_ylim(0, 1)
        w = 1.0/len(cols)
        for i, c in enumerate(cols):
            axs.add_patch(Rectangle((i*w, 0.34), w*0.93, 0.5, facecolor=c, edgecolor="#d5d5d5", lw=0.5))
            axs.text(i*w+w*0.46, 0.22, c.lstrip("#"), ha="center", va="top", fontsize=5.5,
                     family="monospace", color="#666")
        axs.text(0, 0.99, name, ha="left", va="top", fontsize=10.5, fontweight="bold")
        axt = fig.add_subplot(grid[r, 1]); axt.axis("off"); axt.set_xlim(0, 1); axt.set_ylim(0, 1)
        axt.text(0, 0.6, why, fontsize=9, color="#555", va="center", wrap=True)
        demo_fn(fig.add_subplot(grid[r, 2]), cols)
    fig.savefig(OUT / fname, dpi=160, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {fname}")


render("rec-qualitative.jpg", "Recommended · Qualitative (muted / Morandi)", QUALITATIVE, demo_qual)
render("rec-sequential.jpg", "Recommended · Sequential (tonal)", SEQUENTIAL, demo_seq)
render("rec-diverging.jpg", "Recommended · Diverging (muted)", DIVERGING, demo_div)
print("done")
