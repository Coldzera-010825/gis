"""Diagnose the Mint Mambo palette: qualitative original vs derived diverging ramps."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

OUT = Path(__file__).resolve().parents[2] / "figures"
RNG = np.random.default_rng(7)

MINT_QUAL    = ["#AD8632", "#1E619C", "#63B6BF", "#77211D", "#A8D2C2", "#CBA5AE", "#DBD6C8"]
MINT_DIV_OLD = ["#1E619C", "#8FB8CC", "#F3EEE0", "#D2B48C", "#AD8632"]   # what was used (invented middles)
MINT_DIV_NEW = ["#1E619C", "#F4EEDF", "#AD8632"]                        # faithful: only the two anchors


def cmap(cs):
    return LinearSegmentedColormap.from_list("c", cs, N=256)


def swatches(ax, cols, title):
    w = 1.0 / len(cols)
    for i, c in enumerate(cols):
        ax.add_patch(Rectangle((i * w, 0.25), w * 0.94, 0.55, facecolor=c, edgecolor="#ccc", lw=0.5))
        ax.text(i * w + w * 0.47, 0.14, c.lstrip("#"), ha="center", va="top", fontsize=6.5,
                family="monospace", color="#666")
    ax.text(0, 0.95, title, ha="left", va="top", fontsize=10, fontweight="bold")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")


def ramp(ax, cols, title):
    grad = np.linspace(0, 1, 256)[None, :]
    ax.imshow(grad, aspect="auto", cmap=cmap(cols), extent=(0, 1, 0, 1))
    ax.text(0, 1.25, title, ha="left", va="bottom", fontsize=10, fontweight="bold", transform=ax.transAxes)
    ax.set_xticks([]); ax.set_yticks([])


fig = plt.figure(figsize=(11, 6.4), facecolor="white")
gs = fig.add_gridspec(4, 2, height_ratios=[1.1, 1, 1, 1], hspace=0.9, wspace=0.2,
                      left=0.04, right=0.97, top=0.92, bottom=0.06)

swatches(fig.add_subplot(gs[0, :]), MINT_QUAL, "Mint Mambo — original QUALITATIVE (categories: PCA, box, bars)")
ramp(fig.add_subplot(gs[1, 0]), MINT_DIV_OLD, "Diverging OLD (used in heatmap/volcano — added cream/tan)")
ramp(fig.add_subplot(gs[1, 1]), MINT_DIV_NEW, "Diverging NEW (faithful: blue ↔ cream ↔ gold only)")

# mini dot-plot demos: old vs new diverging
for k, (cols, lab) in enumerate([(MINT_DIV_OLD, "old"), (MINT_DIV_NEW, "new")]):
    ax = fig.add_subplot(gs[2:, k])
    xs, ys, ss, cs = [], [], [], []
    rng2 = np.random.default_rng(3)
    for i in range(7):
        for j in range(6):
            xs.append(j); ys.append(i); ss.append(14 + rng2.uniform(0, 9) ** 1.4 * 8)
            cs.append(rng2.uniform(-.05, .05))
    ax.scatter(xs, ys, s=ss, c=cs, cmap=cmap(cols), vmin=-.05, vmax=.05,
               edgecolor="#7a7a7a", linewidth=0.35)
    ax.set_title(f"dot plot · diverging {lab}", loc="left", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

fig.suptitle("Mint Mambo — is the diverging variant faithful?", x=0.04, ha="left",
             fontsize=13, fontweight="bold")
fig.savefig(OUT / "mint-compare.jpg", dpi=160, facecolor="white", bbox_inches="tight")
plt.close(fig)
print("saved mint-compare.jpg")
