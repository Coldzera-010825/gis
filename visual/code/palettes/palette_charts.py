"""Per-palette research showcases for Piece 10 — one distinct chart per palette,
plus a multi-panel 'versatility' sheet for the star palette (Mint Mambo).

Outputs into ../../figures:
  mm-versatility.jpg          Mint Mambo across 4 chart types (scatter/bars/ridgeline/dotplot)
  demo-morandi-stack.jpg      Morandi -> stacked proportion bars
  demo-slateclay-violin.jpg   Slate & Clay -> violin + box
  demo-dustybloom-ridge.jpg   Dusty Bloom -> ridgeline
  demo-rosesage-volcano.jpg   Rose-Sage (diverging) -> volcano plot
  demo-tealtonal-lines.jpg    Teal Tonal -> multi-line with CI bands
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import gaussian_kde

OUT = Path(__file__).resolve().parents[2] / "figures"
RNG = np.random.default_rng(7)

plt.rcParams.update({
    "font.size": 9.5, "axes.edgecolor": "#a6a6a6", "axes.linewidth": 0.7,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": "#5a5a5a", "ytick.color": "#5a5a5a",
    "axes.labelcolor": "#333", "axes.titlecolor": "#222", "figure.facecolor": "white",
})

MINT     = ["#AD8632", "#1E619C", "#63B6BF", "#77211D", "#A8D2C2", "#CBA5AE"]
MINT_DIV = ["#1E619C", "#8FB8CC", "#F3EEE0", "#D2B48C", "#AD8632"]
MORANDI  = ["#B4A79A", "#8C9AA0", "#A88B7D", "#9DA88B", "#B0929B", "#C7B7A3"]
SLATE    = ["#576B8E", "#7EA03C", "#B27276", "#7BA887", "#E5B327", "#741B2D"]
BLOOM    = ["#C45863", "#F1B8CC", "#8EBFE3", "#B8ABC5", "#9E9DA0"]
ROSESAGE = ["#B27276", "#DDBFC1", "#F2ECE4", "#B6CFC7", "#5A8A82"]
TEAL     = ["#EAF0F1", "#9BC0C7", "#6BA3AD", "#417D89", "#2A5A64"]


def cmap(cs):
    return LinearSegmentedColormap.from_list("c", cs, N=256)


def save(fig, name):
    fig.savefig(OUT / name, dpi=160, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {name}")


# ---------------------------------------------------------------- chart builders
def chart_scatter(ax, colors, labels=None):
    cents = [(-3.2, 2.2), (2.6, 2.8), (-2.0, -2.6), (3.0, -1.4), (0.2, 4.0), (-4.2, -0.4)]
    labels = labels or [f"Group {i+1}" for i in range(len(colors))]
    for c, col, nm in zip(cents, colors, labels):
        n = RNG.integers(260, 460)
        pts = RNG.normal(c, (1.15, 0.95), (n, 2))
        ax.scatter(pts[:, 0], pts[:, 1], s=6, color=col, alpha=0.62, edgecolor="none")
    ax.set_xlabel("UMAP_1"); ax.set_ylabel("UMAP_2"); ax.set_xticks([]); ax.set_yticks([])


def chart_grouped_bars(ax, colors):
    groups = ["Ctrl", "T1", "T2", "T3"]
    series = ["A", "B", "C"]
    vals = RNG.uniform(4, 18, (len(series), len(groups)))
    x = np.arange(len(groups)); w = 0.26
    for i, col in enumerate(colors[:len(series)]):
        ax.bar(x + (i-1)*w, vals[i], w, color=col, edgecolor="white", linewidth=0.6, label=series[i])
    ax.set_xticks(x, groups); ax.set_ylabel("Value")
    ax.grid(axis="y", color="#eee", linewidth=0.8); ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper right")


def chart_ridgeline(ax, colors):
    cats = ["c1", "c2", "c3", "c4", "c5", "c6"][:len(colors)]
    xs = np.linspace(-3, 8, 300)
    for i, (col, nm) in enumerate(zip(colors, cats)):
        samp = RNG.normal(i * 0.9, 0.7, 300)
        dens = gaussian_kde(samp)(xs); dens = dens / dens.max() * 1.5
        ax.fill_between(xs, i, i + dens, color=col, alpha=0.8, lw=1.0, edgecolor="white", zorder=len(colors)-i)
    ax.set_yticks(range(len(cats)), cats); ax.set_xlabel("Expression"); ax.set_ylabel("")
    for s in ["left"]:
        ax.spines[s].set_visible(False)


def chart_dotplot(ax, div_colors):
    rows = ["MITOTIC", "APICAL", "ENDOCYT", "YAP1_UP", "P53_DN", "TEAD4", "NFKB1"]
    cols = ["MED12", "MED19", "CCNC", "MED13", "MED15", "MED24"]
    xs, ys, ss, cs = [], [], [], []
    for i in range(len(rows)):
        for j in range(len(cols)):
            xs.append(j); ys.append(i); ss.append(14 + RNG.uniform(0, 9)**1.4*8)
            cs.append(RNG.uniform(-.05, .05))
    sc = ax.scatter(xs, ys, s=ss, c=cs, cmap=cmap(div_colors), vmin=-.05, vmax=.05,
                    edgecolor="#7a7a7a", linewidth=0.35)
    ax.set_xticks(range(len(cols)), cols, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(rows)), rows, fontsize=7)
    ax.set_xlim(-.6, len(cols)-.4); ax.set_ylim(-.6, len(rows)-.4); ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)


# ---------------------------------------------------------------- 1. Mint Mambo versatility
fig, axes = plt.subplots(2, 2, figsize=(11.5, 9))
chart_scatter(axes[0, 0], MINT, ["Cardio", "Fibro", "Myeloid", "Endo", "Tcell", "Bcell"])
axes[0, 0].set_title("(a) cell-type UMAP", loc="left", fontsize=10.5)
chart_grouped_bars(axes[0, 1], MINT)
axes[0, 1].set_title("(b) grouped bars", loc="left", fontsize=10.5)
chart_ridgeline(axes[1, 0], MINT)
axes[1, 0].set_title("(c) ridgeline", loc="left", fontsize=10.5)
chart_dotplot(axes[1, 1], MINT_DIV)
axes[1, 1].set_title("(d) signature dot plot (diverging variant)", loc="left", fontsize=10.5)
fig.suptitle("Mint Mambo — one palette, many chart types (its versatility)",
             x=0.02, ha="left", fontsize=14, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.96))
save(fig, "mm-versatility.jpg")

# ---------------------------------------------------------------- 2. Morandi stacked proportion
fig, ax = plt.subplots(figsize=(7.6, 4.6))
groups = ["Wenzhou", "Ganzhou", "Bristol", "Beijing", "Cardiff", "Nanchang"]
comp = RNG.dirichlet(np.ones(len(MORANDI)) * 1.3, len(groups)).T * 100
bottom = np.zeros(len(groups))
labels = ["Water", "Forest", "Cropland", "Grass", "Built", "Bare"]
for row, col, lb in zip(comp, MORANDI, labels):
    ax.bar(groups, row, bottom=bottom, color=col, edgecolor="white", linewidth=0.7, label=lb)
    bottom += row
ax.set_ylabel("Land-cover share (%)"); ax.set_ylim(0, 100)
ax.legend(frameon=False, fontsize=8, ncol=6, loc="upper center", bbox_to_anchor=(0.5, 1.16))
ax.set_title("Morandi — stacked proportion bars", loc="left", fontsize=11)
save(fig, "demo-morandi-stack.jpg")

# ---------------------------------------------------------------- 3. Slate & Clay violin+box
fig, ax = plt.subplots(figsize=(7.8, 4.8))
cats = ["ST16", "ST17", "ST18", "ST80", "ST117", "ST203"]
data = [RNG.normal(m, s, 160) for m, s in zip([28, 24, 22, 34, 40, 36], [6, 5, 5, 8, 7, 9])]
parts = ax.violinplot(data, showextrema=False, widths=0.85)
for b, col in zip(parts["bodies"], SLATE):
    b.set_facecolor(col); b.set_alpha(0.35); b.set_edgecolor(col)
bp = ax.boxplot(data, widths=0.18, patch_artist=True, showfliers=False,
                medianprops=dict(color="white", linewidth=1.3),
                whiskerprops=dict(color="#666"), capprops=dict(color="#666"))
for patch, col in zip(bp["boxes"], SLATE):
    patch.set_facecolor(col); patch.set_edgecolor(col)
ax.set_xticks(range(1, len(cats)+1), cats); ax.set_ylabel("ISL3 copy per genome")
ax.grid(axis="y", color="#eee", linewidth=0.8); ax.set_axisbelow(True)
ax.set_title("Slate & Clay — violin + box", loc="left", fontsize=11)
save(fig, "demo-slateclay-violin.jpg")

# ---------------------------------------------------------------- 4. Dusty Bloom ridgeline
fig, ax = plt.subplots(figsize=(7.4, 5.0))
chart_ridgeline(ax, BLOOM)
ax.set_title("Dusty Bloom — ridgeline", loc="left", fontsize=11)
save(fig, "demo-dustybloom-ridge.jpg")

# ---------------------------------------------------------------- 5. Rose-Sage volcano
fig, ax = plt.subplots(figsize=(6.8, 5.4))
n = 2000
lfc = RNG.normal(0, 1.6, n)
p = 10 ** (-np.abs(RNG.normal(0, 1.4, n)) * (1 + np.abs(lfc) / 3))
neglogp = -np.log10(p)
sc = ax.scatter(lfc, neglogp, c=lfc, cmap=cmap(ROSESAGE), vmin=-4, vmax=4,
                s=10, alpha=0.7, edgecolor="none")
ax.axvline(-1, ls="--", color="#bbb", lw=0.8); ax.axvline(1, ls="--", color="#bbb", lw=0.8)
ax.axhline(-np.log10(0.05), ls="--", color="#bbb", lw=0.8)
ax.set_xlabel("log$_2$ fold change"); ax.set_ylabel("-log$_{10}$ p")
cb = fig.colorbar(sc, ax=ax, fraction=0.045, pad=0.02); cb.set_label("log$_2$FC", fontsize=8)
ax.set_title("Rose-Sage (diverging) — volcano plot", loc="left", fontsize=11)
save(fig, "demo-rosesage-volcano.jpg")

# ---------------------------------------------------------------- 6. Teal Tonal lines + CI
fig, ax = plt.subplots(figsize=(7.4, 4.8))
x = np.linspace(0, 12, 60)
for i, col in enumerate(TEAL):
    base = np.sin(x / 2 + i * 0.5) * (1 + i * 0.2) + i * 0.8 + 3
    noise = 0.25 + 0.05 * i
    ax.plot(x, base, color=col, lw=2.0, label=f"scenario {i+1}")
    ax.fill_between(x, base - noise, base + noise, color=col, alpha=0.18)
ax.set_xlabel("Lead time (months)"); ax.set_ylabel("Predicted index")
ax.legend(frameon=False, fontsize=8, ncol=5, loc="upper left")
ax.grid(color="#f0f0f0", linewidth=0.8); ax.set_axisbelow(True)
ax.set_title("Teal Tonal — multi-line with CI bands", loc="left", fontsize=11)
save(fig, "demo-tealtonal-lines.jpg")
print("done")
