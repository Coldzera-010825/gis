"""Beautiful, research-realistic palette showcases for Piece 10 (rework).

Curated muted/sophisticated palettes (extracted from journal figures) applied to
real chart types, plus a 'craft of pairing' panel that teaches harmony.

Outputs into ../../figures:
  show-scatter.jpg     UMAP-style scatter (analogous green-blue)
  show-bars.jpg        grouped bars (earthy qualitative)
  show-box.jpg         box + jitter (muted qualitative)
  show-dotplot.jpg     signature dot plot (mint-mambo diverging)
  show-harmony.jpg     the craft: saturated vs muted, analogous, tonal, accent+neutral
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

OUT = Path(__file__).resolve().parents[2] / "figures"
RNG = np.random.default_rng(42)

# a clean, modern rc — thin spines, soft greys, airy
plt.rcParams.update({
    "font.size": 10,
    "axes.edgecolor": "#9a9a9a",
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.color": "#5a5a5a",
    "ytick.color": "#5a5a5a",
    "axes.labelcolor": "#333333",
    "axes.titlecolor": "#222222",
    "figure.facecolor": "white",
})

# ---- curated palettes (muted, journal-grade), extracted from the references ----
P_UMAP    = ["#3D9F3C", "#9ED17B", "#367DB0", "#9DC7DD"]                 # green-blue analogous
P_EARTHY  = ["#576B8E", "#7EA03C", "#B27276", "#7BA887", "#E5B327", "#741B2D"]  # ST earthy
P_DUSTY   = ["#C45863", "#8EBFE3", "#B8ABC5", "#7BA887", "#E5B327", "#9E9DA0"]  # soft mixed
MINT_DIV  = ["#1E619C", "#8FB8CC", "#F3EEE0", "#D2B48C", "#AD8632"]      # mint-mambo diverging


def save(fig, name):
    fig.savefig(OUT / name, dpi=160, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {name}")


# ---------------------------------------------------------------- 1. UMAP scatter
fig, ax = plt.subplots(figsize=(6.6, 5.4))
names = ["Cardiomyocytes", "Fibroblasts", "Myeloid cells", "Endothelial cells"]
cent = [(-3.2, 2.4), (2.6, 3.0), (-2.0, -2.8), (3.2, -1.6)]
for c, col, nm in zip(cent, P_UMAP, names):
    n = RNG.integers(500, 900)
    pts = RNG.normal(c, (1.3, 1.0), (n, 2)) + RNG.normal(0, 0.4, (n, 2))
    ax.scatter(pts[:, 0], pts[:, 1], s=6, color=col, alpha=0.65,
               edgecolor="none", label=nm, rasterized=True)
ax.set_xlabel("UMAP_1"); ax.set_ylabel("UMAP_2")
ax.set_xticks([]); ax.set_yticks([])
leg = ax.legend(loc="upper left", frameon=False, fontsize=9, markerscale=2.4,
                handletextpad=0.3, labelspacing=0.5)
ax.set_title("Analogous green–blue  ·  cell-type UMAP", loc="left", fontsize=11)
save(fig, "show-scatter.jpg")

# ---------------------------------------------------------------- 2. grouped bars
fig, ax = plt.subplots(figsize=(7.4, 4.6))
groups = ["Site A", "Site B", "Site C", "Site D", "Site E"]
series = ["Spring", "Summer", "Autumn"]
vals = RNG.uniform(4, 20, (len(series), len(groups)))
x = np.arange(len(groups)); w = 0.26
for i, (s, col) in enumerate(zip(series, P_EARTHY[:3])):
    ax.bar(x + (i - 1) * w, vals[i], w, label=s, color=col,
           edgecolor="white", linewidth=0.6)
ax.set_xticks(x, groups); ax.set_ylabel("NDVI × 100")
ax.legend(frameon=False, fontsize=9, loc="upper right", ncol=3)
ax.grid(axis="y", color="#ececec", linewidth=0.8); ax.set_axisbelow(True)
ax.set_title("Muted earthy qualitative  ·  grouped bars", loc="left", fontsize=11)
save(fig, "show-bars.jpg")

# ---------------------------------------------------------------- 3. box + jitter
fig, ax = plt.subplots(figsize=(7.4, 4.8))
cats = ["ST16", "ST17", "ST18", "ST80", "ST117", "ST203"]
data = [RNG.normal(m, s, 90) for m, s in
        zip([28, 24, 22, 34, 40, 36], [6, 5, 5, 8, 7, 9])]
bp = ax.boxplot(data, patch_artist=True, widths=0.6, showfliers=False,
                medianprops=dict(color="#333", linewidth=1.3),
                whiskerprops=dict(color="#888", linewidth=1.0),
                capprops=dict(color="#888", linewidth=1.0))
for patch, col in zip(bp["boxes"], P_EARTHY):
    patch.set_facecolor(col); patch.set_alpha(0.55); patch.set_edgecolor(col)
for i, (d, col) in enumerate(zip(data, P_EARTHY), start=1):
    ax.scatter(RNG.normal(i, 0.05, len(d)), d, s=7, color=col, alpha=0.5,
               edgecolor="none", zorder=3)
ax.set_xticks(range(1, len(cats) + 1), cats)
ax.set_ylabel("Estimated ISL3 copy per genome")
ax.grid(axis="y", color="#ececec", linewidth=0.8); ax.set_axisbelow(True)
ax.set_title("Muted qualitative  ·  box + jitter", loc="left", fontsize=11)
save(fig, "show-box.jpg")

# ---------------------------------------------------------------- 4. dot plot (diverging)
fig, ax = plt.subplots(figsize=(7.2, 5.4))
sigs = ["MITOTIC_SPINDLE", "APICAL_JUNCTION", "ENDOCYTOSIS", "YAP1_UP",
        "P53_DN", "TEAD4_A549", "NFKB1_ACTIVE", "TWIST1_UP"]
conds = ["MED12", "MED19", "CCNC", "MED13", "MED15", "MED24"]
cmap = LinearSegmentedColormap.from_list("mint", MINT_DIV, N=256)
xs, ys, ss, cs = [], [], [], []
for i in range(len(sigs)):
    for j in range(len(conds)):
        xs.append(j); ys.append(i)
        ss.append(15 + RNG.uniform(0, 10) ** 1.4 * 10)
        cs.append(RNG.uniform(-0.05, 0.05))
sc = ax.scatter(xs, ys, s=ss, c=cs, cmap=cmap, vmin=-0.05, vmax=0.05,
                edgecolor="#7a7a7a", linewidth=0.4)
ax.set_xticks(range(len(conds)), conds, rotation=45, ha="right", fontsize=8)
ax.set_yticks(range(len(sigs)), sigs, fontsize=8)
ax.set_xlim(-0.6, len(conds) - 0.4); ax.set_ylim(-0.6, len(sigs) - 0.4)
ax.tick_params(length=0)
for s in ax.spines.values():
    s.set_visible(False)
cb = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
cb.set_label("Score KD–NTC", fontsize=8); cb.ax.tick_params(labelsize=7)
ax.set_title("Mint-mambo diverging  ·  signature dot plot", loc="left", fontsize=11)
save(fig, "show-dotplot.jpg")

# ---------------------------------------------------------------- 5. harmony craft
fig, axes = plt.subplots(2, 2, figsize=(11, 7))

# (a) saturated vs muted — the single biggest lever
axa = axes[0, 0]
sat = ["#E51919", "#1552E5", "#12A312", "#F0A000"]
mut = ["#C45863", "#576B8E", "#7BA887", "#E5B327"]
for k, col in enumerate(sat):
    axa.add_patch(Rectangle((k, 1.1), 0.9, 0.8, color=col))
for k, col in enumerate(mut):
    axa.add_patch(Rectangle((k, 0.1), 0.9, 0.8, color=col))
axa.text(-0.15, 1.5, "saturated", ha="right", va="center", fontsize=9, color="#999")
axa.text(-0.15, 0.5, "muted", ha="right", va="center", fontsize=9, color="#333", fontweight="bold")
axa.set_xlim(-1.6, 4); axa.set_ylim(0, 2); axa.axis("off")
axa.set_title("(a) Desaturate — the biggest lever", loc="left", fontsize=10.5)

# (b) analogous — neighbouring hues
axb = axes[0, 1]
ana = ["#2A6F97", "#2C7DA0", "#468FAF", "#61A5C2", "#89C2D9", "#A9D6E5"]
for k, col in enumerate(ana):
    axb.add_patch(Rectangle((k, 0.5), 0.95, 1.0, color=col))
axb.set_xlim(0, len(ana)); axb.set_ylim(0, 2); axb.axis("off")
axb.set_title("(b) Analogous — neighbouring hues, calm & cohesive", loc="left", fontsize=10.5)

# (c) tonal / monochromatic — one hue, varying lightness
axc = axes[1, 0]
tonal = ["#EAF0F1", "#C9DDE0", "#9BC0C7", "#6BA3AD", "#417D89", "#2A5A64"]
for k, col in enumerate(tonal):
    axc.add_patch(Rectangle((k, 0.5), 0.95, 1.0, color=col))
axc.set_xlim(0, len(tonal)); axc.set_ylim(0, 2); axc.axis("off")
axc.set_title("(c) Tonal — one hue, light→dark (elegant sequential)", loc="left", fontsize=10.5)

# (d) accent + neutral — greys carry, one colour points
axd = axes[1, 1]
neut = ["#DBD6C8", "#C9C4B8", "#B3AEA3", "#9E9DA0", "#858692"]
for k, col in enumerate(neut):
    axd.add_patch(Rectangle((k, 0.5), 0.95, 1.0, color=col))
axd.add_patch(Rectangle((len(neut) + 0.3, 0.5), 0.95, 1.0, color="#C45863"))
axd.text(len(neut) + 0.77, 0.32, "accent", ha="center", va="top", fontsize=8, color="#C45863")
axd.set_xlim(0, len(neut) + 1.6); axd.set_ylim(0, 2); axd.axis("off")
axd.set_title("(d) Accent + neutral — one saturated colour points, greys carry", loc="left", fontsize=10.5)

fig.suptitle("The craft of pairing: why the palettes above look good",
             x=0.02, ha="left", fontsize=13, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.96))
save(fig, "show-harmony.jpg")
print("done")
