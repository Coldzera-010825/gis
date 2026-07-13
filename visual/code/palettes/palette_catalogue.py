"""Palette catalogue for Piece 10 — four families, research-realistic demos.

Renders four review overviews, each showing curated palettes with swatches +
a demo in a genuine research chart role:

  palette-qualitative.jpg   distinct hues  -> multi-class PCA scatter
  palette-sequential.jpg    light -> dark  -> interpolated raster surface
  palette-diverging.jpg     two-ended      -> gene-signature dot plot (size + colour)
  palette-gis.jpg           geospatial     -> terrain / bathymetry / land cover / anomaly / perceptual
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.patches import Rectangle

OUT = Path(__file__).resolve().parents[2] / "figures"
RNG = np.random.default_rng(20260713)


def H(s):
    return ["#" + c.strip() for c in s.split(",")]


def cmap(colors):
    return LinearSegmentedColormap.from_list("c", colors, N=256)


# ---------------------------------------------------------------- palettes
QUALITATIVE = [
    ("Jewel 8",   "1F4E7A,E6AB4D,452A52,9A3F00,3E4796,D70006,276633,960448"),
    ("Clinical",  "00468B,ED0000,42B540,0099B4,925E9F,FDAF91,AD002A"),
    ("Nature",    "E64B35,4DBBD5,00A087,3C5488,F39B7F,8491B4,91D1C2"),
    ("Earthy",    "AD8632,1E619C,63B6BF,77211D,A8D2C2,CBA5AE"),
    ("Muted",     "6C8EBF,C1846D,7FA870,9A7AA0,D4B15F,6BA3A0"),
]
SEQUENTIAL = [
    ("Amber Night", "FFF4C1,F36352,19141A"),
    ("Deep Ocean",  "F9F3D2,266292,372435"),
    ("Ember",       "FFFFFF,FFEFB8,F05A5F,833584"),
    ("Steel Blue",  "F7FBFF,9ECAE1,4292C6,08519C,08306B"),
    ("Forest",      "F7FCF5,A1D99B,41AB5D,238B45,00441B"),
]
DIVERGING = [
    ("Navy-Crimson",   "112D61,4472A8,FFFFFF,B0505A,660514"),
    ("Crimson-Azure",  "CB183C,FFC587,FFFFFF,7FB8DA,0370B3"),
    ("Violet-Wine",    "7E33A9,DC9EC8,FFFFFF,B07385,5D2740"),
    ("Earth-Teal",     "8C510A,D8B365,F6E8C3,C7EAE5,5AB4AC,01665E"),
    ("Magenta-Green",  "C51B7D,F1B6DA,F7F7F7,B8E186,4D9221"),
]
# GIS: (name, hex, demo-kind)
GIS = [
    ("Hypsometric (elevation)", "2E6F40,86A96A,E4D6A0,C29A5B,8A5A2B,FFFFFF", "terrain"),
    ("Bathymetry (depth)",      "E1F5FE,81D4FA,29B6F6,0277BD,01467E,002B4D", "bathy"),
    ("Land cover (categorical)","4A90D9,2E7D32,C5E17A,C2B280,B0413E,EDEDED", "landcover"),
    ("Temp anomaly (diverging)","2166AC,92C5DE,F7F7F7,F4A582,B2182B", "anomaly"),
    ("Viridis (perceptual)",    "440154,414487,2A788E,22A884,7AD151,FDE725", "raster"),
    ("Cividis (CVD-safe)",      "00224E,35456C,666970,978F78,DECD5A,FFEA46", "raster"),
]

LANDCOVER_NAMES = ["Water", "Forest", "Cropland", "Bare", "Urban", "Cloud"]


# ---------------------------------------------------------------- shared data
_CENT = RNG.normal(0, 4, (8, 2))


def synth_dem(nx=90, ny=70, seed=1):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:1:complex(0, ny), 0:1:complex(0, nx)]
    z = (1.4 * np.exp(-((xx - 0.35)**2 + (yy - 0.6)**2) / 0.05)
         + 1.0 * np.exp(-((xx - 0.7)**2 + (yy - 0.35)**2) / 0.03)
         + 0.4 * np.sin(xx * 6) * np.cos(yy * 5))
    z += 0.06 * rng.standard_normal(z.shape)
    return z


# ---------------------------------------------------------------- demos
def demo_qual(ax, colors):
    for i, c in enumerate(colors):
        pts = RNG.normal(_CENT[i % len(_CENT)], 0.7, (28, 2))
        ax.scatter(pts[:, 0], pts[:, 1], c=c, s=12, alpha=0.85,
                   edgecolor="white", linewidth=0.2)
    ax.set_xticks([]); ax.set_yticks([])


def demo_seq(ax, colors):
    field = synth_dem()
    ax.imshow(field, cmap=cmap(colors), aspect="auto", origin="lower")
    ax.contour(field, levels=6, colors="k", linewidths=0.25, alpha=0.35)
    ax.set_xticks([]); ax.set_yticks([])


def demo_div_dotplot(ax, colors):
    sigs = ["MITOTIC", "APICAL", "ENDOCYT", "YAP1_UP", "P53_DN", "TEAD4", "NFKB1"]
    conds = ["MED12", "MED19", "CCNC", "MED13", "MED15", "MED24"]
    score = RNG.uniform(-0.05, 0.05, (len(sigs), len(conds)))
    fdr = RNG.uniform(0, 10, (len(sigs), len(conds)))
    xs, ys, ss, cs = [], [], [], []
    for i in range(len(sigs)):
        for j in range(len(conds)):
            xs.append(j); ys.append(i)
            ss.append(12 + fdr[i, j] * 20); cs.append(score[i, j])
    ax.scatter(xs, ys, s=ss, c=cs, cmap=cmap(colors), vmin=-0.05, vmax=0.05,
               edgecolor="grey", linewidth=0.3)
    ax.set_xticks(range(len(conds))); ax.set_xticklabels(conds, fontsize=6, rotation=45, ha="right")
    ax.set_yticks(range(len(sigs))); ax.set_yticklabels(sigs, fontsize=6)
    ax.set_xlim(-0.6, len(conds) - 0.4); ax.set_ylim(-0.6, len(sigs) - 0.4)
    ax.tick_params(length=0)


def demo_gis(ax, colors, kind):
    if kind == "landcover":
        lc = RNG.integers(0, len(colors), (14, 18))
        # smooth into contiguous patches
        from scipy.ndimage import median_filter
        lc = median_filter(lc, size=3)
        ax.imshow(lc, cmap=ListedColormap(colors), aspect="auto", origin="lower")
    elif kind == "bathy":
        depth = -synth_dem(seed=3)
        ax.imshow(depth, cmap=cmap(colors), aspect="auto", origin="lower")
    elif kind == "anomaly":
        an = synth_dem(seed=5) - synth_dem(seed=6)
        vmax = np.abs(an).max()
        ax.imshow(an, cmap=cmap(colors), vmin=-vmax, vmax=vmax, aspect="auto", origin="lower")
    else:  # terrain, raster
        z = synth_dem(seed=1 if kind == "terrain" else 7)
        ax.imshow(z, cmap=cmap(colors), aspect="auto", origin="lower")
        if kind == "terrain":
            ax.contour(z, levels=7, colors="k", linewidths=0.25, alpha=0.3)
    ax.set_xticks([]); ax.set_yticks([])


# ---------------------------------------------------------------- render
def swatch_row(fig, gridcell, name, cols):
    axs = fig.add_subplot(gridcell); axs.axis("off")
    axs.set_xlim(0, 1); axs.set_ylim(0, 1)
    w = 1.0 / len(cols)
    for i, c in enumerate(cols):
        axs.add_patch(Rectangle((i * w, 0.30), w * 0.93, 0.52, facecolor=c,
                                edgecolor="#cfcfcf", lw=0.5))
        axs.text(i * w + w * 0.46, 0.18, c.lstrip("#"), ha="center", va="top",
                 fontsize=6, family="monospace", color="#555")
    axs.text(0, 0.97, name, ha="left", va="top", fontsize=10.5, fontweight="bold")


def render_simple(fname, title, palettes, demo_fn):
    n = len(palettes)
    fig = plt.figure(figsize=(12.5, 1.55 * n + 0.5), facecolor="white")
    grid = fig.add_gridspec(n, 2, width_ratios=[3.6, 2.4], hspace=0.7, wspace=0.14,
                            left=0.03, right=0.97, top=0.90, bottom=0.04)
    fig.suptitle(title, x=0.03, ha="left", fontsize=14, fontweight="bold")
    for r, (name, hx) in enumerate(palettes):
        cols = H(hx)
        swatch_row(fig, grid[r, 0], name, cols)
        demo_fn(fig.add_subplot(grid[r, 1]), cols)
    fig.savefig(OUT / fname, dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {fname}")


def render_gis(fname, title, palettes):
    n = len(palettes)
    fig = plt.figure(figsize=(12.5, 1.55 * n + 0.5), facecolor="white")
    grid = fig.add_gridspec(n, 2, width_ratios=[3.6, 2.4], hspace=0.7, wspace=0.14,
                            left=0.03, right=0.97, top=0.90, bottom=0.04)
    fig.suptitle(title, x=0.03, ha="left", fontsize=14, fontweight="bold")
    for r, (name, hx, kind) in enumerate(palettes):
        cols = H(hx)
        swatch_row(fig, grid[r, 0], name, cols)
        demo_gis(fig.add_subplot(grid[r, 1]), cols, kind)
    fig.savefig(OUT / fname, dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {fname}")


render_simple("palette-qualitative.jpg",
              "Qualitative  |  distinct hues -> categories (multi-class scatter)",
              QUALITATIVE, demo_qual)
render_simple("palette-sequential.jpg",
              "Sequential  |  light -> dark -> ordered magnitude (interpolated surface)",
              SEQUENTIAL, demo_seq)
render_simple("palette-diverging.jpg",
              "Diverging  |  two-ended through neutral -> centered data (signature dot plot)",
              DIVERGING, demo_div_dotplot)
render_gis("palette-gis.jpg",
           "Geospatial maps  |  terrain / depth / land cover / anomaly / perceptual",
           GIS)
print("done")
