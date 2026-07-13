---
title: "Piece 10 · Scientific colour palettes — and where to use them"
description: A palette is a tool matched to a data type. Four families (qualitative, sequential, diverging, and geospatial-map), each with curated hex sets shown on the research chart they actually belong on, plus the code and the rules.
---

# Scientific colour palettes

```{figure} figures/palette-gis.jpg
:width: 100%
:alt: Six geospatial map palettes shown on terrain, bathymetry, land-cover, anomaly and perceptual demos

Geospatial map palettes, each on the map type it is built for — hypsometric terrain,
bathymetric depth, categorical land cover, a temperature-anomaly diverging map, and the
perceptually-uniform viridis / cividis for continuous rasters.
```

## 1 · The problem with most palette lists

Search "scientific colour palettes" and you get long strips of hex codes. What you
almost never get is the part that actually matters: **which chart type each palette
belongs on, and why.** A palette is not decoration — it is an *encoding*, matched to a
*data type*. Use the wrong family and a correct analysis reads as a wrong one.

There are only a few families to learn. Match the data to the family, and colour stops
being guesswork.

| Family | Looks like | Encodes | Natural chart |
| --- | --- | --- | --- |
| **Qualitative** | several distinct hues, no order | discrete **categories** | grouped bars, multi-class scatter, category maps |
| **Sequential** | one/multi hue, light → dark | ordered **magnitude** | heatmaps, rasters, choropleths |
| **Diverging** | two hues through a neutral midpoint | data with a **centre** (0, mean) | correlation, z-score, anomaly, up/down |
| **Geospatial map** | family chosen *per map layer* | terrain, depth, cover, change | thematic maps |

Every hex set below is shown **on the chart it belongs on**, with the code that draws it.

## 2 · The one helper you need

Turning a list of hex colours into a matplotlib colormap is a one-liner — the whole
tutorial reuses it:

```python
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

def cmap(hexes):                       # continuous: sequential / diverging / terrain
    return LinearSegmentedColormap.from_list("c", hexes, N=256)

# ListedColormap(hexes)                # discrete: qualitative / land cover
```

## 3 · Family 1 — Qualitative (categories)

```{figure} figures/palette-qualitative.jpg
:width: 100%
:alt: Five qualitative palettes on multi-class scatter demos

Five qualitative palettes on the honest test of a categorical palette — a multi-class
scatter, where every group must stay distinct.
```

Qualitative palettes carry **no order**, so the only requirements are that the hues be
distinguishable (including for colour-vision deficiency) and roughly equal in
weight — no single category should shout. The demo simply maps one colour per group:

```python
colors = ["#1F4E7A", "#E6AB4D", "#452A52", "#9A3F00", "#3E4796",
          "#D70006", "#276633", "#960448"]                      # "Jewel 8"
for i, cluster in enumerate(clusters):                          # one colour per class
    ax.scatter(cluster[:, 0], cluster[:, 1], color=colors[i],
               s=12, alpha=0.85, edgecolor="white", linewidth=0.2)
```

Rules of thumb:

- **≤ 8 categories.** Beyond that, hues stop being separable — merge classes or
  facet instead of adding colours.
- **Avoid red–green pairings** as the primary contrast (~8% of men can't split them);
  the *Clinical* and *Nature* sets keep red and green far apart in the ordering.
- **Reserve grey** for "other" / non-significant, not for a real category.

## 4 · Family 2 — Sequential (ordered magnitude)

```{figure} figures/palette-sequential.jpg
:width: 100%
:alt: Five sequential palettes on interpolated-surface demos

Five sequential palettes on an interpolated raster surface. The single rule that makes
a sequential palette work: **luminance increases monotonically** from one end to the
other, so higher values always look "more".
```

Sequential palettes encode magnitude, so light must map to low and dark to high (or the
reverse) **without reversal**. The demo is an interpolated field with contour overlay:

```python
colors = ["#F9F3D2", "#266292", "#372435"]                     # "Deep Ocean", light→dark
ax.imshow(surface, cmap=cmap(colors), origin="lower")
ax.contour(surface, levels=6, colors="k", linewidths=0.25, alpha=0.35)
```

Rules of thumb:

- **Monotone luminance.** If the middle is lighter or darker than an end, the eye reads
  a false peak. (This is why classic rainbow / `jet` is banned for magnitude — see §7.)
- **Multi-hue is fine** (*Ember*, *Amber Night*) as long as luminance still climbs — it
  even adds discriminative power, the trick behind viridis.
- **Start near-white** for print, so the lightest values don't disappear on paper.

## 5 · Family 3 — Diverging (data with a centre)

```{figure} figures/palette-diverging.jpg
:width: 100%
:alt: Five diverging palettes on a gene-signature dot plot

Five diverging palettes on a signature dot plot — the format where colour carries a
signed score and dot size carries significance. This is the natural home of a diverging
palette: data with a meaningful zero.
```

Diverging palettes have **two saturated ends and a neutral middle**, for data where the
centre means something — a correlation of 0, a z-score, an anomaly, an up/down-regulation
score. The demo echoes the journal dot plot: **size = −log₁₀(FDR), colour = score**:

```python
colors = ["#112D61", "#4472A8", "#FFFFFF", "#B0505A", "#660514"]   # "Navy-Crimson"
lim = 0.05                                                          # symmetric limits!
ax.scatter(xs, ys,
           s=12 + neglog10_fdr * 20,        # dot size  = significance
           c=score, cmap=cmap(colors),      # dot colour = signed score
           vmin=-lim, vmax=lim,             # centre the scale on 0
           edgecolor="grey", linewidth=0.3)
```

Rules of thumb:

- **Symmetric limits, always** (`vmin=-lim, vmax=+lim`). If the scale isn't centred, the
  neutral colour drifts off zero and the figure lies.
- **Equal-luminance, equal-saturation ends**, so neither pole dominates — *Navy-Crimson*
  and *Crimson-Azure* are tuned for this; a red-vs-pale-blue pairing is not balanced.
- **Never** use a diverging palette for data without a centre — it invents a midpoint.

## 6 · Family 4 — Geospatial map palettes

The hero figure. Maps are special because the palette family is chosen **per layer**, and
a few conventions are near-universal:

```python
# terrain (hypsometric tint): greens → tans → browns → snow-white
dem_cmap = cmap(["#2E6F40", "#86A96A", "#E4D6A0", "#C29A5B", "#8A5A2B", "#FFFFFF"])
ax.imshow(dem, cmap=dem_cmap, origin="lower")
ax.contour(dem, levels=7, colors="k", linewidths=0.25, alpha=0.3)

# land cover (categorical, semantic colours): water/forest/crop/bare/urban/cloud
lc_cmap = ListedColormap(["#4A90D9", "#2E7D32", "#C5E17A",
                          "#C2B280", "#B0413E", "#EDEDED"])
ax.imshow(landcover_ids, cmap=lc_cmap, origin="lower")           # ids are 0..n-1

# continuous scientific raster (LST, NDVI, pollutant): perceptually uniform
ax.imshow(field, cmap="viridis")        # or cmap(["#440154", ..., "#FDE725"])
```

Map-specific rules:

- **Terrain → hypsometric tints** (low green, high brown, peaks white). Readers decode
  elevation pre-attentively; don't fight the convention.
- **Continuous rasters → perceptually-uniform** maps (viridis, cividis, or Crameri's
  scientific colour maps). They keep equal data steps looking equal and survive
  greyscale + colour-blind printing. `cividis` is the CVD-safe default.
- **Land cover → semantic categorical** colours: water blue, vegetation green, built-up
  red/grey. A land-cover map in arbitrary hues is unreadable.
- **Change / anomaly → diverging**, centred on zero (warming red, cooling blue).
- **Depth / bathymetry → sequential blues**, dark = deep.

## 7 · The one hard rule: never rainbow a magnitude

The classic rainbow / `jet` colormap fails every test above: its luminance is
non-monotone (a bright yellow band invents a false ridge), and it is not colour-blind
safe. For any ordered or continuous quantity, reach for a **sequential** or
**perceptually-uniform** map instead. Rainbow survives only as a *qualitative* set of
well-separated categorical hues — never as a scale.

## 8 · Using palettes in R

The same families, two `ggplot2` scales:

```r
# continuous (sequential / diverging / terrain): interpolate a hex vector
scale_fill_gradientn(colours = c("#F9F3D2", "#266292", "#372435"))
scale_fill_gradient2(low = "#2166AC", mid = "#F7F7F7", high = "#B2182B",
                     midpoint = 0)                     # diverging, centred

# qualitative: map hues to discrete categories
scale_colour_manual(values = c("#1F4E7A", "#E6AB4D", "#452A52", "#9A3F00"))

# perceptually-uniform, built in
scale_fill_viridis_c()                                 # or option = "cividis"
```

## 9 · The palette library (extensible)

Every set on this page collected into one place — copy what you need, and the catalogue
grows as new palettes are added. Full rendering source:
[`palette_catalogue.py`](./code/palettes/palette_catalogue.py).

```python
PALETTES = {
  "qualitative": {
    "Jewel 8":  ["#1F4E7A","#E6AB4D","#452A52","#9A3F00","#3E4796","#D70006","#276633","#960448"],
    "Clinical": ["#00468B","#ED0000","#42B540","#0099B4","#925E9F","#FDAF91","#AD002A"],
    "Nature":   ["#E64B35","#4DBBD5","#00A087","#3C5488","#F39B7F","#8491B4","#91D1C2"],
    "Earthy":   ["#AD8632","#1E619C","#63B6BF","#77211D","#A8D2C2","#CBA5AE"],
    "Muted":    ["#6C8EBF","#C1846D","#7FA870","#9A7AA0","#D4B15F","#6BA3A0"],
  },
  "sequential": {
    "Amber Night": ["#FFF4C1","#F36352","#19141A"],
    "Deep Ocean":  ["#F9F3D2","#266292","#372435"],
    "Ember":       ["#FFFFFF","#FFEFB8","#F05A5F","#833584"],
    "Steel Blue":  ["#F7FBFF","#9ECAE1","#4292C6","#08519C","#08306B"],
    "Forest":      ["#F7FCF5","#A1D99B","#41AB5D","#238B45","#00441B"],
  },
  "diverging": {
    "Navy-Crimson":  ["#112D61","#4472A8","#FFFFFF","#B0505A","#660514"],
    "Crimson-Azure": ["#CB183C","#FFC587","#FFFFFF","#7FB8DA","#0370B3"],
    "Violet-Wine":   ["#7E33A9","#DC9EC8","#FFFFFF","#B07385","#5D2740"],
    "Earth-Teal":    ["#8C510A","#D8B365","#F6E8C3","#C7EAE5","#5AB4AC","#01665E"],
    "Magenta-Green": ["#C51B7D","#F1B6DA","#F7F7F7","#B8E186","#4D9221"],
  },
  "gis": {
    "Hypsometric":  ["#2E6F40","#86A96A","#E4D6A0","#C29A5B","#8A5A2B","#FFFFFF"],
    "Bathymetry":   ["#E1F5FE","#81D4FA","#29B6F6","#0277BD","#01467E","#002B4D"],
    "Land cover":   ["#4A90D9","#2E7D32","#C5E17A","#C2B280","#B0413E","#EDEDED"],
    "Temp anomaly": ["#2166AC","#92C5DE","#F7F7F7","#F4A582","#B2182B"],
    "Viridis":      ["#440154","#414487","#2A788E","#22A884","#7AD151","#FDE725"],
    "Cividis":      ["#00224E","#35456C","#666970","#978F78","#DECD5A","#FFEA46"],
  },
}
```

## 10 · Run it yourself

Full source: [`palette_catalogue.py`](./code/palettes/palette_catalogue.py) — needs
`numpy`, `matplotlib`, `scipy`.

```bash
python palette_catalogue.py
# → the four family overview figures on this page
```
