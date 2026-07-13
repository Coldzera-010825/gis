---
title: "Piece 09 · Correlation heatmaps, from basic to Mantel networks"
description: Correlation visualization in two ecosystems — annotated/triangular/clustered heatmaps in Python, and the ecology-standard Mantel-network figure in R with linkET (the maintained successor to ggcor).
---

# Correlation heatmaps

```{figure} figures/corr-linket-mantel.jpg
:width: 100%
:alt: A Mantel-test network: soil-chemistry correlation heatmap with linked species blocks

The finished figure: the ecology-standard **Mantel network**. A correlation heatmap of
soil-chemistry variables (square size + fill = Pearson r) on the right; on the left,
four species blocks linked to the chemistry by curved couples whose colour encodes the
Mantel-test p-value and whose width encodes the Mantel r.
```

## 1 · Why this figure

A correlation heatmap is the fastest way to show "what moves with what." But the
research-grade version answers a harder question — *do whole blocks of one dataset
(species communities) relate to variables in another (soil chemistry)?* — using a
**Mantel test**, and draws both the within-dataset correlations and the
between-dataset links in one figure. This piece climbs from the plain heatmap
(Python) to that Mantel network (R).

## 2 · Python track, level 1 — the annotated heatmap

The workhorse: `seaborn.heatmap` on a `DataFrame.corr()`, diverging colormap centred at
zero, square cells, annotated. The simulated data is an environmental table (climate /
soil / vegetation variables) with a latent-factor correlation structure:

```python
corr = df.corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1,
            center=0, square=True, linewidths=0.5)
```

```{figure} figures/corr-sub-basic.jpg
:width: 60%
:alt: Full annotated correlation heatmap

The full matrix. Three rules make it paper-grade: a **diverging** colormap (never
sequential — zero must be neutral), `vmin/vmax = ±1` (fixed scale, comparable across
figures), and `center=0`.
```

## 3 · Python track, level 2 — triangle + significance

A correlation matrix is symmetric, so half of it is redundant ink. Mask one triangle,
and annotate the other with **significance stars** — because a correlation without its
p-value is not evidence:

```python
mask = np.triu(np.ones_like(corr, dtype=bool))       # hide the upper triangle
labels = corr.round(2).astype(str) + pvals.map(stars)  # "0.72***"
sns.heatmap(corr, mask=mask, annot=labels, fmt="", cmap="RdBu_r", center=0)
```

```{figure} figures/corr-sub-triangle.jpg
:width: 60%
:alt: Lower-triangle correlation heatmap with significance stars

Lower triangle only, with `*`/`**`/`***` at p < .05/.01/.001. Half the ink, twice the
information.
```

## 4 · Python track, level 3 — the clustermap

Ordering matters: a correlation heatmap in arbitrary column order hides its blocks.
`seaborn.clustermap` hierarchically reorders rows and columns so correlated variables
sit together, with dendrograms showing the grouping — the same idea as Piece 08's
clustermap, applied to the correlation matrix itself:

```python
sns.clustermap(corr, cmap="RdBu_r", vmin=-1, vmax=1, center=0,
               annot=True, fmt=".2f", dendrogram_ratio=(0.12, 0.12))
```

```{figure} figures/corr-sub-clustermap.jpg
:width: 62%
:alt: Hierarchically clustered correlation heatmap

The dendrograms surface the structure automatically: the soil-fertility block
(Organic C, Total N, pH) and the climate block (Temp, Rainfall, Humidity) fall out as
neighbours without being told.
```

## 5 · R track — a note on ggcor → linkET

The classic "correlation heatmap + Mantel links" figure was popularised by the R
package **ggcor**. ggcor is now **unmaintained** and breaks under modern ggplot2
(≥ 3.5): its `geom_square()` fails inside ggplot2's stricter `resolve_rect()`.

The fix is not to pin an old ggplot2 — it is **[linkET](https://github.com/Hy4m/linkET)**,
the maintained rewrite **by the same author** (Houyun Huang). The API maps almost
one-to-one:

| ggcor (unmaintained) | linkET (maintained) |
| --- | --- |
| `quickcor(x)` | `qcorrplot(correlate(x))` |
| `geom_square()` *(broken on ggplot2 4.x)* | `geom_square()` *(works)* |
| `anno_link()` | `geom_couple()` |
| `spec.select` / `p.value` | `spec_select` / `p` |

Install it from GitHub (pure R, no compiler needed):

```r
remotes::install_github("Hy4m/linkET")
```

## 6 · R track, level 1 — the linkET heatmap

`qcorrplot()` draws the signature look: each cell a **square whose size and fill both
encode r**, with `geom_mark()` overprinting the value and significance stars:

```r
library(linkET)
qcorrplot(correlate(varechem), type = "upper", diag = FALSE) +
  geom_square() +
  geom_mark(sig_level = c(0.05, 0.01, 0.001), mark = c("*", "**", "***")) +
  scale_fill_gradientn(colours = RdBu, limits = c(-1, 1))
```

```{figure} figures/corr-linket-heat.jpg
:width: 62%
:alt: linkET qcorrplot square heatmap of soil chemistry

Square *size* encodes |r| on top of fill colour — a redundant encoding that survives
greyscale printing, the linkET signature. Data: vegan's `varechem` soil chemistry.
```

## 6.5 · The Mantel test in one paragraph

A **Mantel test** measures whether two distance matrices — computed over the *same
samples* — are correlated. Here: does the community-composition distance between plots
(from a block of species) track their soil-chemistry distance? It answers "does this
species group respond to the environment?" with a single r and p per group — which is
exactly what the couples in the hero figure encode.

## 7 · R track, level 2 — the Mantel network (the hero)

Group the species into blocks, Mantel-test each against the chemistry, bin r and p into
readable classes, then draw the couples on top of the heatmap with `geom_couple()`:

```r
mantel <- mantel_test(varespec, varechem,
    spec_select = list(Spec01 = 1:7, Spec02 = 8:18,
                       Spec03 = 19:37, Spec04 = 38:44)) |>
  mutate(rd = cut(r, c(-Inf, 0.2, 0.4, Inf), labels = c("< 0.2", "0.2-0.4", ">= 0.4")),
         pd = cut(p, c(-Inf, 0.01, 0.05, Inf), labels = c("< 0.01", "0.01-0.05", ">= 0.05")))

qcorrplot(correlate(varechem), type = "upper", diag = FALSE) +
  geom_square() +
  geom_couple(aes(colour = pd, size = rd), data = mantel, curvature = 0.1) +
  scale_size_manual(values = c(0.5, 1.2, 2.2)) +
  scale_colour_manual(values = c("#D95F02", "#1B9E77", "grey70"))
```

The result is the figure at the top: heatmap on the right, four species blocks on the
left, each couple's **colour = Mantel p** (significant links stand out in colour, the
non-significant majority recede to grey) and **width = Mantel r**. It is dense but every
channel is doing work — the reason this single figure has become an ecology-paper
staple.

## 8 · Design notes

- **Diverging colormap, fixed ±1, centred at 0** — for every correlation figure, in
  either language.
- **Never show r without p.** Stars (Python) or couple colour (linkET) carry the
  significance; a bare matrix overstates weak correlations.
- **Order is information.** Cluster the matrix (Python `clustermap`, or linkET's
  reordering) so blocks are visible.
- **Prefer the maintained package.** When a beloved tool (ggcor) breaks, look for the
  author's successor (linkET) before pinning old dependencies.

## 9 · Run it yourself

Python: [`corr_python.py`](./code/corrheatmap/corr_python.py) (`seaborn`, `scipy`).
R: [`corr_linket.R`](./code/corrheatmap/corr_linket.R) (`linkET`, `vegan`, `dplyr`).

```bash
python corr_python.py           # the three Python heatmaps
Rscript corr_linket.R           # the two linkET figures
```

For a broad, example-driven tour of R plotting (including correlation and heatmap
packages), see also:
[**R语言绘图总汇 — 覆盖教程所有图形**](https://zhuanlan.zhihu.com/p/555785174).
