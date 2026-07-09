---
title: "Piece 03 · Facility-equity residual diagnostics"
description: A three-panel composite — violins, a hybrid correlation matrix and a bubble matrix — for reading facility-provision residuals across 339 cities, deconstructed panel by panel.
---

# Facility-equity residual diagnostics

```{figure} figures/facility-main.jpg
:width: 100%
:alt: Facility residual visualization, main figure

The finished composite. **(a)** violin + swarm distributions of provision residuals for
ten facility types; **(b)** a hybrid correlation matrix — scatter below the diagonal,
densities on it, Spearman ρ heat cells above; **(c)** a city × facility bubble matrix
that pinpoints which city is anomalous in which amenity.
```

## 1 · Why this figure

A common question in urban-equity work: after controlling for city size and wealth,
**which cities have more (or fewer) schools, hospitals, parks… than expected?** The
model's *residuals* carry the answer, but a residual table with 339 cities × 10
facility types is unreadable. This composite answers three questions at once:

| Panel | Question |
| --- | --- |
| (a) violins | *How are residuals distributed per facility — symmetric? heavy-tailed? biased?* |
| (b) correlation matrix | *Do amenities over/under-provide together?* (e.g. do food & shopping track each other?) |
| (c) bubble matrix | *Which specific city deviates in which specific facility?* |

That order — distribution → association → case-level anomaly — mirrors exactly how a
referee reads a residual analysis.

## 2 · The data

The tutorial version simulates residuals with a realistic correlation structure: five
latent city factors (think "economic level", "compactness", "ageing"…) load onto ten
facilities, plus noise and a handful of injected city-level anomalies:

```python
FACILITIES = ["Education", "Food", "Shopping", "Living",
              "Sports and\nEntertainment", "Parks", "Culture and\nLeisure",
              "Public\nTransit", "Senior Care", "Healthcare"]

def simulate_residuals(n_cities=339, seed=20260629):
    rng = np.random.default_rng(seed)
    latent = rng.normal(size=(n_cities, 5))          # 5 latent city factors
    weights = np.array([...])                        # 10 × 5 loading matrix
    noise = rng.normal(scale=0.75, size=(n_cities, len(FACILITIES)))
    residual = latent @ weights.T + noise

    # inject ~90 high-magnitude city-facility anomalies
    for _ in range(90):
        i, j = rng.integers(0, n_cities), rng.integers(0, len(FACILITIES))
        residual[i, j] += rng.choice([-1, 1]) * rng.uniform(1.7, 3.3)

    residual = (residual - residual.mean(axis=0)) / residual.std(axis=0)
    residual = sign_log(residual * 1.8)              # tame the heavy tails
    return pd.DataFrame(residual, columns=FACILITIES)
```

The one transform worth stealing is **sign-log**: real residuals have heavy tails, and
a linear axis wastes 80 % of its range on a few extremes. Sign-log compresses magnitude
while preserving sign:

```python
def sign_log(values):
    return np.sign(values) * np.log1p(np.abs(values))
```

## 3 · Panel (a): violin + swarm distributions

Horizontal violins (one per facility), each overlaid with the raw cities as a jittered
swarm — the violin gives shape, the swarm gives honesty (n is visible, outliers are
individually visible):

```python
def draw_violin_panel(ax, data):
    y_positions = np.arange(len(FACILITIES))
    values = [data[col].to_numpy() for col in FACILITIES]
    cmap = plt.get_cmap("PuOr_r")
    colors = [cmap(i / (len(FACILITIES) - 1)) for i in range(len(FACILITIES))]

    parts = ax.violinplot(values, positions=y_positions, vert=False,
                          widths=0.72, showmedians=True, showextrema=False)
    for body, color in zip(parts["bodies"], colors):
        body.set_facecolor(color); body.set_edgecolor(color)
        body.set_alpha(0.35)

    rng = np.random.default_rng(42)
    for y, col, color in zip(y_positions, FACILITIES, colors):
        jitter = rng.normal(scale=0.065, size=len(data))
        ax.scatter(data[col], y + jitter, s=8, color=color,
                   alpha=0.34, linewidths=0, zorder=3)

    ax.axvline(0, color="#6b6b6b", linewidth=0.9)   # the "as expected" reference
    ax.invert_yaxis()
```

```{figure} figures/facility-sub-violin.jpg
:width: 62%
:alt: Panel (a) rendered alone — violins with jittered city swarms

Panel (a) on its own. Each violin is one facility's residual distribution across 339
cities; the jittered dots are the cities themselves. Everything right of the zero line
is over-provision.
```

Reading aids that cost three lines but carry the panel:

- the **vertical zero line** — everything right of it is over-provision, left is
  under-provision;
- **`invert_yaxis()`** so the facility order matches panel (c)'s y-axis;
- violin alpha at 0.35 so the swarm points stay visible *through* the body.

## 4 · Panel (b): the hybrid correlation matrix

A 10 × 10 grid of tiny axes built with a **nested gridspec**, using each triangle for a
different encoding — scatter (raw evidence) below the diagonal, distribution on it, and
a coloured Spearman-ρ cell (the statistic) above:

```python
def draw_corr_panel(fig, outer_spec, data):
    n = len(FACILITIES)
    inner = outer_spec.subgridspec(n, n, wspace=0.04, hspace=0.04)
    norm = Normalize(vmin=-1, vmax=1)
    cmap = plt.get_cmap("PuOr")

    for i, row in enumerate(FACILITIES):
        for j, col in enumerate(FACILITIES):
            ax = fig.add_subplot(inner[i, j])
            if i == j:          # diagonal: density silhouette
                counts, edges = np.histogram(data[col], bins=26, density=True)
                centers = (edges[:-1] + edges[1:]) / 2
                ax.fill_between(centers, 0, counts / counts.max(),
                                color="#8064a2", alpha=0.22)
            elif i > j:         # lower triangle: raw scatter
                ax.scatter(data[col], data[row], s=5,
                           color="#e7a53b", alpha=0.45, linewidths=0)
            else:               # upper triangle: ρ as a coloured cell
                r, p = corr_and_pvalue(data[col], data[row])
                ax.set_facecolor(cmap(norm(r)))
                ax.text(0.5, 0.56, f"{r:.3f}\n{p_stars(p)}",
                        ha="center", va="center", transform=ax.transAxes,
                        fontsize=6.7)
            ax.set_xticks([]); ax.set_yticks([])
```

Significance stars are the usual thresholds:

```python
def p_stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
```

```{figure} figures/facility-sub-corr.jpg
:width: 92%
:alt: Panel (b) rendered alone — the hybrid correlation matrix

Panel (b) on its own: raw scatter below the diagonal, density silhouettes on it,
Spearman ρ heat cells with significance stars above — three encodings of the same
pairwise structure in one matrix.
```

:::{note}
**Why Spearman, not Pearson?** Residuals after sign-log are still not guaranteed
Gaussian, and equity questions care about *monotone* co-provision ("cities generous in
X tend to be generous in Y"), not linearity. Rank correlation is the honest default —
with a plain-`pandas` rank fallback in the script when `scipy` is absent.
:::

## 5 · Panel (c): the bubble matrix

339 city columns would be unreadable, so the panel **curates** its x-axis: the ~36
strongest anomalies plus a random slice of mid-range cities, shuffled so the eye
doesn't read a false gradient:

```python
def draw_bubble_panel(ax, data, n_shown=72):
    z = (data - data.mean()) / data.std()
    strength = z.abs().max(axis=1)
    top = list(strength.nlargest(n_shown // 2).index)      # loudest anomalies
    middle_pool = strength.sort_values().index[n_shown : n_shown + 180]
    middle = list(rng.choice(middle_pool, size=n_shown - len(top), replace=False))
    city_order = top + middle
    rng.shuffle(city_order)
```

Each city × facility cell becomes one bubble; **colour carries sign and magnitude,
size carries magnitude only** (redundant encoding — deliberately, so the pattern
survives greyscale printing):

```python
    sizes.append(8 + min(abs(value), 3.0) ** 1.7 * 19)   # perceptual size ramp

    scatter = ax.scatter(xs, ys, s=sizes, c=vals, cmap="PuOr",
                         vmin=-2.6, vmax=2.6, alpha=0.86, linewidths=0)
```

The `** 1.7` exponent over-drives area growth to compensate for the human tendency to
under-estimate area differences; the `min(..., 3.0)` cap stops one monster outlier from
dwarfing the panel. A size legend is faked with empty scatters:

```python
    for size, label in zip(legend_sizes, ["|z|=0.5", "|z|=1", "|z|=2", "|z|=3"]):
        ax.scatter([], [], s=size, color="#777777", alpha=0.55, label=label)
    ax.legend(title="Bubble size", loc="center left",
              bbox_to_anchor=(1.10, 0.42), frameon=False)
```

```{figure} figures/facility-sub-bubble.jpg
:width: 100%
:alt: Panel (c) rendered alone — the city by facility bubble matrix

Panel (c) on its own: 72 curated cities × 10 facilities. Colour carries sign and
magnitude, bubble size repeats magnitude — the anomalies pop out even in greyscale.
```

## 6 · Composition: one `GridSpec`, three panels

The layout is a 2 × 2 grid where the bubble matrix spans the full bottom row:

```python
fig = plt.figure(figsize=(13.8, 10.2), dpi=160)
grid = fig.add_gridspec(2, 2,
                        width_ratios=[1.0, 2.12],   # violins narrow, matrix wide
                        height_ratios=[1.18, 1.0],
                        wspace=0.14, hspace=0.30,
                        left=0.07, right=0.86, top=0.94, bottom=0.12)

ax_violin = fig.add_subplot(grid[0, 0]);  draw_violin_panel(ax_violin, data)
draw_corr_panel(fig, grid[0, 1], data)                    # nested subgridspec
ax_bubble = fig.add_subplot(grid[1, :]);  draw_bubble_panel(ax_bubble, data)
```

`right=0.86` reserves a margin strip for the two colourbars and the size legend, so
nothing overlaps the panels. Panel tags **(a) (b) (c)** are placed in axes/figure
coordinates — the journal-figure convention that lets captions reference panels.

## 7 · Palette note: one hue system for the whole figure

All three panels share the **PuOr** diverging colormap (purple ↔ orange):

- diverging is the *only* correct family for residuals — zero must be visually neutral;
- PuOr stays distinguishable for the two common colour-vision deficiencies, unlike
  red–green;
- panel (a) uses `PuOr_r` merely to index facilities along the same hue wheel, so the
  figure feels like one system rather than three charts glued together.

## 8 · Run it yourself

Full source: [`facility_residual_visualization.py`](./code/facility_residual_visualization.py)
— needs `numpy`, `pandas`, `matplotlib`; `scipy` optional (Spearman p-values). The
per-panel figures on this page are rendered by [`render_subfigures.py`](./code/render_subfigures.py).

```bash
python facility_residual_visualization.py
# → writes facility_residual_visualization.png into ./codex_outputs/
```
