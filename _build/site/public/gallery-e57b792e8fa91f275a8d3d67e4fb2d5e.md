---
title: "Visual Gallery 🎨"
description: A growing collection of research-visualization tutorials — each piece deconstructs one publication-style figure into sub-plots, code and palette choices.
---

# Visual Gallery 🎨

A practice collection for **geospatial research visualization**. Each *piece* takes one
publication-style figure and rebuilds it from scratch: every sub-plot gets its own code
walkthrough, then everything is composed into the final layout — with notes on **why**
the design works and where it is useful in real research.

> The goal is rehearsal: building a vocabulary of figure types and colour palettes now,
> so the strongest of them can be applied to actual research output later.

## Pieces

| # | Tutorial | Tools | What it teaches |
| --- | --- | --- | --- |
| 01 | [3D research figures with ggcube](https://github.com/Coldzera-010825/coldzera.github.io/blob/main/assets/reports/ggcube-3d-viz.Rmd) *(R Markdown source; rendered report lives on my personal site, Projects → 09)* | R · ggplot2 · ggcube | 3D scatter / surfaces / bars / space–time paths with `coord_3d()` |
| 02 | [Circular lollipop + categorical heatmap](./circular-lollipop.md) | Python · matplotlib | Polar sample rings, wedge heatmaps, lollipop encoding, marginal summaries, 8 palettes |
| 03 | [Facility-equity residual diagnostics](./facility-residuals.md) | Python · matplotlib · scipy | Violin panel, hybrid correlation matrix, bubble matrix, `GridSpec` composition |

## How each tutorial is organised

1. **The figure** — what it shows and when you would reach for it.
2. **The data** — a simulated dataset shaped like the real research data.
3. **Sub-plots, one at a time** — each panel isolated with its own code and design notes.
4. **Composition** — how the panels are assembled into one figure.
5. **Palettes** — the colour decisions, with alternatives where relevant.

All scripts are self-contained (`numpy` / `pandas` / `matplotlib`, no exotic
dependencies) and each page links to the full source file.
