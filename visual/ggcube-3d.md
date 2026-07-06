---
title: "Piece 01 · 3D research figures with ggcube (R)"
description: An overview of the ggcube piece — five 3D figure types built with ggplot2 grammar in R. The full step-by-step tutorial lives as a rendered R Markdown report on my personal site.
---

# 3D research figures with ggcube

```{figure} figures/ggcube-dem.jpg
:width: 100%
:alt: Simulated DEM surface rendered with ggcube

A simulated DEM surface (`geom_surface_3d()` with hillshade lighting) — one of five 3D
figures the piece walks through.
```

## What this piece covers

This piece is the only R entry in the gallery (the others are Python). It explores
[**ggcube**](https://matthewkling.github.io/ggcube/) — an R package by Matthew Kling
that extends ggplot2 into three dimensions: keep `ggplot()`, `aes()` and the layer
grammar, add a `z` aesthetic and `coord_3d()`, and swap `geom_*()` for the matching
`geom_*_3d()`.

The tutorial rebuilds five figure types end to end, each mapped to a research use case:

| Figure | ggcube layer | Where it earns its place |
| --- | --- | --- |
| 3D scatter | `geom_point_3d()` | three-variable station / sample data |
| Multivariate scatter | `geom_point_3d()` + colour | adding a fourth variable honestly |
| DEM surface | `geom_surface_3d()` + `light()` | terrain, continuous fields, hillshading |
| 3D bars & columns | `geom_bar_3d()` / `geom_col_3d()` | site × month grids, 2D histograms |
| Space–time path | `geom_path_3d()` + `position_on_face()` | storm tracks, propagation, with a floor "shadow" |

plus a **stacked-surface** finale — three drought-index sheets offset along `z` in one
cube, faceted into 3 × 3 monthly small multiples:

```{figure} figures/ggcube-facets.jpg
:width: 88%
:alt: 3x3 small multiples, three stacked index layers per panel

Small multiples with three stacked index sheets per panel — `facet_wrap()` works
unchanged under `coord_3d()`.
```

## Key ideas worth stealing

- **One grammar, one learning curve.** Everything is ordinary ggplot2 code; the 3D
  behaviour comes only from `coord_3d(yaw, pitch, roll, dist, ratio)` and the `_3d`
  geoms — so themes, scales, legends and faceting all keep working.
- **Light is a scale.** `light(direction, mode, contrast)` gives surfaces terrain-style
  hillshading without leaving the plot pipeline.
- **Faces are free real estate.** `position_on_face("zmin")` projects a layer onto a
  cube face — used here to give the space–time path a ground shadow.
- **Stacked sheets need thin relief.** For multi-layer cubes, map colour to the value
  but height to `base + value × small_factor`, or upper sheets will occlude lower ones.

```{figure} figures/ggcube-path.jpg
:width: 88%
:alt: Space-time propagation path with ground shadow

A propagation trajectory in (lon, lat, time) space, with its projection drawn on the
floor face as a reading aid.
```

## Read the full tutorial

The complete walkthrough — all code, package installation, figure-by-figure notes and
the palette choices — is a rendered R Markdown report on my personal site:

**→ [Open the full report · Visualizing Geo-Science Results in 3D with ggcube](https://coldzera-010825.github.io/assets/reports/ggcube-3d-viz.html)**

- R Markdown source: [`ggcube-3d-viz.Rmd` on GitHub](https://github.com/Coldzera-010825/coldzera.github.io/blob/main/assets/reports/ggcube-3d-viz.Rmd)
- Package docs: [matthewkling.github.io/ggcube](https://matthewkling.github.io/ggcube/)
