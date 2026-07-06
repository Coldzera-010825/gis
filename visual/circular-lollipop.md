---
title: "Piece 02 · Circular lollipop + categorical heatmap"
description: One polar figure that carries three continuous variables, two categorical variables and their marginal summaries — a full code deconstruction with eight palette variants.
---

# Circular lollipop + categorical heatmap

```{figure} figures/lollipop-main.jpg
:width: 100%
:alt: Circular lollipop + categorical heatmap, main figure

The finished figure: 64 samples wrap around a shared polar axis. Three outer **lollipop
rings** encode continuous variables (Age, Education, Area); two inner **heatmap rings**
encode categorical variables (Training, Machines). Box-plot and stacked-bar summaries
dock on the left, and the hole in the middle holds the legend.
```

## 1 · Why this figure

Survey and household data are awkward to plot: every sample carries **numeric and
categorical attributes at once** (a farmer has an age, an education level, a plot area,
a yes/no training record, a machine count…). The usual answer is five separate panels —
which breaks the most important thing, *sample identity*: you can no longer see that the
40-year-old with 15 years of schooling is the same person who owns 5 machines.

This figure keeps identity intact. Each sample owns one **angular slot**, and every ring
reads a different variable off that same slot:

- **Categorical → colour** (heatmap wedges, inner rings) — pre-attentive, no scale needed;
- **Continuous → radial length** (lollipop stems, outer rings) — position along a common
  scale, the most accurate visual channel after aligned position.

One glance connects five attributes of the same sample; one sweep around the circle
scans the whole cohort.

## 2 · The data

The dataset is simulated but shaped like real farm-household survey data — including a
realistic dependency (younger, better-educated farmers are more likely to have attended
training):

```python
def simulate_data(n=64, seed=20260629):
    rng = np.random.default_rng(seed)
    ids = [str(i) for i in range(1, n + 1)]

    age = np.clip(rng.normal(50, 13, n), 20, 80)
    education = np.clip(rng.normal(8.5, 4.0, n), -2, 17)
    area = np.clip(rng.gamma(shape=2.2, scale=0.32, size=n), 0, 1.5)
    training_prob = np.clip(0.22 + education / 25 + (age < 45) * 0.18, 0.1, 0.9)
    training = rng.binomial(1, training_prob)
    machines = rng.choice([1, 2, 3, 4, 5], n, p=[0.30, 0.24, 0.20, 0.16, 0.10])

    return pd.DataFrame({
        "ID": ids, "Age": age, "Education": education,
        "Area": area, "Training": training, "Machines": machines,
    })
```

Swapping in a real CSV is trivial — the plotting code only needs those six columns.

## 3 · Ring geometry: the figure's skeleton

Everything hangs on two small helpers. `build_rings()` stacks the five rings outward
from the centre (categorical rings innermost, continuous rings outside), and
`value_to_radius()` linearly maps a data value into a ring's radial band:

```python
def build_rings(ring_thickness, ring_gap):
    start = 2.05                          # radius of the central hole (legend space)
    rings = {}
    for name in ["Machines", "Training", "Area", "Education", "Age"]:
        rings[name] = (start, start + ring_thickness)
        start += ring_thickness + ring_gap
    return rings

def value_to_radius(value, value_limits, radius_limits):
    v_min, v_max = value_limits
    r_in, r_out = radius_limits
    value = np.clip(value, v_min, v_max)
    return r_in + (value - v_min) / (v_max - v_min) * (r_out - r_in)
```

The angular skeleton is just as simple. The circle is *not* fully closed — samples span
from 90° to −180° (three quarters of the circle), leaving the left quadrant free for the
marginal summaries:

```python
edges = np.linspace(90, -180, n + 1)      # n+1 boundaries → n angular slots
centers = (edges[:-1] + edges[1:]) / 2    # slot centres, one per sample
```

:::{tip}
Note the figure is drawn on a **plain Cartesian axis** (`ax.set_aspect("equal")`,
`ax.axis("off")`) rather than matplotlib's `projection="polar"`. Wedge patches and
`cos/sin` maths give far more layout freedom — insets, labels and the legend can be
placed anywhere.
:::

## 4 · Sub-plot A: the categorical heatmap rings

A "heatmap" on a circle is nothing more than one coloured `Wedge` per sample per ring.
The whole sub-chart is one helper plus two calls inside the sample loop:

```python
from matplotlib.patches import Wedge

def add_wedge(ax, r_in, r_out, theta1, theta2, color,
              edgecolor="black", lw=0.25, zorder=1):
    ax.add_patch(Wedge(
        center=(0, 0), r=r_out, theta1=theta1, theta2=theta2,
        width=r_out - r_in,               # ring thickness → annular wedge
        facecolor=color, edgecolor=edgecolor, linewidth=lw, zorder=zorder,
    ))

# inside the per-sample loop:
m_col = colors[f"Mach_{int(row['Machines'])}"]           # 5-step colour ramp
add_wedge(ax, rings["Machines"][0], rings["Machines"][1], theta1, theta2, m_col)

t_col = colors["Train_Yes"] if int(row["Training"]) == 1 else colors["Train_No"]
add_wedge(ax, rings["Training"][0], rings["Training"][1], theta1, theta2, t_col)
```

Design notes:

- **Machines** (ordinal, 1–5) uses a *sequential* five-colour ramp — the eye should read
  "more machines = deeper colour".
- **Training** (binary) uses two *contrasting* hues — a yes/no should never look like a
  gradient.
- The thin black wedge edges (`lw=0.25`) are what makes 64 tiny cells legible instead of
  mushy.

## 5 · Sub-plot B: the lollipop rings

Each continuous ring draws, per sample, a **stem** from the ring's inner radius out to
the value's radius, capped with a **dot**. Faint grid arcs and white-on-colour tick
labels make the ring readable as an actual axis:

```python
for ring_name, data_col, limits, _ in continuous_specs:
    r_in, r_out = rings[ring_name]
    # empty wedge = light grid cell behind each sample
    add_wedge(ax, r_in, r_out, theta1, theta2, "none",
              edgecolor=colors["Grid"], lw=0.25, zorder=0)

    r_val = value_to_radius(float(row[data_col]), limits, (r_in, r_out))
    theta_c = np.deg2rad(centers[i])

    # the stem …
    ax.plot([r_in * np.cos(theta_c), r_val * np.cos(theta_c)],
            [r_in * np.sin(theta_c), r_val * np.sin(theta_c)],
            color=colors[ring_name], lw=1.25, solid_capstyle="round", zorder=3)
    # … and the head
    ax.plot(r_val * np.cos(theta_c), r_val * np.sin(theta_c), "o",
            color=colors[ring_name], markersize=3.7,
            markeredgecolor="white", markeredgewidth=0.25, zorder=4)
```

Why lollipops instead of bars? At 64 samples a radial bar ring turns into a solid
colour band; the thin stem + dot keeps each sample's value readable and the dot gives
the eye a precise mark to compare against the grid arcs.

Radial tick labels are drawn once per ring at 94° (just past the first sample), as
white text on a colour-chip background so they survive crossing the stems:

```python
def add_tick_labels(ax, ring_name, rings, value_limits, ticks, color):
    r_in, r_out = rings[ring_name]
    for tick in ticks:
        radius = value_to_radius(tick, value_limits, (r_in, r_out))
        theta = np.deg2rad(94)
        ax.text(radius * np.cos(theta), radius * np.sin(theta), f"{tick:g}",
                ha="center", va="center", fontsize=7, color="white",
                bbox={"facecolor": color, "edgecolor": "none",
                      "pad": 1.2, "alpha": 0.82}, zorder=5)
```

## 6 · Sub-plot C: the marginal summaries

The gap left by the open quadrant hosts one summary per ring, aligned with its ring's
radius via `ax.inset_axes(..., transform=ax.transData)` — so the Age summary sits
exactly at the Age ring's radial distance:

```python
ax_age = ax.inset_axes([-rings["Age"][1], inset_y, inset_w, inset_h],
                       transform=ax.transData)
create_vertical_swarm(ax_age, df["Age"], colors["Age"], "Age", "years",
                      [20, 80], [30, 40, 50, 60, 70])
```

Continuous variables get a **box plot + jittered swarm** (distribution *and* raw
points):

```python
ax_sub.boxplot(data, positions=[0], widths=0.34, patch_artist=True,
               showfliers=False,
               boxprops={"facecolor": color, "alpha": 0.24,
                         "edgecolor": color, "linewidth": 1.1},
               medianprops={"color": color, "linewidth": 1.6})
jitter = rng.normal(0, 0.075, len(data))
ax_sub.scatter(jitter, data, s=12, color=color, alpha=0.52,
               edgecolor="white", linewidth=0.25, zorder=3)
```

Categorical variables get a **stacked percent bar** with in-bar labels:

```python
def create_stacked_percent_bar(ax_sub, counts, title):
    bottom = 0
    for label, percent, color in counts:
        ax_sub.bar(0, percent, bottom=bottom, width=0.8,
                   color=color, edgecolor="white", linewidth=0.5)
        if percent >= 8:                       # only label slices that fit
            ax_sub.text(0, bottom + percent / 2, f"{percent:.1f}",
                        ha="center", va="center", fontsize=9)
        bottom += percent
```

The same palette entries are reused (ring colour = summary colour), so the marginal
panels need no extra legend — the association is carried by colour alone.

## 7 · Composition: why it all goes in one figure

The assembly order in `create_plot()` matters less than the *shared coordinate logic*:

1. one Cartesian axis, aspect-equal, axes off;
2. angular slots assigned once (`edges`, `centers`) and reused by every ring;
3. categorical wedges first (zorder 1), grid cells behind (0), stems and dots on top
   (3–4), tick chips above all (5);
4. summaries and the centre legend as inset axes, so they inherit figure export
   settings for free.

Because every mark derives from the same `centers[i]`, sample identity is preserved
by construction — that is the entire argument for combining the charts rather than
faceting them.

## 8 · Palettes: one figure, eight skins

The script separates *all* colour decisions into a `COLOR_SCHEMES` dict — swapping the
whole look of the figure is a one-integer change (`create_plot(df, scheme_id=5)`):

```python
COLOR_SCHEMES = {
    1: {
        "Name": "teal-gold",
        "Age": "#294E5C", "Education": "#2FA69A", "Area": "#E8C15A",
        "Train_Yes": "#F4A259", "Train_No": "#E76F51",
        "Mach_1": "#294E5C", "Mach_2": "#2FA69A", "Mach_3": "#8FB681",
        "Mach_4": "#E8C15A", "Mach_5": "#F4A259",
        "Grid": "#D8D4CC",
    },
    # … 7 more schemes
}
```

Each scheme follows the same three rules:

- the three **continuous rings** get clearly separated hues (they are different
  variables, not a gradient);
- **Training** yes/no gets a strong hue contrast;
- **Machines 1–5** gets a five-step ramp that reads as ordered.

::::{grid} 2 2 4 4

:::{figure} figures/lollipop-scheme-01.jpg
:alt: teal-gold
01 · teal-gold
:::

:::{figure} figures/lollipop-scheme-02.jpg
:alt: wine-orange
02 · wine-orange
:::

:::{figure} figures/lollipop-scheme-03.jpg
:alt: coral-meadow
03 · coral-meadow
:::

:::{figure} figures/lollipop-scheme-04.jpg
:alt: emerald-violet
04 · emerald-violet
:::

:::{figure} figures/lollipop-scheme-05.jpg
:alt: ocean-magenta
05 · ocean-magenta
:::

:::{figure} figures/lollipop-scheme-06.jpg
:alt: forest-clay
06 · forest-clay
:::

:::{figure} figures/lollipop-scheme-07.jpg
:alt: royal-lime
07 · royal-lime
:::

:::{figure} figures/lollipop-scheme-08.jpg
:alt: slate-rose
08 · slate-rose
:::

::::

A few observations from comparing them:

- **teal-gold (01)** and **slate-rose (08)** are the safest for print — muted hues, the
  ramp stays legible in greyscale.
- **ocean-magenta (05)** borrows the classic `#003F5C → #FFA600` data-viz ramp; the
  Machines ring reads best of all eight.
- **coral-meadow (03)** splits warm (continuous) vs cool (categorical) — a useful trick
  when you want the two ring families to feel like two systems.
- High-chroma schemes like **royal-lime (07)** work on screens and slides but risk
  overpowering the thin stems on paper.

## 9 · Run it yourself

Full source: [`circular_lollipop_heatmap.py`](./code/circular_lollipop_heatmap.py) —
self-contained, only `numpy`, `pandas`, `matplotlib`.

```bash
python circular_lollipop_heatmap.py
# → writes the CSV + all eight scheme PNGs into ./codex_outputs/
```
