"""Render each tutorial sub-plot as a standalone figure.

Reuses the plotting helpers from the two main scripts so the tutorial's
per-section code is shown with its actual visual output.

Outputs (into ../figures):
  lollipop-sub-heatmap.jpg    Piece 02 / sub-plot A: categorical heatmap rings
  lollipop-sub-lollipop.jpg   Piece 02 / sub-plot B: continuous lollipop rings
  lollipop-sub-summaries.jpg  Piece 02 / sub-plot C: marginal summaries
  facility-sub-violin.jpg     Piece 03 / panel (a): violin + swarm
  facility-sub-corr.jpg       Piece 03 / panel (b): hybrid correlation matrix
  facility-sub-bubble.jpg     Piece 03 / panel (c): bubble matrix
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import circular_lollipop_heatmap as cl
import facility_residual_visualization as fr

OUT = Path(__file__).resolve().parent.parent / "figures"
OUT.mkdir(exist_ok=True)


def save(fig, name):
    fig.savefig(OUT / name, dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {name}")


# ---------------------------------------------------------------- Piece 02
df = cl.simulate_data(n=64)
colors = cl.COLOR_SCHEMES[1]
rings = cl.build_rings(ring_thickness=1.15, ring_gap=0.18)
n = len(df)
edges = np.linspace(90, -180, n + 1)
centers = (edges[:-1] + edges[1:]) / 2

CONTINUOUS_SPECS = [
    ("Area", "Area", (0, 1.5), [0.3, 0.6, 1.0, 1.3]),
    ("Education", "Education", (-2, 17), [0, 5, 10, 15]),
    ("Age", "Age", (20, 80), [30, 50, 70]),
]


def polar_canvas(outer_r):
    fig = plt.figure(figsize=(9, 9), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(-outer_r - 0.7, outer_r + 0.7)
    ax.set_ylim(-outer_r - 0.7, outer_r + 0.7)
    return fig, ax


# --- sub-plot A: the two categorical heatmap rings alone
fig, ax = polar_canvas(rings["Training"][1])
for i, row in df.reset_index(drop=True).iterrows():
    theta1, theta2 = edges[i + 1], edges[i]
    m_col = colors[f"Mach_{int(row['Machines'])}"]
    cl.add_wedge(ax, rings["Machines"][0], rings["Machines"][1], theta1, theta2, m_col, lw=0.23)
    t_col = colors["Train_Yes"] if int(row["Training"]) == 1 else colors["Train_No"]
    cl.add_wedge(ax, rings["Training"][0], rings["Training"][1], theta1, theta2, t_col, lw=0.23)
ax.text(0, 0, "Machines (inner)\nTraining (outer)", ha="center", va="center", fontsize=12, color="#444444")
save(fig, "lollipop-sub-heatmap.jpg")

# --- sub-plot B: the three continuous lollipop rings alone
fig, ax = polar_canvas(rings["Age"][1])
for ring_name, _, limits, ticks in CONTINUOUS_SPECS:
    r_in, r_out = rings[ring_name]
    cl.add_ring_guides(ax, r_in, r_out)
    cl.add_tick_labels(ax, ring_name, rings, limits, ticks, colors[ring_name])
for i, row in df.reset_index(drop=True).iterrows():
    theta1, theta2 = edges[i + 1], edges[i]
    theta_c = np.deg2rad(centers[i])
    for ring_name, data_col, limits, _ in CONTINUOUS_SPECS:
        r_in, r_out = rings[ring_name]
        cl.add_wedge(ax, r_in, r_out, theta1, theta2, "none", edgecolor=colors["Grid"], lw=0.25, zorder=0)
        r_val = cl.value_to_radius(float(row[data_col]), limits, (r_in, r_out))
        ax.plot([r_in * np.cos(theta_c), r_val * np.cos(theta_c)],
                [r_in * np.sin(theta_c), r_val * np.sin(theta_c)],
                color=colors[ring_name], lw=1.25, solid_capstyle="round", zorder=3)
        ax.plot(r_val * np.cos(theta_c), r_val * np.sin(theta_c), "o",
                color=colors[ring_name], markersize=3.7,
                markeredgecolor="white", markeredgewidth=0.25, zorder=4)
ax.text(0, 0, "Area (inner)\nEducation (middle)\nAge (outer)", ha="center", va="center", fontsize=12, color="#444444")
save(fig, "lollipop-sub-lollipop.jpg")

# --- sub-plot C: the marginal summaries as a standalone row
fig, axes = plt.subplots(1, 5, figsize=(12, 4.2), facecolor="white",
                         gridspec_kw={"wspace": 0.55})
cl.create_vertical_swarm(axes[0], df["Age"], colors["Age"], "Age", "years", [20, 80], [30, 40, 50, 60, 70])
cl.create_vertical_swarm(axes[1], df["Education"], colors["Education"], "Education", "years", [-2, 17], [0, 5, 10, 15])
cl.create_vertical_swarm(axes[2], df["Area"], colors["Area"], "Area", "ha", [0, 1.5], [0.3, 0.6, 1.0, 1.3])
train_counts = df["Training"].value_counts(normalize=True).mul(100)
cl.create_stacked_percent_bar(axes[3], [
    ("0", train_counts.get(0, 0), colors["Train_No"]),
    ("1", train_counts.get(1, 0), colors["Train_Yes"]),
], "Training(%)")
mach_counts = df["Machines"].value_counts(normalize=True).mul(100)
cl.create_stacked_percent_bar(axes[4], [
    (str(i), mach_counts.get(i, 0), colors[f"Mach_{i}"]) for i in [1, 2, 3, 4, 5]
], "Machines(%)")
save(fig, "lollipop-sub-summaries.jpg")

# ---------------------------------------------------------------- Piece 03
data = fr.simulate_residuals()

# --- panel (a): violin + swarm standalone
fig = plt.figure(figsize=(6.4, 7.2), facecolor="white")
ax = fig.add_axes([0.30, 0.09, 0.66, 0.86])
fr.draw_violin_panel(ax, data)
ax.set_yticklabels(fr.FACILITIES, fontsize=8)
save(fig, "facility-sub-violin.jpg")

# --- panel (b): hybrid correlation matrix standalone
from matplotlib import cm
from matplotlib.colors import Normalize

fig = plt.figure(figsize=(9.6, 8.4), facecolor="white")
grid = fig.add_gridspec(1, 1, left=0.10, right=0.88, top=0.94, bottom=0.10)
nfac = len(fr.FACILITIES)
inner = grid[0, 0].subgridspec(nfac, nfac, wspace=0.04, hspace=0.04)
norm = Normalize(vmin=-1, vmax=1)
cmap = plt.get_cmap("PuOr")
for i, row_name in enumerate(fr.FACILITIES):
    for j, col_name in enumerate(fr.FACILITIES):
        ax = fig.add_subplot(inner[i, j])
        if i == j:
            vals = data[col_name].to_numpy()
            counts, bin_edges = np.histogram(vals, bins=26, density=True)
            cs = (bin_edges[:-1] + bin_edges[1:]) / 2
            counts = counts / counts.max() if counts.max() else counts
            ax.fill_between(cs, 0, counts, color="#8064a2", alpha=0.22)
            ax.plot(cs, counts, color="#6a4c93", linewidth=1.1)
            ax.set_xlim(data.min().min(), data.max().max())
            ax.set_ylim(0, 1.1)
        elif i > j:
            ax.scatter(data[col_name], data[row_name], s=5, color="#e7a53b", alpha=0.45, linewidths=0)
            ax.set_xlim(data.min().min(), data.max().max())
            ax.set_ylim(data.min().min(), data.max().max())
        else:
            r, p = fr.corr_and_pvalue(data[col_name], data[row_name])
            ax.set_facecolor(cmap(norm(r)))
            ax.text(0.5, 0.56, f"{r:.3f}\n{fr.p_stars(p)}", ha="center", va="center",
                    transform=ax.transAxes, fontsize=6.7, color="#1f1f1f")
        ax.set_xticks([])
        ax.set_yticks([])
        if i == nfac - 1:
            ax.set_xlabel(col_name, rotation=45, ha="right", fontsize=7)
        if j == 0:
            ax.set_ylabel(row_name, rotation=0, ha="right", va="center", fontsize=7)
        for spine in ax.spines.values():
            spine.set_color("#cfcfcf")
            spine.set_linewidth(0.65)
cax = fig.add_axes([0.905, 0.30, 0.015, 0.42])
scalar = cm.ScalarMappable(norm=norm, cmap=cmap)
scalar.set_array([])
cbar = fig.colorbar(scalar, cax=cax)
cbar.set_label("Spearman correlation", fontsize=8)
cbar.ax.tick_params(labelsize=7)
save(fig, "facility-sub-corr.jpg")

# --- panel (c): bubble matrix standalone
fig = plt.figure(figsize=(13.2, 4.6), facecolor="white")
ax = fig.add_axes([0.05, 0.24, 0.82, 0.70])
fr.draw_bubble_panel(ax, data)
save(fig, "facility-sub-bubble.jpg")

print("done")
