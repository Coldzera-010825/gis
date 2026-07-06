import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from matplotlib.colors import Normalize


try:
    from scipy.stats import spearmanr
except Exception:  # pragma: no cover - optional dependency
    spearmanr = None


FACILITIES = [
    "Education",
    "Food",
    "Shopping",
    "Living",
    "Sports and\nEntertainment",
    "Parks",
    "Culture and\nLeisure",
    "Public\nTransit",
    "Senior Care",
    "Healthcare",
]

SHORT_NAMES = ["EDU", "FOOD", "SHOP", "LIV", "SPORT", "PARK", "CULT", "TRANS", "SEN", "HLTH"]


def sign_log(values):
    return np.sign(values) * np.log1p(np.abs(values))


def simulate_residuals(n_cities=339, seed=20260629):
    rng = np.random.default_rng(seed)
    cities = [f"City{i:03d}" for i in range(1, n_cities + 1)]

    latent = rng.normal(size=(n_cities, 5))
    weights = np.array(
        [
            [0.85, -0.35, 0.10, 0.10, 0.35],
            [-0.65, 0.85, -0.10, -0.20, -0.15],
            [-0.45, 0.80, 0.15, -0.10, -0.10],
            [0.25, -0.20, 0.85, 0.20, 0.20],
            [-0.10, 0.25, 0.25, 0.70, -0.15],
            [0.20, -0.20, 0.45, 0.75, -0.10],
            [0.45, -0.25, 0.30, 0.70, 0.15],
            [0.25, -0.25, 0.50, 0.45, 0.05],
            [0.05, 0.00, 0.20, 0.05, 0.75],
            [0.45, -0.35, 0.55, -0.15, 0.65],
        ]
    )

    noise = rng.normal(scale=0.75, size=(n_cities, len(FACILITIES)))
    residual = latent @ weights.T + noise

    # Add a few high-magnitude city-facility anomalies so the bubble map has structure.
    for _ in range(90):
        i = rng.integers(0, n_cities)
        j = rng.integers(0, len(FACILITIES))
        residual[i, j] += rng.choice([-1, 1]) * rng.uniform(1.7, 3.3)

    residual = (residual - residual.mean(axis=0)) / residual.std(axis=0)
    residual = sign_log(residual * 1.8)
    return pd.DataFrame(residual, columns=FACILITIES, index=cities)


def corr_and_pvalue(x, y):
    if spearmanr is not None:
        r, p = spearmanr(x, y)
        return float(r), float(p)

    ranks = pd.DataFrame({"x": x, "y": y}).rank()
    r = float(ranks["x"].corr(ranks["y"]))
    return r, math.nan


def p_stars(p):
    if math.isnan(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def draw_violin_panel(ax, data):
    y_positions = np.arange(len(FACILITIES))
    values = [data[col].to_numpy() for col in FACILITIES]
    cmap = plt.get_cmap("PuOr_r")
    colors = [cmap(i / (len(FACILITIES) - 1)) for i in range(len(FACILITIES))]

    parts = ax.violinplot(
        values,
        positions=y_positions,
        vert=False,
        widths=0.72,
        showmeans=False,
        showmedians=True,
        showextrema=False,
    )

    for body, color in zip(parts["bodies"], colors):
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.35)
        body.set_linewidth(1.2)

    if "cmedians" in parts:
        parts["cmedians"].set_color("#4a3f62")
        parts["cmedians"].set_linewidth(1.0)

    rng = np.random.default_rng(42)
    for y, col, color in zip(y_positions, FACILITIES, colors):
        jitter = rng.normal(scale=0.065, size=len(data))
        ax.scatter(
            data[col],
            y + jitter,
            s=8,
            color=color,
            alpha=0.34,
            linewidths=0,
            zorder=3,
        )

    ax.axvline(0, color="#6b6b6b", linewidth=0.9)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([])
    ax.set_xlabel("Residual (Sign-Log)", fontsize=9)
    ax.set_ylabel("")
    ax.set_xlim(-2.4, 2.4)
    ax.invert_yaxis()
    ax.text(-0.12, 1.02, "(a)", transform=ax.transAxes, fontsize=16, fontweight="bold")
    ax.tick_params(axis="x", labelsize=8)


def draw_corr_panel(fig, outer_spec, data):
    n = len(FACILITIES)
    inner = outer_spec.subgridspec(n, n, wspace=0.04, hspace=0.04)
    norm = Normalize(vmin=-1, vmax=1)
    cmap = plt.get_cmap("PuOr")

    for i, row in enumerate(FACILITIES):
        for j, col in enumerate(FACILITIES):
            ax = fig.add_subplot(inner[i, j])

            if i == j:
                vals = data[col].to_numpy()
                counts, edges = np.histogram(vals, bins=26, density=True)
                centers = (edges[:-1] + edges[1:]) / 2
                counts = counts / counts.max() if counts.max() else counts
                ax.fill_between(centers, 0, counts, color="#8064a2", alpha=0.22)
                ax.plot(centers, counts, color="#6a4c93", linewidth=1.1)
                ax.set_xlim(data.min().min(), data.max().max())
                ax.set_ylim(0, 1.1)
            elif i > j:
                ax.scatter(
                    data[col],
                    data[row],
                    s=5,
                    color="#e7a53b",
                    alpha=0.45,
                    linewidths=0,
                )
                ax.set_xlim(data.min().min(), data.max().max())
                ax.set_ylim(data.min().min(), data.max().max())
            else:
                r, p = corr_and_pvalue(data[col], data[row])
                ax.set_facecolor(cmap(norm(r)))
                ax.text(
                    0.5,
                    0.56,
                    f"{r:.3f}\n{p_stars(p)}",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=6.7,
                    color="#1f1f1f",
                )

            ax.set_xticks([])
            ax.set_yticks([])

            if i == n - 1:
                ax.set_xlabel(col, rotation=45, ha="right", fontsize=7)
            if j == 0:
                ax.set_ylabel(row, rotation=0, ha="right", va="center", fontsize=7)

            for spine in ax.spines.values():
                spine.set_color("#cfcfcf")
                spine.set_linewidth(0.65)

    fig.text(0.318, 0.948, "(b)", fontsize=16, fontweight="bold")

    cax = fig.add_axes([0.922, 0.585, 0.014, 0.315])
    scalar = cm.ScalarMappable(norm=norm, cmap=cmap)
    scalar.set_array([])
    cbar = fig.colorbar(scalar, cax=cax)
    cbar.set_label("Spearman correlation", fontsize=8)
    cbar.ax.tick_params(labelsize=7)


def draw_bubble_panel(ax, data, n_shown=72):
    # Pick representative cities: strongest positive/negative anomalies plus a random middle sample.
    z = (data - data.mean()) / data.std()
    strength = z.abs().max(axis=1)
    top = list(strength.nlargest(n_shown // 2).index)
    middle_pool = strength.sort_values().index[n_shown : n_shown + 180]
    rng = np.random.default_rng(29)
    middle = list(rng.choice(middle_pool, size=n_shown - len(top), replace=False))
    city_order = top + middle
    rng.shuffle(city_order)

    selected = z.loc[city_order, FACILITIES]
    y_order = SHORT_NAMES[::-1]
    y_lookup = {name: i for i, name in enumerate(y_order)}

    xs = []
    ys = []
    vals = []
    sizes = []

    for x, city in enumerate(city_order):
        for facility, short in zip(FACILITIES, SHORT_NAMES):
            value = float(selected.loc[city, facility])
            xs.append(x)
            ys.append(y_lookup[short])
            vals.append(value)
            sizes.append(8 + min(abs(value), 3.0) ** 1.7 * 19)

    scatter = ax.scatter(
        xs,
        ys,
        s=sizes,
        c=vals,
        cmap="PuOr",
        vmin=-2.6,
        vmax=2.6,
        alpha=0.86,
        linewidths=0,
    )

    ax.set_xlim(-0.8, len(city_order) - 0.2)
    ax.set_ylim(-0.8, len(y_order) - 0.2)
    ax.set_xticks(np.arange(len(city_order)))
    ax.set_xticklabels(city_order, rotation=82, ha="right", fontsize=6.5)
    ax.set_yticks(np.arange(len(y_order)))
    ax.set_yticklabels(y_order, fontsize=8)
    ax.grid(True, color="#eeeeee", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.text(-0.03, 1.03, "(c)", transform=ax.transAxes, fontsize=16, fontweight="bold")

    cbar = plt.colorbar(scatter, ax=ax, fraction=0.018, pad=0.012)
    cbar.set_label("Residual column z-score", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    legend_sizes = [8 + v**1.7 * 19 for v in [0.5, 1, 2, 3]]
    for size, label in zip(legend_sizes, ["|z|=0.5", "|z|=1", "|z|=2", "|z|=3"]):
        ax.scatter([], [], s=size, color="#777777", alpha=0.55, label=label)
    ax.legend(
        title="Bubble size",
        loc="center left",
        bbox_to_anchor=(1.10, 0.42),
        frameon=False,
        fontsize=7,
        title_fontsize=8,
        borderaxespad=0,
    )


def main():
    data = simulate_residuals()

    fig = plt.figure(figsize=(13.8, 10.2), dpi=160)
    grid = fig.add_gridspec(
        2,
        2,
        width_ratios=[1.0, 2.12],
        height_ratios=[1.18, 1.0],
        wspace=0.14,
        hspace=0.30,
        left=0.07,
        right=0.86,
        top=0.94,
        bottom=0.12,
    )

    ax_violin = fig.add_subplot(grid[0, 0])
    draw_violin_panel(ax_violin, data)

    draw_corr_panel(fig, grid[0, 1], data)

    ax_bubble = fig.add_subplot(grid[1, :])
    draw_bubble_panel(ax_bubble, data)

    out_dir = Path(__file__).resolve().parent / "codex_outputs"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"{Path(__file__).stem}.png"
    fig.savefig(out_path, dpi=300)
    print(f"Saved figure to: {out_path}")


if __name__ == "__main__":
    main()
