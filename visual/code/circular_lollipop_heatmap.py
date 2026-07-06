from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle, Wedge


COLOR_SCHEMES = {
    1: {
        "Name": "teal-gold",
        "Age": "#294E5C",
        "Education": "#2FA69A",
        "Area": "#E8C15A",
        "Train_Yes": "#F4A259",
        "Train_No": "#E76F51",
        "Mach_1": "#294E5C",
        "Mach_2": "#2FA69A",
        "Mach_3": "#8FB681",
        "Mach_4": "#E8C15A",
        "Mach_5": "#F4A259",
        "Grid": "#D8D4CC",
    },
    2: {
        "Name": "wine-orange",
        "Age": "#6B0F43",
        "Education": "#B2072C",
        "Area": "#FF7F20",
        "Train_Yes": "#E85D04",
        "Train_No": "#0F4C5C",
        "Mach_1": "#6B0F43",
        "Mach_2": "#7F1D1D",
        "Mach_3": "#C2410C",
        "Mach_4": "#F2A900",
        "Mach_5": "#17A2A5",
        "Grid": "#D8D4CC",
    },
    3: {
        "Name": "coral-meadow",
        "Age": "#F94144",
        "Education": "#F3722C",
        "Area": "#F8961E",
        "Train_Yes": "#FF7F50",
        "Train_No": "#D90429",
        "Mach_1": "#90BE6D",
        "Mach_2": "#43AA8B",
        "Mach_3": "#4D908E",
        "Mach_4": "#577590",
        "Mach_5": "#277DA1",
        "Grid": "#D8D4CC",
    },
    4: {
        "Name": "emerald-violet",
        "Age": "#5F0F7B",
        "Education": "#EF476F",
        "Area": "#E85D04",
        "Train_Yes": "#45CDB2",
        "Train_No": "#12A66A",
        "Mach_1": "#5F0F7B",
        "Mach_2": "#7B2CBF",
        "Mach_3": "#9D4EDD",
        "Mach_4": "#C77DFF",
        "Mach_5": "#E0AAFF",
        "Grid": "#D8D4CC",
    },
    5: {
        "Name": "ocean-magenta",
        "Age": "#003F5C",
        "Education": "#BC5090",
        "Area": "#FFA600",
        "Train_Yes": "#FF6361",
        "Train_No": "#58508D",
        "Mach_1": "#003F5C",
        "Mach_2": "#2F4B7C",
        "Mach_3": "#665191",
        "Mach_4": "#A05195",
        "Mach_5": "#FFA600",
        "Grid": "#D8D4CC",
    },
    6: {
        "Name": "forest-clay",
        "Age": "#2D3A3A",
        "Education": "#C8553D",
        "Area": "#F2C14E",
        "Train_Yes": "#C8553D",
        "Train_No": "#4F6D7A",
        "Mach_1": "#124559",
        "Mach_2": "#598392",
        "Mach_3": "#AEC3B0",
        "Mach_4": "#EFF6E0",
        "Mach_5": "#C8553D",
        "Grid": "#D8D4CC",
    },
    7: {
        "Name": "royal-lime",
        "Age": "#3D155F",
        "Education": "#DF678C",
        "Area": "#E8D21D",
        "Train_Yes": "#7A9E7E",
        "Train_No": "#3D155F",
        "Mach_1": "#3D155F",
        "Mach_2": "#513B56",
        "Mach_3": "#7A9E7E",
        "Mach_4": "#B6C649",
        "Mach_5": "#E8D21D",
        "Grid": "#D8D4CC",
    },
    8: {
        "Name": "slate-rose",
        "Age": "#293241",
        "Education": "#EE6C4D",
        "Area": "#98C1D9",
        "Train_Yes": "#E0FBFC",
        "Train_No": "#3D5A80",
        "Mach_1": "#293241",
        "Mach_2": "#3D5A80",
        "Mach_3": "#98C1D9",
        "Mach_4": "#E0FBFC",
        "Mach_5": "#EE6C4D",
        "Grid": "#D8D4CC",
    }
}


def simulate_data(n=64, seed=20260629):
    rng = np.random.default_rng(seed)
    ids = [str(i) for i in range(1, n + 1)]

    age = np.clip(rng.normal(50, 13, n), 20, 80)
    education = np.clip(rng.normal(8.5, 4.0, n), -2, 17)
    area = np.clip(rng.gamma(shape=2.2, scale=0.32, size=n), 0, 1.5)
    training_prob = np.clip(0.22 + education / 25 + (age < 45) * 0.18, 0.1, 0.9)
    training = rng.binomial(1, training_prob)
    machines = rng.choice([1, 2, 3, 4, 5], n, p=[0.30, 0.24, 0.20, 0.16, 0.10])

    return pd.DataFrame(
        {
            "ID": ids,
            "Age": age,
            "Education": education,
            "Area": area,
            "Training": training,
            "Machines": machines,
        }
    )


def build_rings(ring_thickness, ring_gap):
    start = 2.05
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


def add_wedge(ax, r_in, r_out, theta1, theta2, color, edgecolor="black", lw=0.25, zorder=1):
    patch = Wedge(
        center=(0, 0),
        r=r_out,
        theta1=theta1,
        theta2=theta2,
        width=r_out - r_in,
        facecolor=color,
        edgecolor=edgecolor,
        linewidth=lw,
        zorder=zorder,
    )
    ax.add_patch(patch)


def add_ring_guides(ax, r_in, r_out, color="#ECE7DF"):
    for radius in np.linspace(r_in, r_out, 4):
        ax.add_patch(
            Wedge(
                center=(0, 0),
                r=radius,
                theta1=-180,
                theta2=90,
                width=0.002,
                facecolor="none",
                edgecolor=color,
                linewidth=0.5,
                zorder=0,
            )
        )


def add_tick_labels(ax, ring_name, rings, value_limits, ticks, color):
    r_in, r_out = rings[ring_name]
    for tick in ticks:
        radius = value_to_radius(tick, value_limits, (r_in, r_out))
        theta = np.deg2rad(94)
        ax.text(
            radius * np.cos(theta),
            radius * np.sin(theta),
            f"{tick:g}",
            ha="center",
            va="center",
            fontsize=7,
            color="white",
            bbox={"facecolor": color, "edgecolor": "none", "pad": 1.2, "alpha": 0.82},
            zorder=5,
        )


def create_vertical_swarm(ax_sub, data, color, title, unit, ylim, ticks):
    rng = np.random.default_rng(1207)
    ax_sub.set_ylim(ylim)
    ax_sub.set_xlim(-0.55, 0.55)
    ax_sub.boxplot(
        data,
        positions=[0],
        widths=0.34,
        patch_artist=True,
        showfliers=False,
        boxprops={"facecolor": color, "alpha": 0.24, "edgecolor": color, "linewidth": 1.1},
        medianprops={"color": color, "linewidth": 1.6},
        whiskerprops={"color": color, "linewidth": 1.0},
        capprops={"color": color, "linewidth": 1.0},
    )

    jitter = rng.normal(0, 0.075, len(data))
    ax_sub.scatter(
        jitter,
        data,
        s=12,
        color=color,
        alpha=0.52,
        edgecolor="white",
        linewidth=0.25,
        zorder=3,
    )

    for tick in ticks:
        ax_sub.axhline(tick, color="#E6E0D8", lw=0.5, zorder=0)
        ax_sub.text(
            0.47,
            tick,
            f"{tick:g}",
            ha="center",
            va="center",
            fontsize=7,
            color="white",
            bbox={"facecolor": "#6d6a5d", "edgecolor": "none", "pad": 1.0, "alpha": 0.82},
            clip_on=False,
            zorder=4,
        )

    ax_sub.set_title(f"{title}\n({unit})", fontsize=10, pad=8)
    ax_sub.set_xticks([])
    ax_sub.set_yticks([])
    for spine in ax_sub.spines.values():
        spine.set_color("#8A8A8A")
        spine.set_linewidth(0.6)


def create_stacked_percent_bar(ax_sub, counts, title):
    bottom = 0
    for label, percent, color in counts:
        ax_sub.bar(0, percent, bottom=bottom, width=0.8, color=color, edgecolor="white", linewidth=0.5)
        if percent >= 8:
            text_color = "white" if label not in {"0", "No"} else "#222222"
            ax_sub.text(0, bottom + percent / 2, f"{percent:.1f}", ha="center", va="center", fontsize=9, color=text_color)
        bottom += percent

    ax_sub.set_title(title, rotation=45, ha="left", y=1.02, fontsize=10)
    ax_sub.set_ylim(0, 100)
    ax_sub.set_xlim(-0.55, 0.55)
    ax_sub.axis("off")


def add_summary_insets(ax, df, rings, colors, ring_thickness):
    inset_y = 0.35
    inset_h = 7.2
    inset_w = ring_thickness * 0.82

    ax_age = ax.inset_axes([-rings["Age"][1], inset_y, inset_w, inset_h], transform=ax.transData)
    create_vertical_swarm(ax_age, df["Age"], colors["Age"], "Age", "years", [20, 80], [30, 40, 50, 60, 70])

    ax_edu = ax.inset_axes([-rings["Education"][1], inset_y, inset_w, inset_h], transform=ax.transData)
    create_vertical_swarm(ax_edu, df["Education"], colors["Education"], "Education", "years", [-2, 17], [0, 5, 10, 15])

    ax_area = ax.inset_axes([-rings["Area"][1], inset_y, inset_w, inset_h], transform=ax.transData)
    create_vertical_swarm(ax_area, df["Area"], colors["Area"], "Area", "ha", [0, 1.5], [0.3, 0.6, 1.0, 1.3])

    train_counts = df["Training"].value_counts(normalize=True).mul(100)
    ax_train = ax.inset_axes([-rings["Training"][1], inset_y, inset_w, inset_h], transform=ax.transData)
    create_stacked_percent_bar(
        ax_train,
        [
            ("0", train_counts.get(0, 0), colors["Train_No"]),
            ("1", train_counts.get(1, 0), colors["Train_Yes"]),
        ],
        "Training(%)",
    )

    mach_counts = df["Machines"].value_counts(normalize=True).mul(100)
    ax_mach = ax.inset_axes([-rings["Machines"][1], inset_y, inset_w, inset_h], transform=ax.transData)
    create_stacked_percent_bar(
        ax_mach,
        [(str(i), mach_counts.get(i, 0), colors[f"Mach_{i}"]) for i in [1, 2, 3, 4, 5]],
        "Machines(%)",
    )


def add_center_legend(ax, colors):
    ax_leg = ax.inset_axes([-1.95, -1.62, 3.9, 3.24], transform=ax.transData)
    ax_leg.axis("off")
    ax_leg.set_xlim(0, 3.9)
    ax_leg.set_ylim(0, 3.24)

    sw = 0.22
    sh = 0.17
    rows = [
        (0.15, 2.85, colors["Age"], "Age"),
        (0.15, 2.55, colors["Education"], "Education"),
        (0.15, 2.25, colors["Area"], "Area"),
        (0.15, 1.72, colors["Train_Yes"], "Training 1"),
        (0.15, 1.42, colors["Train_No"], "Training 0"),
    ]

    for x, y, color, label in rows:
        ax_leg.add_patch(Rectangle((x, y - sh / 2), sw, sh, facecolor=color, edgecolor="none"))
        ax_leg.text(x + 0.34, y, label, va="center", fontsize=9)

    ax_leg.text(1.72, 2.85, "Machines", va="center", fontsize=9, fontweight="bold")
    for idx, machine in enumerate([1, 2, 3, 4, 5]):
        y = 2.55 - idx * 0.3
        ax_leg.add_patch(Rectangle((1.72, y - sh / 2), sw, sh, facecolor=colors[f"Mach_{machine}"], edgecolor="none"))
        ax_leg.text(2.06, y, str(machine), va="center", fontsize=9)

    ax_leg.text(1.72, 1.05, "Outer rings", va="center", fontsize=9, fontweight="bold")
    ax_leg.plot([1.72, 2.38], [0.74, 0.74], color=colors["Age"], lw=1.5)
    ax_leg.plot(2.38, 0.74, "o", color=colors["Age"], ms=4)
    ax_leg.text(2.55, 0.74, "lollipop value", va="center", fontsize=9)


def create_plot(df, scheme_id=1, ring_thickness=1.15, ring_gap=0.18, output_path=None):
    colors = COLOR_SCHEMES.get(scheme_id, COLOR_SCHEMES[1])
    rings = build_rings(ring_thickness, ring_gap)
    n = len(df)

    fig = plt.figure(figsize=(14, 14), facecolor="white", dpi=160)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_aspect("equal")
    ax.axis("off")

    edges = np.linspace(90, -180, n + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    outer_r = rings["Age"][1]
    ax.set_xlim(-outer_r - 1.8, outer_r + 1.0)
    ax.set_ylim(-outer_r - 1.5, outer_r + 1.1)

    continuous_specs = [
        ("Area", "Area", (0, 1.5), [0.3, 0.6, 1.0, 1.3]),
        ("Education", "Education", (-2, 17), [0, 5, 10, 15]),
        ("Age", "Age", (20, 80), [30, 50, 70]),
    ]

    for ring_name, _, limits, ticks in continuous_specs:
        r_in, r_out = rings[ring_name]
        add_ring_guides(ax, r_in, r_out)
        add_tick_labels(ax, ring_name, rings, limits, ticks, colors[ring_name])

    for i, row in df.reset_index(drop=True).iterrows():
        theta1 = edges[i + 1]
        theta2 = edges[i]
        theta_center = np.deg2rad(centers[i])

        m_col = colors[f"Mach_{int(row['Machines'])}"]
        add_wedge(ax, rings["Machines"][0], rings["Machines"][1], theta1, theta2, m_col, lw=0.23, zorder=1)

        t_col = colors["Train_Yes"] if int(row["Training"]) == 1 else colors["Train_No"]
        add_wedge(ax, rings["Training"][0], rings["Training"][1], theta1, theta2, t_col, lw=0.23, zorder=1)

        for ring_name, data_col, limits, _ in continuous_specs:
            r_in, r_out = rings[ring_name]
            add_wedge(ax, r_in, r_out, theta1, theta2, "none", edgecolor=colors["Grid"], lw=0.25, zorder=0)
            r_val = value_to_radius(float(row[data_col]), limits, (r_in, r_out))
            ax.plot(
                [r_in * np.cos(theta_center), r_val * np.cos(theta_center)],
                [r_in * np.sin(theta_center), r_val * np.sin(theta_center)],
                color=colors[ring_name],
                lw=1.25,
                solid_capstyle="round",
                zorder=3,
            )
            ax.plot(
                r_val * np.cos(theta_center),
                r_val * np.sin(theta_center),
                "o",
                color=colors[ring_name],
                markersize=3.7,
                markeredgecolor="white",
                markeredgewidth=0.25,
                zorder=4,
            )

        text_radius = rings["Age"][1] + 0.34
        x_text = text_radius * np.cos(theta_center)
        y_text = text_radius * np.sin(theta_center)
        rot_deg = centers[i] - 90
        if centers[i] < -90:
            rot_deg += 180
        ax.text(x_text, y_text, str(row["ID"]), rotation=rot_deg, ha="center", va="center", fontsize=7.2, color="#222222")

    add_summary_insets(ax, df, rings, colors, ring_thickness)
    add_center_legend(ax, colors)

    palette_name = colors.get("Name", f"scheme-{scheme_id}")
    ax.text(
        -outer_r - 1.25,
        outer_r + 0.45,
        f"Circular lollipop + categorical heatmap | {palette_name}",
        fontsize=16,
        fontweight="bold",
    )
    ax.text(
        -outer_r - 1.25,
        outer_r + 0.10,
        f"Simulated data: {n} samples, 3 continuous rings, 2 categorical rings",
        fontsize=10,
    )

    if output_path is None:
        out_dir = Path(__file__).resolve().parent / "codex_outputs"
        out_dir.mkdir(exist_ok=True)
        output_path = out_dir / f"circular_lollipop_heatmap_scheme_{scheme_id:02d}.png"

    fig.savefig(output_path, dpi=300, facecolor="white")
    plt.close(fig)
    return Path(output_path)


def main():
    df = simulate_data(n=190)
    out_dir = Path(__file__).resolve().parent / "codex_outputs"
    out_dir.mkdir(exist_ok=True)

    data_path = out_dir / "circular_lollipop_heatmap_data.csv"
    df.to_csv(data_path, index=False, encoding="utf-8-sig")

    figure_paths = []
    for scheme_id in sorted(COLOR_SCHEMES):
        figure_path = create_plot(
            df,
            scheme_id=scheme_id,
            ring_thickness=1.15,
            ring_gap=0.18,
            output_path=out_dir / f"circular_lollipop_heatmap_scheme_{scheme_id:02d}.png",
        )
        figure_paths.append(figure_path)

    print(f"Saved simulated data to: {data_path}")
    for figure_path in figure_paths:
        print(f"Saved figure to: {figure_path}")


if __name__ == "__main__":
    main()
