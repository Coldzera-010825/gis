"""Node-link (neuron-and-edge) MLP diagrams in pure matplotlib.

Three renders, one function:
  nl-sub-nodes.jpg  step 1 - neuron positions only
  nl-sub-edges.jpg  step 2 - uniform connections
  nl-mlp-main.jpg   step 3 - weight-encoded edges + labels (the finished figure)
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

OUT = Path(__file__).resolve().parents[2] / "figures"

LAYERS = [4, 6, 6, 3]                                # neurons per layer
LAYER_NAMES = ["Input", "Hidden 1", "Hidden 2", "Output"]
NODE_COLORS = ["#4C86C6", "#57A773", "#57A773", "#EE6C4D"]
X_GAP, Y_GAP, R = 2.4, 1.0, 0.30                     # spacing and neuron radius


def layer_positions(layer_sizes, x_gap=X_GAP, y_gap=Y_GAP):
    """Column x = layer index; neurons vertically centred per layer."""
    positions = []
    for i, n in enumerate(layer_sizes):
        ys = np.arange(n, dtype=float) * y_gap
        ys -= ys.mean()                              # centre every column on y=0
        positions.append([(i * x_gap, y) for y in ys])
    return positions


def draw_network(ax, layer_sizes, draw_edges=False, weighted=False,
                 labels=False, seed=42):
    pos = layer_positions(layer_sizes)
    rng = np.random.default_rng(seed)

    # ---- edges first, so nodes paint on top of them
    if draw_edges:
        for li in range(len(layer_sizes) - 1):
            for (x1, y1) in pos[li]:
                for (x2, y2) in pos[li + 1]:
                    if weighted:                      # simulated weight in [-1, 1]
                        w = rng.uniform(-1, 1)
                        color = "#4C86C6" if w > 0 else "#EE6C4D"
                        lw, alpha = 0.4 + 2.2 * abs(w), 0.28 + 0.45 * abs(w)
                    else:
                        color, lw, alpha = "#9aa3ab", 0.7, 0.55
                    ax.plot([x1 + R, x2 - R], [y1, y2],
                            color=color, lw=lw, alpha=alpha, zorder=1)

    # ---- neurons
    for li, layer in enumerate(pos):
        for (x, y) in layer:
            ax.add_patch(Circle((x, y), R, facecolor=NODE_COLORS[li],
                                edgecolor="white", lw=1.4, zorder=3))

    # ---- annotations
    if labels:
        for k, (x, y) in enumerate(pos[0], start=1):
            ax.text(x, y, f"$x_{k}$", ha="center", va="center",
                    fontsize=10, color="white", zorder=4)
        for k, (x, y) in enumerate(pos[-1], start=1):
            ax.text(x, y, f"$y_{k}$", ha="center", va="center",
                    fontsize=10, color="white", zorder=4)
        y_cap = max(y for layer in pos for (_, y) in layer) + 1.0
        for li, name in enumerate(LAYER_NAMES[: len(layer_sizes)]):
            ax.text(li * X_GAP, y_cap, name, ha="center", va="bottom",
                    fontsize=11, color="#444444")

    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(-0.9, (len(layer_sizes) - 1) * X_GAP + 0.9)
    ymin = min(y for layer in pos for (_, y) in layer) - 0.9
    ymax = max(y for layer in pos for (_, y) in layer) + 1.8
    ax.set_ylim(ymin, ymax)


def render(name, **kwargs):
    fig, ax = plt.subplots(figsize=(9, 5.6), facecolor="white")
    draw_network(ax, LAYERS, **kwargs)
    fig.savefig(OUT / name, dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {name}")


if __name__ == "__main__":
    render("nl-sub-nodes.jpg")                                        # step 1
    render("nl-sub-edges.jpg", draw_edges=True)                       # step 2
    render("nl-mlp-main.jpg", draw_edges=True, weighted=True, labels=True)  # final
