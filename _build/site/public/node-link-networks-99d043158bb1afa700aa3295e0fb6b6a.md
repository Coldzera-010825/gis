---
title: "Piece 05 · Node-link neural-network diagrams"
description: The classic neurons-and-edges network figure, built two ways — from scratch in matplotlib (with weight-encoded edges) and in publication-grade TikZ — plus the no-code option, NN-SVG.
---

# Node-link neural-network diagrams

```{figure} figures/nl-mlp-main.jpg
:width: 88%
:alt: A 4-6-6-3 MLP drawn in matplotlib with weight-encoded edges

The finished matplotlib figure: a 4–6–6–3 MLP. Edge **colour** carries the weight's
sign (blue positive, orange negative — the same diverging logic as Piece 03) and edge
**thickness** its magnitude.
```

## 1 · Why this figure

Piece 04 drew the **volume style** — 3D slabs for convolutional feature maps. Its
sister is the **node-link style**: neurons as circles, layers as columns, every
connection drawn. This is the figure for *fully-connected* content — MLP heads,
conceptual "what is a neural network" panels, attention/embedding sketches — anywhere
the reader should see **individual units and their connections**, not tensor shapes.

Two builds of the same diagram below, because the two tools serve different moments:

| Track | Tool | When |
| --- | --- | --- |
| A | matplotlib, from scratch | you want full control, weight encodings, or to embed it beside your other Python figures |
| B | TikZ (LaTeX) | the figure goes straight into a manuscript and must match its typography |

## 2 · Track A, step 1: neurons on a grid

The whole layout is one idea: **layer index → x, neuron index → y, centred per
column**. Everything else is drawing circles:

```python
LAYERS = [4, 6, 6, 3]                                # neurons per layer
NODE_COLORS = ["#4C86C6", "#57A773", "#57A773", "#EE6C4D"]
X_GAP, Y_GAP, R = 2.4, 1.0, 0.30

def layer_positions(layer_sizes, x_gap=X_GAP, y_gap=Y_GAP):
    positions = []
    for i, n in enumerate(layer_sizes):
        ys = np.arange(n, dtype=float) * y_gap
        ys -= ys.mean()                              # centre every column on y=0
        positions.append([(i * x_gap, y) for y in ys])
    return positions

# neurons: one Circle per (layer, unit)
for li, layer in enumerate(pos):
    for (x, y) in layer:
        ax.add_patch(Circle((x, y), R, facecolor=NODE_COLORS[li],
                            edgecolor="white", lw=1.4, zorder=3))
```

```{figure} figures/nl-sub-nodes.jpg
:width: 74%
:alt: Step 1 — neuron circles positioned by layer, no edges yet

Step 1 rendered: 19 circles, four centred columns. `ys -= ys.mean()` is the only
"layout algorithm" needed — it vertically centres layers of different sizes.
```

## 3 · Track A, step 2: connect the layers

A fully-connected network is a double loop over adjacent layers. Draw edges **before**
nodes (lower `zorder`) so circles cover the line endpoints; trimming `R` off each end
keeps arrows from poking into the circles:

```python
for li in range(len(layer_sizes) - 1):
    for (x1, y1) in pos[li]:
        for (x2, y2) in pos[li + 1]:
            ax.plot([x1 + R, x2 - R], [y1, y2],
                    color="#9aa3ab", lw=0.7, alpha=0.55, zorder=1)
```

```{figure} figures/nl-sub-edges.jpg
:width: 74%
:alt: Step 2 — uniform grey connections between all adjacent layers

Step 2 rendered: 87 uniform edges. Thin (`lw=0.7`) and translucent (`alpha=0.55`) is
the difference between a diagram and a hairball.
```

## 4 · Track A, step 3: make the edges say something

A node-link figure can carry real information: map each connection's **weight** to
colour (sign) and linewidth (magnitude). With a trained model you would read
`model.coefs_` / `state_dict()`; here the weights are simulated:

```python
w = rng.uniform(-1, 1)                                # or a real trained weight
color = "#4C86C6" if w > 0 else "#EE6C4D"             # sign → diverging hue
lw, alpha = 0.4 + 2.2 * abs(w), 0.28 + 0.45 * abs(w)  # magnitude → thickness
ax.plot([x1 + R, x2 - R], [y1, y2], color=color, lw=lw, alpha=alpha, zorder=1)
```

Add input/output labels inside the circles and a caption per column, and you get the
hero figure at the top of the page. The full script is
[`mlp_matplotlib.py`](./code/nodelink/mlp_matplotlib.py) — ~90 lines, `numpy` +
`matplotlib` only, with `draw_edges` / `weighted` / `labels` flags so all three steps
come from one function.

## 5 · Track B: the TikZ route

For manuscripts, the standard is TikZ — vector output, LaTeX fonts, and a `\foreach`
grammar that mirrors the Python loops one-to-one. This is the style popularised by
[Izaak Neutelings' neural-network TikZ examples](https://tikz.net/neural_networks/),
here reduced to a self-contained minimum:

```latex
\tikzset{
  neuron in/.style={circle, draw=colin, fill=blue!18, minimum size=0.72cm},
  link/.style={-{Latex[length=1.5mm]}, gray!55, thin},
}

% neurons: one \foreach per layer, vertically centred
\foreach \i in {1,...,4}
  \node[neuron in]  (I\i) at (0, {\i - 2.5}) {$x_{\i}$};
\foreach \i in {1,...,5}
  \node[neuron hid] (H\i) at (1, {\i - 3.0}) {$h^{(1)}_{\i}$};

% fully-connected links: nested \foreach per layer pair
\foreach \i in {1,...,4} \foreach \j in {1,...,5}
  \draw[link] (I\i) -- (H\j);
```

```{figure} figures/nl-tikz.jpg
:width: 74%
:alt: The same MLP compiled from TikZ

The TikZ render (`pdflatex mlp_tikz.tex`): named nodes (`I3`, `H2`…) make every neuron
addressable — annotating a single connection later is one `\draw` line. Compiles with
TinyTeX; same toolchain as Piece 04.
```

Full source: [`mlp_tikz.tex`](./code/nodelink/mlp_tikz.tex). Note the symmetry with
Track A: `\foreach \i` ↔ the Python `for` loops, `(0, {\i - 2.5})` ↔ `ys -= ys.mean()`.
Learn one, and you know both.

## 6 · When you don't need code: NN-SVG

Honest tooling advice: if you just need a clean node-link (or LeNet/AlexNet-style)
figure *right now*, skip the code entirely and use
[**NN-SVG**](https://alexlenail.me/NN-SVG/) by Alex Lenail
([GitHub](https://github.com/alexlenail/NN-SVG)) — a browser app where you set layer
sizes with sliders and export publication-ready SVG in seconds. It has three modes
(FCNN node-link, LeNet slab, AlexNet block) and is my own go-to for quick drafts.

The trade-off is the usual one: no version control, no weight encodings, no batch
regeneration when the architecture changes. For a figure that will be revised under
review, the code tracks above pay for themselves; for a one-off slide, NN-SVG wins.

Model-driven alternatives, if the diagram should come *from* a real network object:
[visualtorch](https://github.com/willyfh/visualtorch) (PyTorch, layered + graph views)
and [visualkeras](https://github.com/paulgavrikov/visualkeras) (Keras).

## 7 · Run it yourself

[`mlp_matplotlib.py`](./code/nodelink/mlp_matplotlib.py) ·
[`mlp_tikz.tex`](./code/nodelink/mlp_tikz.tex) ·
[`build_all.py`](./code/nodelink/build_all.py)

```bash
python build_all.py
# → the three matplotlib steps + the compiled TikZ figure
```
