---
title: "Piece 04 · Neural-network architecture diagrams with PlotNeuralNet"
description: Publication-grade CNN / U-Net architecture diagrams from Python-generated LaTeX — built in three steps, from a five-layer chain to a full encoder–decoder with skip connections.
---

# Neural-network architecture diagrams

```{figure} figures/nn-unet-main.jpg
:width: 100%
:alt: U-Net architecture diagram rendered with PlotNeuralNet

The finished figure: a U-Net — four encoder blocks, a bottleneck, four decoder blocks
with skip connections, and a softmax head — declared in ~40 lines of Python and rendered
by LaTeX/TikZ as crisp vector graphics.
```

## 1 · Why this tool

Every deep-learning paper needs an architecture figure, and hand-drawing one in
PowerPoint or Inkscape ends the same way: misaligned slabs, inconsistent perspective,
and a full redraw every time a channel count changes.
[**PlotNeuralNet**](https://github.com/HarisIqbal88/PlotNeuralNet) (MIT, by Haris Iqbal)
takes a different route:

- you **declare** the architecture as a Python list of layer primitives;
- the library emits **LaTeX/TikZ** code;
- `pdflatex` renders it into a **vector PDF** — the same 3D-slab style used in many
  CVPR/NeurIPS papers.

Change `n_filer=512` to `1024` and recompile: the figure updates itself. The diagram is
*code*, so it versions, diffs and reviews like code.

## 2 · Setup

Clone the repo and make a working folder inside it (the library resolves its LaTeX
helpers relative to the project root):

```bash
git clone https://github.com/HarisIqbal88/PlotNeuralNet.git
cd PlotNeuralNet && mkdir mydiagrams
```

You also need a LaTeX distribution with `pdflatex` ([TinyTeX](https://yihui.org/tinytex/)
is enough — install the `standalone` and `import` packages via
`tlmgr install standalone import`).

Every diagram script starts with the same boilerplate — three header items that
`to_generate()` will turn into the LaTeX preamble, colour definitions and
`tikzpicture` opening:

```python
import sys
sys.path.append('../')             # reach pycore/ from mydiagrams/
from pycore.tikzeng import *       # layer primitives
from pycore.blocks import *        # multi-layer factories

arch = [
    to_head('..'),                 # path to the repo's LaTeX layer definitions
    to_cor(),                      # colour palette
    to_begin(),                    # \begin{tikzpicture}
    # ... layers go here ...
    to_end(),                      # \end{tikzpicture}
]
```

## 3 · Step 1: a minimal chain — the five primitives

Almost every CNN figure is built from five verbs: `to_input` (a raster image),
`to_Conv` (an orange slab), `to_Pool` (a shrinking red slab), `to_SoftMax` (the output
column) and `to_connection` (an arrow). Positioning is relative: `to="(conv1-east)"`
docks a layer onto the east face of the previous one, and `offset="(1.5,0,0)"` adds a
gap that `to_connection` then bridges:

```python
arch = [
    to_head('..'), to_cor(), to_begin(),

    to_input('../examples/fcn8s/cats.jpg'),

    # conv -> pool -> conv -> pool -> softmax
    to_Conv("conv1", s_filer=224, n_filer=64, offset="(0,0,0)", to="(0,0,0)",
            width=2, height=40, depth=40, caption="Conv1"),
    to_Pool("pool1", offset="(0,0,0)", to="(conv1-east)",
            width=1, height=32, depth=32, opacity=0.5),

    to_Conv("conv2", s_filer=112, n_filer=128, offset="(1.5,0,0)", to="(pool1-east)",
            width=3.5, height=32, depth=32, caption="Conv2"),
    to_connection("pool1", "conv2"),
    to_Pool("pool2", offset="(0,0,0)", to="(conv2-east)",
            width=1, height=25, depth=25, opacity=0.5),

    to_SoftMax("soft1", s_filer=10, offset="(2,0,0)", to="(pool2-east)",
               caption="SoftMax"),
    to_connection("pool2", "soft1"),

    to_end(),
]
```

```{figure} figures/nn-sub-minimal.jpg
:width: 62%
:alt: Minimal five-layer CNN chain

Step 1 rendered: input image, two conv slabs with pooling, softmax output. The three
size parameters map to semantics — `height`/`depth` visualise the spatial resolution,
`width` the channel count, and the printed labels come from `s_filer` (spatial size)
and `n_filer` (filters).
```

:::{tip}
Read the slab geometry as a legend: as the network deepens, `height`/`depth` shrink
(pooling) while `width` grows (more channels). Keeping that mapping consistent is what
makes these figures readable at a glance.
:::

## 4 · Step 2: the encoder — write the motif once, repeat it

A real encoder repeats one motif — *double conv + pool* — with growing channels.
`to_ConvConvRelu()` draws the fused double-conv slab; `pycore.blocks` then packages the
whole motif as `block_2ConvPool()`, a factory returning a *list* of primitives, spliced
into `arch` with Python's `*` unpacking. Each block chains onto the previous via
`botton=` / `top=` names:

```python
# block 1, written by hand: double conv + relu, then a pooling slab
to_ConvConvRelu(name='ccr_b1', s_filer=500, n_filer=(64, 64),
                offset="(0,0,0)", to="(0,0,0)",
                width=(2, 2), height=40, depth=40),
to_Pool(name="pool_b1", offset="(0,0,0)", to="(ccr_b1-east)",
        width=1, height=32, depth=32, opacity=0.5),

# blocks 2-4: the same motif, one factory call each
*block_2ConvPool(name='b2', botton='pool_b1', top='pool_b2',
                 s_filer=256, n_filer=128, offset="(1,0,0)",
                 size=(32, 32, 3.5), opacity=0.5),
*block_2ConvPool(name='b3', botton='pool_b2', top='pool_b3',
                 s_filer=128, n_filer=256, offset="(1,0,0)",
                 size=(25, 25, 4.5), opacity=0.5),
*block_2ConvPool(name='b4', botton='pool_b3', top='pool_b4',
                 s_filer=64, n_filer=512, offset="(1,0,0)",
                 size=(16, 16, 5.5), opacity=0.5),
```

```{figure} figures/nn-sub-encoder.jpg
:width: 82%
:alt: Four-block encoder

Step 2 rendered: the contracting path. Four double-conv blocks, each halving the
spatial size (`s_filer` 500 → 256 → 128 → 64) and doubling the channels (`n_filer`
64 → 128 → 256 → 512) — the size tuple `(height, depth, width)` shrinks in space and
thickens in channels accordingly.
```

## 5 · Step 3: bottleneck, decoder and skip connections

Two new ingredients complete the U-Net. `block_Unconv()` is the decoder mirror of
`block_2ConvPool` (unpool + residual conv + double conv), and `to_skip()` draws the
long U-shaped arrows between mirrored encoder/decoder layers — referencing layers *by
name*, which is why consistent naming (`ccr_b3` ↔ `ccr_res_b7`) pays off:

```python
# bottleneck
to_ConvConvRelu(name='ccr_b5', s_filer=32, n_filer=(1024, 1024),
                offset="(2,0,0)", to="(pool_b4-east)",
                width=(8, 8), height=8, depth=8, caption="Bottleneck"),
to_connection("pool_b4", "ccr_b5"),

# decoder: each block mirrors one encoder block, plus a skip arrow
*block_Unconv(name="b6", botton="ccr_b5", top='end_b6', s_filer=64,
              n_filer=512, offset="(2.1,0,0)", size=(16, 16, 5.0), opacity=0.5),
to_skip(of='ccr_b4', to='ccr_res_b6', pos=1.25),
*block_Unconv(name="b7", botton="end_b6", top='end_b7', s_filer=128,
              n_filer=256, offset="(2.1,0,0)", size=(25, 25, 4.5), opacity=0.5),
to_skip(of='ccr_b3', to='ccr_res_b7', pos=1.25),
# ... b8, b9 follow the same pattern ...

# segmentation head
to_ConvSoftMax(name="soft1", s_filer=512, offset="(0.75,0,0)",
               to="(end_b9-east)", width=1, height=40, depth=40, caption="SOFT"),
to_connection("end_b9", "soft1"),
```

The full result is the hero figure at the top of this page: encoder and decoder
mirror each other in size, the four blue skip arrows connect them, and the growing
slabs of the expanding path retrace the contracting path in reverse.

## 6 · The build pipeline

Three deterministic steps, scriptable end to end:

```text
python step3_unet.py        →  step3_unet.tex      (declare)
pdflatex step3_unet.tex     →  step3_unet.pdf      (render, vector)
pymupdf / pdftoppm          →  step3_unet.jpg      (rasterise for the web)
```

The PDF is the publication artefact — infinitely zoomable, embeds cleanly in a LaTeX
manuscript. The JPG conversion only exists for pages like this one. All three steps for
all three figures are wrapped in [`build_all.py`](./code/plotneuralnet/build_all.py).

## 7 · Design notes

- **The geometry is the message.** Height/depth = spatial resolution, width = channels.
  Resist decorating: the default palette (orange conv, red pool, blue unpool) is already
  the de-facto standard readers recognise from papers.
- **Name layers like variables.** Skips and connections reference names; a naming
  convention (`ccr_b3`, `pool_b3`, `ccr_res_b7`) makes the mirror structure of an
  encoder–decoder self-documenting.
- **Declare, don't draw.** Because the figure is generated, architecture revisions are
  one-line edits — the reason to prefer this over any WYSIWYG tool for work that will
  be revised under review.
- **Where it fits my research:** CNN / U-Net style figures for remote-sensing
  segmentation and geospatial deep-learning pipelines — the exact models used in urban
  land-use and heat-mapping work.

## 8 · Run it yourself

The three step scripts and the build script:
[`step1_minimal.py`](./code/plotneuralnet/step1_minimal.py) ·
[`step2_encoder.py`](./code/plotneuralnet/step2_encoder.py) ·
[`step3_unet.py`](./code/plotneuralnet/step3_unet.py) ·
[`build_all.py`](./code/plotneuralnet/build_all.py)

```bash
# from inside PlotNeuralNet/mydiagrams/
python build_all.py
# → step*.tex, step*.pdf and web JPGs
```

*Package: [PlotNeuralNet](https://github.com/HarisIqbal88/PlotNeuralNet) by Haris Iqbal, MIT license.*
