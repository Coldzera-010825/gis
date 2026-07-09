---
title: "Piece 06 · SHAP interpretability for geospatial ML"
description: Explaining an urban-heat XGBoost model with SHAP — global bar, beeswarm, dependence and waterfall plots, manual panel composition, and how the display evolves for LSTM-style sequence models.
---

# SHAP interpretability for geospatial ML

```{figure} figures/shap-main-beeswarm.jpg
:width: 92%
:alt: SHAP beeswarm summary of an urban-heat XGBoost model

The signature figure: a SHAP **beeswarm**. One dot per neighbourhood per feature;
x-position is the feature's contribution to predicted land-surface temperature,
colour is the feature's own value. High NDVI (red, left) cools; high building
density (red, right) warms — importance, direction and distribution in one panel.
```

## 1 · Why this figure

A trained model that predicts well but explains nothing is half a result. In
geospatial ML — heat-risk regression, land-use classification, accessibility
modelling — reviewers now expect the *why*: which drivers matter, in which
direction, and for which cases. **SHAP** (SHapley Additive exPlanations) answers all
three with one idea: every prediction is decomposed into per-feature contributions
that provably sum back to the prediction.

This tutorial explains a simulated but realistically-structured **urban heat**
model — the exact workflow (XGBoost + `TreeExplainer`) used in my own heat-risk
research — then walks the four canonical SHAP figures.

## 2 · The data and the model

Eight drivers of land-surface temperature with realistic signs and one deliberate
interaction (openness only cools where there is vegetation):

```python
lst = (
    34.0
    - 6.5 * ndvi                       # vegetation cools
    + 4.8 * build                      # built-up warms
    - 9.0 * albedo                     # bright surfaces cool
    + 0.35 * np.minimum(water_d, 6)    # further from water = warmer (saturates)
    + 0.9 * np.log1p(pop)              # anthropogenic heat
    + 0.06 * road
    - 1.6 * svf * ndvi                 # open + green interact
    - 0.012 * elev
    + RNG.normal(0, 0.8, n)
)
```

Train the model, then compute SHAP values — for tree ensembles `TreeExplainer` is
exact and fast, and the modern API returns a single `Explanation` object that every
plot consumes:

```python
model = xgb.XGBRegressor(n_estimators=400, max_depth=4, learning_rate=0.06,
                         subsample=0.9, colsample_bytree=0.9).fit(X, y)

explainer = shap.TreeExplainer(model)
sv = explainer(X)          # Explanation: (2400 samples x 8 features)
```

## 3 · Global ranking: the bar plot

The simplest question first — *what matters overall?* Mean |SHAP| per feature, in
the target's own units (°C here), which is the advantage over gain-based
importances:

```python
shap.plots.bar(sv, max_display=8, show=False)
```

```{figure} figures/shap-sub-bar.jpg
:width: 66%
:alt: Mean absolute SHAP value per feature

NDVI moves predictions by 1.23 °C on average — readable directly in physical units,
something `feature_importances_` can never give you.
```

## 4 · The beeswarm: ranking + direction + distribution

The bar plot hides *direction*. The beeswarm (the hero figure above) adds it: each
feature row is a strip of all 2,400 neighbourhoods, coloured by feature value. Read
it as three questions per row — is the strip wide (important)? does red sit left or
right (direction)? is it skewed (nonlinear / threshold behaviour)?

```python
shap.plots.beeswarm(sv, max_display=8, show=False)
```

One honest caveat worth writing in any paper: SHAP explains **the model**, not
nature. A confounded model produces confidently wrong explanations — the beeswarm
inherits whatever the model learned.

## 5 · Dependence: one feature in depth

The beeswarm compresses each feature to a strip; the dependence scatter expands one
feature back out. Colouring by a second feature exposes interactions — here NDVI's
cooling is steeper where the sky view factor is high (our simulated
`svf × ndvi` term, recovered by the model):

```python
shap.plots.scatter(sv[:, "NDVI"], color=sv[:, "Sky view factor"], show=False)
```

```{figure} figures/shap-sub-dependence.jpg
:width: 68%
:alt: SHAP dependence plot for NDVI coloured by sky view factor

The x-axis is the feature's value, the y-axis its SHAP contribution — an empirical
partial-effect curve, with vertical colour separation revealing the interaction.
```

## 6 · Waterfall: explaining a single case

Global plots convince reviewers; local plots convince stakeholders. The waterfall
decomposes one prediction — here the hottest neighbourhood in the dataset — from the
dataset mean down to its individual drivers:

```python
hot_i = int(np.argmax(model.predict(X)))   # the hottest neighbourhood
shap.plots.waterfall(sv[hot_i], max_display=9, show=False)
```

```{figure} figures/shap-sub-waterfall.jpg
:width: 70%
:alt: SHAP waterfall plot for the hottest neighbourhood

“Why is this place 5 °C above average?” — sparse vegetation, dense building stock
and low albedo, each quantified. This is the figure for community-facing heat-equity
communication.
```

## 7 · Composing panels for a paper

SHAP's plot functions draw on the current axes, so `plt.sca()` (plus the `ax=`
argument most `shap.plots.*` now accept) lets you assemble multi-panel layouts like
any other matplotlib figure:

```python
fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.8))
plt.sca(axes[0]); shap.plots.bar(sv, max_display=8, show=False, ax=axes[0])
plt.sca(axes[1]); shap.plots.scatter(sv[:, "NDVI"], color=sv[:, "Sky view factor"],
                                     show=False, ax=axes[1])
axes[0].set_title("(a) Global importance", loc="left")
axes[1].set_title("(b) NDVI dependence", loc="left")
```

```{figure} figures/shap-sub-compose.jpg
:width: 100%
:alt: Bar and dependence plots composed into one two-panel figure

The (a)/(b) two-panel format most journals expect — global ranking beside the
mechanism of the top driver.
```

## 8 · Beyond trees: what changes for LSTM-style models

For sequence models (LSTM/GRU/Transformers) the *idea* — additive attribution —
survives, but the display evolves, because attributions become
**per-time-step × per-feature**:

- **SHAP still works** (`DeepExplainer` / `GradientExplainer`), but a beeswarm can't
  hold a lookback window; the standard rendering becomes a **temporal attribution
  heatmap** — x = time step, y = variable, colour = attribution — "the model looked
  at rainfall from t−3 to t−1".
- **Gradient methods** (Integrated Gradients, DeepLIFT via
  [captum](https://github.com/pytorch/captum)) are the more common choice in the
  hydrology/meteorology LSTM literature, with the same heatmap presentation.
- **Attention weights**, when the architecture has them, are the cheapest
  interpretability — plot them directly as a curve or heatmap over the sequence.

Rule of thumb: tabular + trees → beeswarm/waterfall; sequences → time-step
attribution heatmaps; images → Grad-CAM overlays (a future piece).

## 9 · Run it yourself

Full source: [`shap_urban_heat.py`](./code/shapviz/shap_urban_heat.py) — needs
`shap`, `xgboost`, `pandas`, `matplotlib`.

```bash
python shap_urban_heat.py
# → the five figures on this page
```
