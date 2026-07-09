---
title: "Piece 07 · The classifier evaluation panel"
description: The four figures every classification paper needs — normalized confusion matrix, one-vs-rest ROC, precision-recall curves and a learning curve — each deconstructed, then composed into one GridSpec panel.
---

# The classifier evaluation panel

```{figure} figures/eval-main-panel.jpg
:width: 100%
:alt: Four-panel classifier evaluation figure

The finished figure: one 2 × 2 panel that answers the four referee questions in
order — *where* does the model confuse classes (a), how well does it *rank* (b),
how does it behave under *class imbalance* (c), and would *more data* help (d).
```

## 1 · Why this figure

Every classification study — land-cover mapping included — faces the same four
questions, and a scattered collection of screenshots answers none of them well.
This piece builds each panel as an **`ax`-taking function**, which is the real
lesson: draw functions that accept an axes object compose into any layout for free.

The simulated task is a five-class land-cover problem
(`Water / Vegetation / Built-up / Bare soil / Cropland`) with a RandomForest —
swap in your own `y_te / y_pred / y_prob` and every panel below works unchanged.

```python
clf = RandomForestClassifier(n_estimators=300, n_jobs=-1).fit(X_tr, y_tr)
y_pred = clf.predict(X_te)
y_prob = clf.predict_proba(X_te)               # ROC / PR need probabilities
y_bin  = label_binarize(y_te, classes=range(5))  # one-vs-rest ground truth
```

## 2 · Panel (a): the confusion matrix, done properly

Three details separate a paper-grade confusion matrix from `plt.imshow(cm)`:
**row-normalize** (`normalize="true"`) so each cell reads as recall share;
**annotate every cell** with adaptive text colour; and label the axes with class
names, not integers:

```python
def draw_confusion(ax):
    cm = confusion_matrix(y_te, y_pred, normalize="true")
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    for i in range(N_CLASSES):
        for j in range(N_CLASSES):
            colour = "white" if cm[i, j] > 0.55 else "#1f3044"
            ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center",
                    fontsize=9, color=colour)
    ax.set_xticks(range(N_CLASSES), CLASSES, rotation=35, ha="right")
    ax.set_yticks(range(N_CLASSES), CLASSES)
    return im
```

```{figure} figures/eval-sub-confusion.jpg
:width: 62%
:alt: Row-normalized annotated confusion matrix

Panel (a) on its own. The off-diagonal cells are the story: 10 % of true
Vegetation is called Cropland — the classic spectral-confusion pair.
```

## 3 · Panel (b): one-vs-rest ROC curves

Multiclass ROC means binarising: one curve per class against the rest, plus the
macro average as the headline number. Always draw the chance diagonal — a curve
without its baseline is unreadable:

```python
def draw_roc(ax):
    for k in range(N_CLASSES):
        fpr, tpr, _ = roc_curve(y_bin[:, k], y_prob[:, k])
        ax.plot(fpr, tpr, color=COLORS[k], lw=1.6,
                label=f"{CLASSES[k]} (AUC={auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], "--", color="#999999", lw=1)   # chance line
```

```{figure} figures/eval-sub-roc.jpg
:width: 62%
:alt: One-vs-rest ROC curves with per-class AUC

Panel (b) on its own — per-class AUC in the legend, macro-AUC as the panel title,
chance line for honesty.
```

## 4 · Panel (c): precision–recall, the imbalance-honest view

ROC flatters models on imbalanced classes because the false-positive rate divides
by the (large) negative pool. PR curves do not — which is why remote-sensing
papers with rare classes (water bodies, bare soil) should always show both. The
baseline here is each class's **prevalence**, not 0.5:

```python
prec, rec, _ = precision_recall_curve(y_bin[:, k], y_prob[:, k])
ap = average_precision_score(y_bin[:, k], y_prob[:, k])
ax.axhline(y_bin.mean(), ls="--", color="#999999")   # prevalence baseline
```

```{figure} figures/eval-sub-pr.jpg
:width: 62%
:alt: Precision-recall curves per class with average precision

Panel (c) on its own. AP (average precision) summarises each curve; the dashed
prevalence line is the "random classifier" reference for PR space.
```

## 5 · Panel (d): the learning curve

The panel reviewers rarely see but always appreciate: performance vs training-set
size, with train/CV bands. It answers "would labelling more data help?" — if the
CV curve is still climbing, yes; if it has flattened into the training curve, the
model, not the data, is the bottleneck:

```python
sizes, tr_scores, va_scores = learning_curve(
    clf, X, y, cv=5, scoring="f1_macro",
    train_sizes=np.linspace(0.08, 1.0, 8), n_jobs=-1)
mean, std = va_scores.mean(axis=1), va_scores.std(axis=1)
ax.plot(sizes, mean, "o-"); ax.fill_between(sizes, mean - std, mean + std, alpha=0.15)
```

```{figure} figures/eval-sub-learning.jpg
:width: 66%
:alt: Learning curve with train and cross-validation bands

Panel (d) on its own. The gap between the curves is variance; the CV curve's slope
at the right edge is the labelling-budget argument.
```

## 6 · Composition: `ax`-taking functions + GridSpec

Because every panel is a `draw_*(ax)` function, the hero figure is ten lines:

```python
fig = plt.figure(figsize=(12.4, 10.2))
grid = fig.add_gridspec(2, 2, wspace=0.30, hspace=0.34)
specs = [(draw_confusion, "(a) Confusion matrix"),
         (draw_roc,       "(b) ROC curves (one-vs-rest)"),
         (draw_pr,        "(c) Precision-recall curves"),
         (draw_learning,  "(d) Learning curve")]
for spec, (fn, title) in zip(grid, specs):
    ax = fig.add_subplot(spec)
    fn(ax)
    ax.set_title(title, loc="left")
```

This is the same composition discipline as Piece 03 (`GridSpec` + one function per
panel) — the pattern scales to any evaluation suite you need.

## 7 · Design notes

- **Row-normalize the confusion matrix** — raw counts conflate class frequency with
  model skill.
- **Show ROC *and* PR** when classes are imbalanced; quote macro-AUC and per-class
  AP, not accuracy.
- **One colour per class, everywhere.** The same five hues bind panels (a)–(c)
  together; a class keeps its colour across every figure of the paper.
- **Baselines in every panel**: chance diagonal (ROC), prevalence line (PR),
  train/CV gap (learning curve). A metric without its baseline is decoration.

## 8 · Run it yourself

Full source: [`classifier_eval.py`](./code/modeleval/classifier_eval.py) — needs
`scikit-learn` and `matplotlib` only.

```bash
python classifier_eval.py
# → the four sub-panels + the composed 2x2 figure
```
