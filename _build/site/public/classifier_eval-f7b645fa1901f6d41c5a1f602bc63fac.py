"""Classifier-evaluation suite for a simulated land-cover task.

Five classes, a RandomForest, and the four figures every classification
paper needs, plus the composed 2x2 layout:

  eval-sub-confusion.jpg   normalized, annotated confusion matrix
  eval-sub-roc.jpg         one-vs-rest ROC curves + macro average
  eval-sub-pr.jpg          precision-recall curves per class
  eval-sub-learning.jpg    learning curve (is more data still helping?)
  eval-main-panel.jpg      all four in one GridSpec figure (the hero)
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (auc, average_precision_score, confusion_matrix,
                             precision_recall_curve, roc_curve)
from sklearn.model_selection import learning_curve, train_test_split
from sklearn.preprocessing import label_binarize

OUT = Path(__file__).resolve().parents[2] / "figures"

CLASSES = ["Water", "Vegetation", "Built-up", "Bare soil", "Cropland"]
COLORS = ["#3D5A80", "#57A773", "#EE6C4D", "#C8963E", "#8064A2"]
N_CLASSES = len(CLASSES)

# ---------------------------------------------------------------- data + model
X, y = make_classification(
    n_samples=4000, n_features=12, n_informative=8, n_redundant=2,
    n_classes=N_CLASSES, n_clusters_per_class=1,
    class_sep=1.15, flip_y=0.06, random_state=7,
)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, stratify=y,
                                          random_state=7)

clf = RandomForestClassifier(n_estimators=300, random_state=7, n_jobs=-1)
clf.fit(X_tr, y_tr)
y_pred = clf.predict(X_te)
y_prob = clf.predict_proba(X_te)
y_bin = label_binarize(y_te, classes=range(N_CLASSES))


# ---------------------------------------------------------------- panels
def draw_confusion(ax):
    cm = confusion_matrix(y_te, y_pred, normalize="true")
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    for i in range(N_CLASSES):
        for j in range(N_CLASSES):
            colour = "white" if cm[i, j] > 0.55 else "#1f3044"
            ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center",
                    fontsize=9, color=colour)
    ax.set_xticks(range(N_CLASSES), CLASSES, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(N_CLASSES), CLASSES, fontsize=8)
    ax.set_xlabel("Predicted", fontsize=9)
    ax.set_ylabel("True", fontsize=9)
    return im


def draw_roc(ax):
    aucs = []
    for k in range(N_CLASSES):
        fpr, tpr, _ = roc_curve(y_bin[:, k], y_prob[:, k])
        aucs.append(auc(fpr, tpr))
        ax.plot(fpr, tpr, color=COLORS[k], lw=1.6,
                label=f"{CLASSES[k]} (AUC={aucs[-1]:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="#999999", lw=1)   # chance line
    ax.set_xlabel("False positive rate", fontsize=9)
    ax.set_ylabel("True positive rate", fontsize=9)
    ax.legend(fontsize=7.5, loc="lower right", frameon=False)
    ax.set_title(f"macro-AUC = {np.mean(aucs):.3f}", fontsize=9, loc="right")


def draw_pr(ax):
    for k in range(N_CLASSES):
        prec, rec, _ = precision_recall_curve(y_bin[:, k], y_prob[:, k])
        ap = average_precision_score(y_bin[:, k], y_prob[:, k])
        ax.plot(rec, prec, color=COLORS[k], lw=1.6,
                label=f"{CLASSES[k]} (AP={ap:.3f})")
    base = y_bin.mean()                                     # prevalence baseline
    ax.axhline(base, ls="--", color="#999999", lw=1)
    ax.set_xlabel("Recall", fontsize=9)
    ax.set_ylabel("Precision", fontsize=9)
    ax.set_ylim(0, 1.03)
    ax.legend(fontsize=7.5, loc="lower left", frameon=False)


def draw_learning(ax):
    sizes, tr_scores, va_scores = learning_curve(
        RandomForestClassifier(n_estimators=150, random_state=7, n_jobs=-1),
        X, y, cv=5, scoring="f1_macro",
        train_sizes=np.linspace(0.08, 1.0, 8), n_jobs=-1,
    )
    for scores, color, label in [(tr_scores, "#EE6C4D", "training"),
                                 (va_scores, "#3D5A80", "cross-validation")]:
        mean, std = scores.mean(axis=1), scores.std(axis=1)
        ax.plot(sizes, mean, "o-", color=color, lw=1.6, ms=4, label=label)
        ax.fill_between(sizes, mean - std, mean + std, color=color, alpha=0.15)
    ax.set_xlabel("Training samples", fontsize=9)
    ax.set_ylabel("Macro F1", fontsize=9)
    ax.legend(fontsize=8, frameon=False, loc="lower right")


def style(ax):
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=8)


def solo(draw_fn, name, w=6.4, h=5.2, colorbar=False):
    fig, ax = plt.subplots(figsize=(w, h), facecolor="white")
    im = draw_fn(ax)
    style(ax)
    if colorbar and im is not None:
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03,
                     label="Row-normalized share")
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {name}")


# ---------------------------------------------------------------- render
solo(draw_confusion, "eval-sub-confusion.jpg", w=6.2, h=5.4, colorbar=True)
solo(draw_roc, "eval-sub-roc.jpg")
solo(draw_pr, "eval-sub-pr.jpg")
solo(draw_learning, "eval-sub-learning.jpg", w=6.6, h=4.6)

# the composed 2x2 panel
fig = plt.figure(figsize=(12.4, 10.2), facecolor="white")
grid = fig.add_gridspec(2, 2, wspace=0.30, hspace=0.34,
                        left=0.08, right=0.97, top=0.95, bottom=0.07)
specs = [(draw_confusion, "(a) Confusion matrix"),
         (draw_roc, "(b) ROC curves (one-vs-rest)"),
         (draw_pr, "(c) Precision-recall curves"),
         (draw_learning, "(d) Learning curve")]
for spec, (fn, title) in zip(grid, specs):
    ax = fig.add_subplot(spec)
    im = fn(ax)
    style(ax)
    ax.set_title(title, loc="left", fontsize=11, pad=10)
    if fn is draw_confusion and im is not None:
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
fig.savefig(OUT / "eval-main-panel.jpg", dpi=150, facecolor="white",
            bbox_inches="tight")
plt.close(fig)
print("saved eval-main-panel.jpg")
print("done")
