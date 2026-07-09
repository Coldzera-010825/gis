"""Clustering-visualization suite, basic -> advanced, on one simulated dataset.

A simulated city-neighbourhood table (socio-environmental indicators) carrying
four latent clusters is visualised progressively:

  clu-sub-scatter.jpg     basic: KMeans colouring on two raw features + centroids
  clu-sub-choosek.jpg     elbow (inertia) + silhouette score vs k
  clu-sub-pca2d.jpg       2D PCA projection, clusters coloured
  clu-sub-pca3d.jpg       3D PCA scatter (first three components)
  clu-sub-manifold.jpg    t-SNE vs UMAP side by side
  clu-sub-silhouette.jpg  per-sample silhouette "knife" plot
  clu-sub-clustermap.jpg  seaborn clustermap (the pheatmap equivalent)
  clu-main-panel.jpg      PCA + silhouette + choose-k composed (the hero)
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.preprocessing import StandardScaler

import umap

OUT = Path(__file__).resolve().parents[2] / "figures"
RNG = np.random.default_rng(20260709)
K = 4
PALETTE = ["#3D5A80", "#57A773", "#EE6C4D", "#C8963E"]

FEATURES = ["NDVI", "Building density", "Income", "Road density",
            "PM2.5", "Green access", "Population", "Land price"]


# ---------------------------------------------------------------- data
def simulate_neighbourhoods(n_per=150):
    """Four archetypal neighbourhood types in 8-D feature space."""
    # cluster centres in standardized-ish space, then add correlated noise
    centres = {
        "Green suburb":  [1.4, -1.2, 0.9, -1.1, -1.2, 1.5, -0.9, 0.4],
        "Dense core":    [-1.3, 1.6, 0.6, 1.4, 1.3, -1.2, 1.5, 1.5],
        "Industrial":    [-0.9, 0.8, -1.3, 1.3, 1.7, -0.9, 0.3, -0.9],
        "Old residential": [0.6, -1.5, -1.4, -0.4, -0.3, 0.3, 1.4, -1.5],
    }
    rows, labels = [], []
    for name, c in centres.items():
        block = RNG.normal(c, 0.5, size=(n_per, len(FEATURES)))
        rows.append(block)
        labels += [name] * n_per
    X = np.vstack(rows)
    df = pd.DataFrame(X, columns=FEATURES)
    return df, np.array(labels)


df, true_labels = simulate_neighbourhoods()
Xs = StandardScaler().fit_transform(df)

km = KMeans(n_clusters=K, n_init=10, random_state=0).fit(Xs)
clusters = km.labels_


def save(fig, name):
    fig.savefig(OUT / name, dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {name}")


def cmap_pts(labels):
    return [PALETTE[i] for i in labels]


# 1) basic: two raw features, coloured by KMeans, centroids marked
fig, ax = plt.subplots(figsize=(6.6, 5.2), facecolor="white")
ax.scatter(df["Building density"], df["NDVI"], c=cmap_pts(clusters),
           s=18, alpha=0.7, edgecolor="white", linewidth=0.3)
cent = StandardScaler().fit(df).inverse_transform(km.cluster_centers_)
ci_b, ci_n = FEATURES.index("Building density"), FEATURES.index("NDVI")
ax.scatter(cent[:, ci_b], cent[:, ci_n], c="black", s=180, marker="X",
           edgecolor="white", linewidth=1.5, zorder=5, label="centroids")
ax.set_xlabel("Building density"); ax.set_ylabel("NDVI")
ax.legend(frameon=False)
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)
save(fig, "clu-sub-scatter.jpg")

# 2) choosing k: elbow + silhouette
ks = range(2, 9)
inertias, sils = [], []
for k in ks:
    m = KMeans(n_clusters=k, n_init=10, random_state=0).fit(Xs)
    inertias.append(m.inertia_)
    sils.append(silhouette_score(Xs, m.labels_))
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2), facecolor="white")
a1.plot(list(ks), inertias, "o-", color="#3D5A80", lw=1.8)
a1.axvline(K, ls="--", color="#EE6C4D"); a1.set_title("Elbow (inertia)", loc="left")
a1.set_xlabel("k"); a1.set_ylabel("Within-cluster SSE")
a2.plot(list(ks), sils, "o-", color="#57A773", lw=1.8)
a2.axvline(K, ls="--", color="#EE6C4D"); a2.set_title("Silhouette score", loc="left")
a2.set_xlabel("k"); a2.set_ylabel("Mean silhouette")
for ax in (a1, a2):
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
save(fig, "clu-sub-choosek.jpg")

# 3) 2D PCA
pca = PCA(n_components=3).fit(Xs)
proj = pca.transform(Xs)
evr = pca.explained_variance_ratio_ * 100
fig, ax = plt.subplots(figsize=(6.6, 5.4), facecolor="white")
ax.scatter(proj[:, 0], proj[:, 1], c=cmap_pts(clusters), s=18, alpha=0.75,
           edgecolor="white", linewidth=0.3)
ax.set_xlabel(f"PC1 ({evr[0]:.1f}%)"); ax.set_ylabel(f"PC2 ({evr[1]:.1f}%)")
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)
save(fig, "clu-sub-pca2d.jpg")

# 4) 3D PCA
fig = plt.figure(figsize=(7.2, 6.0), facecolor="white")
ax = fig.add_subplot(111, projection="3d")
ax.scatter(proj[:, 0], proj[:, 1], proj[:, 2], c=cmap_pts(clusters),
           s=16, alpha=0.75, edgecolor="white", linewidth=0.2)
ax.set_xlabel(f"PC1 ({evr[0]:.1f}%)"); ax.set_ylabel(f"PC2 ({evr[1]:.1f}%)")
ax.set_zlabel(f"PC3 ({evr[2]:.1f}%)")
ax.view_init(elev=18, azim=-60)
save(fig, "clu-sub-pca3d.jpg")

# 5) t-SNE vs UMAP
ts = TSNE(n_components=2, perplexity=30, random_state=0, init="pca").fit_transform(Xs)
um = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=0).fit_transform(Xs)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 5.0), facecolor="white")
a1.scatter(ts[:, 0], ts[:, 1], c=cmap_pts(clusters), s=16, alpha=0.75,
           edgecolor="white", linewidth=0.2)
a1.set_title("t-SNE", loc="left"); a1.set_xticks([]); a1.set_yticks([])
a2.scatter(um[:, 0], um[:, 1], c=cmap_pts(clusters), s=16, alpha=0.75,
           edgecolor="white", linewidth=0.2)
a2.set_title("UMAP", loc="left"); a2.set_xticks([]); a2.set_yticks([])
save(fig, "clu-sub-manifold.jpg")

# 6) per-sample silhouette knife plot
sil_vals = silhouette_samples(Xs, clusters)
sil_avg = silhouette_score(Xs, clusters)
fig, ax = plt.subplots(figsize=(6.8, 5.4), facecolor="white")
y_lower = 0
for i in range(K):
    vals = np.sort(sil_vals[clusters == i])
    y_upper = y_lower + len(vals)
    ax.fill_betweenx(np.arange(y_lower, y_upper), 0, vals,
                     facecolor=PALETTE[i], alpha=0.8, edgecolor="none")
    ax.text(-0.05, (y_lower + y_upper) / 2, str(i), va="center", ha="right")
    y_lower = y_upper + 12
ax.axvline(sil_avg, color="#EE6C4D", ls="--", label=f"mean = {sil_avg:.2f}")
ax.set_xlabel("Silhouette coefficient"); ax.set_ylabel("Samples grouped by cluster")
ax.set_yticks([]); ax.legend(frameon=False, loc="lower right")
for s in ["top", "right", "left"]:
    ax.spines[s].set_visible(False)
save(fig, "clu-sub-silhouette.jpg")

# 7) seaborn clustermap (pheatmap-style): standardized features, both dendrograms
row_colors = pd.Series(clusters, index=df.index).map(dict(enumerate(PALETTE)))
cg = sns.clustermap(
    pd.DataFrame(Xs, columns=FEATURES), cmap="vlag", center=0,
    row_colors=row_colors, figsize=(8.2, 8.6), yticklabels=False,
    dendrogram_ratio=(0.14, 0.10), cbar_pos=(0.02, 0.83, 0.03, 0.13),
)
cg.ax_heatmap.set_xlabel("Feature")
cg.fig.savefig(OUT / "clu-sub-clustermap.jpg", dpi=150, facecolor="white",
               bbox_inches="tight")
plt.close(cg.fig)
print("saved clu-sub-clustermap.jpg")

# 8) hero composed panel: PCA 2D + silhouette + choose-k(silhouette)
fig = plt.figure(figsize=(13.2, 4.8), facecolor="white")
grid = fig.add_gridspec(1, 3, wspace=0.28, left=0.06, right=0.98,
                        top=0.90, bottom=0.14)
axp = fig.add_subplot(grid[0, 0])
axp.scatter(proj[:, 0], proj[:, 1], c=cmap_pts(clusters), s=14, alpha=0.75,
            edgecolor="white", linewidth=0.2)
axp.set_xlabel(f"PC1 ({evr[0]:.1f}%)"); axp.set_ylabel(f"PC2 ({evr[1]:.1f}%)")
axp.set_title("(a) PCA projection", loc="left")

axs = fig.add_subplot(grid[0, 1])
y_lower = 0
for i in range(K):
    vals = np.sort(sil_vals[clusters == i])
    y_upper = y_lower + len(vals)
    axs.fill_betweenx(np.arange(y_lower, y_upper), 0, vals,
                      facecolor=PALETTE[i], alpha=0.8)
    y_lower = y_upper + 12
axs.axvline(sil_avg, color="#EE6C4D", ls="--")
axs.set_xlabel("Silhouette coefficient"); axs.set_yticks([])
axs.set_title("(b) Silhouette", loc="left")

axk = fig.add_subplot(grid[0, 2])
axk.plot(list(ks), sils, "o-", color="#57A773", lw=1.8)
axk.axvline(K, ls="--", color="#EE6C4D")
axk.set_xlabel("k"); axk.set_ylabel("Mean silhouette")
axk.set_title("(c) Choosing k", loc="left")
for ax in (axp, axs, axk):
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
save(fig, "clu-main-panel.jpg")
print("done")
