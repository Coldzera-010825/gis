---
title: "Piece 08 · Clustering visualization, basic to advanced"
description: One neighbourhood dataset, visualised from a raw KMeans scatter up to PCA/t-SNE/UMAP projections, silhouette diagnostics and a seaborn clustermap — with the R pheatmap equivalent at the end.
---

# Clustering visualization

```{figure} figures/clu-main-panel.jpg
:width: 100%
:alt: PCA projection, silhouette plot and choose-k curve in one panel

The finished panel: the three figures that turn "I ran KMeans" into "the clusters
are real and there are four of them" — a PCA projection (a), a per-sample silhouette
plot (b), and the silhouette-vs-k curve that justifies k = 4 (c).
```

## 1 · Why this matters

Clustering is unsupervised, so the visualization *is* the result — there is no
accuracy score to fall back on. This piece walks one dataset from the most naive plot
to the diagnostics a reviewer expects, answering three questions in order: *can I see
structure? how many clusters? are they any good?*

The data is a simulated **city-neighbourhood** table — 600 neighbourhoods, eight
socio-environmental indicators (NDVI, building density, income, PM2.5…), built around
four archetypes (green suburb, dense core, industrial, old residential). One dataset
carries the whole tutorial:

```python
Xs = StandardScaler().fit_transform(df)      # always standardize before distance-based clustering
km = KMeans(n_clusters=4, n_init=10, random_state=0).fit(Xs)
clusters = km.labels_
```

:::{important}
**Standardize first.** KMeans, PCA, t-SNE and UMAP all use distances; a feature
measured in thousands (land price) would otherwise drown one measured in [0, 1]
(NDVI). `StandardScaler` is not optional.
:::

## 2 · Level 1 — the raw scatter

The simplest clustering figure: pick two interpretable features, colour points by
cluster label, mark the centroids. It is limited (only 2 of 8 dimensions) but it is
where every analysis starts:

```python
ax.scatter(df["Building density"], df["NDVI"], c=[PALETTE[i] for i in clusters],
           s=18, alpha=0.7, edgecolor="white", linewidth=0.3)
# centroids, mapped back to the original feature units
cent = StandardScaler().fit(df).inverse_transform(km.cluster_centers_)
ax.scatter(cent[:, b], cent[:, n], c="black", s=180, marker="X", zorder=5)
```

```{figure} figures/clu-sub-scatter.jpg
:width: 62%
:alt: KMeans clusters on building density vs NDVI with centroids

Green suburbs (high NDVI, low density) separate cleanly here; other pairs of clusters
overlap — because two features can't show eight-dimensional separation. That is the
motivation for everything below.
```

## 3 · Level 2 — how many clusters?

Never trust a `k` you didn't test. Two standard curves: the **elbow** (within-cluster
SSE, look for the kink) and the **silhouette score** (look for the peak). Here both
agree on 4:

```python
for k in range(2, 9):
    m = KMeans(n_clusters=k, n_init=10, random_state=0).fit(Xs)
    inertias.append(m.inertia_)
    sils.append(silhouette_score(Xs, m.labels_))
```

```{figure} figures/clu-sub-choosek.jpg
:width: 100%
:alt: Elbow curve and silhouette-score curve, both marking k=4

The elbow bends and the silhouette peaks at the same k — the reassuring case. When
they disagree, prefer the silhouette (it measures separation, not just compactness).
```

## 4 · Level 3 — PCA, seeing all dimensions at once

To see all eight dimensions of separation, project to the directions of greatest
variance. Always print the **explained-variance ratio** on the axes — a projection
without it is unfalsifiable:

```python
pca = PCA(n_components=3).fit(Xs)
proj = pca.transform(Xs)
evr = pca.explained_variance_ratio_ * 100
ax.set_xlabel(f"PC1 ({evr[0]:.1f}%)"); ax.set_ylabel(f"PC2 ({evr[1]:.1f}%)")
```

```{figure} figures/clu-sub-pca2d.jpg
:width: 60%
:alt: 2D PCA projection coloured by cluster

Two components capture ~67% of variance and the four clusters fall apart cleanly —
the honest evidence that the structure is real, not a 2-feature artefact.
```

For a third component, matplotlib's 3D axes add depth (the same viewpoint control as
Piece 01's ggcube):

```python
ax = fig.add_subplot(111, projection="3d")
ax.scatter(proj[:, 0], proj[:, 1], proj[:, 2], c=colors, s=16)
ax.view_init(elev=18, azim=-60)
```

```{figure} figures/clu-sub-pca3d.jpg
:width: 60%
:alt: 3D PCA scatter of the first three components

PC3 adds the separation the 2D view flattens — useful for exploration, though for a
paper a well-chosen 2D projection usually communicates better (occlusion is real in 3D).
```

## 5 · Level 4 — nonlinear projections: t-SNE vs UMAP

PCA is linear; manifold methods bend space to preserve *local* neighbourhoods, often
separating clusters PCA leaves touching. t-SNE and UMAP are the two standards — worth
showing side by side because they trade off differently:

```python
ts = TSNE(n_components=2, perplexity=30, init="pca").fit_transform(Xs)
um = umap.UMAP(n_neighbors=15, min_dist=0.1).fit_transform(Xs)
```

```{figure} figures/clu-sub-manifold.jpg
:width: 100%
:alt: t-SNE and UMAP projections side by side

Both isolate the four clusters as islands. Caveat for captions: **distances between
t-SNE/UMAP clusters are not meaningful** — only the grouping is. UMAP better preserves
global layout and is far faster on large data; t-SNE remains the familiar default.
```

## 6 · Level 5 — silhouette diagnostics

The number `silhouette_score = 0.53` hides the distribution. The per-sample
silhouette "knife" plot shows every point's fit: wide blades = tight clusters,
negative values = misassigned points:

```python
sil_vals = silhouette_samples(Xs, clusters)
for i in range(K):
    vals = np.sort(sil_vals[clusters == i])
    ax.fill_betweenx(np.arange(lo, lo + len(vals)), 0, vals, facecolor=PALETTE[i])
ax.axvline(silhouette_score(Xs, clusters), color="#EE6C4D", ls="--")
```

```{figure} figures/clu-sub-silhouette.jpg
:width: 60%
:alt: Per-sample silhouette knife plot for the four clusters

Four blades, all mostly right of the mean line, few negatives — the visual proof that
k = 4 produced clean clusters, not just a good average number.
```

## 7 · Level 6 — the clustermap (a heatmap with dendrograms)

The clustermap fuses a standardized heatmap with hierarchical clustering on **both**
axes — rows (neighbourhoods) reordered so similar ones sit together, columns
(features) grouped by co-variation. The left colour strip ties rows back to the KMeans
labels. This is the Python equivalent of R's **pheatmap**:

```python
row_colors = pd.Series(clusters).map(dict(enumerate(PALETTE)))
sns.clustermap(pd.DataFrame(Xs, columns=FEATURES), cmap="vlag", center=0,
               row_colors=row_colors, dendrogram_ratio=(0.14, 0.10))
```

```{figure} figures/clu-sub-clustermap.jpg
:width: 66%
:alt: seaborn clustermap with row colour strip matching KMeans clusters

The row dendrogram recovers the four blocks *without being told the labels* — and the
colour strip confirms they match KMeans. The feature dendrogram (top) reveals which
indicators move together: PM2.5 with road/building density, NDVI with green access.
```

## 8 · Composition and the hero panel

The three most defensible figures — PCA projection, silhouette, choose-k — compose
into one review-ready panel via the same `GridSpec` discipline as Pieces 03 and 07:

```python
fig = plt.figure(figsize=(13.2, 4.8))
grid = fig.add_gridspec(1, 3, wspace=0.28)
# (a) PCA scatter, (b) silhouette blades, (c) silhouette-vs-k
```

## 9 · The R route: pheatmap

The clustermap above is the Python answer; in the R ecosystem the classic tool is
**pheatmap** ("pretty heatmap"), which produces the same dendrogram-flanked,
annotation-strip heatmap and is ubiquitous in bioinformatics and ecology papers:

```r
library(pheatmap)
pheatmap(
  scale(mat),                       # standardized matrix (rows = samples)
  clustering_distance_rows = "euclidean",
  clustering_method = "ward.D2",    # linkage
  annotation_row = cluster_df,      # colour strip = cluster labels
  color = colorRampPalette(c("#3D5A80", "white", "#B2182B"))(100),
  show_rownames = FALSE
)
```

For a thorough, example-driven tour of `pheatmap` (and the wider R plotting
ecosystem), see this well-known walkthrough:
[**R语言绘图总汇 — 覆盖教程所有图形**](https://zhuanlan.zhihu.com/p/555785174).

## 10 · Run it yourself

Full source: [`clustering_viz.py`](./code/clustering/clustering_viz.py) — needs
`scikit-learn`, `seaborn`, `umap-learn`, `matplotlib`.

```bash
python clustering_viz.py
# → the eight figures on this page
```
