# K-Means Clustering

K-Means groups every pixel into one of **K** clusters by spectral similarity,
producing an unsupervised classification of the scene.

## How it works

Each pixel's spectrum is a point in band-space. K-Means picks K initial
centroids, then repeats two steps until the centroids stop moving (or a limit
is hit): assign each pixel to its nearest centroid (Euclidean distance), then
recompute each centroid as the mean of its assigned pixels.

The output is a single-band **label image** where each pixel's value is its
cluster index (0…K−1); nodata pixels are set to −1. The final cluster centroid
spectra are stored and viewable. Bad bands are excluded from the distance
computation.

The result is added as a new dataset named `K-Means Labels (k=K): <source>`.

## Using the tool

:::{figure} ../../_static/tutorials/t5_kmeans_dialog.png
:width: 45%
:align: center
:alt: The K-means dialog with input dataset and cluster count
:::

Choose an **Input Dataset**, enter **K clusters**, and click **OK**. The run
proceeds in the background. Click **View Centroids** to open any stored result
and plot its centroid spectra.

### Advanced Options

:::{figure} ../../_static/tutorials/t5_kmeans_advanced.png
:width: 45%
:align: center
:alt: K-means advanced options
:::

Expand **Advanced Options** to control the fit (all optional; sensible defaults
are used when blank):

- **Initialization Method** — `k-means++` (smart default), `random`, or
  `manual` (pick your own starting spectra; this hides the seed/initializations
  fields since the start is fixed).
- **Number of Initializations** — how many times to re-run with different
  starts, keeping the best.
- **Max Iterations** — iteration cap per run.
- **Convergence Tolerance** — stop once centroids move less than this.
- **Random Seed** — fix for reproducible results.
- **Algorithm** — `lloyd` (classic) or `elkan` (faster on well-separated
  clusters, more memory).

:::{figure} ../../_static/tutorials/t5_kmeans_labels.png
:width: 85%
:align: center
:alt: A five-cluster K-means label image with a categorical colormap
:::

Cluster indices are arbitrary — cluster 3 is not "vegetation" until you check.
Use **View Centroids** to plot each cluster's mean spectrum and identify it.

```{tip}
On a cube with hundreds of bands, run {doc}`MNF <mnf>` first and cluster the
leading components. K-means partitions *total* variance, and on a
high-resolution scene most of that is within-surface variation rather than
between-material difference —
{doc}`Lab A <../../tutorials/labs/lab-aviris-ng-urban>` shows what the raw-cube
result looks like.
```

## See also

- {doc}`Tutorial 5 — Classifying a Scene <../../tutorials/05-classification>`
- {doc}`Interactive Scatter Plot <interactive-scatter-plot>`
