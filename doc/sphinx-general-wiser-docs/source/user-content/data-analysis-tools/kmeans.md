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

Choose an **Input Dataset**, enter **K clusters**, and click **OK**. The run
proceeds in the background. Click **View Centroids** to open any stored result
and plot its centroid spectra.

### Advanced Options

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
