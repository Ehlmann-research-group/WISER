# Tutorial 5 — Classifying a Scene

**Goal:** find structure in feature space with the interactive scatter plot,
then let K-means partition the scene without being told what to look for.

**Data:** `caltech_4_100_150_nm.hdr` (from {doc}`Tutorial 1 <01-first-look>`).

---

## Part A — Feature space

### Step 1 — Plot two bands against each other

1. Open the campus scene.
2. Right-click the image and choose **Data Analysis ▸ Interactive Scatter
   Plot** (also at **Tools ▸ Data Analysis ▸ Interactive Scatter Plot**).
3. At the bottom of the window set:
   - **X Axis Band** — `caltech_4_100_150_nm`, **Band 2: 702.42 nm**
   - **Y Axis Band** — the same dataset, **Band 3: 852.68 nm**
   - **Render Onto** — the same dataset
4. Click **Create Plot**.

:::{figure} ../_static/tutorials/t8_scatter_empty.png
:width: 70%
:align: center
:alt: The interactive scatter plot before plotting, showing the band selectors
:::

All three datasets must share the same width and height, since points are
matched to pixels by position.

### Step 2 — Read the cloud

:::{figure} ../_static/tutorials/t8_scatter_plot.png
:width: 70%
:align: center
:alt: Density scatter plot of the 702 nm band against the 852 nm band
:::

Every pixel is one point, colored by how many pixels share that spot (the
default is a **density** plot; **To Scatter** switches to plain dots, easier to
read where the population is sparse).

The shape is textbook:

- **The diagonal ridge** from the origin to the upper right is the **soil
  line** — surfaces whose 702 nm and 852 nm reflectances rise together. Roofs,
  roads and bare ground lie along it, dark ones at the bottom, bright at the
  top.
- **The plume rising above the diagonal** at low 702 nm values is
  **vegetation**: high near-infrared, suppressed red edge. NDVI is, in effect,
  a measure of how far a point sits above that line.

### Step 3 — Link the plot back to the image

The polygon selector is always live.

1. Click around the vegetation plume to drop vertices; finish with a
   **double-click** or **Enter**.
2. The selected points are outlined, the **N pts** counter updates, and the
   matching pixels are marked on the **Render Onto** image.

:::{figure} ../_static/tutorials/t8_scatter_selection.png
:width: 70%
:align: center
:alt: A polygon selection around the vegetation plume
:::

:::{figure} ../_static/tutorials/t8_scatter_highlight.png
:width: 90%
:align: center
:alt: The selected feature-space pixels highlighted in red across the image
:::

The highlighted pixels land exactly on the tree crowns, hedges and lawns. You
went from *a shape in feature space* to *a place on the ground* without knowing
in advance what either was.

Click **Create ROI from Selection** to turn that into a Region of Interest, and
everything in {doc}`Tutorial 3 <03-regions-of-interest>` becomes available.

**Escape** or **Clear selection** starts over.

---

## Part B — K-means

### Step 4 — Run it

Where the scatter plot compares two bands, K-means uses **all** of them: each
pixel is a point in *n*-band space, and the algorithm partitions those points
into **K** clusters, iterating until the centers stop moving.

1. **Tools ▸ Data Analysis ▸ K-means**.
2. **Input Dataset** — `caltech_4_100_150_nm`.
3. **K clusters** — `5`.

   :::{figure} ../_static/tutorials/t5_kmeans_dialog.png
   :width: 45%
   :align: center
   :alt: The K-means dialog with five clusters requested
   :::

4. Expand **Advanced Options** and set **Random Seed** to `42`.

   :::{figure} ../_static/tutorials/t5_kmeans_advanced.png
   :width: 45%
   :align: center
   :alt: K-means advanced options: init method, iterations, tolerance, seed, algorithm
   :::

   The seed matters more than it looks. K-means starts from randomly chosen
   centers, so two unseeded runs on identical data give different — and
   differently *numbered* — clusters. Fix the seed and your figure is
   reproducible.

5. Click **OK**. The run proceeds in the background; watch it in the
   **Activity Monitor**.

### Step 5 — Read the labels

The result is a single-band **label image**, `K-Means Labels (k=5): ...`, whose
pixel values are cluster indices 0…4. Display it with a categorical colormap
such as **tab10**.

:::{figure} ../_static/tutorials/t5_kmeans_labels.png
:width: 90%
:align: center
:alt: The five-cluster K-means label image over the Caltech scene
:::

The clusters recover the scene's structure: bright roofs, darker roofs, road
and parking surfaces, canopy, and shadow.

```{important}
**Cluster colors and numbers mean nothing on their own.** K-means is
unsupervised: it finds groups, it does not name them. Cluster 3 is not
"vegetation" until you check. Click **View Centroids** to plot the mean
spectrum of each cluster — the one that climbs steeply from 702 to 852 nm is
your canopy class.
```

### Step 6 — Choosing K

There is no correct K. Ways to decide:

- **Run several.** Try K = 3, 5 and 8 and see which boundaries persist.
- **Look at the centroids.** If two clusters have near-identical spectra, K is
  too high.
- **Count your classes.** If you know the scene holds four materials, start at
  four and add one for shadow.
- **Reduce dimensions first.** On a cube with hundreds of bands, run
  {doc}`PCA or MNF <06-pca-mnf>` and cluster the leading components: faster,
  and far less prone to chasing noise.

That last point is not a detail. {doc}`Lab A <labs/lab-aviris-ng-urban>` runs
K-means on a raw 425-band cube and the result is visibly speckled — the method
partitions *total* variance, and on a high-resolution scene most of that is
within-surface variation rather than between-material difference.

The other Advanced Options:

| Option | Effect |
|---|---|
| **Initialization** | `k-means++` (default), `random`, or `manual` — pick your own starting spectra, which turns K-means into a semi-supervised classifier |
| **Number of Initializations** | Re-runs from different starts, keeps the best; raise when results look unstable |
| **Max Iterations / Convergence Tolerance** | Stopping rules |
| **Algorithm** | `lloyd` (classic) or `elkan` (faster on well-separated clusters, more memory) |

Bad bands are excluded from the distance computation, and nodata pixels are
labeled −1.

---

## What you can now do

- Read a two-band feature space and recognize the soil line
- Move between feature space and image space in both directions
- Run a reproducible unsupervised classification
- Identify what a cluster actually is, rather than assuming

---

**Next:** {doc}`Tutorial 6 — PCA and MNF <06-pca-mnf>` — compress hundreds of
bands into the few that carry the information.
