# Principal Component Analysis (PCA)

PCA reduces a hyperspectral cube to a smaller set of bands ordered by
**variance**: the first principal component captures the most variation in the
data, the second the next most, and so on. Bands deemed *bad* are dropped
before the fit, so only good bands contribute.

## How it works

The covariance matrix of the (good-band) spectra is eigendecomposed. Each
eigenvector is a *principal component* — a direction in spectral space — and
its eigenvalue is the variance the data has along that direction. The cube is
projected onto the top *N* eigenvectors (largest eigenvalues), producing a new
dataset with *N* PCA bands ordered from most to least variance.

The result is added as a new dataset named `PCA on <source>`.

## Using the tool

:::{figure} ../../_static/tutorials/t6_pca_dialog.png
:width: 45%
:align: center
:alt: The PCA dialog with number of components and estimator matrix
:::

Right-click a raster image and choose **PCA**. In the dialog, set **Number of
Components** (how many PCA bands to keep) and click **OK**. The spin box
defaults to the maximum — the count of good bands. **Estimator Matrix** offers
only *Covariance* and is disabled.

When the run finishes, a PCA metadata widget and a **scree plot** (eigenvalue
vs. component index) open automatically. Click **View Past Results** to reopen
the scree plot for any earlier run; it helps you judge how many components are
worth keeping.

:::{figure} ../../_static/tutorials/lab_avng_scree.png
:width: 80%
:align: center
:alt: A PCA scree plot on log axes for a 425-band AVIRIS-NG cube
:::

A scree plot from a real 425-band cube: eigenvalues fall four orders of
magnitude by component 50, with the elbow near component 10–15. The 372
components shown are 425 bands minus the 53 flagged bad.

## See also

- {doc}`Tutorial 6 — PCA and MNF <../../tutorials/06-pca-mnf>`
- {doc}`Lab A <../../tutorials/labs/lab-aviris-ng-urban>` — PCA on a full
  AVIRIS-NG cube, with the composite and the scree plot read in detail
- {doc}`Minimum Noise Fraction <mnf>` — orders by signal-to-noise instead
