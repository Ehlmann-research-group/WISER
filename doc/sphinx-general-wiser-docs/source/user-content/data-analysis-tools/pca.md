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

Right-click a raster image and choose **PCA**. In the dialog, set **Number of
Components** (how many PCA bands to keep) and click **OK**. The spin box
defaults to the maximum — the count of good bands. **Estimator Matrix** offers
only *Covariance* and is disabled.

When the run finishes, a PCA metadata widget and a **scree plot** (eigenvalue
vs. component index) open automatically. Click **View Past Results** to reopen
the scree plot for any earlier run; it helps you judge how many components are
worth keeping.

<!-- PICTURE: Screenshot of the right-click context menu on a raster view with
     the "PCA" action highlighted. -->

<!-- PICTURE: Screenshot of the PCA dialog showing the Number of Components spin
     box, the (disabled) Estimator Matrix dropdown, and the View Past Results /
     OK / Cancel buttons. Label each control. -->

<!-- PICTURE: Example scree plot, showing eigenvalues on a log y-axis dropping
     off sharply, with the "elbow" annotated to indicate where added components
     stop contributing meaningful variance. -->
