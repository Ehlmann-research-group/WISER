# Minimum Noise Fraction (MNF)

MNF reduces a hyperspectral cube to a smaller set of bands ordered by
**signal-to-noise ratio** rather than by raw variance (as PCA does), so the
first MNF bands hold the cleanest information and the last hold mostly noise.

## How it works

1. **Estimate noise.** Noise is approximated with the *shift-difference*
   method: each pixel is subtracted from its neighbor one row below. Adjacent
   pixels should be nearly identical, so what remains is largely sensor noise.
2. **Whiten by noise.** The noise covariance is computed and used to build a
   whitening transform that rescales every band so the estimated noise has
   equal (unit) variance in all directions.
3. **Eigendecomposition.** A standard eigendecomposition (PCA) is run on the
   whitened data covariance. Because the data is already noise-normalized, the
   resulting components are ranked by signal-to-noise ratio in descending
   order.
4. **Project.** The original cube is projected onto the top *N* eigenvectors,
   producing a new dataset with *N* MNF bands.

The result is added as a new dataset named `MNF, Img: <source>`.

## Using the tool

:::{figure} ../../_static/tutorials/t6_mnf_dialog.png
:width: 45%
:align: center
:alt: The MNF dialog with dataset and component count
:::

Pick a **Dataset**, set **Num Components** (the number of MNF bands to keep),
and click **OK**. The spin box defaults to the maximum allowed for that
dataset — the count of good bands, capped by the available noise samples — so
that ceiling is visible at a glance.

Click **View Past Results** to reopen any previous run and view its scree plot
(eigenvalue vs. component index), which helps you judge how many components are
worth keeping.

## See also

- {doc}`Tutorial 6 — PCA and MNF <../../tutorials/06-pca-mnf>`
- {doc}`Principal Component Analysis <pca>` — orders by variance instead
- {doc}`MTMF <mtmf>` — runs a full MNF internally
