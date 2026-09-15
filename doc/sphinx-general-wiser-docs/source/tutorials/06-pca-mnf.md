# Tutorial 6 — PCA and MNF

**Goal:** compress a cube into a handful of bands, judge how many are worth
keeping, and know when to reach for the Minimum Noise Fraction (MNF) instead of
Principal Component Analysis (PCA).

**Data:** `caltech_425_7_7_nm.hdr` and `caltech_4_100_150_nm.hdr`.

---

## Why reduce dimensions

A 425-band cube is mostly redundant: neighboring bands measure nearly the same
thing, so the information lives in far fewer dimensions than the band count
suggests. Reducing them first makes every later step cheaper and less noisy — classifiers and unmixing models chase sensor noise
when handed 400 correlated channels.

Two transforms, two different orderings:

| | Orders components by | Use it when |
|---|---|---|
| **PCA** | **Variance** | A quick look, display, or feeding a classifier |
| **MNF** | **Signal-to-noise ratio** | Noise varies between bands — which on a real spectrometer it always does |

The distinction matters. PCA calls a component important if the data spreads
out along it — and a noisy band spreads out. MNF estimates the noise first,
rescales so noise is equal in every direction, *then* decomposes. Its leading
components carry real signal; its trailing components are where the noise went.

---

## Step 1 — Run PCA

1. Open a dataset.
2. Right-click the image and choose **Data Analysis ▸ Principal Component
   Analysis**, or use the **Main Menu** at **Tools ▸ Data Analysis ▸ Principal
   Component Analysis**. The **PCA Dialog** opens.

:::{figure} ../_static/tutorials/t6_pca_dialog.png
:width: 45%
:align: center
:alt: The PCA dialog with number of components and estimator matrix
:::

3. **Number of Components** defaults to the maximum, which is the count of good
   bands. Leave it there for a first run, since you decide how many to keep
   *after* seeing the scree plot.
4. **Estimator Matrix** offers only *Covariance* and is disabled, so there is
   nothing to set.
5. Click **OK**. **View Past Results** in the same dialog brings back the
   output of an earlier run without recomputing it.

WISER drops bad bands, eigendecomposes the covariance matrix of what remains,
and projects the cube onto the leading eigenvectors. The result is added as
**`PCA on <source>`**, ordered most to least variance.

```{note}
PCA is fast even on a large cube. On the 680 × 500 × 425 AVIRIS-NG scene in
{doc}`Lab A <labs/lab-aviris-ng-urban>` it completes in about **7 seconds** and
returns 372 components — 425 bands minus the 53 flagged bad.
```

---

## Step 2 — Look at the components

1. Switch to the result with **Select dataset to view** on the Main Toolbar.
2. Click **Band chooser**, select **Grayscale**, set **Grayscale Band** to
   band 0, and click **OK**.
3. Step through bands 1, 2 and 3 the same way.
4. To build a false-color composite, select **RGB** in the **Band Chooser** and
   set **Red Band**, **Green Band** and **Blue Band** to bands 0, 1 and 2.

```{admonition} Interpretation
:class: note
Band 0 is almost always overall brightness. Everything in a scene reflects more
or less light in total, so the largest single direction of variance tends to be
albedo.

Bands 1, 2 and 3 look different, because later components carry the
*differences* between materials rather than their brightness. That is why a
false-color composite of the first three components often separates surfaces
that look alike in true color. Lab A shows that on a full cube.
```

```{note}
**Principal components have no physical units.** A PCA band is a projection
onto an eigenvector, so its values are not reflectance and its sign is
arbitrary — an inverted-looking component is not a bug. Interpret them by what
they separate, and go back to the original cube whenever you need real spectra.
```

---

## Step 3 — Read the scree plot

When the run finishes, WISER opens a **scree plot**: eigenvalue against
component index, on a log scale. It answers "how many components should I
keep?".

Look for the **elbow** — where the curve flattens. Components before it hold
structure; after it, noise. On the AVIRIS-NG scene in Lab A the eigenvalues
fall four orders of magnitude by component 50, with the elbow around component
10–15: of 372 components, roughly the first dozen carry the scene.

Click **View Past Results** in the PCA dialog to reopen the scree plot for any
earlier run without recomputing.

---

## Step 4 — MNF

MNF is at **Tools ▸ Data Analysis ▸ Minimum Noise Fraction**.

:::{figure} ../_static/tutorials/t6_mnf_dialog.png
:width: 45%
:align: center
:alt: The MNF dialog with dataset and component count
:::

1. Pick the **Dataset** — the dropdown starts on *(no data)*, so choose it
   explicitly.
2. **Num Components** defaults to the maximum for that dataset: the good-band
   count, capped by the number of noise samples available.
3. Click **OK**. The result is **`MNF, Img: <source>`**.

How WISER estimates the noise decides whether the result is meaningful:

1. **Shift difference.** Each pixel is subtracted from its neighbor one row
   below. Adjacent pixels should be nearly the same, so what remains is mostly
   sensor noise.
2. **Whitening.** The noise covariance rescales every band so estimated noise
   has unit variance in all directions.
3. **Eigendecomposition** of the whitened data covariance.
4. **Projection** onto the top *N* eigenvectors.

```{note}
The shift-difference estimate assumes neighboring pixels are similar. It works
on a scene with smooth spatial structure and misleads on one that is busy at
the pixel scale — a striped scene, a dense urban scene at meter resolution, or
an image with strong along-track banding. If MNF results look wrong, check this
assumption first.
```

Use MNF's leading components as the input to K-means, unmixing, or matched
filtering. That is exactly what {doc}`MTMF <07-detection>` does internally.

---

## Which one, and how many

| Situation | Reach for |
|---|---|
| Quick visual survey of a big cube | **PCA**, 3–5 components as false color |
| Input to a classifier or unmixing | **MNF**, components up to the elbow |
| Reducing noise before a detection run | **MNF** — that is what it is for |
| You need real reflectance values back | Neither — go to the original cube |

Both tools drop bands flagged bad in the header before fitting, so a cube with
a proper bad-band list gives better results than one without.

---

## What you can now do

- Run PCA and MNF and say which one a task calls for
- Use a scree plot to choose a component count
- Recognize a brightness component and interpret later ones
- Say when the shift-difference noise estimate is not trustworthy

---

**Next:** {doc}`Tutorial 7 — Finding a Known Material <07-detection>`.
