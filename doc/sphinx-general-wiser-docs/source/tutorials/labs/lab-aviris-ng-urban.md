# Lab A — Urban Vegetation and Materials with AVIRIS-NG

- **Field:** urban ecology, land-cover mapping, remote-sensing methods
- **Instrument:** AVIRIS-NG — 425 bands, 377–2500 nm at ~5 nm, ~5 m pixels
- **Prerequisites:** {doc}`Tutorials 1–7 <../index>`
- **Time:** 2–3 hours

This is the lab to start with. The data is one download with no account, the
scene is a place you can look up, and every figure below was produced by
running the steps in WISER.

---

## The question

An urban scene is a mosaic of a few materials — vegetation, concrete, asphalt,
roofing, water — packed at a scale of metres. A colour photograph separates
some of them and confuses the rest: a green roof and a green tree are the same
colour, and two roofs of different material can look identical.

With 425 contiguous bands each of those materials has a *shape*, not just a
colour. In this lab you use that shape to:

1. read the physical signature of four surfaces,
2. map vegetation quantitatively across the scene,
3. find structure without being told what to look for, and
4. compress 425 bands into the handful that carry the information.

---

## Get the data

**Scene:** `ang20171108t184227` — AVIRIS-NG over Pasadena, California,
8 November 2017, subset to the Caltech campus and its surroundings.

Download both files from JPL — no account, no login:

```bash
cd src/test_utils/test_datasets

curl -O https://avng.jpl.nasa.gov/pub/DThompson/istutor/ang20171108t184227_corr_v2p13_subset_bil
curl -O https://avng.jpl.nasa.gov/pub/DThompson/istutor/ang20171108t184227_corr_v2p13_subset_bil.hdr
```

The data file is **551 MB**; the header is 15 KB. Put them in the same
directory — WISER finds the data file from the header, or the header from the
data file, but only if they sit together.

```{admonition} If your download arrives with an .img extension
:class: note
Some browsers add one. `ang20171108t184227_corr_v2p13_subset_bil.img` beside
`....hdr` works exactly the same way — ENVI data files may be named with no
extension, `.img`, or `.dat`. What matters is that the base name matches and
both files are in the same folder.
```

That path is git-ignored, so the scene will not be committed if you work in a
clone of the WISER repository.

**Reference spectra** (used in Part 5): the
[USGS Spectral Library Version 7](https://dx.doi.org/10.5066/F7RR1WDJ), or the
smaller `src/test_utils/test_spectra/usgs_resampHeadwallSWIR.hdr` that ships
with WISER.

### What you are opening

| | |
|---|---|
| Dimensions | 680 samples × 500 lines × **425 bands** |
| Wavelengths | 376.9 – 2500.5 nm, mean sampling **5.01 nm** |
| Data type | 32-bit float, BIL interleave |
| Units | Surface reflectance (`corr` = atmospherically corrected) |
| Bad bands | **53 of 425** flagged in the header's `bbl` |
| Georeferencing | UTM Zone 11N, WGS-84, 2 m map info, rotation −2° |
| Nodata | `-9999` |

The 53 flagged bands are the water-vapour regions near 1400 nm and 1900 nm,
plus the noisy ends of the range. Every WISER tool drops them automatically.

---

## Part 1 — Open and orient (20 min)

1. **File ▸ Open...** and select either file. Leave the type on **All
   supported files**.
2. Turn on **Context**, **Zoom**, **Spectrum Plot** and **Dataset Info**, then
   **Zoom to fit**.
3. Open the **contrast stretch** and click **2.5% linear**.

:::{figure} ../../_static/tutorials/lab_avng_truecolour.png
:width: 95%
:align: center
:alt: The AVIRIS-NG Caltech scene in WISER, true colour, all panes visible
:::

Opening a 551 MB cube takes a few seconds; after that, panning and zooming are
immediate, because WISER reads bands on demand rather than holding the whole
cube in memory.

You are looking at the Caltech campus and the blocks around it: the athletic
field and pools at the bottom, the olive walk running through the middle, and
Pasadena's street grid. The black margins on the left and right are outside the
flight line — nodata, and they matter in Part 3.

### Which band is which

Open the **band chooser**. Every band is labelled with its wavelength, which is
how you pick bands by physics rather than by index.

:::{figure} ../../_static/tutorials/lab_avng_band_chooser.png
:width: 45%
:align: center
:alt: The band chooser listing AVIRIS-NG bands by wavelength
:::

The bands this lab uses:

| Wavelength | Band index | Why |
|---|---|---|
| 482 nm | 21 | Blue |
| 552 nm | 35 | Green |
| 662 nm | 57 | Red — chlorophyll absorption |
| 858 nm | 96 | NIR — the top of the red edge |
| 1649 nm | 254 | SWIR-1 |
| 2200 nm | 364 | SWIR-2 — clay/carbonate region |

### A composite true colour cannot give you

Set the band chooser to **RGB** with **2200 / 1649 / 858 nm** and re-apply a
2.5% stretch.

:::{figure} ../../_static/tutorials/lab_avng_swir.png
:width: 95%
:align: center
:alt: SWIR false-colour composite of the same scene
:::

Vegetation turns deep blue — it absorbs strongly in both SWIR bands but stays
bright in the NIR. Roofs and pavement turn yellow and tan, and now they
*separate from each other* in a way they never do in true colour. Nothing here
is new information: it was in the cube all along, and choosing three different
bands is all it took to see it.

**Deliverable 1:** the true-colour and SWIR composites of the same area, and
two sentences on what the SWIR composite distinguishes that true colour does
not.

---

## Part 2 — Four surfaces, four spectra (30 min)

Click each of these pixels and **collect** the spectrum
({doc}`Tutorial 2 <../02-spectra>`). Rename and recolour each one from the list
below the plot, or they will all be drawn the same.

| Surface | Pixel (x, y) | What it is |
|---|---|---|
| Vegetation | 330, 141 | Dense canopy |
| Swimming pool | 298, 428 | Open water |
| Roof | 462, 301 | Bright reflective roof |
| Asphalt | 631, 129 | Paved surface |

:::{figure} ../../_static/tutorials/lab_avng_spectra_plot.png
:width: 95%
:align: center
:alt: Four AVIRIS-NG spectra: vegetation, swimming pool, roof and asphalt
:::

### Read them

This one figure contains most of what optical remote sensing rests on.

**Vegetation** (green) is dark and flat through the visible, then jumps almost
vertically at about 700 nm from 0.04 to 0.71 — the **red edge**, the boundary
between chlorophyll absorption and the cellular scattering of leaf mesophyll.
It stays high across the NIR plateau, then steps down twice: the **leaf-water
absorptions** at roughly 1400 nm and 1900 nm, which are so deep they coincide
with the flagged atmospheric bands, leaving the gaps you see. Between them sit
the two SWIR shoulders near 1650 nm and 2200 nm.

**Water** (blue) is the mirror image: brightest in the blue at about 0.38,
falling steadily through the green and red, and essentially **zero beyond
750 nm**. Liquid water absorbs the near-infrared almost completely. Any pixel
that is dark in the NIR and bright in the blue is water, and no other common
surface behaves that way.

**The roof** (red) is bright — around 0.95 — and nearly flat from 500 nm to
1800 nm before declining through the SWIR. High and featureless is the
signature of a broadband reflector.

**Asphalt** (grey) is dark in the visible, around 0.07, and rises steadily all
the way to 0.45 in the SWIR. That steady climb is what separates it from the
roof: both are "grey" to the eye, and they are nothing alike past 1000 nm.

```{admonition} Why the gaps are not a defect
:class: note
The three breaks in every line are the 53 bands the header flags bad. At those
wavelengths atmospheric water vapour absorbs nearly all the signal, so nothing
reliable about the surface survives. WISER omits them from the plot and every
tool excludes them from its computation. A spectrum drawn straight through
those regions is showing you the atmosphere, not the ground.
```

**Deliverable 2:** the four spectra on one labelled plot, with the red edge,
the two leaf-water absorptions and water's NIR cutoff annotated.

---

## Part 3 — Map the vegetation (30 min)

The vegetation and asphalt spectra diverge hardest between 662 nm and 858 nm.
Turn that into a number for every pixel.

1. **Tools ▸ Band math...**
2. Expression:

   ```text
   (nir - red) / (nir + red)
   ```

3. Bind `nir` to **Band 96: 857.69 nm** and `red` to **Band 57: 662.35 nm**,
   both as type **Image Band**. Name the result `NDVI`.

:::{figure} ../../_static/tutorials/lab_avng_bandmath.png
:width: 95%
:align: center
:alt: The band math dialog with NDVI bound to the 858 nm and 662 nm bands
:::

Unlike a 4-band sensor, AVIRIS-NG lets you use the textbook wavelengths
directly — no substituting a red-edge band for red and hoping.

### The result will look wrong. That is the lesson.

:::{figure} ../../_static/tutorials/lab_avng_ndvi_unstretched.png
:width: 95%
:align: center
:alt: The NDVI result with the default stretch: a uniform green field
:::

Flat, uniform green. Nothing is broken — the **stretch** is wrong.

The flight-line edges contain nodata, and where a near-zero denominator falls
in the margin the ratio blows up. Across this scene NDVI actually spans

| | value |
|---|---|
| minimum | **−3.35** |
| 2.5th percentile | −0.03 |
| median | +0.17 |
| 97.5th percentile | +0.83 |
| maximum | +0.93 |

A default 100% linear stretch maps −3.35 to black and +0.93 to white, so the
entire meaningful range is squeezed into the top quarter of the ramp.

Fix it: open the **contrast stretch** and click **2.5% linear**, or set the
**Minimum** and **Maximum** limits to −0.2 and 0.9 explicitly. Then choose the
**RdYlGn** colormap — a diverging map, because NDVI has a meaningful zero.

:::{figure} ../../_static/tutorials/lab_avng_ndvi.png
:width: 95%
:align: center
:alt: The NDVI map after a 2.5% linear stretch, resolving individual tree crowns
:::

Now it is a map. Individual tree crowns resolve as separate green blobs, the
street trees line up along the avenues, the athletic field and the lawns are
solid green, and the buildings and roads are red.

```{admonition} The general rule
:class: important
**Any computed product needs its stretch set before you read it.** The values
that come out of band math, an index, a band-depth calculation or a detection
score have no reason to fill a display range sensibly, and one nodata pixel at
a flight-line edge can flatten everything else. Check the histogram in the
stretch dialog before you conclude anything from a computed image.
```

### Compare side by side

Split the main view with the **grid** button and show the true-colour cube in
one panel and NDVI in the other.

:::{figure} ../../_static/tutorials/lab_avng_ndvi_vs_rgb.png
:width: 95%
:align: center
:alt: True colour and NDVI side by side in a 1x2 grid
:::

**Deliverable 3:** the NDVI map with your stretch limits stated, the
before/after pair, and the fraction of the scene above a canopy threshold you
justify from the histogram.

---

## Part 4 — Structure without supervision (40 min)

### 4a. Feature space

Open **Data Analysis ▸ Interactive Scatter Plot**, set **X** to band 57
(662 nm), **Y** to band 96 (858 nm), render onto the cube, and **Create Plot**.

:::{figure} ../../_static/tutorials/lab_avng_scatter.png
:width: 75%
:align: center
:alt: Density scatter plot of 662 nm against 858 nm for the whole scene
:::

Two structures, and both mean something:

- **The diagonal ridge** is the **soil line** — surfaces whose red and NIR
  reflectance rise together. Roofs, roads and bare ground all lie along it,
  dark ones near the origin and bright ones far out.
- **The near-vertical plume** climbing off the origin is **vegetation**: NIR
  rising to 0.6 and beyond while red stays below 0.1. NDVI is, geometrically, a
  measure of how far a point sits above the soil line.

Lasso the plume. The point count updates and the matching pixels are marked on
the image.

:::{figure} ../../_static/tutorials/lab_avng_scatter_selection.png
:width: 75%
:align: center
:alt: A polygon selection around the vegetation plume, 21,174 points selected
:::

:::{figure} ../../_static/tutorials/lab_avng_scatter_highlight.png
:width: 95%
:align: center
:alt: The selected feature-space pixels highlighted across the scene
:::

Here 21,174 pixels fall inside the polygon, and they land on the canopy. You
went from a shape in feature space to a place on the ground without deciding in
advance what either was. **Create ROI from Selection** turns that into a
Region of Interest you can use anywhere else.

### 4b. K-means over all 425 bands

**Tools ▸ Data Analysis ▸ K-means**, input the cube, **K = 6**, and under
**Advanced Options** set **Random Seed** to 42 so your result is reproducible.

:::{figure} ../../_static/tutorials/lab_avng_kmeans_dialog.png
:width: 45%
:align: center
:alt: The K-means dialog set to six clusters
:::

On this cube — 680 × 500 × 425 — the run took **16 seconds**.

:::{figure} ../../_static/tutorials/lab_avng_kmeans.png
:width: 95%
:align: center
:alt: Six-cluster K-means labels over the AVIRIS-NG scene
:::

```{admonition} Read this result critically
:class: important
It is speckled. Clusters change from pixel to pixel across surfaces that are
obviously uniform on the ground, and the six classes do not map cleanly onto
six materials.

That is what K-means does to a high-resolution scene with 425 correlated,
partly noisy bands: it partitions total spectral variance, and at 5 m most of
that variance is *within-surface* — illumination, shadow, sub-pixel mixing,
sensor noise — rather than *between-material*. The fix is Part 5: reduce to the
components that carry signal, then cluster those. Compare the two and you will
see the difference immediately.

Cluster indices are also arbitrary. Use **View Centroids** to plot each
cluster's mean spectrum and find out what it actually is before naming it.
```

**Deliverable 4:** the feature-space selection and its image highlight, the
K-means map, and a paragraph on why the clustering is noisier than the scene.

---

## Part 5 — Compress 425 bands (30 min)

Right-click the image and choose **PCA**. Leave the component count at its
maximum and click **OK**.

:::{figure} ../../_static/tutorials/lab_avng_pca_dialog.png
:width: 45%
:align: center
:alt: The PCA dialog
:::

The run took **7 seconds** and produced **372 components** — 425 bands minus
the 53 flagged bad. The scree plot opens automatically.

:::{figure} ../../_static/tutorials/lab_avng_scree.png
:width: 80%
:align: center
:alt: PCA scree plot on log axes, showing eigenvalues falling four orders of magnitude
:::

Read it on the log axis. The first eigenvalue is above 1, and by component 50
they are down near 10⁻⁵ — **four orders of magnitude**. The elbow is somewhere
around component 10 to 15. Everything past roughly component 50 is noise, and
the fact that 372 components exist tells you nothing about how many matter.

A false-colour composite of PC1/PC2/PC3:

:::{figure} ../../_static/tutorials/lab_avng_pc_composite.png
:width: 95%
:align: center
:alt: A PC1/PC2/PC3 false-colour composite
:::

Vegetation is green, roofs magenta and pink, pavement blue-purple — and roofs
that were indistinguishable in true colour now differ. Three numbers per pixel,
carrying most of what 425 bands had to say.

:::{figure} ../../_static/tutorials/lab_avng_pc1.png
:width: 95%
:align: center
:alt: The first principal component in greyscale
:::

PC1 is overall brightness, as it almost always is: everything reflects more or
less light in total, so that is the largest single direction of variance.

### Now do Part 4b again, properly

Run **MNF** ({doc}`Tutorial 6 <../06-pca-mnf>`), keep the components up to the
elbow, and run K-means on the MNF result instead of the raw cube. Compare
against the map you made in Part 4b.

**Deliverable 5:** the scree plot with your chosen component count marked and
justified, the PC composite, and the two K-means maps side by side with an
explanation of the difference.

---

## Questions to answer

1. A green roof and a tree are the same colour in the true-colour composite.
   Name two bands that separate them and explain the physics of each.
2. Your NDVI map has a value of −3.35 somewhere. Where does that come from, and
   why does it not appear once you apply a 2.5% stretch?
3. PCA gave 372 components for a 425-band cube. Where did the other 53 go, and
   why is dropping them the right behaviour rather than a loss?
4. K-means on the raw cube is noisier than K-means on MNF components. Explain
   why in terms of what each method is ranking.
5. The pool spectrum is essentially zero beyond 750 nm. What does that imply
   for detecting *shallow* or *turbid* water, as opposed to a clean deep pool?

---

## Going further

- **Map impervious surface.** Combine the NDVI mask with a SWIR brightness
  threshold and estimate the impervious fraction per block — the standard input
  to urban runoff and heat-island models.
- **Look for roofing materials.** Collect roof spectra, continuum-remove them,
  and run {doc}`SFF <../07-detection>` over 2000–2400 nm. Asphalt shingle,
  membrane, tile and gravel differ there.
- **Find the stressed trees.** Compare NDVI against a red-edge position index
  computed from the 700–740 nm bands, which AVIRIS-NG samples finely enough to
  resolve. Red-edge position shifts before NDVI drops.
- **Compare against EMIT.** {doc}`Lab C <lab-emit-dust>` uses the same
  measurement from orbit at 60 m. Run the same NDVI on both and see what 5 m
  resolves that 60 m averages away.
