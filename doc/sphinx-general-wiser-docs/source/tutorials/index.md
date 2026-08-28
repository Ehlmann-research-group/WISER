# Tutorials

Two sets of hands-on material.

The **Getting Started** tutorials run on data that ships with the WISER source,
so there is nothing to download. Work through them in order and you will have
used every major tool in the application on a real scene.

The **Applied Labs** each take one field's question to a full dataset you
download yourself. They are written to work as course labs — each has
deliverables and questions — and equally as worked references for what WISER
can do.

Every screenshot in the Getting Started series, and in Labs A, B, C and F,
was produced by driving WISER through the steps described. Nothing is a mockup.

---

## Getting Started

Seven short tutorials, roughly two hours end to end.

```{list-table}
:header-rows: 1
:widths: 6 32 42 20

* - #
  - Tutorial
  - What you learn
  - Data
* - 1
  - {doc}`Your First Scene <01-first-look>`
  - Open a file, arrange the panes, choose bands, apply a contrast stretch
  - `caltech_4_100_150_nm`
* - 2
  - {doc}`Reading Spectra <02-spectra>`
  - Pull spectra from pixels, collect and compare them, load a mineral library, remove the continuum
  - `caltech_425_7_7_nm`
* - 3
  - {doc}`Regions of Interest <03-regions-of-interest>`
  - Define classes, extract mean signatures, export spectra and geometry
  - `caltech_4_100_150_nm`
* - 4
  - {doc}`Band Math: Mapping Vegetation <04-band-math-ndvi>`
  - Write and bind an expression, build a vegetation index, threshold it
  - `caltech_4_100_150_nm`
* - 5
  - {doc}`Classifying a Scene <05-classification>`
  - Read feature space in the scatter plot, run K-means, identify clusters
  - `caltech_4_100_150_nm`
* - 6
  - {doc}`PCA and MNF <06-pca-mnf>`
  - Reduce dimensions, read a scree plot, choose between variance and SNR ordering
  - `caltech_425_7_7_nm`
* - 7
  - {doc}`Finding a Known Material <07-detection>`
  - SAM, SFF, MTMF and linear unmixing, and which to reach for
  - `caltech_15_20_22_bb`
```

### The data

Everything the Getting Started series needs is already in your WISER checkout:

```
src/test_utils/test_datasets/     rasters
src/test_utils/test_spectra/      spectral libraries
```

These are the project's unit-test fixtures, so they are small on purpose — a
150 × 150 pixel campus scene, a 7 × 7 pixel 425-band cube, a 20 × 22 pixel SWIR
subset with a bad-band list. They are real data, and they are enough to
exercise every tool.

If you installed WISER from a release rather than from source, get them by
cloning the repository:

```bash
git clone https://github.com/Ehlmann-research-group/WISER.git
```

```{admonition} Small fixtures, real scenes
:class: tip
The fixtures are chosen so the tutorials run instantly for everyone, with no
download. They are deliberately not impressive to look at. When you want to see
what the same tools do on a full 425-band airborne cube — real red edges, real
scree plots, real mineral separations — go to
{doc}`Lab A <labs/lab-aviris-ng-urban>`, which is built on exactly that.
```

---

## Applied Labs

```{list-table}
:header-rows: 1
:widths: 6 30 28 36

* -
  - Lab
  - Field
  - Data
* - A
  - {doc}`Urban Vegetation and Materials with AVIRIS-NG <labs/lab-aviris-ng-urban>`
  - Urban ecology, remote-sensing methods
  - AVIRIS-NG — 551 MB, no account
* - B
  - {doc}`Mineral Mapping at Cuprite <labs/lab-cuprite-minerals>`
  - Economic geology, alteration mapping
  - AVIRIS-Classic — no account
* - C
  - {doc}`Martian Mineralogy with CRISM <labs/lab-mars-crism>`
  - Planetary science, astrobiology
  - MRO/CRISM MTRDR — no account
* - D
  - {doc}`Surface Mineralogy with EMIT <labs/lab-emit-dust>`
  - Earth system science, climate forcing
  - EMIT L2A — Earthdata Login
* - E
  - {doc}`Phytoplankton and Coastal Water with PACE <labs/lab-pace-phytoplankton>`
  - Biological oceanography, water quality
  - PACE/OCI L2 — Earthdata Login
* - F
  - {doc}`Close-Range Materials Imaging <labs/lab-materials-imaging>`
  - Materials, inspection, cultural heritage
  - Ships with WISER
```

```{admonition} For instructors
:class: tip
Each lab lists **deliverables** and **questions to answer** and is sized for a
2–3 hour session. **Lab F** runs entirely on bundled data, which makes it the
safest choice when you cannot rely on students having network access or disk
space. **Labs A, B and C** need no account of any kind; **D** and **E** need a
free NASA Earthdata Login, worth having students create in advance.
```

---

## Where to go from here

- {doc}`User Manual <../user-content/user-manual>` — the reference for every
  tool, dialog and option
- {doc}`Data Analysis Tools <../user-content/data-analysis-tools/data-analysis-tools>` —
  what each algorithm does and how it is implemented
- {doc}`Extending WISER <../extending-wiser/index>` — add your own analysis as
  a plugin

```{toctree}
:hidden:

01-first-look
02-spectra
03-regions-of-interest
04-band-math-ndvi
05-classification
06-pca-mnf
07-detection
labs/index
```
