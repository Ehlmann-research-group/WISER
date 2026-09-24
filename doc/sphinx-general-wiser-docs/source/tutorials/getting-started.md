# Tutorials

Eight short tutorials, a little over two hours end to end, on data that ships
with the WISER source. Nothing to download. Work through them in order and you
will have used every major tool in the application on a real scene.

```{admonition} What you are actually working with
:class: note
A GIS layer stores one value per pixel, or three if it is a color image. The
scenes here store hundreds. Every pixel holds a full spectrum — a measurement of
how much light that patch of ground reflected at each of 4, 285 or 425
wavelengths — and WISER is built to get at it.

That changes what the software is for. You are not arranging layers to make a
picture. You are reading a measurement out of any pixel you click
({doc}`Tutorial 2 <02-spectra>`), averaging it over an area to get a signature
you can trust ({doc}`Tutorial 3 <03-regions-of-interest>`), comparing it against
a laboratory reference to identify what is on the ground
({doc}`Tutorial 7 <07-detection>`), and letting the shapes of those spectra sort
the scene into classes on their own ({doc}`Tutorial 5 <05-classification>`).

Keep the Spectrum Plot pane open throughout. Most of what makes a result
believable is visible there and nowhere else.
```

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
* - 8
  - {doc}`Bench and Close-Range Data <08-bench-and-close-range>`
  - The same tools on a laboratory cube, and what a bench instrument needs from you
  - `circuit_4_100_150_um`
```

## The data

Everything these tutorials need is already in your WISER checkout:

```
src/test_utils/test_datasets/     rasters
src/test_utils/test_spectra/      spectral libraries
```

These are the project's unit-test fixtures, so they are small on purpose — a
150 × 150 pixel campus scene, a 7 × 7 pixel 425-band cube, a 20 × 22 pixel
short-wave infrared subset with a bad-band list. They are real data, and they are enough to
exercise every tool.

If you installed WISER from a release rather than from source, get them by
cloning the repository:

```bash
git clone https://github.com/Ehlmann-research-group/WISER.git
```

```{admonition} Small fixtures, real scenes
:class: note
The fixtures are chosen so the tutorials run instantly for everyone, with no
download. They are deliberately not impressive to look at. When you want to see
what the same tools do on a full 425-band airborne cube — real red edges, real
scree plots, real mineral separations — go to
{doc}`Lab A <labs/lab-aviris-ng-urban>`, which is built on exactly that.
```

---

**Next:** the {doc}`Labs <labs/index>` take these tools to full public
datasets, in a form you can adapt into your own lab.

```{toctree}
:hidden:

01-first-look
02-spectra
03-regions-of-interest
04-band-math-ndvi
05-classification
06-pca-mnf
07-detection
08-bench-and-close-range
```
