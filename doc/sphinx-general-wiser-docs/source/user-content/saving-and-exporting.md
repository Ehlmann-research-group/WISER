# Saving and Exporting

WISER has five different "save" operations, and picking the wrong one is a
common source of frustration. This page says what each produces.

| You want | Use | Produces |
|---|---|---|
| A dataset as a raster file, optionally subset | **Save dataset as...** | An image file with all its bands and metadata |
| A picture of what you are looking at | **Export RGB image** | A PNG / TIFF / JPEG of the rendered view |
| Your whole session | **Save Project As...** | A `.wiserproj` |
| Spectra as numbers | Spectrum list or ROI context menu | ASCII text |
| A publication figure of a plot | Right-click the plot ▸ **Export plot to image...** | EPS / PDF / PNG / SVG |

---

## Saving a dataset

Two routes to the same dialog:

- Right-click the image ▸ **Save as...**
- **File ▸ Save dataset as...** ▸ choose the dataset

:::{figure} ../_static/images/save_img.png
:width: 55%
:align: center
:alt: The save-image dialog
:::

This writes a **raster file**: the actual band values, with wavelengths, bad
bands and georeferencing, readable by WISER, ENVI, GDAL or anything else.

Click **Show Advanced** for the rest:

:::{figure} ../_static/images/save_dataset.png
:width: 55%
:align: center
:alt: The save-dataset dialog with advanced options expanded
:::

| Option | Use |
|---|---|
| **Data description** | Free text written into the header |
| **Data ignore value** | The value downstream tools should treat as nodata |
| **Dimensions** tab | **Spatial subsetting** — write out a pixel window rather than the whole scene |
| **Wavelengths** | **Spectral subsetting** — choose which bands to write, and set the bad-band list |
| **Default display bands** | The RGB or grayscale combination the file recommends when reopened |

```{tip}
Spatial and spectral subsetting here is the practical answer to a scene too
large to work with. Cut a full flight line down to your study area and the
wavelength range you need, save that, and every subsequent analysis — band math
especially — becomes tractable.
```

---

## Exporting a picture of the view

Right-click the image ▸ **Export RGB image**, then one of:

- **Export visible image area to RGB image...** — exactly what is on screen, at
  the current zoom, bands and stretch
- **Export full image extent to RGB image...** — the whole scene, with the
  current bands and stretch applied

Formats are **PNG**, **TIFF** and **JPEG**.

```{important}
This writes **display values** — 8-bit RGB after the contrast stretch — not
data. It is the right thing for a figure, a slide or a report, and the wrong
thing for anything you intend to analyse further. Use **Save dataset as...**
for that.
```

---

## Exporting spectra

- **Right-click a spectrum in the list ▸ Save to file...** — one spectrum as
  text
- **ROI context menu ▸ Export all spectra in ROI...** — every pixel's spectrum
  in a region, as text
- **Right-click the plot ▸ Export plot to image...** — the plot as a figure, in
  EPS, PDF, PNG or SVG at 72, 100 or 300 dpi

See {doc}`Spectra and Spectral Libraries <spectra-and-libraries>` and
{doc}`Regions of Interest <regions-of-interest>`.

---

## Exporting ROIs

- **Right-click an ROI ▸ Export ROI...** — one ROI as GeoJSON
- Image context menu ▸ **Export all ROIs...** — every ROI at once

Read them back with **File ▸ Import regions of interest...**.

---

## Saving the session

**File ▸ Save Project As...** writes everything — datasets, ROIs, collected
spectra, contrast stretches, saved band-math expressions, analysis runs,
imported libraries and custom coordinate systems — to a single `.wiserproj`.

Choose **Save self-contained** when the project needs to travel: it copies the
imagery into the project file rather than pointing at files on your disk.

Full details, including what happens to derived products when you exclude a
dataset, are in {doc}`Saving and Opening Projects <projects>`.

---

## See also

- {doc}`Saving and Opening Projects <projects>`
- {doc}`Mosaic <mosaic>` — exporting a composite of several scenes
- {doc}`Display and Contrast Stretch <display-and-stretch>` — what "display
  values" means
