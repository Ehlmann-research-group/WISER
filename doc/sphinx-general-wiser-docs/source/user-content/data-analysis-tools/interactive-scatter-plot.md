# Interactive Scatter Plot

The Interactive Scatter Plot plots two bands against each other — every pixel
becomes one point, positioned by its value in the **X band** (horizontal) and
**Y band** (vertical). Its real power is the two-way link to the image: draw a
shape around a cluster of points and the matching pixels light up in the image,
so you can discover structure in *feature space* and instantly see *where* it
lives in the scene.

Open it from a raster view's **Data Analysis ▸ Interactive Scatter Plot** menu.

## The core idea

A scatter plot of two bands reveals groupings that a single band can't: soil,
vegetation, water, or a mineral often form distinct clouds of points. Because
WISER keeps each point tied to its source pixel, selecting a cloud on the plot
highlights those exact pixels back on the image (you can also turn a
selection into an ROI for use elsewhere).

## Setting up a plot

At the bottom of the window:

1. **X Axis Band** — pick a **Dataset** and a **Band #** (the spin box and the
   band dropdown stay in sync).
2. **Y Axis Band** — pick a dataset and band. The X and Y bands may come from
   two *different* images.
3. **Render Onto** — choose the image whose view should display the highlighted
   pixels of your selection.
4. Click **Create Plot**.

All three datasets (X, Y, and Render Onto) must have the **same width and
height**, since points are matched to pixels by position; mismatched dimensions
raise an error. The plot is computed in the background (a loading spinner shows
meanwhile), and pixels with NaN values are dropped.

:::{figure} ../../_static/tutorials/t8_scatter_plot.png
:width: 70%
:align: center
:alt: A density scatter plot of two bands showing the soil line and a vegetation plume
:::

## Reading the plot

By default you get a **density scatter plot**: instead of drawing millions of
overlapping dots, color encodes how many pixels fall on each spot, with a
colorbar labeled *Number of points per spectral value*. Bright/dense regions
are common pixel values. Use **To Scatter** to switch to a plain dot plot, and
**To Density** to switch back. **To Scatter** is useful for areas that do not
have a dense enough pixel population to show up when **To Density** is selected.

## Selecting points

The plot always has a **polygon selector** active:

- Click to drop vertices around a cluster; finish with a **double-click** or
  **Enter**.
- Selected points are outlined on the plot, the **point count** updates, and
  the corresponding pixels are drawn on the **Render Onto** image in your
  highlight color.
- Press **Escape** or **Clear selection** to start over.

:::{figure} ../../_static/tutorials/t8_scatter_selection.png
:width: 70%
:align: center
:alt: A polygon selection around a cluster, with the selected points outlined
:::

:::{figure} ../../_static/tutorials/t8_scatter_highlight.png
:width: 85%
:align: center
:alt: The selected feature-space pixels highlighted back on the image
:::

## Controls reference

The strip just below the toolbar:

- **N pts** — live count of selected points.
- **Clear selection** — remove the polygon and all highlights.
- **Create ROI from Selection** — save the selected pixels as a Region of
  Interest (named *Scatter Plot Selection*) for use in other tools.
- **Color Map** — choose the colormap used for the density plot (any Matplotlib
  colormap; a preview is shown).
- **Axes Limits** — set the X/Y min and max manually, or reset to the data
  range (with a 10% margin).
- **To Scatter / To Density** — toggle the plot style.
- **Highlight** — pick the color used both to outline selected points and to
  mark them on the image.

The standard Matplotlib navigation toolbar (pan, zoom, save image) sits above
the strip.

## See also

- {doc}`Tutorial 5 — Classifying a Scene <../../tutorials/05-classification>`
- {doc}`Regions of Interest <../regions-of-interest>`
