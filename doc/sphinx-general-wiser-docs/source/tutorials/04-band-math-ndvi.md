# Tutorial 4 — Band Math: Mapping Vegetation

**Goal:** compute a vegetation index over the campus scene, display it, and
read the result as a canopy map.

**Data:** `caltech_4_100_150_nm.hdr` (from {doc}`Tutorial 1 <01-first-look>`).

---

## The idea

{doc}`Tutorial 3 <03-regions-of-interest>` showed the canopy spectrum dropping
at 702 nm and climbing at 852 nm while roofs and asphalt stayed flat. A
**normalized difference** turns that contrast into one number per pixel:

$$\text{NDVI} = \frac{\rho_{\text{NIR}} - \rho_{\text{red}}}{\rho_{\text{NIR}} + \rho_{\text{red}}}$$

Dividing by the sum normalizes out brightness, so a shadowed leaf and a sunlit
leaf score alike. Values run −1 to +1: dense green vegetation high, bare soil
and pavement near zero, water below zero.

```{admonition} A note on which "red" you have
:class: note

Textbook NDVI uses a red band near **670 nm**, at the bottom of the chlorophyll
absorption. The reddest band in this cube is **702 nm**, on the shoulder of the
red edge where a leaf is already brightening. Substituting it gives a
**red-edge NDVI**: absolute values come out lower than a 670 nm NDVI would, and
it saturates differently over dense canopy. It separates vegetation from
everything else just as cleanly, which is what you want here — but do not
compare these numbers against published 670 nm NDVI values.

Choosing bands by what they physically measure, not by what an index is
conventionally called, is the whole job. {doc}`Lab A <labs/lab-aviris-ng-urban>`
runs the same index on a 425-band cube where the textbook wavelengths are
available.
```

---

## Step 1 — Write the expression

1. Open the campus scene as in {doc}`Tutorial 1 <01-first-look>`.
2. From the **Main Menu** at the top of the window, go to **Tools** and select
   **Band math...** from the dropdown list. The **Band Math** dialog opens.
3. In the **Expression:** box at the top, type:

   ```text
   (nir - red) / (nir + red)
   ```

4. Press **Enter** on your keyboard, or click elsewhere in the dialog. WISER
   parses what you typed and adds a row under **Variable bindings:** for every
   name it does not recognize as a function — here, `nir` and `red`.

---

## Step 2 — Bind the variables

The **Variable bindings:** table has three columns: **Variable** (the name from
your expression), **Type** (what kind of thing it binds to), and **Variable
Assignments** (which dataset and band).

1. Check that **Type** reads **Image Band** for both rows. That is the default.
2. Scroll right in the table to reach the **Variable Assignments** column, which
   holds two dropdowns per row: the dataset, then the band within it.
3. In the `nir` row, set the dataset to `caltech_4_100_150_nm` and the band to
   **Band 3: 852.68 nm**.
4. In the `red` row, set the same dataset and **Band 2: 702.42 nm**.
5. Type `NDVI` into **Result name (optional):** near the bottom. Without a name
   the result is harder to find in the dataset list later.

:::{figure} ../_static/tutorials/t4_bandmath_dialog.png
:width: 90%
:align: center
:alt: The band math dialog with the NDVI expression and both variables bound
:::

Two things on this screen are worth pausing over:

- To the right of the **Expression:** box, WISER reports the **result type and
  size** — `Result: Image Band, 150x150 (87.9KB)`. Band math is not streamed for
  every case, so check this before running an expression on a full flight line.
- **Toggle Help** opens the operator reference in a panel on the right. Read it,
  because the built-in function set is deliberately small.

6. Click **OK**. After a moment the result is added as a new dataset named
   **NDVI**.

---

## Step 3 — Display it meaningfully

A vegetation index in grayscale wastes the fact that it has a meaningful zero.

1. Use the **Select dataset to view** dropdown on the Main Toolbar to switch the
   main window to the new **NDVI** dataset. It is drawn in grayscale, and the
   Spectrum Plot will no longer plot a spectrum for the pixels you click,
   because NDVI has only one band.
2. Click **Band chooser** on the Main Toolbar. Select **Grayscale** in the
   **General** section, set **Grayscale Band** to band 0, tick **Use a
   colormap**, choose **RdYlGn** from the dropdown, and click **OK**.
3. Click **Stretch builder**, select **Linear Stretch**, click **2.5% linear**,
   and click **OK**.

:::{figure} ../_static/tutorials/t4_ndvi.png
:width: 90%
:align: center
:alt: The NDVI result with a red-yellow-green diverging colormap
:::

```{admonition} Interpretation
:class: note
Every street tree resolves as an individual green crown, the hedgerows show as
continuous green lines, and the lawn in the south-east as a solid block. Roofs,
roads and parking areas are flat pale yellow. The index has separated the
vegetation from everything else using two bands, where the true-color image
needed you to recognize shapes.
```

```{note}
**Set the stretch on any computed product before you read it.** Index values
have no reason to fill the display range sensibly, and a single extreme pixel
at a scene edge can flatten everything else into one color. Check the
histogram in the stretch dialog first. {doc}`Lab A <labs/lab-aviris-ng-urban>`
shows what this looks like when it goes wrong.
```

---

## Step 4 — Compare side by side

1. Click **Split/unsplit the main view** on the Main Toolbar, the grid button,
   and choose **1 row x 2 columns** from its menu. The main window splits in two.
2. Use the **Select dataset to view** dropdown above the left panel to show
   `caltech_4_100_150_nm`, and the one above the right panel to show `NDVI`.

:::{figure} ../_static/tutorials/t4_ndvi_vs_rgb.png
:width: 90%
:align: center
:alt: True-color image and NDVI side by side in a 1x2 grid
:::

When every open dataset has the same width and height, **Link view scrolling**
on the Main Toolbar ties the panels together: pan or zoom one and the others
follow. The status bar confirms with `Linked view scrolling is ON`.

```{note}
In grid view the band chooser and contrast stretch controls move from the main
toolbar to a strip above **each** panel, so you can set them per panel.
```

---

## Step 5 — Threshold it

To go from a continuous index to a canopy mask, run one more expression:

```text
ndvi > 0.35
```

Open **Tools ▸ Band math...** again, type the expression, and bind `ndvi` as an
**Image Band** pointing at band 0 of the NDVI dataset. Comparison operators
return 1 where the test passes and 0 where it fails, so the result is a binary
canopy mask you can count, export, or use to restrict another analysis.

Pick the threshold from your own data rather than from a paper — collect an ROI
over known canopy and another over known pavement
({doc}`Tutorial 3 <03-regions-of-interest>`), look at where their NDVI values
separate, and cut there.

---

## What band math can and cannot do

**Operators:** `+` `-` `*` `/`, `**` (power, so `x ** 0.5` is a square root),
unary `-`, and the comparisons `==` `!=` `<` `>` `<=` `>=`.

**Built-in functions:** `sin`, `cos`, `tan`, `arcsin`, `arccos`, `arctan`,
`arctan2`, `dotprod`. That is the whole list: there is no built-in `sqrt`,
`log` or `exp`. Use `** 0.5` for a square root; for anything else a
{doc}`band-math plugin <../extending-wiser/bandmath_plugins>` adds functions
without rebuilding WISER.

**Variables** bind to a whole **image cube**, a single **image band**, or a
**spectrum**, and names are case-insensitive.

Expressions can be saved and reloaded, and **Enable Batch Processing** applies
one expression across every raster in a folder. Full reference:
{doc}`Band Math <../user-content/band-math>`.

---

## What you can now do

- Write and bind a band-math expression
- Choose bands on physical grounds and say what that choice costs
- Display an index with a colormap and stretch that reflect its zero point
- Compare two datasets in a linked grid
- Threshold a continuous index into a mask

---

**Next:** {doc}`Tutorial 5 — Classifying a Scene <05-classification>` — let the
data find its own classes.
