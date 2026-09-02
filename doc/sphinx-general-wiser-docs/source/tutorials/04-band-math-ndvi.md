# Tutorial 4 — Band Math: Mapping Vegetation

**Goal:** compute a vegetation index over the campus scene, display it, and
read the result as a canopy map.

**Data:** `caltech_4_100_150_nm.hdr` (from {doc}`Tutorial 1 <01-first-look>`).

**Time:** about 10 minutes.

---

## The idea

{doc}`Tutorial 3 <03-regions-of-interest>` showed the canopy spectrum dropping
at 702 nm and climbing at 852 nm while roofs and asphalt stayed flat. A
**normalised difference** turns that contrast into one number per pixel:

$$\text{NDVI} = \frac{\rho_{\text{NIR}} - \rho_{\text{red}}}{\rho_{\text{NIR}} + \rho_{\text{red}}}$$

Dividing by the sum normalises out brightness, so a shadowed leaf and a sunlit
leaf score alike. Values run −1 to +1: dense green vegetation high, bare soil
and pavement near zero, water below zero.

```{admonition} A note on which "red" you have
:class: note

Textbook NDVI uses a red band near **670 nm**, at the bottom of the chlorophyll
absorption. The reddest band in this cube is **702 nm**, on the shoulder of the
red edge where a leaf is already brightening. Substituting it gives a
**red-edge NDVI**: absolute values come out lower than a 670 nm NDVI would, and
it saturates differently over dense canopy. It separates vegetation from
everything else just as cleanly, which is what we need here — but do not
compare these numbers against published 670 nm NDVI values.

Choosing bands by what they physically measure, not by what an index is
conventionally called, is the whole job. {doc}`Lab A <labs/lab-aviris-ng-urban>`
runs the same index on a 425-band cube where the textbook wavelengths are
available.
```

---

## Step 1 — Write the expression

1. Open the campus scene.
2. **Tools ▸ Band math...**
3. In the **Expression** box, type:

   ```text
   (nir - red) / (nir + red)
   ```

4. Press **Enter** or click away. WISER parses the expression and adds a row to
   **Variable bindings** for every name it does not recognise as a function —
   here, `nir` and `red`.

---

## Step 2 — Bind the variables

1. Leave both rows on type **Image Band**.
2. For `nir`, choose `caltech_4_100_150_nm` and **Band 3: 852.68 nm**.
3. For `red`, choose the same dataset and **Band 2: 702.42 nm**.
4. Type `NDVI` in **Result name**.

:::{figure} ../_static/tutorials/t4_bandmath_dialog.png
:width: 90%
:align: center
:alt: The band math dialog with the NDVI expression and both variables bound
:::

Two things on this screen are worth pausing over:

- Above the table, WISER reports the **result type and size** —
  `Result: Image Band, 150x150 (87.9KB)`. Band math is not streamed for every
  case, so check this before running an expression on a full flight line.
- **Toggle Help** opens the operator reference on the right. Read it: the
  built-in set is deliberately small.

5. Click **OK**. The result is added as a new dataset named **NDVI**.

---

## Step 3 — Display it meaningfully

A vegetation index in greyscale wastes the fact that it has a meaningful zero.

1. Open the **band chooser** for the NDVI dataset.
2. Select **Grayscale**, band 0, and the **RdYlGn** colormap.
3. Apply a **2.5% linear** contrast stretch.

:::{figure} ../_static/tutorials/t4_ndvi.png
:width: 90%
:align: center
:alt: The NDVI result with a red-yellow-green diverging colormap
:::

**Read the map.** Every street tree resolves as an individual green crown, the
hedgerows show as continuous green lines, the lawn in the south-east as a solid
block, and roofs, roads and parking areas as flat pale yellow.

```{important}
**Set the stretch on any computed product before you read it.** Index values
have no reason to fill the display range sensibly, and a single extreme pixel
at a scene edge can flatten everything else into one colour. Check the
histogram in the stretch dialog first. {doc}`Lab A <labs/lab-aviris-ng-urban>`
shows what this looks like when it goes wrong.
```

---

## Step 4 — Compare side by side

1. Click the **grid** button and set the layout to **1 × 2**.
2. Use the dataset chooser above the left panel to show
   `caltech_4_100_150_nm`, and the right panel to show `NDVI`.

:::{figure} ../_static/tutorials/t4_ndvi_vs_rgb.png
:width: 90%
:align: center
:alt: True-colour image and NDVI side by side in a 1x2 grid
:::

When every open dataset has the same width and height, the **link** button ties
the panels together: pan or zoom one and the others follow.

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

Bind `ndvi` as an **Image Band**, band 0 of the NDVI dataset. Comparison
operators return 1 where the test passes and 0 where it fails, so the result is
a binary canopy mask you can count, export, or use to restrict another
analysis.

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
