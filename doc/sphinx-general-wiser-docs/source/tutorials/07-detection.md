# Tutorial 7 — Finding a Known Material

**Goal:** go from "what is in this scene?" to "**where is this specific
substance?**" using WISER's four matching and unmixing tools, and know which
one to reach for.

**Data:** `caltech_15_20_22_bb.hdr` — a 20 × 22 pixel, 15-band short-wave
infrared (SWIR) subset
(1308–1454 nm) carrying a **bad-band list**, so you can see band flagging in
action. Reference spectra from
`src/test_utils/test_spectra/usgs_resampHeadwallSWIR.hdr`.

---

## Four tools, four questions

| Tool | Answers | Needs |
|---|---|---|
| **Spectral Angle Mapper (SAM)** | Which reference does this pixel most resemble in *shape*? | One or more reference spectra |
| **Spectral Feature Fitting (SFF)** | Do this pixel's *absorption features* match the reference's? | A reference **and** a diagnostic wavelength range |
| **Mixture-Tuned Matched Filter (MTMF)** | How much of this one target is present, against an unknown background? | One target spectrum |
| **Linear Unmixing** | What fraction of each of these known materials is in this pixel? | Endmembers for **every** material present |

The ordering is roughly "how much you have to know in advance". SAM asks the
least; unmixing asks the most and, in exchange, gives you abundances.

---

## Step 1 — Load the scene and a library

1. **File ▸ Open...** → `caltech_15_20_22_bb.hdr`.
2. Open the **Dataset Info** pane and expand the dataset. Its header carries
   `bbl = { 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1 }` — bands 8 and 9 are
   flagged bad. Every tool below drops them automatically.
3. In the Spectrum Plot toolbar, **Load or import spectra ▸ Load spectral
   library...** and choose `usgs_resampHeadwallSWIR.hdr`.

---

## Step 2 — Spectral Angle Mapper

SAM treats each spectrum as a vector in *n*-band space and measures the
**angle** between pixel and reference:

$$\theta = \arccos\!\left(\frac{\mathbf{p}\cdot\mathbf{r}}{\lVert\mathbf{p}\rVert\,\lVert\mathbf{r}\rVert}\right)$$

Because an angle depends only on direction, **SAM ignores brightness**. A
sunlit and a shadowed patch of the same mineral give the same small angle,
which makes it the workhorse for scenes with topography or uneven illumination.
Smaller angle, better match.

Open **Tools ▸ Data Analysis ▸ Spectral Angle Mapper**.

:::{figure} ../_static/tutorials/t7_sam_dialog.png
:width: 90%
:align: center
:alt: The Spectral Angle Mapper dialog with a USGS mineral library as reference
:::

1. **Select Target Type** — **Image Cube**, then pick the dataset. (Choosing
   **Spectrum** compares one spectrum against the references and opens a ranked
   table — a fast way to identify a spectrum you just collected.)
2. **Min/Max Wavelength** and **Units** restrict the comparison. Leaving both
   at `0.0` uses the full overlap.
3. **Reference Library Selection** — a USGS mineral library is preloaded. Tick
   it, or use **Add Library**, **Add Spectrum** (from a text file), or **Add
   Collected Spectrum** to use a signature you measured yourself in
   {doc}`Tutorial 3 <03-regions-of-interest>`.
4. **Initial Angle (°)** is the detection threshold, default 5°. Each reference
   also carries its own threshold, so you can be strict about one mineral and
   loose about another.
5. Click **Run SAM**.

Two datasets come back, one band per reference:

- **`SAM Angle, Img: <source>`** — the angle in degrees at every pixel
- **`SAM CLS, Img: <source>`** — a boolean map of `angle < threshold`

Display the **angle** image with a colormap and stretch it before trusting the
classification: the threshold is a decision you are making, and the angle image
shows you what you are deciding about. Every reference is interpolated onto the
target's wavelength grid first, so a library resampled to a different sensor
still works.

---

## Step 3 — Spectral Feature Fitting

SFF asks a narrower question: forget the overall spectrum, do the **absorption
features** line up?

It continuum-removes both spectra — dividing out the smooth upper envelope so
only the dips remain — inverts the result so absorptions become peaks, then
finds the single **scale factor** that best matches their depths by least
squares.

:::{figure} ../_static/tutorials/t7_sff_dialog.png
:width: 90%
:align: center
:alt: The Spectral Feature Fitting dialog
:::

The inputs mirror SAM's, with one difference that decides whether the run is
worth anything:

```{important}
**Set the wavelength range to bracket the feature you care about.** SFF over a
full spectral range averages your diagnostic band in with everything else. Over
a window around a known absorption it is far more specific than SAM. Common
ones: kaolinite 2160/2200 nm, alunite 2170 nm, calcite 2340 nm, gypsum
1750 nm, chlorophyll 670 nm.
```

Three outputs, one band per reference:

- **`SFF RMSE`** — root-mean-square fit error; **lower is better**
- **`SFF SCALE`** — fitted feature depth, loosely an abundance indicator
- **`SFF CLS`** — boolean `RMSE < threshold` (default 0.03)

Read RMSE and SCALE **together**. A low RMSE with a near-zero scale means "the
feature is absent, and its absence fits well" — a good fit to nothing.

---

## Step 4 — Mixture-Tuned Matched Filter

MTMF is for the case where you know your target but nothing about the
background — the usual situation in target detection.

:::{figure} ../_static/tutorials/t7_mtmf_dialog.png
:width: 55%
:align: center
:alt: The MTMF dialog showing input cube, noise method and target
:::

1. **Select Image Cube** — the dataset to search.
2. **Noise method**:
   - **Image Cube Based** — shift difference within the scene; pick a
     direction (Down/Up/Left/Right)
   - **Dark Image Based** — a separate dark/noise dataset with the same bands
   - **ROI Based** — an ROI over a region you consider background, plus the
     dataset it comes from
3. **Target** — the reference spectrum to detect.
4. Click **OK**.

Internally MTMF runs a full MNF ({doc}`Tutorial 6 <06-pca-mnf>`) so the noise
is white, projects the target into that space, and computes a matched-filter
score per pixel: near 0 for background, near 1 for a pure target pixel. It also
computes each pixel's **infeasibility** — how far its spectrum is from any
physically plausible mixture of background and target.

```{important}
**A high score alone is not a detection.** The matched filter produces false
positives on spectra that happen to project well onto the target direction. The
mixture-tuning step separates them: a real detection has a **high score and low
infeasibility**. Scoring pixels without checking feasibility is the classic way
to over-report a target.
```

Output is one float32 image per target, `MTMF [target]: <source>`, nodata as
`NaN`.

---

## Step 5 — Linear Unmixing

Unmixing assumes each pixel is a weighted sum of a few pure **endmember**
spectra and solves for the weights.

:::{figure} ../_static/tutorials/t7_unmix_dialog.png
:width: 55%
:align: center
:alt: The Linear Unmixing dialog with endmember list and Sum to Unity
:::

1. **Input Dataset** — the cube.
2. Build the **Endmembers** list (at least two): **Add Collected Spectrum** for
   signatures measured in-app, **Import Spectrum** for a text file.
3. Optionally tick **Sum to Unity** and set its weight. This softly forces each
   pixel's abundances to add to 1 — right when you believe your endmembers
   cover everything in the scene, wrong when they do not.
4. Click **OK**.

The result carries **one abundance band per endmember**, in the order listed,
plus a final **RMSE band**.

```{important}
**Read the RMSE band first.** It is the per-pixel reconstruction error, and it
tells you where your endmember set fails to explain the data — usually because
a material is present that you did not include. Abundances in high-RMSE areas
are not meaningful. Unmixing is the tool that most rewards checking its own
residual.
```

Endmembers must share the input's wavelength grid, and a band flagged bad in
*either* the dataset or an endmember is excluded from the fit.

---

## Choosing between them

- You have a spectrum and want to know **what it is** → SAM in **Spectrum**
  mode against a library, then read the ranked table.
- You want a map of **where a material is**, illumination varies, and you have
  no background information → **SAM**, inspecting the angle image before
  thresholding.
- You know the **diagnostic absorption** and want specificity → **SFF** over a
  narrow window.
- You have **one target**, an unknown background, and want an abundance-like
  score → **MTMF**, checking feasibility.
- You have endmembers for **everything** and want fractions → **Linear
  Unmixing**, checking RMSE.

None of these tools proves a material is present. They rank pixels by how well
they match a hypothesis you supplied. Confirmation comes from the spectrum
itself — go back to the Spectrum Plot, look at the pixels a tool flagged, and
check that the absorptions you expect are actually there.

---

## What you can now do

- Run all four matching and unmixing tools
- Pick the one that suits what you know going in
- Set a wavelength window around a diagnostic feature
- Read the diagnostic outputs — angle, RMSE, infeasibility — rather than only
  the classification

---

**Next:** the {doc}`Labs <labs/index>` take these tools to real,
downloadable scenes.
