# Lab B — Mineral Mapping at Cuprite, Nevada

- **Field:** economic geology, alteration mapping
- **Instrument:** AVIRIS-Classic — 224 bands, 379–2498 nm at ~9.5 nm, 15.2 m
- **Prerequisites:** {doc}`Tutorials 1–7 <../index>`, and {doc}`Lab A <lab-aviris-ng-urban>` for the mechanics on a large cube
- **Time:** 2–3 hours

---

## The setting

Cuprite, Nevada is the reference site of imaging spectroscopy. A hydrothermal
system altered the volcanic rocks into concentric mineral zones, vegetation
cover is close to nil, and the outcrops are large enough to resolve at 15 m.
Almost every method paper in the field has been demonstrated here, so your
results have a large literature to check against.

The alteration zoning, from the centre of each hydrothermal centre outwards:

| Zone | Diagnostic mineral | Key absorption |
|---|---|---|
| Silicified core | opal / chalcedony | broad, weak features |
| Opalised | **alunite** | 2170 nm (sharp) |
| Argillic | **kaolinite** | **doublet** at 2160 + 2200 nm |
| Outer / unaltered | **muscovite / illite** | 2200 nm (broader, asymmetric) |
| Playa and carbonate units | **calcite** | 2340 nm |
| Locally | **buddingtonite** (NH₄-feldspar) | 2120 nm |

Those wavelengths are the whole lab.

---

## Get the data

**Scene:** `f230918t01p00r11_rfl` — an orthocorrected AVIRIS-Classic
**reflectance** flight line flown on 18 September 2023, published open (no
login) in a public directory on JPL's AVIRIS distribution server. It runs
110 km north–south through southwestern Nevada and crosses Cuprite about
40 % of the way along.

The whole line is **10.2 GB**. You only need the part over Cuprite, and because
the file is **BIL** interleaved every image line is one contiguous block of
bytes — so a single ranged download gets you a usable cube.

**1. Fetch the header** (a few kB, and it carries the 224 wavelengths):

```bash
curl -O https://popo.jpl.nasa.gov/pub/RKokaly/f230918t01p00r11_rfl.hdr
```

**2. Fetch lines 2400–3799** — 1400 lines centred on the district, 2.05 GB,
a few minutes on a decent connection:

```bash
curl -r 3513753600-5563443199 \
  https://popo.jpl.nasa.gov/pub/RKokaly/f230918t01p00r11_rfl \
  -o f230918t01p00r11_rfl_cuprite
```

```{admonition} Where that byte range comes from
:class: tip
One line is `samples × bands × 4 bytes` = 1634 × 224 × 4 = **1 464 064 bytes**.
Line 2400 therefore starts at 2400 × 1 464 064 = 3 513 753 600, and line 3800
starts one byte past the end of the range. Change the two line numbers and
recompute if you want a different window.
```

**3. Rename the header to match, and edit three things in it:**

```bash
mv f230918t01p00r11_rfl.hdr f230918t01p00r11_rfl_cuprite.hdr
```

| Field | Change it to | Why |
|---|---|---|
| `lines` | `1400` | you downloaded 1400 of the 7293 lines |
| `map info` 5th value | `4168685.200` | the new upper-left northing: 4205165.2 − 2400 × 15.2 |
| *(add a line)* | `data ignore value = -9999` | **the archive header omits it** |

That last one matters. The orthocorrection pads the rotated flight line with
**−9999**, and here that fill is over half the frame. Undeclared, it drags every
contrast stretch and every statistic with it. WISER will also accept it after
the fact through **Edit dataset...** in the Dataset Info pane.

The result is a 1634 × 1400 × 224 cube. Valid data occupies samples ~220–1000;
everything outside is fill.

```{admonition} Use reflectance, not radiance
:class: warning
Radiance carries the solar spectrum and the atmosphere in it, so its absorption
features are mostly not the surface's. Every method here assumes atmospherically
corrected **reflectance** — that is what the `_rfl` suffix means. Check the file
name and header description first.
```

**Reference spectra (optional, for Part 3):** download `usgs_splib07.zip` from
the [USGS Spectral Library Version 7 release](https://dx.doi.org/10.5066/F7RR1WDJ).
Use the version **convolved to AVIRIS-Classic** — the library ships copies
resampled to AVIRIS-Classic, HyMap, Hyperion, CRISM, M3 and VIMS, and matching
the sensor saves a resampling step and a class of subtle errors.

---

## Part 1 — Orient yourself (20 min)

1. Open `f230918t01p00r11_rfl_cuprite.hdr` and turn on all four panes.
2. Use **Choose Visible-Light Bands** (660 / 550 / 480 nm) and apply a **2.5%
   linear** stretch.

:::{figure} ../../_static/tutorials/lab_cuprite_truecolour.png
:width: 100%
:align: center
:alt: Cuprite in true colour, a near-featureless beige desert
:::

**This is the point of the lab.** One of the most intensely studied
hydrothermal systems on Earth, and in true colour it is beige gravel. Nothing
about the mineral zoning is visible.

3. Now build a SWIR composite: red **2200 nm**, green **2170 nm**, blue
   **2340 nm**, and stretch it 2.5% linear again.

:::{figure} ../../_static/tutorials/lab_cuprite_swir.png
:width: 100%
:align: center
:alt: The same area as a 2200/2170/2340 nm composite, still largely grey
:::

Barely better. That is not a mistake — **neighbouring SWIR bands are strongly
correlated**, so an RGB composite built from three of them is close to grey no
matter how you stretch each channel independently.

4. Reopen the stretch dialog and choose **Decorrelation Stretch** (it is only
   enabled for 3-band displays).

:::{figure} ../../_static/tutorials/lab_cuprite_decorr.png
:width: 100%
:align: center
:alt: The same three bands after a decorrelation stretch, showing strong colour separation
:::

The decorrelation stretch rotates the three display bands onto their principal
axes, stretches *those*, and rotates back — removing exactly the correlation
that made the image grey. The alteration zones separate into distinct colours.
Compare this with the band-depth map in Part 3: the dark-blue patches here are
the alunite-rich ground.

```{note}
Colour in a decorrelation stretch is **relative**, not diagnostic. It tells you
*that* two areas differ spectrally, never *which mineral* either one is. That
requires the spectra themselves — Part 2.
```

```{admonition} No bad-band list in this file
:class: warning
Unlike many AVIRIS products, this header carries **no `bbl` entry**, so nothing
marks the 1400 nm and 1900 nm water-vapour regions as unusable. They are still
unusable. Keep them out of every wavelength range you give an analysis tool,
and expect to see them as noise spikes in Part 2.
```

**Deliverable 1:** true-colour, plain SWIR and decorrelation-stretched
composites of the same area, exported with **Export RGB image ▸ Export visible
image area**, plus two sentences on what the decorrelation stretch shows that
the other two do not.

---

## Part 2 — Identify minerals by hand (40 min)

Colour separation tells you units exist. Only the spectra name them.

1. Click across the bright altered ground and collect spectra. Set **Number of
   pixels to average** to a 3 × 3 **median** — a single AVIRIS pixel at 2200 nm
   is noisy.

The four spectra below come from pixels chosen by band position, not by eye:

| Mineral | Pixel (x, y) | Deepest SWIR band |
|---|---|---|
| Alunite | 353, 993 | 2170 nm |
| Kaolinite | 663, 745 | 2210 nm, with a 2160 shoulder |
| Muscovite | 266, 1033 | 2200 nm |
| Calcite | 253, 364 | 2339 nm |

:::{figure} ../../_static/tutorials/lab_cuprite_spectra_window.png
:width: 100%
:align: center
:alt: The Cuprite scene with four spectra collected, listed in the Spectra pane
:::

The status bar confirms which pixel you are on — `Pixel: (253, 364)` above —
and the **Spectra and Spectral Libraries** pane lists what you have collected
so far. Check both before reading the plot.

:::{figure} ../../_static/tutorials/lab_cuprite_spectra_plot.png
:width: 100%
:align: center
:alt: Four full-range Cuprite spectra with large noise spikes at 1400 and 1900 nm
:::

Two things to read off the full-range plot. The **1400 and 1900 nm spikes** are
the water-vapour regions — atmospheric correction cannot recover them and the
values there are meaningless. And the diagnostic mineral features are the small
wiggles past 2000 nm, dwarfed at this scale by overall brightness differences.

2. Set the plot's x-axis range to **2000–2500 nm** (the gear icon in the
   Spectrum Plot toolbar) so the SWIR features fill the frame.

:::{figure} ../../_static/tutorials/lab_cuprite_swir_spectra.png
:width: 100%
:align: center
:alt: The same four spectra restricted to 2000-2500 nm, showing distinct absorption bands
:::

Now each mineral is obvious:

- **Alunite** (red) — deepest at **2170 nm**, with a second minimum near 2210
- **Kaolinite** (blue) — a **doublet**, 2160 and a deeper 2205
- **Muscovite** (green) — a single band at **2200 nm**, plus a 2350 secondary
- **Calcite** (purple) — one broad, deep band at **2340 nm**, bright elsewhere

3. To measure rather than eyeball the bands, right-click a spectrum ▸
   **Continuum Removal: Single Spectrum**. WISER adds the continuum-removed
   spectrum and the convex hull it divided by.

:::{figure} ../../_static/tutorials/lab_cuprite_continuum.png
:width: 100%
:align: center
:alt: The kaolinite spectrum, its convex hull, and the continuum-removed result
:::

:::{figure} ../../_static/tutorials/lab_cuprite_continuum_swir.png
:width: 100%
:align: center
:alt: The continuum-removed kaolinite spectrum zoomed to 2000-2500 nm, showing the doublet
:::

Zoomed in, the kaolinite doublet is unambiguous — **2160** and a deeper
**2210** — and the band depths are now directly readable as a fraction of the
continuum (about 0.29 and 0.32 here).

```{admonition} Continuum removal fits the hull over the whole spectrum
:class: warning
WISER removes the continuum across every band in the spectrum, and the noise
spike at 1400 nm becomes a hull vertex — visible in the first figure as the
peak the hull is pinned to. The result is still correct *within* the SWIR, but
do not read the 1300–2000 nm part of a continuum-removed AVIRIS spectrum.
```

4. Confirm each: import the USGS library and run **SAM in Spectrum mode**
   ({doc}`Tutorial 7 <../07-detection>`) with your collected spectrum as the
   target. Read the ranked match table.

**Deliverable 2:** a continuum-removed plot of four identified spectra, each
labelled with its mineral and the wavelength you used to call it.

---

## Part 3 — Map the minerals (60 min)

### 3a. Band depth — the direct measurement

Before reaching for a classifier, map a single absorption. A **band depth** is
the diagnostic band divided by a straight continuum drawn between two shoulders
either side of it, and WISER's band math computes it in one expression.

For alunite, use 2100 nm and 2250 nm as shoulders and 2170 nm as the centre.
The 2170 band sits 0.467 of the way between them in wavelength, so the
continuum at that point is `0.533 × a + 0.467 × b`:

1. **Tools ▸ Band Math**, expression:

   ```text
   1 - c / (0.533 * a + 0.467 * b)
   ```

2. Bind `a` → **2100 nm** (band 183), `b` → **2250 nm** (band 198),
   `c` → **2170 nm** (band 190). Name the result `AluniteBD2170`.

:::{figure} ../../_static/tutorials/lab_cuprite_bandmath.png
:width: 80%
:align: center
:alt: The WISER band math dialog with the band-depth expression and its three bindings
:::

3. Display the result with a colormap and a 2.5% linear stretch.

:::{figure} ../../_static/tutorials/lab_cuprite_bd2170.png
:width: 100%
:align: center
:alt: The 2170 nm alunite band-depth map, showing two bright alteration centres
:::

The two bright lobes are the opalised cores of the hydrothermal centres, and
band depth reaches **0.35** — a 35 % absorption, which is very strong. Unlike
the decorrelation stretch, this image is **quantitative**: the value at each
pixel is a physical measurement you can threshold, compare between scenes, or
check against a laboratory spectrum.

Repeat for the other three minerals by moving the centre and shoulders:

| Mineral | Centre | Shoulders | Weight on the upper shoulder |
|---|---|---|---|
| Alunite | 2170 nm | 2100 / 2250 | 0.467 |
| Kaolinite–muscovite | 2200 nm | 2130 / 2280 | 0.467 |
| Calcite | 2340 nm | 2260 / 2400 | 0.571 |

```{admonition} Band depth does not separate kaolinite from muscovite
:class: warning
Both absorb at 2200 nm, so one band-depth map lights up for both. Separating
them needs the *shape* of the feature — the 2160 shoulder — which is what SFF
in 3c is for, or a ratio of the 2160 and 2200 depths.
```

### 3b. Spectral Angle Mapper

1. **Tools ▸ Data Analysis ▸ Spectral Angle Mapper**, target **Image Cube**.
2. Add the USGS library; tick alunite, kaolinite, muscovite and calcite.
3. **Wavelength range: 2000–2400 nm.** This is what makes SAM work here — over
   the full range the albedo and iron-oxide variation in the visible dominates
   the angle and swamps the clay signal.
4. Start at the default 5° threshold and **Run SAM**.

Display the **`SAM Angle`** image first, with a colormap and a tight stretch.
Only then look at **`SAM CLS`**. Adjust each mineral's threshold and re-run
until the classified areas match the outcrops visible in the decorrelation
stretch, and until alunite lands where your 3a band-depth map is brightest.

### 3c. Spectral Feature Fitting

Repeat with SFF, one feature at a time:

| Mineral | SFF window |
|---|---|
| Alunite | 2120–2220 nm |
| Kaolinite | 2120–2250 nm |
| Muscovite | 2150–2250 nm |
| Calcite | 2280–2400 nm |

Compare **`SFF RMSE`** against **`SAM Angle`**. SFF should separate kaolinite
from muscovite better than SAM does — the two have similar overall SWIR shape
but different feature *structure*, which is exactly the distinction SFF makes.

**Deliverable 3:** the alunite band-depth map from 3a alongside SAM and SFF
maps for the same four minerals, and a paragraph on where the three disagree
and which you trust there. Band depth measures one absorption and nothing else,
so where it and a classifier diverge, one of them is telling you something the
other cannot see.

---

## Part 4 — Endmembers from the data itself (40 min)

Library spectra are laboratory measurements of pure samples. Field pixels are
mixtures under a real atmosphere. Pull the endmembers out of the scene instead.

1. Run **MNF** and keep components up to the scree-plot elbow
   ({doc}`Tutorial 6 <../06-pca-mnf>`).
2. Open the **Interactive Scatter Plot** on MNF band 1 against MNF band 2.
   Mixtures fall inside the convex hull of the pure materials, so the
   **corners** of the point cloud are your candidate endmembers.
3. Lasso each corner and **Create ROI from Selection**.
4. Collect each ROI's **mean spectrum**.
5. Run **Linear Unmixing** with those spectra as endmembers.

**Read the RMSE band before the abundance bands.** High residual marks pixels
your endmember set cannot explain — usually a material you missed. Add an
endmember and re-run until the residual is flat.

**Deliverable 4:** abundance maps and the RMSE map, plus a comparison of your
image-derived endmembers against the USGS library spectra for the same
minerals. Explain any differences (grain size, mixing, residual atmosphere,
illumination).

---

## Questions to answer

1. Kaolinite and muscovite both put their deepest band at 2200 nm, and a
   single band-depth map cannot tell them apart. Which of SAM and SFF separates
   them better, and why does that follow from how each works?
2. The decorrelation stretch made the alteration zones obvious, but you were
   told not to read mineralogy from its colours. Why not — what exactly does a
   colour in that image correspond to?
3. You get a high SAM score for buddingtonite in an area with no other
   alteration minerals. What would you check before reporting it?
4. Your unmixing RMSE is high across a whole playa. Give two possible causes
   and say how you would tell them apart.
5. Why does restricting the wavelength range change a SAM result at all, given
   that SAM is supposed to be insensitive to brightness?

---

## Going further

- Compare your alteration map against the published USGS Cuprite maps at
  [crustal.usgs.gov/speclab](https://crustal.usgs.gov/speclab/).
- Repeat with an **AVIRIS-NG** scene (5 nm sampling instead of 10 nm) and see
  which mineral separations improve.
- Run the same analysis on a **radiance** product and document how the results
  degrade — a useful demonstration of why atmospheric correction matters.
