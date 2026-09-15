# Lab B — Mineral Mapping at Cuprite, Nevada

- **Field:** economic geology, alteration mapping
- **Instrument:** AVIRIS-Classic (Airborne Visible/Infrared Imaging
  Spectrometer), 224 bands, 379–2498 nm at ~9.5 nm, 15.2 m
- **Prerequisites:** {doc}`Tutorials 1–7 <../index>`, and {doc}`Lab A <lab-aviris-ng-urban>` for the mechanics on a large cube

```{admonition} You will need to download data for this lab
:class: note
**Get the data** below has two routes: one command that fetches only the part
you need, or a click-to-download link for the whole flight line.
```

---

## The setting

Cuprite, Nevada is a common reference site for imaging spectroscopy. A hydrothermal
system altered the volcanic rocks into concentric mineral zones, vegetation
cover is close to nil, and the outcrops are large enough to resolve at 15 m.
Many method papers use it as their demonstration site, so your results have a
substantial literature to check against. The most detailed mineral map of the
site was made from AVIRIS data much like yours
([Swayze et al. 2014](https://doi.org/10.2113/econgeo.109.5.1179)), and it is
worth looking at once you have your own maps.

The alteration zoning, from the center of each hydrothermal center outwards:

| Zone | Diagnostic mineral | Key absorption |
|---|---|---|
| Silicified core | opal / chalcedony | broad, weak features |
| Opalised | **alunite** | 2170 nm (sharp) |
| Argillic | **kaolinite** | **doublet** at 2160 + 2200 nm |
| Outer / unaltered | **muscovite / illite** | 2200 nm (broader, asymmetric) |
| Playa and carbonate units | **calcite** | 2340 nm |
| Locally | **buddingtonite** (NH₄-feldspar) | 2120 nm |

The rest of the lab works from those wavelengths.

---

## Get the data

**Scene:** `f230918t01p00r11_rfl`, an orthocorrected AVIRIS-Classic
**reflectance** flight line flown on 18 September 2023, published open (no
login) in a public directory on JPL's AVIRIS distribution server. It runs
110 km north–south through southwestern Nevada and crosses Cuprite about
40 % of the way along.

The whole line is **10.2 GB**, and you only need the part over Cuprite. Take
either route below: the first downloads a fifth as much but needs a terminal,
the second is three clicks and a long wait.

### Route 1 — fetch only the part you need (2.05 GB)

Because the file is **band-interleaved-by-line** (BIL), every image line is one
contiguous
block of bytes, so a single ranged request gets you a usable cube.

**1. Fetch the header** (a few kB, and it carries the 224 wavelengths):

```bash
curl -O https://popo.jpl.nasa.gov/pub/RKokaly/f230918t01p00r11_rfl.hdr
```

**2. Fetch lines 2400–3799** — 1400 lines centered on the district, 2.05 GB,
a few minutes on a decent connection:

```bash
curl -r 3513753600-5563443199 \
  https://popo.jpl.nasa.gov/pub/RKokaly/f230918t01p00r11_rfl \
  -o f230918t01p00r11_rfl_cuprite
```

```{admonition} Where that byte range comes from
:class: note
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

The orthocorrection pads the rotated flight line with **−9999**, and here that
fill is over half the frame. Undeclared, it drags every contrast stretch and
every statistic with it. WISER will also accept it after the fact through
**Edit dataset...** in the Dataset Info pane.

The result is a 1634 × 1400 × 224 cube. Valid data occupies samples ~220–1000;
everything outside is fill.

### Route 2 — download the whole flight line (10.2 GB)

If you would rather not use a terminal, take both files straight from the
archive. Save them side by side and leave the names alone: the header that
ships with the file already describes the full line correctly, so the only edit
is the `data ignore value` line noted below.

- [`f230918t01p00r11_rfl`](https://popo.jpl.nasa.gov/pub/RKokaly/f230918t01p00r11_rfl)
  — the data, **10.2 GB**
- [`f230918t01p00r11_rfl.hdr`](https://popo.jpl.nasa.gov/pub/RKokaly/f230918t01p00r11_rfl.hdr)
  — the header, 5 KB

Both are in the same [public directory](https://popo.jpl.nasa.gov/pub/RKokaly/)
if you would rather browse.

```{admonition} Route 2 shifts every y coordinate by 2400
:class: note
The pixel coordinates quoted throughout this lab refer to the 1400-line subset
from Route 1, whose first line is line 2400 of the full flight line. If you
took Route 2, add **2400** to every *y* value: the alunite pixel listed as
(353, 993) is at (353, 3393) in the whole line. The *x* values are the same
either way.

The archive header also omits `data ignore value = -9999`, so add that line
whichever route you take, or set it through **Edit dataset...** once the file
is open.
```

```{admonition} Use reflectance, not radiance
:class: note
Radiance carries the solar spectrum and the atmosphere in it, so its absorption
features are mostly not the surface's. Every method here assumes atmospherically
corrected **reflectance**, which is what the `_rfl` suffix means. Check the file
name and header description first.
```

**Reference spectra (optional, for Part 3):** download `usgs_splib07.zip` from
the [USGS Spectral Library Version 7 release](https://dx.doi.org/10.5066/F7RR1WDJ).
Use the version **convolved to AVIRIS-Classic**. The library ships copies
resampled to AVIRIS-Classic, HyMap, Hyperion, CRISM, M3 and VIMS, and matching
the sensor saves a resampling step and a class of subtle errors.

---

## Part 1 — Orient yourself

1. Open `f230918t01p00r11_rfl_cuprite.hdr` with **File ▸ Open...**, then click
   the four pane toggles on the Main Toolbar.
2. Use **Choose Visible-Light Bands** (660 / 550 / 480 nm) and apply a **2.5%
   linear** stretch.

:::{figure} ../../_static/tutorials/lab_cuprite_truecolour.png
:width: 100%
:align: center
:alt: Cuprite in true color, a near-featureless beige desert
:::

Cuprite in true color is beige gravel. None of the zoning in the table above
is visible here, and no stretch of these three bands will bring it out.

3. Now build a short-wave infrared (SWIR) composite: red **2200 nm**, green
   **2170 nm**, blue
   **2340 nm**, and stretch it 2.5% linear again.

:::{figure} ../../_static/tutorials/lab_cuprite_swir.png
:width: 100%
:align: center
:alt: The same area as a 2200/2170/2340 nm composite, still largely gray
:::

Barely better, and that is not a mistake: neighboring SWIR bands are strongly
correlated, so an RGB composite built from three of them is close to gray no
matter how you stretch each channel independently.

4. Reopen the stretch dialog and choose **Decorrelation Stretch** (it is only
   enabled for 3-band displays).

:::{figure} ../../_static/tutorials/lab_cuprite_decorr.png
:width: 100%
:align: center
:alt: The same three bands after a decorrelation stretch, showing strong color separation
:::

The decorrelation stretch rotates the three display bands onto their principal
axes, stretches *those*, and rotates back, removing the correlation that made
the image gray. The alteration zones separate into distinct colors.
Compare this with the band-depth map in Part 3: the dark-blue patches here are
the alunite-rich ground.

```{note}
Color in a decorrelation stretch is **relative**, not diagnostic. It tells you
*that* two areas differ spectrally, never *which mineral* either one is. That
requires the spectra themselves, in Part 2.
```

```{admonition} No bad-band list in this file
:class: note
Unlike many AVIRIS products, this header carries **no `bbl` entry**, so nothing
marks the 1400 nm and 1900 nm water-vapor regions as unusable. They are still
unusable. Keep them out of every wavelength range you give an analysis tool,
and expect to see them as noise spikes in Part 2.
```

**Deliverable 1:** true-color, plain SWIR and decorrelation-stretched
composites of the same area, exported with **Export RGB image ▸ Export visible
image area**, plus two sentences on what the decorrelation stretch shows that
the other two do not.

---

## Part 2 — Identify minerals by hand

A decorrelation stretch separates the ground into units without identifying
them. Naming the minerals takes the spectra.

1. Click across the bright altered ground and collect spectra. Set **Number of
   pixels to average** to a 3 × 3 **median**, because a single AVIRIS pixel at
   2200 nm is noisy.

Your own clicking will find the altered ground. The four spectra below were
picked more systematically, by searching the whole cube for the deepest
absorption at each diagnostic wavelength — the measurement Part 3a formalizes as
band depth. Averaging three channels rather than reading one mattered: the
single-channel version returns detector spikes rather than minerals, with a
maximum of 0.094 against 0.034 for the averaged version, and the spike pixels
fell in no coherent spatial pattern. The coordinates are Route 1's; add 2400 to each *y* if
you downloaded the whole flight line:

| Mineral | Pixel (x, y) | Deepest SWIR band |
|---|---|---|
| Alunite | 353, 993 | 2170 nm |
| Kaolinite | 663, 745 | 2200 nm, with a 2160 shoulder |
| Muscovite | 266, 1033 | 2200 nm |
| Calcite | 253, 364 | 2339 nm |

:::{figure} ../../_static/tutorials/lab_cuprite_spectra_window.png
:width: 100%
:align: center
:alt: The Cuprite scene with four spectra collected, listed in the Spectra pane
:::

The status bar confirms which pixel you are on (`Pixel: (253, 364)` above), and
the **Spectra and Spectral Libraries** pane lists what you have collected
so far. Check both before reading the plot.

:::{figure} ../../_static/tutorials/lab_cuprite_spectra_plot.png
:width: 100%
:align: center
:alt: Four full-range Cuprite spectra with large noise spikes at 1400 and 1900 nm
:::

The spikes at 1400 and 1900 nm are the water-vapor regions. Atmospheric
correction cannot recover them, so the values there mean nothing. The mineral
features are the small wiggles past 2000 nm, dwarfed at this scale by the
differences in overall brightness.

2. Set the plot's x-axis range to **2000–2500 nm** (click **Configure** in the
   Spectrum Plot toolbar) so the SWIR features fill the frame.

:::{figure} ../../_static/tutorials/lab_cuprite_swir_spectra.png
:width: 100%
:align: center
:alt: The same four spectra restricted to 2000-2500 nm, showing distinct absorption bands
:::

Each mineral now shows a distinct absorption:

- **Alunite** (red) — deepest at **2170 nm**, with a second minimum near 2210
- **Kaolinite** (blue) — a **doublet**, 2160 and a deeper 2200
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

Zoomed in, the kaolinite doublet is unambiguous: 2160 and a deeper 2200. The
band depths are now readable directly as a fraction of the continuum, about
0.29 and 0.32 here.

```{admonition} Continuum removal fits the hull over the whole spectrum
:class: note
WISER removes the continuum across every band in the spectrum, and the noise
spike at 1400 nm becomes a hull vertex, visible in the first figure as the
peak the hull is pinned to. The result is still correct *within* the SWIR, but
do not read the 1300–2000 nm part of a continuum-removed AVIRIS spectrum.
```

4. Confirm each: import the USGS library and run **SAM in Spectrum mode**
   ({doc}`Tutorial 7 <../07-detection>`) with your collected spectrum as the
   target. Read the ranked match table.

**Deliverable 2:** a continuum-removed plot of four identified spectra, each
labeled with its mineral and the wavelength you used to call it.

---

## Part 3 — Map the minerals

### 3a. Band depth

Before reaching for a classifier, map a single absorption. A **band depth** is
the diagnostic band divided by a straight continuum drawn between two shoulders
either side of it, and WISER's band math computes it in one expression.

For alunite, use 2100 nm and 2250 nm as shoulders and 2170 nm as the center.
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
:alt: The 2170 nm alunite band-depth map, showing two bright alteration centers
:::

The two bright lobes are the opalised cores of the hydrothermal centers. Band
depth reaches 0.35 there, meaning the 2170 nm channel sits 35 % below the
continuum drawn across it. Each pixel now holds a measurement rather than a
display effect, so you can threshold it, compare it against another scene, or
check it against a laboratory spectrum.

Repeat for the other three minerals by moving the center and shoulders:

| Mineral | Center | Shoulders | Weight on the upper shoulder |
|---|---|---|---|
| Alunite | 2170 nm | 2100 / 2250 | 0.467 |
| Kaolinite–muscovite | 2200 nm | 2130 / 2280 | 0.467 |
| Calcite | 2340 nm | 2260 / 2400 | 0.571 |

```{admonition} Band depth does not separate kaolinite from muscovite
:class: note
Both absorb at 2200 nm, so one band-depth map lights up for both. Separating
them needs the *shape* of the feature, the 2160 shoulder: either SFF in 3c, or
a ratio of the 2160 and 2200 depths.
```

### 3b. Spectral Angle Mapper

1. **Tools ▸ Data Analysis ▸ Spectral Angle Mapper**, target **Image Cube**.
2. Add the USGS library; tick alunite, kaolinite, muscovite and calcite.
3. **Wavelength range: 2000–2400 nm.** Over the full range the albedo and
   iron-oxide variation in the visible dominates the angle and swamps the clay
   signal.
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

Compare **`SFF RMSE`** against **`SAM Angle`**. SAM compares the whole spectral
vector and reports the angle between it and the reference, which makes it
insensitive to brightness but blind to which features account for the shape. SFF
continuum-removes both spectra inside the window you choose and fits the
reference's absorption to the target's, so it is scored on the depth, width and
position of one feature. Kaolinite and muscovite share an overall SWIR slope and
a 2200 nm minimum, and only kaolinite has the 2160 nm shoulder, so SFF over a
2120–2250 nm window should separate them better than SAM does.

### 3d. One map, four minerals

So far you have four separate maps, one per mineral. A geologist wants one map
with four colors. Band math will build it, because `SAM CLS` bands are 1 where
the mineral was detected and 0 where it was not, and you can use those as
switches.

The catch is overlap. A pixel can pass the threshold for both kaolinite and
muscovite, so you cannot simply add the four bands together — a pixel that is
both would score 2 + 3 = 5 and come out as a mineral you never mapped. Instead,
give the minerals a precedence order and let each one claim only the pixels the
ones before it did not:

1. Open **Tools ▸ Band math...** and type:

   ```text
   a + (1 - a) * k * 2 + (1 - a) * (1 - k) * m * 3 + (1 - a) * (1 - k) * (1 - m) * c * 4
   ```

2. Bind `a`, `k`, `m` and `c` to the alunite, kaolinite, muscovite and calcite
   bands of your **`SAM CLS`** dataset, all as **Image Band**.
3. Name the result `MineralClasses` and click **OK**.
4. Switch to it with **Select dataset to view**, open the **Band chooser**,
   select **Grayscale**, tick **Use a colormap** and choose a categorical
   colormap such as **tab10**. Click **OK**.

Every pixel now holds 0 for unclassified, 1 for alunite, 2 for kaolinite, 3 for
muscovite or 4 for calcite, drawn in four distinct colors.

```{admonition} Interpretation
:class: note
Read the precedence order as part of your method, not as a detail. You put
alunite first, so anywhere alunite and kaolinite were both detected is now
colored alunite, and the kaolinite class is really "kaolinite where alunite was
not." Reorder the expression and the boundaries between zones move. Say which
order you used when you present the map.

This is also why the map is worth less than the four maps it came from. A single
color per pixel throws away the fact that a pixel matched two minerals, which is
usually the interesting thing about it. Keep both.
```

### 3e. Let the classifier find the zones

The map in 3d needed you to name four minerals first. K-means does not.

1. Run **Tools ▸ Data Analysis ▸ K-means** on the cube with **K clusters** set
   to 6, and a fixed **Random Seed**.
2. Display the labels with a categorical colormap, as in 3d.
3. Reopen the **K-means Dialog** and click **View Centroids**. Each cluster's
   mean spectrum is plotted together.
4. Work along the plot and name each cluster from its SWIR features: a minimum
   at 2170 nm is alunite, 2200 nm with a 2160 shoulder is kaolinite, 2200 nm
   without the shoulder is muscovite, 2340 nm is calcite. Clusters with no SWIR
   feature are unaltered ground.

```{admonition} Interpretation
:class: note
This is the step that separates a spectral image from a picture. The classifier
grouped pixels by the shape of their spectra without being told what any mineral
looks like, and **View Centroids** hands you the average spectrum of each group
so you can identify it afterwards. Compare the result against your 3d map: where
the two agree you have a mineral zone that shows up whether or not you went
looking for it.
```

**Deliverable 3:** the alunite band-depth map from 3a, SAM and SFF maps for the
same four minerals, and the combined mineral map from 3d. Add a paragraph on
where the methods disagree and which you trust there, and name the precedence
order you used in 3d.

---

## Part 4 — Endmembers from the data itself

Library spectra are laboratory measurements of pure samples. Field pixels are
mixtures under a real atmosphere. Pull the endmembers out of the scene instead.

1. Run **Tools ▸ Data Analysis ▸ Minimum Noise Fraction**. Set **Choose
   Dataset** to the Cuprite cube, leave **Num Components** at its default for a
   first pass, and click **OK**. Keep components up to the scree-plot elbow
   ({doc}`Tutorial 6 <../06-pca-mnf>`).
2. Right-click the image and choose **Data Analysis ▸ Interactive Scatter
   Plot**. Set **X Axis Band** to MNF band 1 and **Y Axis Band** to MNF band 2,
   set **Render Onto** to the MNF result, and click **Create Plot**.
   Mixtures fall inside the convex hull of the pure materials, so the
   **corners** of the point cloud are your candidate endmembers.
3. Lasso each corner and **Create ROI from Selection**.
4. Collect each ROI's **mean spectrum**.
5. Run **Tools ▸ Data Analysis ▸ Linear Unmixing**. Set **Input Dataset** to
   the Cuprite cube, click **Add Collected Spectrum** once per endmember to
   load the ROI means, and click **OK**.

**Read the RMSE band before the abundance bands.** High residual marks pixels
your endmember set cannot explain, usually a material you missed. Add an
endmember and re-run until the residual is flat.

**Deliverable 4:** abundance maps and the RMSE map, plus a comparison of your
image-derived endmembers against the USGS library spectra for the same
minerals. Explain any differences (grain size, mixing, residual atmosphere,
illumination).

---

## Questions to answer

1. Put your SAM result and your SFF result for kaolinite side by side and find
   somewhere they disagree. The two are measuring different things about the
   same spectrum. What would you look at to decide which one to trust there?
2. The decorrelation stretch in Part 1 made the alteration zones jump out, but
   the text warned you not to read mineralogy off its colors. What does a color
   in that image actually correspond to?
3. You get a strong buddingtonite match in a spot with no other alteration
   minerals around it. Before you tell anyone, what would you want to check?
4. Your unmixing RMSE is high across a whole playa. Give two things that could
   cause that and say how you would tell which it was.
5. SAM is supposed to ignore brightness, yet changing the wavelength range you
   run it over changes the answer. Why should that be, if brightness is not what
   it is comparing?

---

## Going further

- Compare your alteration map against the published USGS Cuprite maps at
  [the USGS Spectroscopy Lab](https://www.usgs.gov/labs/spectroscopy-lab).
- Repeat with an **AVIRIS-NG** scene (~5 nm instead of ~9.5 nm) and see
  which mineral separations improve.
- Run the same analysis on a **radiance** product and document how the results
  degrade.
