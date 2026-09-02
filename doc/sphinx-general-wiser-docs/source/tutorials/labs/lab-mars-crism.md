# Lab C — Martian Mineralogy with CRISM

- **Field:** planetary science, astrobiology
- **Instrument:** CRISM (Compact Reconnaissance Imaging Spectrometer for Mars)
  on Mars Reconnaissance Orbiter. Map-projected cubes carry 489 bands over
  436–3897 nm, at 18 m/pixel (full-resolution targeted, `FRT`) or 36 m
  (half-resolution long, `HRL`)
- **Prerequisites:** {doc}`Tutorials 1–7 <../index>`; {doc}`Lab B <lab-cuprite-minerals>` helps

```{admonition} You will need to download data for this lab
:class: note
You will need to download some CRISM data to do this lab. **Get the data**
below has click-to-download links and the equivalent commands.
```

---

## The question

Jezero Crater held a lake. A delta built out into it, and the surrounding
watershed carries olivine- and carbonate-bearing units. Carbonate forms in the
presence of water and preserves biosignatures well, which is a large part of
why *Perseverance* landed there in 2021. The detections were made from orbit,
with CRISM, before any lander confirmed them.

In this lab you make those detections yourself.

| Mineral | Diagnostic absorptions | Why it matters |
|---|---|---|
| **Olivine** | broad compound band centred ~1000 nm | Primary igneous; unweathered |
| **Mg-carbonate** | paired bands near **2310** and **2510 nm** | Aqueous alteration; biosignature host |
| **Fe/Mg-smectite** | ~2300 nm with a 1400/1900 nm hydration pair | Prolonged water–rock interaction |
| **Pyroxene** | broad bands near 1000 and 2000 nm | Primary igneous |

Carbonate and Fe/Mg-smectite both absorb near 2300 nm. Separating them is the
analytical crux, and it is done on the **2510 nm** band: carbonate has it,
smectite does not.

---

## Get the data

CRISM data are free and need no login. The figures in this lab come from
**`HRL000040FF`**, which covers the western delta at 36 m/pixel and is the
smallest of the Jezero cubes at 640 MB.

**Click to download.** Save both into the same directory:

- [`HRL000040FF_07_IF183J_MTR3.IMG`](https://pds-geosciences.wustl.edu/mro/mro-m-crism-5-rdr-mptargeted-v1/mrocr_4001/mtrdr/2007/2007_029/hrl000040ff/HRL000040FF_07_IF183J_MTR3.IMG)
  — the cube, 640 MB
- [`HRL000040FF_07_IF183J_MTR3.HDR`](https://pds-geosciences.wustl.edu/mro/mro-m-crism-5-rdr-mptargeted-v1/mrocr_4001/mtrdr/2007/2007_029/hrl000040ff/HRL000040FF_07_IF183J_MTR3.HDR)
  — the header, 15 KB

The [full product directory](https://pds-geosciences.wustl.edu/mro/mro-m-crism-5-rdr-mptargeted-v1/mrocr_4001/mtrdr/2007/2007_029/hrl000040ff/)
holds the browse images and summary products alongside them.

**Or from a terminal:**

```bash
B=https://pds-geosciences.wustl.edu/mro/mro-m-crism-5-rdr-mptargeted-v1
curl -O $B/mrocr_4001/mtrdr/2007/2007_029/hrl000040ff/HRL000040FF_07_IF183J_MTR3.HDR
curl -O $B/mrocr_4001/mtrdr/2007/2007_029/hrl000040ff/HRL000040FF_07_IF183J_MTR3.IMG
```

Keep both files in the same directory. Other cubes over the same target:

| Product | Resolution | Size | Coverage |
|---|---|---|---|
| **`HRL000040FF`** | 36 m | 640 MB | 18.28–18.73°N, 77.28–77.56°E |
| `FRT000047A3` | 18 m | 1.3 GB | 18.44–18.69°N, 77.38–77.64°E |
| `FRT00005C5E` | 18 m | 1.5 GB | 18.37–18.64°N, 77.28–77.56°E |

The two **FRT** cubes are full-resolution and worth the extra download if you
want to resolve individual delta foresets; the workflow below is identical.

To find others yourself, search the
[Mars Orbital Data Explorer](https://ode.rsl.wustl.edu/mars/) for **CRISM
MTRDR** over Jezero (18.38°N, 77.58°E) and take the **`*_IF*_MTR3.IMG`** cube
with its **`.HDR`**. Check the footprint before you download: many products
whose names look similar are nowhere near the crater.

Optionally also take the **`*_SR*`** refined summary-parameter product and the
**`*_BR*`** browse products, for cross-checking.

```{admonition} Why MTRDR and not TRDR
:class: note
**MTRDR** (Map-projected Targeted Reduced Data Record) products are
analysis-ready: map-projected, photometrically and atmospherically corrected,
with noisy channels and detector overlap already handled. Raw **TRDR** products
need the CRISM Analysis Toolkit before any of this works. Start with MTRDR.
```

The `*_WV*` text file lists the wavelength for each channel; the `.HDR` already
carries them, so WISER plots against wavelength without it.

---

## Part 1 — Open and orient

1. **File ▸ Open...** → the `_IF*_MTR3.IMG` file. WISER reads it as an ENVI
   raster; the `.HDR` beside it supplies the wavelengths.

```{admonition} Open the .IMG, not the .HDR
:class: note
For most ENVI datasets either file works. These products name the header
`.HDR` in capitals, and GDAL will tell you to select the data file instead.
Open the `.IMG`.
```

2. The file names its own default bands, so WISER shows a composite as soon as
   it opens. Apply a **2.5% linear** stretch.

:::{figure} ../../_static/tutorials/lab_crism_default.png
:width: 100%
:align: center
:alt: The Jezero MTRDR cube in the bands its header nominates
:::

3. Now set the standard **"FAL"** browse combination — R = 2529 nm (band 303),
   G = 1506 nm (band 148), B = 1080 nm (band 83) — and stretch again.

:::{figure} ../../_static/tutorials/lab_crism_fal.png
:width: 100%
:align: center
:alt: Jezero Crater in the CRISM FAL false-color composite, with the delta clearly visible
:::

The **delta** is the fan in the lower half of the scene, its channels picked
out in gray-green against the pink-red crater floor. Color differences here
are already mineralogical, but as at Cuprite, color alone names nothing.

4. Click across the scene and watch the spectra. Note the vertical striping:
   CRISM's detector produces column-correlated noise, visible in single-pixel
   spectra and in the band-depth maps you will make in Part 3.

```{admonition} CRISM spectra need averaging
:class: note
A single CRISM pixel is often too noisy to identify a 1–2% deep carbonate band.
Set **Number of pixels to average** to a 5 × 5 **median** before identifying
anything, and prefer ROI mean spectra
({doc}`Tutorial 3 <../03-regions-of-interest>`) over single clicks.
```

```{admonition} Ignore everything past ~2600 nm
:class: note
The header flags **no bad bands**, but the channels beyond about 2.6 µm are at
the detector's long-wavelength edge and are not usable; values there swing by
hundreds of I/F units and will wreck an autoscaled plot. Pin the spectrum
plot's x-axis (gear icon ▸ axis range) to something like **900–2650 nm** before
you read anything.
```

**Deliverable 1:** a false-color image with the delta marked, and one raw
single-pixel spectrum beside one 5 × 5 median from the same pixel.

---

## Part 2 — Find the carbonate pair, then clean it up

These three spectra come from pixels on the crater margin, plotted over
900–2650 nm with the axes pinned:

| Label | Pixel (x, y) |
|---|---|
| Carbonate A | 123, 298 |
| Carbonate B | 182, 229 |
| Carbonate C | 106, 371 |

:::{figure} ../../_static/tutorials/lab_crism_spectra_window.png
:width: 100%
:align: center
:alt: The Jezero scene with three carbonate spectra collected
:::

:::{figure} ../../_static/tutorials/lab_crism_spectra_plot.png
:width: 100%
:align: center
:alt: Three CRISM spectra from the Jezero margin over 900-2650 nm
:::

At this scale the mineral bands are barely perceptible. Tighten the axes onto
2150–2600 nm:

:::{figure} ../../_static/tutorials/lab_crism_carbonate_pair.png
:width: 100%
:align: center
:alt: The same three spectra zoomed to 2150-2600 nm, showing paired absorptions near 2310 and 2510 nm
:::

Now the **pair** is unmistakable: a minimum near **2310 nm**, a recovery through
2360–2410, and a second minimum near **2510 nm**. Both bands are only about
**2–3 % deep**. A smectite would give you the first and not the second.

```{admonition} Why this has to be measured rather than looked at
:class: note
You cannot eyeball a 2 % band across a million pixels, and at that depth noise
and real signal look alike in any single spectrum. The band-depth maps in
Part 3 measure it instead. A detection is then judged on whether the deep
pixels form a coherent unit, not on depth alone.
```

### Ratioing

Column noise, residual atmosphere and a broad ferric slope sit on top of the
mineral bands. Planetary spectroscopists remove them by **ratioing**: divide
the spectrum of interest by one of spectrally bland ground **in the same
detector columns**.

1. Draw an ROI over a bland, dusty area with no obvious absorptions, spanning the
   same image columns as your area of interest.
2. Draw a second ROI over the unit you want to characterize.
3. Collect both mean spectra.
4. **Tools ▸ Band math...**:

   ```text
   target / background
   ```

   Bind both variables as type **Spectrum**.

```{admonition} What a ratio costs you
:class: note
A ratioed spectrum is not reflectance. Band *depths* are relative to your
denominator, so they are not comparable to laboratory values, and any feature
your denominator contains shows up **inverted** in the result. Always report
which region you divided by, and keep the unratioed spectrum alongside.
```

**Deliverable 2:** one ratioed spectrum showing a clear mineral band, with the
denominator ROI identified.

---

## Part 3 — Band-depth maps

Planetary work usually maps **band depth**: how deep an absorption is relative
to a continuum drawn across it. For a band at $\lambda_c$ with shoulders at
$\lambda_s$ and $\lambda_l$:

$$D = 1 - \frac{R_{\lambda_c}}{(1-f)\,R_{\lambda_s} + f\,R_{\lambda_l}},
\qquad f = \frac{\lambda_c - \lambda_s}{\lambda_l - \lambda_s}$$

$f$ places the continuum at the band center. Take the **2510 nm** band, the one
that separates carbonate from smectite, with shoulders at 2400 and
2600 nm. The nearest CRISM channels are 2397.2, 2509.7 and 2602.1 nm, so
$f = 0.549$:

1. **Tools ▸ Band Math**, expression:

   ```text
   1 - c / (0.451 * a + 0.549 * b)
   ```

2. Bind `a` → **2400 nm** (band 283), `b` → **2600 nm** (band 314),
   `c` → **2510 nm** (band 300). Name it `CarbonateBD2510`.

:::{figure} ../../_static/tutorials/lab_crism_bandmath.png
:width: 80%
:align: center
:alt: The band math dialog with the 2510 nm band-depth expression
:::

3. Display it with a sequential colormap and a **2.5% linear** stretch; a
   full-range stretch shows nothing when the feature is 2 % deep.

:::{figure} ../../_static/tutorials/lab_crism_carbonate.png
:width: 100%
:align: center
:alt: The 2510 nm band-depth map, showing a coherent sinuous carbonate unit
:::

The bright material forms a coherent, sinuous unit following the crater margin
and the delta front. A 2 %-deep band in scattered pixels would be noise; the
same depth mapping onto a geological contact is a mineral. The vertical
striping is CRISM column noise, and separating the two is much of the skill
here.

This is the Jezero **marginal carbonate** unit. Orbital carbonate detections
like this one were part of the case for landing *Perseverance* here, and the
rover has since sampled carbonate-bearing rock in the margin.

Now build the other indices the same way:

| Index | Center | Shoulders | $f$ | What it shows |
|---|---|---|---|---|
| `BD2310` | 2310 nm (band 270) | 2230 / 2400 (bands 258 / 283) | 0.480 | carbonate **and** Fe/Mg-smectite |
| `BD2510` | 2510 nm (band 300) | 2400 / 2600 (bands 283 / 314) | 0.549 | carbonate only |
| `BD1050` | 1050 nm (band 78) | 860 / 1470 (bands 54 / 142) | 0.309 | olivine |

**Now separate carbonate from smectite.** Both light up at 2310 nm; only
carbonate lights up at 2510 nm:

```text
(d2310 > 0.02) * (d2510 > 0.01)
```

The product is 1 only where both tests pass. Choose the two thresholds from
your own band-depth histograms, not from these example numbers. On
`HRL000040FF` the 99th percentile of each index is around 0.019 and 0.014, so
those two example thresholds already select a small fraction of the scene.

```{admonition} Average channels, do not trust one
:class: note
Single CRISM channels carry spikes that a naive band depth reads as a 9 %
absorption. The published CRISM summary parameters average several channels at
each shoulder and center for exactly this reason. If your map is speckled with
extreme single pixels, that is what you are seeing. Median-filter the cube
first ({doc}`Filters <../../user-content/filters>`) or widen the index.
```

**Deliverable 3:** band-depth maps for 2310 nm, 2510 nm and 1050 nm, plus the
combined carbonate mask, with your thresholds stated and justified.

---

## Part 4 — Confirm with spectral matching

Band-depth maps are indices; they can be fooled. Confirm them.

1. Draw ROIs on the pixels your carbonate mask flagged and on olivine-rich
   areas; collect their mean spectra.
2. Import CRISM-convolved USGS library spectra: the
   [USGS Spectral Library Version 7](https://dx.doi.org/10.5066/F7RR1WDJ)
   ships versions resampled to **CRISM**.
3. Run **SFF** ({doc}`Tutorial 7 <../07-detection>`) with the range set to
   **2200–2600 nm** for carbonate and **800–1300 nm** for olivine.
4. Cross-check against the MTRDR **`_SR`** product if you downloaded it: its
   `BD2500` and `OLINDEX` bands are the mission's own versions of what you just
   computed.

**Deliverable 4:** a figure comparing your ROI mean spectrum against the best
library match, diagnostic bands annotated, and a statement of whether your
detection holds.

---

## Questions to answer

1. Why is the 2500 nm band, rather than the deeper 2300 nm band, the one that
   identifies carbonate?
2. Your 2300 nm band-depth map lights up along one image column across the
   whole scene. What is that, and how would you confirm it?
3. What does a ratioed spectrum let you claim, and what does it stop you from
   claiming?
4. Olivine band depth increases with both abundance **and** grain size. What
   does that do to a map you might want to read as abundance?

---

## Going further

- Repeat over **Nili Fossae** (`FRT00003E12` and neighbors), the largest
  carbonate exposure known on Mars, and Jezero's watershed.
- Compare a CRISM detection against **Perseverance** ground truth: SuperCam and
  SHERLOC results for the Jezero delta are published, giving a rare
  orbit-to-ground check.
- Try **Mawrth Vallis** for a layered kaolinite-over-smectite stratigraphy and
  map the contact.
