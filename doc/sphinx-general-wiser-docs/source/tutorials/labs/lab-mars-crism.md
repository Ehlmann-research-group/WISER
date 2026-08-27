# Lab C — Martian Mineralogy with CRISM

- **Field:** planetary science, astrobiology
- **Instrument:** MRO/CRISM — 545 bands, 362–3920 nm, ~18 m/pixel targeted
- **Prerequisites:** {doc}`Tutorials 1–7 <../index>`; {doc}`Lab B <lab-cuprite-minerals>` helps
- **Time:** 2–3 hours

```{note}
Capture your own figures as you work — the WISER team has not shot screenshots
for this lab. {doc}`Lab A <lab-aviris-ng-urban>` shows every dialog involved.
```

---

## The question

Jezero Crater held a lake. A delta built out into it, and the surrounding
watershed carries olivine- and carbonate-bearing units. Carbonate forms in the
presence of water and preserves biosignatures well, which is a large part of
why *Perseverance* landed there in 2021 — the detections were made from orbit,
with CRISM, before any lander confirmed them.

In this lab you make those detections yourself.

| Mineral | Diagnostic absorptions | Why it matters |
|---|---|---|
| **Olivine** | broad compound band centred ~1000 nm | Primary igneous; unweathered |
| **Mg-carbonate** | paired bands at **2300** and **2500 nm** | Aqueous alteration; biosignature host |
| **Fe/Mg-smectite** | 2300 nm with a 1400/1900 nm hydration pair | Prolonged water–rock interaction |
| **Pyroxene** | broad bands near 1000 and 2000 nm | Primary igneous |

Carbonate and Fe/Mg-smectite both absorb near 2300 nm. Separating them is the
analytical crux, and it is done on the **2500 nm** band: carbonate has it,
smectite does not.

---

## Get the data

CRISM data are free and need no login.

1. Go to the [Mars Orbital Data Explorer](https://ode.rsl.wustl.edu/mars/).
2. Search for **CRISM MTRDR** products over Jezero Crater (about 18.4°N,
   77.7°E). `FRT0000B8C2` and `HRL000040FF` both cover the delta.
3. Download the **`*_IF*_MTR3.IMG`** cube and its **`.HDR`** — the
   map-projected, atmospherically corrected I/F hyperspectral cube. Keep both
   files in the same directory.
4. Optionally also take the **`*_SR*`** refined summary-parameter product and
   the **`*_BR*`** browse products, for cross-checking.

```{admonition} Why MTRDR and not TRDR
:class: tip
**MTRDR** (Map-projected Targeted Reduced Data Record) products are
analysis-ready: map-projected, photometrically and atmospherically corrected,
with noisy channels and detector overlap already handled. Raw **TRDR** products
need the CRISM Analysis Toolkit before any of this works. Start with MTRDR.
```

The `*_WV*` text file lists the wavelength for each channel.

---

## Part 1 — Open and orient (25 min)

1. **File ▸ Open...** → the `_IF*_MTR3.IMG` file (or its `.HDR`). WISER reads
   it as an ENVI raster.
2. If WISER cannot work the format out, set the file type explicitly rather
   than leaving it on **All supported files** — see
   {doc}`Opening Data Files <../../user-content/opening-data-files>`.
3. Build a **false-colour composite**. CRISM's standard "FAL" browse
   combination is roughly R = 2529 nm, G = 1506 nm, B = 1080 nm. Apply a 2.5%
   linear stretch.
4. Click across the scene and watch the spectra. Note the vertical striping —
   CRISM's detector produces column-correlated noise, visible in single-pixel
   spectra.

```{admonition} CRISM spectra need averaging
:class: important
A single CRISM pixel is often too noisy to identify a 1–2% deep carbonate band.
Set **Number of pixels to average** to a 5 × 5 **median** before identifying
anything, and prefer ROI mean spectra
({doc}`Tutorial 3 <../03-regions-of-interest>`) over single clicks.
```

**Deliverable 1:** a false-colour image with the delta marked, and one raw
single-pixel spectrum beside one 5 × 5 median from the same pixel.

---

## Part 2 — The ratio trick (30 min)

Column noise, residual atmosphere and a broad ferric slope sit on top of the
mineral bands. Planetary spectroscopists remove them by **ratioing**: divide
the spectrum of interest by one of spectrally bland ground **in the same
detector columns**.

1. Draw an ROI over a bland, dusty area — no obvious absorptions — spanning the
   same image columns as your area of interest.
2. Draw a second ROI over the unit you want to characterise.
3. Collect both mean spectra.
4. **Tools ▸ Band math...**:

   ```text
   target / background
   ```

   Bind both variables as type **Spectrum**.

```{admonition} What a ratio costs you
:class: warning
A ratioed spectrum is not reflectance. Band *depths* are relative to your
denominator, so they are not comparable to laboratory values, and any feature
your denominator contains shows up **inverted** in the result. Always report
which region you divided by, and keep the unratioed spectrum alongside.
```

**Deliverable 2:** one ratioed spectrum showing a clear mineral band, with the
denominator ROI identified.

---

## Part 3 — Band-depth maps (45 min)

Planetary work usually maps **band depth**: how deep an absorption is relative
to a continuum drawn across it. For a band at $\lambda_c$ with shoulders at
$\lambda_s$ and $\lambda_l$:

$$D = 1 - \frac{R_{\lambda_c}}{a\,R_{\lambda_s} + b\,R_{\lambda_l}}$$

Setting $a = b = 0.5$ (shoulders equally spaced) makes this straightforward
band math. Use the band chooser to find the band index nearest each wavelength.

**Carbonate, 2300 nm:**

```text
1 - c2300 / (0.5 * s2250 + 0.5 * s2350)
```

**Carbonate, 2500 nm — the discriminator:**

```text
1 - c2500 / (0.5 * s2450 + 0.5 * s2530)
```

**Olivine, ~1000 nm:**

```text
1 - c1050 / (0.5 * s0860 + 0.5 * s1470)
```

Bind each variable as an **Image Band** at the wavelength its name gives.

Display each band-depth map with a sequential colormap and a **tight stretch**
— these features are a few percent deep, so a 100% linear stretch shows you
nothing. (Lab A demonstrates what a badly stretched computed product looks
like.)

**Now separate carbonate from smectite.** Both light up at 2300 nm; only
carbonate lights up at 2500 nm:

```text
(d2300 > 0.02) * (d2500 > 0.01)
```

The product is 1 only where both tests pass. Choose the two thresholds from
your own band-depth histograms, not from these example numbers.

**Deliverable 3:** band-depth maps for 2300 nm, 2500 nm and 1000 nm, plus the
combined carbonate mask, with your thresholds stated and justified.

---

## Part 4 — Confirm with spectral matching (30 min)

Band-depth maps are indices; they can be fooled. Confirm them.

1. Draw ROIs on the pixels your carbonate mask flagged and on olivine-rich
   areas; collect their mean spectra.
2. Import CRISM-convolved USGS library spectra — the
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

- Repeat over **Nili Fossae** (`FRT00003E12` and neighbours) — the largest
  carbonate exposure known on Mars, and Jezero's watershed.
- Compare a CRISM detection against **Perseverance** ground truth: SuperCam and
  SHERLOC results for the Jezero delta are published, giving a rare
  orbit-to-ground check.
- Try **Mawrth Vallis** for a layered kaolinite-over-smectite stratigraphy and
  map the contact.
