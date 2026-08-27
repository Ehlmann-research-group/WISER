# Lab B — Mineral Mapping at Cuprite, Nevada

- **Field:** economic geology, alteration mapping
- **Instrument:** AVIRIS-Classic — 224 bands, 400–2500 nm, 20 m
- **Prerequisites:** {doc}`Tutorials 1–7 <../index>`, and {doc}`Lab A <lab-aviris-ng-urban>` for the mechanics on a large cube
- **Time:** 2–3 hours

```{note}
The figures in this lab come from your own run — the WISER team has not shot
screenshots for it, because the scene is a 600 MB download. The workflow,
parameters and expected mineralogy below are all specified; capture your own
figures as you go. {doc}`Lab A <lab-aviris-ng-urban>` shows what each dialog
looks like.
```

---

## The setting

Cuprite, Nevada is the reference site of imaging spectroscopy. A hydrothermal
system altered the volcanic rocks into concentric mineral zones, vegetation
cover is close to nil, and the outcrops are large enough to resolve at 20 m.
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

**Scene:** `f970619t01p02_r02` — AVIRIS-Classic, 19 June 1997, Cuprite.

1. [AVIRIS free data](https://aviris.jpl.nasa.gov/data/free_data.html) —
   download the **reflectance** product for run `f970619t01p02_r02`
   (`..._sc03.a.rfl` and its header), about 600 MB.
2. Or search the [AVIRIS Data Portal](https://aviris.jpl.nasa.gov/dataportal/)
   for `Cuprite` and take any AVIRIS-Classic **L2 reflectance** scene.
3. An orthocorrected L2 collection is mirrored at the
   [ORNL DAAC](https://daac.ornl.gov/AVIRIS/guides/AVIRIS-Classic_L2_Reflectance.html)
   (free Earthdata Login).

Put the data file and its `.hdr` in the same directory.

**Reference spectra:** download `usgs_splib07.zip` from the
[USGS Spectral Library Version 7 release](https://dx.doi.org/10.5066/F7RR1WDJ).
Use the version **convolved to AVIRIS-Classic** — the library ships copies
resampled to AVIRIS-Classic, HyMap, Hyperion, CRISM, M3 and VIMS, and matching
the sensor saves a resampling step and a class of subtle errors.

```{admonition} Use reflectance, not radiance
:class: warning
Radiance carries the solar spectrum and the atmosphere in it, so its absorption
features are mostly not the surface's. Every method here assumes atmospherically
corrected **reflectance**. Check the file name and header description first.
```

---

## Part 1 — Orient yourself (20 min)

1. Open the reflectance file and turn on all four panes.
2. Use **Choose Visible-Light Bands** for a true-colour view, then build a SWIR
   composite: red ≈ 2200 nm, green ≈ 2170 nm, blue ≈ 2340 nm. Apply a **2.5%
   linear** stretch.

The alteration zones appear as colour patterns invisible in true colour — the
same effect as the SWIR composite in {doc}`Lab A <lab-aviris-ng-urban>`.

3. In **Dataset Info**, note the bad-band list: AVIRIS flags the 1400 nm and
   1900 nm water-vapour regions. Confirm the gaps appear in the spectrum plot.

**Deliverable 1:** true-colour and SWIR composites of the same area, exported
with **Export RGB image ▸ Export visible image area**, plus two sentences on
what the SWIR composite shows that true colour does not.

---

## Part 2 — Identify minerals by hand (40 min)

1. Click across the bright altered ground and collect spectra. Set **Number of
   pixels to average** to a 3 × 3 **median** — a single AVIRIS pixel at 2200 nm
   is noisy.
2. Set the plot's x-axis range to **2000–2400 nm** so the SWIR features fill
   the frame.
3. Right-click the plot ▸ **Continuum Removal: Collected Spectra**.

Read the continuum-removed spectra against the table above:

- **Single sharp band at 2170 nm** → alunite
- **Doublet, 2160 and 2200 nm**, the 2200 nm side deeper → kaolinite
- **Single band near 2200 nm**, broader and asymmetric → muscovite/illite
- **Band at 2340 nm** → calcite
- **Band at 2120 nm** → buddingtonite

4. Confirm each: import the USGS library and run **SAM in Spectrum mode**
   ({doc}`Tutorial 7 <../07-detection>`) with your collected spectrum as the
   target. Read the ranked match table.

**Deliverable 2:** a continuum-removed plot of four identified spectra, each
labelled with its mineral and the wavelength you used to call it.

---

## Part 3 — Map the minerals (60 min)

### 3a. Spectral Angle Mapper

1. **Tools ▸ Data Analysis ▸ Spectral Angle Mapper**, target **Image Cube**.
2. Add the USGS library; tick alunite, kaolinite, muscovite and calcite.
3. **Wavelength range: 2000–2400 nm.** This is what makes SAM work here — over
   the full range the albedo and iron-oxide variation in the visible dominates
   the angle and swamps the clay signal.
4. Start at the default 5° threshold and **Run SAM**.

Display the **`SAM Angle`** image first, with a colormap and a tight stretch.
Only then look at **`SAM CLS`**. Adjust each mineral's threshold and re-run
until the classified areas match the outcrops visible in the SWIR composite.

### 3b. Spectral Feature Fitting

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

**Deliverable 3:** SAM and SFF mineral maps for the same four minerals, and a
paragraph on where they disagree and which you trust there.

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

1. Alunite and kaolinite both absorb near 2200 nm. Which of SAM and SFF
   separates them better, and why does that follow from how each works?
2. You get a high SAM score for buddingtonite in an area with no other
   alteration minerals. What would you check before reporting it?
3. Your unmixing RMSE is high across a whole playa. Give two possible causes
   and say how you would tell them apart.
4. Why does restricting the wavelength range change a SAM result at all, given
   that SAM is supposed to be insensitive to brightness?

---

## Going further

- Compare your alteration map against the published USGS Cuprite maps at
  [crustal.usgs.gov/speclab](https://crustal.usgs.gov/speclab/).
- Repeat with an **AVIRIS-NG** scene (5 nm sampling instead of 10 nm) and see
  which mineral separations improve.
- Run the same analysis on a **radiance** product and document how the results
  degrade — a useful demonstration of why atmospheric correction matters.
