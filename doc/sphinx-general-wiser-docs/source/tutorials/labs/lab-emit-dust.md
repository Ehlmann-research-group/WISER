# Lab D — Surface Mineralogy from Orbit with EMIT

- **Field:** Earth system science, arid-land geology, climate forcing
- **Instrument:** EMIT (Earth Surface Mineral Dust Source Investigation) on the
  International Space Station, 285 bands, 381–2493 nm at ~7.5 nm, 60 m
- **Prerequisites:** {doc}`Tutorials 1–7 <../index>`
- **Time:** 2 hours

```{admonition} You will need to download data for this lab
:class: note
You will need to download some EMIT data to do this lab. You will also need to
[create a free Earthdata account](https://urs.earthdata.nasa.gov/users/new) in order to access and
download it, so do that before you start. Everything is done through the
browser; **Get the data** below has the search links and the steps.

No screenshots are shipped for this lab, because the figure harness cannot
authenticate to Earthdata, so capture your own as you work.
{doc}`Lab A <lab-aviris-ng-urban>` shows the same workflow on airborne data,
including every dialog you will use here.
```

---

## The question

Windblown mineral dust warms or cools the atmosphere depending on what it is
made of. Iron-oxide-rich dust absorbs sunlight and warms; clay-rich dust
scatters and cools. Before EMIT, the mineral composition of the world's dust
source regions was largely guessed at, and climate models carried that guess as
a major uncertainty.

EMIT was flown to measure it. In this lab you do the core EMIT measurement on
one scene: separate iron oxides from clays and carbonates, and say what that
means for the dust that region emits.

| Mineral group | Diagnostic feature | Radiative effect of the dust |
|---|---|---|
| **Hematite** | broad ~860 nm; steep red slope | Strongly absorbing — warming |
| **Goethite** | broad ~920 nm; steep red slope | Absorbing — warming |
| **Kaolinite** | doublet 2160 + 2200 nm | Scattering — cooling |
| **Illite / muscovite** | 2200 nm | Scattering — cooling |
| **Calcite / dolomite** | 2340 nm | Scattering |
| **Gypsum** | 1750 nm, 2210 nm | Scattering |

---

## Get the data

**Scene:** any EMIT **Level-2A (L2A) Reflectance** granule over a desert — the Mojave and
Sonoran deserts, the Sahara, the Arabian Peninsula, the Taklamakan, or the Lake
Eyre basin.

1. [Create a free Earthdata account](https://urs.earthdata.nasa.gov/users/new).
2. Search for **`EMITL2ARFL`** in
   [Earthdata Search](https://search.earthdata.nasa.gov/), or browse the
   [EMIT L2A collection page](https://www.earthdata.nasa.gov/data/catalog/lpcloud-emitl2arfl-001).
3. Download the **`EMIT_L2A_RFL_*.nc`** file. Each is around 1.8 GB — pick one
   scene deliberately, checking the browse image for low cloud and low
   vegetation.

Each granule ships three NetCDF files: `RFL` (reflectance — the one you want),
`RFLUNCERT` (per-band uncertainty) and `MASK` (cloud and quality flags).

```{admonition} Confirm your NetCDF reading first
:class: note
A cropped, real EMIT L2A granule ships with the WISER source at
`src/test_utils/test_datasets/EMIT_L2A_RFL_001_20241006T165148_2428011_003_crop.nc`
(Imperial Valley, California, October 2024). It is cut to 32 × 32 pixels and 3
bands, so it will not carry this lab's science — but opening it takes seconds
and confirms your NetCDF path works before you spend an hour on a 1.8 GB
download.
```

**Reference spectra:** the
[USGS Spectral Library Version 7](https://dx.doi.org/10.5066/F7RR1WDJ). EMIT's
sampling is close to AVIRIS-NG's, so the AVIRIS-convolved version is a
reasonable match; the detection tools resample references onto the target grid
regardless.

---

## Part 1 — Open the granule (20 min)

1. **File ▸ Open...** → the `EMIT_L2A_RFL_*.nc` file.
2. A NetCDF file may hold several variables. When WISER asks, choose the
   **reflectance** sub-dataset.
3. Turn on all four panes and apply a 2.5% linear stretch.

```{admonition} EMIT L2A is not map-projected
:class: warning
Standard EMIT L2A granules are **spatially raw** — delivered in the
instrument's acquisition geometry, not on a map grid. Geolocation arrives in a
separate array, not as a simple geotransform. That is fine for the spectroscopy
here, but do not treat pixel positions as map coordinates, and orthorectify
before overlaying anything geographic. WISER's
{doc}`Georeferencer <../../user-content/spatial-tools>` handles the alignment
if you need it.
```

4. Check the wavelength coverage in **Dataset Info**: 285 bands over
   381–2493 nm, with the 1400 nm and 1900 nm water-vapour regions flagged.

**Deliverable 1:** true-colour and SWIR-composite views of your scene, and the
granule's acquisition date, location and solar geometry from its metadata.

---

## Part 2 — Mask what you cannot use (20 min)

Mineral mapping only works on exposed soil and rock. Remove everything else, or
you will map vegetation as clay.

1. Compute NDVI with band math ({doc}`Tutorial 4 <../04-band-math-ndvi>`).
   EMIT's sampling lets you use the textbook wavelengths:

   ```text
   (nir - red) / (nir + red)
   ```

   Bind `nir` to the band nearest **860 nm** and `red` to the band nearest
   **670 nm**.

2. Build a bare-ground mask:

   ```text
   ndvi < 0.15
   ```

3. Check the granule's **`MASK`** file for cloud and cirrus flags, and exclude
   those areas too.

**Deliverable 2:** your bare-ground mask, with the NDVI threshold justified
from the NDVI histogram rather than assumed.

---

## Part 3 — Iron oxides in the visible/NIR (30 min)

Hematite and goethite have broad crystal-field absorptions in the visible/NIR
and a steep rise across the red — why iron-rich soils look red.

1. Collect spectra from several reddish and several pale areas.
2. Set the plot x-axis to **400–1300 nm** and continuum-remove.

   Hematite's band centres near **860 nm**, goethite's near **920 nm**. The
   difference is small and the bands are broad, so use ROI mean spectra, not
   single pixels.

3. Map them with band depth:

   ```text
   1 - c860 / (0.5 * s700 + 0.5 * s1300)
   ```

   and the same with `c920` for goethite.

4. Cross-check with a redness ratio:

   ```text
   r700 / r500
   ```

**Deliverable 3:** hematite and goethite band-depth maps over the bare-ground
mask, plus continuum-removed spectra showing the band-centre difference you are
relying on.

---

## Part 4 — Clays, carbonates and sulphates in the SWIR (40 min)

1. Set the plot x-axis to **2000–2400 nm**, collect spectra across the scene,
   and continuum-remove.
2. Identify features against the table at the top of this lab.
3. Run **SFF** ({doc}`Tutorial 7 <../07-detection>`) with the USGS library, one
   narrow window per mineral:

   | Mineral | Window |
   |---|---|
   | Kaolinite | 2120–2250 nm |
   | Illite / muscovite | 2150–2250 nm |
   | Calcite | 2280–2400 nm |
   | Gypsum | 1700–1800 nm |

4. Where you have good endmembers, run **Linear Unmixing** for fractional
   abundances, and read the RMSE band first.

**Deliverable 4:** a mineral map of your scene, and the fraction of unmasked
area assigned to each group.

---

## Part 5 — Say what it means (20 min)

1. Estimate the areal fraction of iron-oxide-rich versus clay/carbonate-rich
   bare ground.
2. Locate the likely emitting surfaces — dry lake beds, alluvial fans, dune
   fields, disturbed agricultural soil.
3. State whether dust emitted from **this** region would tend to warm or cool,
   and how confident you are.

**Deliverable 5:** a paragraph answering (3), with your uncertainties named
explicitly: mixed pixels, the masking you applied, grain-size effects on band
depth, and residual atmospheric correction error.

---

## Questions to answer

1. Band depth responds to grain size as well as abundance. What does that do to
   a claim that "this area is 40% hematite"?
2. Why does this lab mask vegetation before mapping minerals rather than after?
3. EMIT samples at ~7.5 nm; a laboratory spectrometer samples at ~1 nm. Which
   features here survive that difference and which are at risk?
4. You detect kaolinite over an irrigated field. Give a mineralogical
   explanation and an artefact explanation, and say how you would distinguish
   them.

---

## Going further

- **EMIT L2B** delivers the mission's own mineral maps. Download it for your
  scene and compare against what you produced.
- **EMIT L2B CH4/CO2** carries methane and carbon-dioxide plume detections — a
  completely different use of the same instrument.
- Compare an EMIT scene against the **AVIRIS-NG** scene in
  {doc}`Lab A <lab-aviris-ng-urban>`: 60 m from orbit versus 5 m from an
  aircraft, and what each resolves.
