# Lab D — Surface Mineralogy from Orbit with EMIT

- **Field:** Earth system science, arid-land geology, climate forcing
- **Instrument:** EMIT (Earth Surface Mineral Dust Source Investigation) on the
  International Space Station, 285 bands, 381–2493 nm at ~7.5 nm, 60 m
- **Prerequisites:** {doc}`Tutorials 1–7 <../index>`

```{admonition} You will need to download data for this lab
:class: note
You will need a [free Earthdata account](https://urs.earthdata.nasa.gov/users/new)
to search for and download EMIT granules, so create one before you start.
Everything is done through the browser; **Get the data** below has the search
links and the steps.

The figures here were captured on `EMIT_L2A_RFL_001_20230804T191650_2321613_007`,
an August 2023 granule over the ranges of southwestern Nevada. Your scene will
look different; the steps and the diagnostic wavelengths do not change.
```

---

## The question

Windblown mineral dust changes how much sunlight reaches the ground, and which
way it pushes depends on what the dust is made of. Iron oxides are dark and
absorb; clays are paler and scatter more of the light back. What that adds up
to across the whole atmosphere is genuinely unsettled. EMIT was flown to help
pin it down: JPL describes the mission's maps as intended to improve forecasts
of ["the role of mineral dust in the radiative forcing (warming or cooling) of
the atmosphere"](https://science.jpl.nasa.gov/projects/EMIT/). Which of the two
it comes out as, globally, is still an open question.

Part of why it is unsettled is that nobody had measured what the world's dust
source regions are actually made of. That is the measurement you are doing here,
on one scene: separate iron oxides from clays and carbonates, and say what that
implies for the dust this region emits.

| Mineral group | Diagnostic feature | Optical behavior |
|---|---|---|
| **Hematite** | broad ~860 nm; steep red slope | Strongly absorbing |
| **Goethite** | broad ~920 nm; steep red slope | Absorbing |
| **Kaolinite** | doublet 2160 + 2200 nm | Scattering |
| **Illite / muscovite** | 2200 nm | Scattering |
| **Calcite / dolomite** | 2340 nm | Scattering |
| **Gypsum** | 1750 nm, 2210 nm | Scattering |

Whether a mineral absorbs or scatters is straightforward; these are the
ten-odd minerals EMIT was built to look for. Turning a map of them into a
number for warming or cooling is the hard part, and not something this lab
attempts.

---

## Get the data

**Scene:** any EMIT **Level-2A (L2A) Reflectance** granule over a desert, such
as the Mojave and Sonoran deserts, the Sahara, the Arabian Peninsula, the
Taklamakan, or the Lake Eyre basin.

1. [Create a free Earthdata account](https://urs.earthdata.nasa.gov/users/new).
2. Search for **`EMITL2ARFL`** in
   [Earthdata Search](https://search.earthdata.nasa.gov/), or browse the
   [EMIT L2A collection page](https://www.earthdata.nasa.gov/data/catalog/lpcloud-emitl2arfl-001).
3. Download the **`EMIT_L2A_RFL_*.nc`** file. Each is around 1.8 GB. Pick one
   scene deliberately, checking the browse image for low cloud and low
   vegetation.

Each granule ships three NetCDF files: `RFL` (reflectance, the one you want),
`RFLUNCERT` (per-band uncertainty) and `MASK` (cloud and quality flags).

If you want something to check your own work against later, EMIT also publishes
a [Level-2B mineral product](https://www.earthdata.nasa.gov/data/catalog/lpcloud-emitl2bmin-001)
for the same scenes, at the same 60 m, which reports an identified mineral and a
band depth per pixel for ten minerals including hematite, goethite, kaolinite
and calcite. Part 4 has you compute band depths by hand; downloading the L2B
granule for your scene lets you put the two side by side in WISER.

```{admonition} Confirm your NetCDF reading first
:class: note
A cropped, real EMIT L2A granule ships with the WISER source at
`src/test_utils/test_datasets/EMIT_L2A_RFL_001_20241006T165148_2428011_003_crop.nc`
(Imperial Valley, California, October 2024). It is cut to 32 × 32 pixels and 3
bands, so it will not carry this lab's science, but opening it takes seconds
and confirms your NetCDF path works before you spend an hour on a 1.8 GB
download.
```

**Reference spectra:** the
[USGS Spectral Library Version 7](https://dx.doi.org/10.5066/F7RR1WDJ). EMIT's
sampling is close to AVIRIS-NG's, so the AVIRIS-convolved version is a
reasonable match; the detection tools resample references onto the target grid
regardless.

---

## Part 1 — Open the granule

1. **File ▸ Open...** → the `EMIT_L2A_RFL_*.nc` file.
2. A NetCDF file holds several variables, so WISER asks which to open. Choose
   **`reflectance`**. The others carry geolocation and per-band metadata.
3. Turn on all four panes, build a true-color composite from about
   660 / 550 / 480 nm, and apply a 2.5% linear stretch.

:::{figure} ../../_static/tutorials/lab_emit_truecolour.png
:width: 100%
:align: center
:alt: EMIT true color over desert ranges, showing brown and tan terrain with some reddening where iron oxides are exposed
:::

True color tells you where the mountains and the fans are, and almost nothing
about what they are made of. Iron oxide reddens some surfaces, which is what
Part 3 sends you back here to sample, but red alone does not tell hematite from
goethite, and the clays, carbonates and sulfates in the table above are all
beige.

```{admonition} EMIT L2A is not map-projected
:class: note
Standard EMIT L2A granules are **spatially raw** — delivered in the
instrument's acquisition geometry, not on a map grid. Geolocation arrives in a
separate array, not as a simple geotransform. That is fine for the spectroscopy
here, but do not treat pixel positions as map coordinates, and orthorectify
before overlaying anything geographic. WISER's
{doc}`Georeferencer <../../user-content/spatial-tools>` handles the alignment
if you need it.
```

4. Build a short-wave infrared (SWIR) composite. Click **Band chooser**, select
   **RGB**, and set **Red Band** to **Band 244: 2200 nm**, **Green Band** to
   **Band 239: 2160 nm** and **Blue Band** to **Band 263: 2340 nm**. Click
   **OK**, then **Stretch builder ▸ Linear Stretch ▸ 2.5% linear**.

:::{figure} ../../_static/tutorials/lab_emit_swir.png
:width: 100%
:align: center
:alt: The same scene in a SWIR composite, where alluvial fans and playa margins separate into distinct colors
:::

The same ground, in three bands chosen for what absorbs there. Clays push the
red channel, kaolinite the green, carbonates the blue, and the fans and playa
margins separate into units true color could not distinguish.

5. Apply a decorrelation stretch to the same three bands: reopen **Stretch
   builder** and select **Decorrelation Stretch** in the **Stretch** section.

:::{figure} ../../_static/tutorials/lab_emit_decorr.png
:width: 100%
:align: center
:alt: The SWIR composite after a decorrelation stretch, with the same units in saturated, strongly separated colors
:::

The decorrelation stretch removes the correlation between the three channels
and exaggerates what is left. It makes boundaries obvious, which is what it is
for. Do not read mineralogy from its colors: the transform is derived from this
scene's own statistics, so the same mineral in another granule can come out a
different color. Use it to decide where to look, then go to the spectra.

6. Check the wavelength coverage in **Dataset Info**: 285 bands over
   381–2493 nm, with the 1400 nm and 1900 nm water-vapor regions flagged.

```{admonition} The flagged regions are gaps, not noise
:class: note
EMIT's own good-wavelength mask flags roughly **1327–1432 nm** and
**1774–1960 nm**, where atmospheric water vapor leaves no usable surface
signal. Two consequences for this lab. Continuum removal across a gap
interpolates over nothing, so keep your windows on one side of it. And gypsum's
1750 nm feature sits right at the edge: its upper shoulder falls inside the
second gap, so the band-depth recipe used elsewhere in this lab cannot be built
for it. Identify gypsum from its 2210 nm feature and the shape of the spectrum
instead.
```

**Deliverable 1:** true-color and SWIR-composite views of your scene, and the
granule's acquisition date, location and solar geometry from its metadata.

---

## Part 2 — Mask what you cannot use

Mineral mapping only works on exposed soil and rock. Remove everything else, or
you will map vegetation as clay.

1. Compute NDVI with band math ({doc}`Tutorial 4 <../04-band-math-ndvi>`).
   EMIT's sampling lets you use the textbook wavelengths:

   ```text
   (nir - red) / (nir + red)
   ```

   Open **Tools ▸ Band math...**, type the expression, press **Enter**, then
   scroll right in **Variable bindings:** to **Variable Assignments**. Bind
   `nir` to **Band 64: 860 nm** and `red` to **Band 38: 660 nm**, the nearest
   EMIT bands to the textbook wavelengths. Give it a **Result name (optional):**
   of `ndvi` and click **OK**.

2. Build a bare-ground mask:

   ```text
   ndvi < 0.15
   ```

3. Check the granule's **`MASK`** file for cloud and cirrus flags, and exclude
   those areas too.
4. Exclude standing water. A brightness test on the SWIR does it:

   ```text
   r1650 > 0.12
   ```

```{admonition} Water will pass for iron oxide if you let it
:class: note
Ranking this scene for iron oxide without a water mask returns brine pools
first. Their reflectance peaks near 570 nm and collapses to about 0.005 in the
SWIR, which reads as a strong red slope and a deep absorption to any index that
only looks at band ratios. They are the brightest thing in the scene by those
measures and they are not mineral. Mask on SWIR brightness before you rank
anything.
```

**Deliverable 2:** your bare-ground mask, with the NDVI threshold justified
from the NDVI histogram rather than assumed, and standing water excluded.

---

## Part 3 — Iron oxides in the visible/NIR

Hematite and goethite have broad crystal-field absorptions in the visible/NIR
and a steep rise across the red, which is why iron-rich soils look red.

1. Collect spectra from several reddish and several pale areas.

:::{figure} ../../_static/tutorials/lab_emit_spectra_plot.png
:width: 100%
:align: center
:alt: Four EMIT spectra over the full 380 to 2490 nm range, showing the iron oxide red slope and the flagged water-vapor gaps
:::

Four single pixels across the full range. The iron-oxide spectrum climbs
steeply through the red and flattens; the three others are brighter in the SWIR
and carry the features Part 4 uses. The two vertical breaks are the flagged
water-vapor regions, and they are why the x-axis windows below stay on one side
of them.

2. Set the plot x-axis to **400–1300 nm** and continuum-remove.

   Hematite's band centers near **860 nm**, goethite's near **920 nm**. The
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
mask, plus continuum-removed spectra showing the band-center difference you are
relying on.

---

## Part 4 — Clays, carbonates and sulfates in the SWIR

1. Set the plot x-axis to **2000–2400 nm**, collect spectra across the scene,
   and continuum-remove.

:::{figure} ../../_static/tutorials/lab_emit_swir_spectra.png
:width: 100%
:align: center
:alt: The same four spectra between 2000 and 2450 nm, where kaolinite shows a doublet, muscovite a single 2200 nm band, calcite a 2340 nm band, and iron oxide nothing
:::

Kaolinite falls to its minimum at 2200 nm with a distinct shoulder at 2160,
the doublet that identifies it. Muscovite/illite reaches the same 2200 nm
minimum with no shoulder, which is how you tell the two apart here. Calcite
ignores 2200 and drops at 2340. Iron oxide is featureless here: whatever is
reddening the visible is not a clay.

2. Identify features against the table at the top of this lab.
3. Run **Tools ▸ Data Analysis ▸ Spectral Feature Fitting**
   ({doc}`Tutorial 7 <../07-detection>`). Set **Select Target Type:** to **Image
   Cube**, tick the USGS library under **Reference Library Selection**, and run
   it once per mineral, setting **Min Wavelength (nm):** and **Max Wavelength
   (nm):** to the window below each time:

   | Mineral | Window |
   |---|---|
   | Kaolinite | 2120–2250 nm |
   | Illite / muscovite | 2150–2250 nm |
   | Calcite | 2280–2400 nm |
   | Gypsum | 1700–1800 nm |

4. Map the clays directly with a band depth, which needs no library at all:

   ```text
   1 - c2200 / ((1 - f) * s2130 + f * s2280)
   ```

   with `f = (2200 - 2130) / (2280 - 2130) = 0.467`.

:::{figure} ../../_static/tutorials/lab_emit_bandmath.png
:width: 100%
:align: center
:alt: The band math dialog holding the 2200 nm band depth expression with its three variables bound
:::

:::{figure} ../../_static/tutorials/lab_emit_clay.png
:width: 100%
:align: center
:alt: The 2200 nm band depth as a map, bright along alluvial fans and playa margins and dark over bare rock
:::

The result is a map of clay absorption strength that owes nothing to a spectral
library. Bright is deep absorption. The fans radiating from the ranges light
up, and so do the playa margins, which is where windblown material is generated
and where it settles.

5. Where you have good endmembers, run **Linear Unmixing** for fractional
   abundances, and read the RMSE band first.

**Deliverable 4:** a mineral map of your scene, and the fraction of unmasked
area assigned to each group.

---

## Part 5 — Say what it means

1. Estimate the areal fraction of iron-oxide-rich versus clay/carbonate-rich
   bare ground.
2. Locate the likely emitting surfaces — dry lake beds, alluvial fans, dune
   fields, disturbed agricultural soil.
3. Say which way you would expect dust from **this** region to lean, absorbing
   or scattering, and how much you would trust that.

**Deliverable 5:** a paragraph answering (3), naming what could undermine it:
mixed pixels, the masking you applied, grain size affecting band depth, and
leftover atmospheric correction error. You are stopping at absorbing versus
scattering; say why getting to warming versus cooling would take more than one
scene.

---

## Questions to answer

1. Your band-depth map is brightest where a mineral's absorption is deepest.
   But a coarse-grained patch and an abundant patch can both look bright. What
   does that do to a sentence like "this area is 40% hematite"?
2. The lab has you mask vegetation in Part 2, before mapping minerals in Parts 3
   and 4. What would have gone wrong if you had done it the other way round?
3. EMIT samples about every 7.5 nm. A laboratory spectrometer samples about
   every 1 nm. Look at the kaolinite doublet near 2160 and 2200 nm in your
   spectra. Is it still two features, and would you have known to call it a
   doublet from EMIT alone?
4. You find what looks like kaolinite over an irrigated field. Give one
   explanation where the mineral is really there and one where something else
   produced that shape, and say what you would look at to tell them apart.
5. If you downloaded the L2B mineral product for your scene, open it next to
   your own band-depth maps. Where do you and the mission disagree, and does
   that make you doubt your bands or their thresholds?

---

## Going further

- **EMIT L2B** delivers the mission's own mineral maps. Download it for your
  scene and compare against what you produced.
- **EMIT L2B CH4/CO2** carries methane and carbon-dioxide plume detections, a
  completely different use of the same instrument.
- Compare an EMIT scene against the **AVIRIS-NG** scene in
  {doc}`Lab A <lab-aviris-ng-urban>`: 60 m from orbit versus 5 m from an
  aircraft, and what each resolves.
- Run a 2170 nm band depth over the granule used for these figures and find its
  strongest pixels. They land on Cuprite, the site {doc}`Lab B
  <lab-cuprite-minerals>` maps from the air at 15 m: about 0.32 against a scene
  median near 0.03. Doing both labs gives you the same alteration system at two
  resolutions from two platforms, which is a direct test of what 60 m pixels
  cost you.
