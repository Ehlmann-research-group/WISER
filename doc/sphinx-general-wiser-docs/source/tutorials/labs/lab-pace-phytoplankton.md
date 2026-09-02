# Lab E — Phytoplankton and Coastal Water with PACE

- **Field:** biological and optical oceanography, water quality
- **Instrument:** OCI (Ocean Color Instrument) on PACE (Plankton, Aerosol,
  Cloud, ocean Ecosystem), hyperspectral 340–895 nm at ~5 nm plus short-wave
  infrared bands, 1 km
- **Prerequisites:** {doc}`Tutorials 1–5 <../index>`

```{admonition} You will need to download data for this lab
:class: note
You will need to download some PACE data to do this lab. You will also need to
[create a free Earthdata account](https://urs.earthdata.nasa.gov/users/new) in order to access and
download it, so do that before you start. Everything is done through the
browser; **Get the data** below has the search links and the steps.

This lab also needs a WISER newer than 3.0b0. PACE stores reflectance as packed
integers with a scale factor, and 3.0b0 reads the stored integers, so every
spectrum comes back in the tens of thousands instead of in reflectance units.
Build from source, or use a later release.
```

---

## The question

Ocean color satellites have counted chlorophyll for decades, but a handful of
broad bands can only tell you *how much* phytoplankton there is. Different
groups — diatoms, cyanobacteria, coccolithophores, dinoflagellates — carry
different accessory pigments with narrow, distinctive absorptions. Resolving
them needs a spectrometer.

PACE flies one. Its Ocean Color Instrument samples continuously from the
ultraviolet into the near-infrared, making it possible to ask *which*
phytoplankton, and to separate their signal from the sediment and dissolved
organic matter that confound coastal water.

| Pigment | Absorption | Found in |
|---|---|---|
| **Chlorophyll-a** | 443 nm and 675 nm | All phytoplankton |
| **Chlorophyll-b** | 470 nm | Green algae, prochlorophytes |
| **Chlorophyll-c** | 460, 630 nm | Diatoms, dinoflagellates |
| **Phycoerythrin** | ~565 nm | Cryptophytes, some cyanobacteria |
| **Phycocyanin** | ~620 nm | **Cyanobacteria — harmful blooms** |
| **Carotenoids** | 490–530 nm | Most groups; photoprotective |

Three things drive color in coastal water and must be told apart:

- **Phytoplankton** — pigment absorptions, and a fluorescence peak near 685 nm
- **Colored dissolved organic matter (CDOM)** — smooth exponential rise
  towards the blue, no features
- **Suspended sediment** — high, broadly flat reflectance rising to the red

---

## Get the data

**Product:** PACE OCI **Level-2 Regional Surface Reflectance** (`SFREFL`) — 122
wavelengths from 346 to 895 nm plus 5 SWIR bands. Level-2 **AOP** (giving
remote-sensing reflectance $R_{rs}$) is the more rigorous choice for open-ocean
work.

1. [Create a free Earthdata account](https://urs.earthdata.nasa.gov/users/new).
2. Search the Ocean Biology Distributed Active Archive Center
   ([OB.DAAC](https://oceancolor.gsfc.nasa.gov/)) or
   [Earthdata Search](https://search.earthdata.nasa.gov/) for **PACE OCI L2
   SFREFL** or **L2 AOP**.
3. Pick a scene over a coastal region with contrast: Chesapeake Bay, the Baltic
   (reliable summer cyanobacteria blooms), Lake Erie (late-summer
   *Microcystis*), the Gulf of Mexico, or the California Current.
4. Choose a **cloud-free** granule. Clouds dominate ocean color scenes, and a
   scene that looks 40% clear in the browse image will be worse in practice.

Data run from March 2024 to the present.

```{admonition} Level 2, not Level 1
:class: note
The water-leaving signal is a few percent of what the satellite measures; the
rest is atmosphere. Ocean color is the application where atmospheric
correction matters most. Use an **L2** product, where it has been done for you.
An L1 radiance scene will show you the atmosphere, not the ocean.
```

---

## Part 1 — Open and orient

1. **File ▸ Open...** → the PACE `.nc` granule. One netCDF file holds many
   variables, so WISER asks which to open. Choose `geophysical_data/Rrs`. The
   rest are geolocation, quality flags and per-band metadata, and none of them
   is the reflectance cube.
2. Build a true-color composite (about 660 / 555 / 443 nm) and apply a **2.5%
   linear** stretch.
3. Water is dark. Open the contrast stretch and set the **Maximum** limit to
   exclude clouds and land, so the stretch is computed on the water alone —
   otherwise a few bright cloud pixels flatten the whole ocean to black. See
   {doc}`Display and Contrast Stretch <../../user-content/display-and-stretch>`.

:::{figure} ../../_static/tutorials/lab_pace_truecolour.png
:width: 100%
:align: center
:alt: PACE true color over coastal water, showing blue open ocean, a turbid coastal band with visible eddies, and black gaps where cloud prevented a retrieval
:::

Most of the frame is black, and that is the ordinary condition for ocean color
at Level 2. About one pixel in six of this granule carries a retrieval; the
rest was cloud, sun glint or otherwise rejected, and the processing left it
empty. Expect to work in the clear part of a scene rather than across all of
it. What survives here is worth the hunt: deep blue open water on the left, a
turbid coastal band threaded with eddies and filaments, and a plume at the
bottom.

**Deliverable 1:** a stretched true-color image in which water structure —
fronts, plumes, blooms — is visible, plus a note on the stretch limits used.

---

## Part 2 — Three water types, three spectra

Draw ROIs ({doc}`Tutorial 3 <../03-regions-of-interest>`) over:

1. **Clear offshore water** — dark, blue
2. **A bloom** — green, or turquoise if it is a coccolithophore
3. **A sediment plume** — brown, usually near a river mouth
4. **Transitional water** — between the first two, which is what most of a
   coastal scene actually is

Collect all four mean spectra.

**Read them:**

- **Clear water** — highest in the blue, falling steeply through green and red.
  Almost all of the signal is molecular scattering.
- **Bloom** — a peak in the green near 550 nm, a trough near 443 nm
  (chlorophyll-a), a second trough near 675 nm, and often a small bump at
  **685 nm**: chlorophyll fluorescence, light re-emitted by the cells.
- **Sediment** — high everywhere, rising towards the red, with pigment features
  weak or absent.

:::{figure} ../../_static/tutorials/lab_pace_spectra_plot.png
:width: 100%
:align: center
:alt: Four remote-sensing reflectance spectra: clear ocean peaking below 440 nm, a transitional type, high chlorophyll peaking near 560 nm, and a sediment plume bright across the whole visible range
:::

Four single pixels from the granule above. Clear ocean is highest below 440 nm
and has fallen to nothing by 550. High chlorophyll is close to its inverse:
suppressed in the blue where chlorophyll-a absorbs, peaking near 560 nm, then
dropping steeply past 580. Sediment is several times brighter than either and
still climbing at 600 nm. The transitional pixel sits between clear water and
chlorophyll, which is where most coastal pixels land.

Note the vertical scale. Remote-sensing reflectance runs to about 0.03 sr⁻¹
here, a few percent of what the instrument measured before atmospheric
correction. If your numbers are in the thousands, you are reading stored
integers rather than reflectance; see the note at the top of this lab.

**Deliverable 2:** the four mean spectra on one labeled plot, each diagnostic
feature annotated.

---

## Part 3 — Chlorophyll and fluorescence

**Blue-green ratio** — the classical chlorophyll algorithm, in band math:

```text
b443 / b555
```

High ratio → clear water; low → more chlorophyll. It is a proxy, not a
concentration, and it fails in coastal water where CDOM also absorbs blue,
which is why the next index exists.

:::{figure} ../../_static/tutorials/lab_pace_bandmath.png
:width: 100%
:align: center
:alt: The band math dialog with the expression blue divided by green, its two variables bound to the 443 and 555 nm bands
:::

Bind `blue` to the 443 nm band and `green` to 555 nm, name the result, and run
it.

:::{figure} ../../_static/tutorials/lab_pace_ratio.png
:width: 100%
:align: center
:alt: The blue-green ratio map, bright over clear offshore water and dark through the coastal plume and the cloud gaps alike
:::

Bright values are clear, blue-dominated water; dark values are where
chlorophyll and sediment have taken the blue out. Read the dark areas
carefully. A pixel with no retrieval carries no value and renders at the same
end of the color scale as a genuinely low ratio, so cloud gaps and productive
water look alike here. Compare against the true-color image before calling any
of it a bloom.

**Fluorescence line height (FLH)** — the 685 nm bump above a baseline between
its shoulders:

```text
b685 - (0.5 * b665 + 0.5 * b710)
```

FLH is far more robust in turbid coastal water, because sediment and CDOM
affect the three bands almost equally and drop out of the difference.

**Cyanobacteria index** — phycocyanin absorbs near 620 nm, and only
cyanobacteria have it:

```text
1 - b620 / (0.5 * b600 + 0.5 * b650)
```

A positive result over a bloom is evidence of a cyanobacterial rather than a
diatom bloom, which matters because cyanobacterial blooms can be toxic.

Display each index with a sequential colormap and a tight stretch.

**Deliverable 3:** the three index maps, and an explanation of where the
blue-green ratio and the fluorescence line height disagree.

---

## Part 4 — Unmix the water

1. Use your three ROI mean spectra from Part 2 as endmembers.
2. Run **Linear Unmixing** ({doc}`Tutorial 7 <../07-detection>`) with **Sum to
   Unity** enabled — you chose endmembers meant to span the scene's water.
3. Read the **RMSE** band first. High residual marks water your three
   endmembers do not describe: a fourth optical type, cloud shadow, or glint.

**Deliverable 4:** abundance maps for the three components, the RMSE map, and a
short account of where the model breaks down.

---

## Part 5 — Cluster the optical types

1. Run **K-means** with K = 5 or 6 on the water pixels
   ({doc}`Tutorial 5 <../05-classification>`), with a fixed random seed.
2. Click **View Centroids** and identify each cluster from its spectrum.
3. Compare the clusters against your index maps.

Optical water-type classification is used operationally to decide **which
algorithm to apply where** — a chlorophyll retrieval tuned for open ocean gives
nonsense in a river plume.

**Deliverable 5:** a labeled optical water-type map with each class named from
its centroid spectrum.

---

## Questions to answer

1. Why does the blue-green ratio overestimate chlorophyll in CDOM-rich water,
   and why is fluorescence line height less affected?
2. The water-leaving signal is a small fraction of what the sensor sees. What
   does that imply about how much a 1% atmospheric-correction error matters
   here compared with a land scene?
3. PACE pixels are about 1 km. What does that do to your ability to map a river
   plume, and what would you need instead?
4. You find high FLH but no phycocyanin absorption. What kind of bloom is it,
   and does it warrant a health advisory?

---

## Going further

- Compare against the **PACE L2 IOP** product — its phytoplankton absorption
  coefficient is the mission's own version of what you estimated.
- Build a time series across a bloom's growth and collapse from consecutive
  granules.
- Compare a PACE scene with an **EMIT** scene of the same coastal water
  ({doc}`Lab D <lab-emit-dust>`): 60 m resolves plume structure PACE averages
  over, at the cost of coverage and revisit.
- Cross-check against in-water measurements from
  [SeaBASS](https://seabass.gsfc.nasa.gov/) if any coincide with your granule.
