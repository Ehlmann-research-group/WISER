# Lab E — Phytoplankton and Coastal Water with PACE

- **Field:** biological and optical oceanography, water quality
- **Instrument:** OCI (Ocean Color Instrument) on PACE (Plankton, Aerosol,
  Cloud, ocean Ecosystem), hyperspectral 340–895 nm at ~5 nm plus short-wave
  infrared bands, 1 km
- **Prerequisites:** {doc}`Tutorials 1–5 <../index>`
- **Time:** 2 hours

```{admonition} You will need to download data for this lab
:class: note
You will need to download some PACE data to do this lab. You will also need to
[create a free Earthdata account](https://urs.earthdata.nasa.gov/users/new) in order to access and
download it, so do that before you start. Everything is done through the
browser; **Get the data** below has the search links and the steps.

No screenshots are shipped for this lab, because the figure harness cannot
authenticate to Earthdata, so capture your own as you work.
{doc}`Lab A <lab-aviris-ng-urban>` shows the same dialogs on airborne data.
```

---

## The question

Ocean colour satellites have counted chlorophyll for decades, but a handful of
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

Three things drive colour in coastal water and must be told apart:

- **Phytoplankton** — pigment absorptions, and a fluorescence peak near 685 nm
- **Coloured dissolved organic matter (CDOM)** (coloured dissolved organic matter) — smooth exponential rise
  towards the blue, no features
- **Suspended sediment** — high, broadly flat reflectance rising to the red

---

## Get the data

**Product:** PACE OCI **Level-2 Regional Surface Reflectance** (`SFREFL`) — 122
wavelengths from 346 to 895 nm plus 5 SWIR bands. Level-2 **AOP** (giving
remote-sensing reflectance $R_{rs}$) is the more rigorous choice for open-ocean
work.

1. [Create a free Earthdata account](https://urs.earthdata.nasa.gov/users/new).
2. Search the Ocean Biology Distributed Active Archive Centre
   ([OB.DAAC](https://oceancolor.gsfc.nasa.gov/)) or
   [Earthdata Search](https://search.earthdata.nasa.gov/) for **PACE OCI L2
   SFREFL** or **L2 AOP**.
3. Pick a scene over a coastal region with contrast: Chesapeake Bay, the Baltic
   (reliable summer cyanobacteria blooms), Lake Erie (late-summer
   *Microcystis*), the Gulf of Mexico, or the California Current.
4. Choose a **cloud-free** granule. Clouds dominate ocean colour scenes, and a
   scene that looks 40% clear in the browse image will be worse in practice.

Data run from March 2024 to the present.

```{admonition} Level 2, not Level 1
:class: warning
The water-leaving signal is a few percent of what the satellite measures; the
rest is atmosphere. Ocean colour is the application where atmospheric
correction matters most. Use an **L2** product, where it has been done for you.
An L1 radiance scene will show you the atmosphere, not the ocean.
```

---

## Part 1 — Open and orient (20 min)

1. **File ▸ Open...** → the PACE `.nc` granule; pick the reflectance
   sub-dataset when WISER asks.
2. Build a true-colour composite (about 665 / 555 / 490 nm) and apply a **2.5%
   linear** stretch.
3. Water is dark. Open the contrast stretch and set the **Maximum** limit to
   exclude clouds and land, so the stretch is computed on the water alone —
   otherwise a few bright cloud pixels flatten the whole ocean to black. See
   {doc}`Display and Contrast Stretch <../../user-content/display-and-stretch>`.

**Deliverable 1:** a stretched true-colour image in which water structure —
fronts, plumes, blooms — is visible, plus a note on the stretch limits used.

---

## Part 2 — Three water types, three spectra (30 min)

Draw ROIs ({doc}`Tutorial 3 <../03-regions-of-interest>`) over:

1. **Clear offshore water** — dark, blue
2. **A bloom** — green, or turquoise if it is a coccolithophore
3. **A sediment plume** — brown, usually near a river mouth

Collect all three mean spectra.

**Read them:**

- **Clear water** — highest in the blue, falling steeply through green and red.
  Almost all of the signal is molecular scattering.
- **Bloom** — a peak in the green near 550 nm, a trough near 443 nm
  (chlorophyll-a), a second trough near 675 nm, and often a small bump at
  **685 nm**: chlorophyll fluorescence, light re-emitted by the cells.
- **Sediment** — high everywhere, rising towards the red, with pigment features
  weak or absent.

**Deliverable 2:** the three mean spectra on one labelled plot, each diagnostic
feature annotated.

---

## Part 3 — Chlorophyll and fluorescence (30 min)

**Blue-green ratio** — the classical chlorophyll algorithm, in band math:

```text
b443 / b555
```

High ratio → clear water; low → more chlorophyll. It is a proxy, not a
concentration, and it fails in coastal water where CDOM also absorbs blue —
which is why the next index exists.

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

## Part 4 — Unmix the water (30 min)

1. Use your three ROI mean spectra from Part 2 as endmembers.
2. Run **Linear Unmixing** ({doc}`Tutorial 7 <../07-detection>`) with **Sum to
   Unity** enabled — you chose endmembers meant to span the scene's water.
3. Read the **RMSE** band first. High residual marks water your three
   endmembers do not describe: a fourth optical type, cloud shadow, or glint.

**Deliverable 4:** abundance maps for the three components, the RMSE map, and a
short account of where the model breaks down.

---

## Part 5 — Cluster the optical types (20 min)

1. Run **K-means** with K = 5 or 6 on the water pixels
   ({doc}`Tutorial 5 <../05-classification>`), with a fixed random seed.
2. Click **View Centroids** and identify each cluster from its spectrum.
3. Compare the clusters against your index maps.

Optical water-type classification is used operationally to decide **which
algorithm to apply where** — a chlorophyll retrieval tuned for open ocean gives
nonsense in a river plume.

**Deliverable 5:** a labelled optical water-type map with each class named from
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
