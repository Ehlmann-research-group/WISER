# Lab E — Phytoplankton and Coastal Water with PACE

- **Field:** biological and optical oceanography, water quality
- **Instrument:** OCI (Ocean Color Instrument) on PACE (Plankton, Aerosol,
  Cloud, ocean Ecosystem), hyperspectral 340–895 nm at ~5 nm plus short-wave
  infrared bands, 1 km
- **Prerequisites:** {doc}`Tutorials 1–5 <../index>`

```{admonition} You will need to download data for this lab
:class: note
This lab uses PACE data you download yourself, and you need a
[free Earthdata account](https://urs.earthdata.nasa.gov/users/new) to access it, so create one
before you start. Everything is done through the
browser; **Get the data** below has the search links and the steps.

This lab also needs a WISER newer than 3.0b0. `Rrs` is
[stored as scaled integers](https://www.earthdata.nasa.gov/data/catalog/ob-cloud-pace-oci-l2-aop-3.1),
with the real value recovered from the variable's `scale_factor` and
`add_offset` attributes. WISER 3.0b0 reads the stored integers and ignores
those attributes, so every spectrum comes back in the tens of thousands
instead of in reflectance units. Build from source, or use a later release.
```

---

## The question

Ocean color satellites have counted chlorophyll for decades, but a handful of
broad bands can only tell you *how much* phytoplankton there is. Different
groups carry different pigments, and those pigments absorb at different
wavelengths, so an instrument that samples the spectrum finely can start to say
something about *which* phytoplankton. How far that can be taken is still being
worked out. The IOCCG's review of the problem concludes that identifying
phytoplankton groups from satellite data
[remains uncertain](https://www.ioccg.org/reports/IOCCG_Report_15_2014.pdf), so
this lab stays well short of it. You will separate a few water types that look
different from each other, not name species.

PACE's Ocean Color Instrument samples from the ultraviolet into the
near-infrared, which is what lets you do all of the band math below on one
scene.

Here are the bands this lab uses, and why each one:

| Band | Used in | Why that wavelength |
|---|---|---|
| **443 nm** | blue-green ratio | Chlorophyll-a absorbs strongly in the blue |
| **555 nm** | blue-green ratio | Near the green reflectance peak of algal water |
| **620 nm** | cyanobacteria index | Phycocyanin, a cyanobacterial pigment, absorbs here |
| **600, 650 nm** | cyanobacteria index | Shoulders either side of 620, for the baseline |
| **665, 710 nm** | fluorescence line height | Shoulders either side of 685, for the baseline |
| **685 nm** | fluorescence line height | Chlorophyll fluorescence emission peak |

You will see three kinds of spectrum in a coastal scene, and most of the work
is telling them apart. Phytoplankton show pigment absorptions and a small peak
near 685 nm. Colored dissolved organic matter (CDOM) shows no features, only a
smooth rise towards the blue. Suspended sediment is high across the visible and
rises towards the red.

---

## Get the data

**Product:** PACE OCI **Level-2 Apparent Optical Properties** (`AOP`), which
carries remote-sensing reflectance $R_{rs}$ in sr⁻¹ across 172 wavelengths from
346 to 719 nm. Every wavelength and value
quoted in this lab assumes AOP.

Alongside `Rrs`, the granule holds `l2_flags` — a per-pixel bitmask marking
cloud, land, glint and the other conditions that stopped a retrieval, and
`nflh`, a normalized fluorescence line height computed by the mission. Both are
[listed in the product documentation](https://www.earthdata.nasa.gov/data/catalog/ob-cloud-pace-oci-l2-aop-3.1),
and both come back in WISER's subdataset list when you open the file. Part 3
computes its own fluorescence index, which gives you something to check `nflh`
against. The **Regional Surface Reflectance** product
(`SFREFL`) — 122 wavelengths from 346 to 895 nm plus 5 SWIR bands — is a
related alternative, but it carries `rhos`, dimensionless surface reflectance,
and has no `Rrs` variable at all.

1. [Create a free Earthdata account](https://urs.earthdata.nasa.gov/users/new).
2. Search the Ocean Biology Distributed Active Archive Center
   ([OB.DAAC](https://oceancolor.gsfc.nasa.gov/)) or
   [Earthdata Search](https://search.earthdata.nasa.gov/) for **PACE OCI L2
   AOP**.
3. Pick a scene over a coastal region with contrast: Chesapeake Bay, the Baltic
   (reliable summer cyanobacteria blooms), Lake Erie (late-summer
   *Microcystis*), the Gulf of Mexico, or the California Current.
4. Choose a **cloud-free** granule. Clouds dominate ocean color scenes, and a
   scene that looks 40% clear in the browse image will be worse in practice.

Data run from March 2024 to the present.

```{admonition} Level 2, not Level 1
:class: note
The water-leaving signal is a few percent of what the satellite measures; the
rest is atmosphere. Use an **L2** product, where it has been done for you.
An L1 radiance scene will show you the atmosphere, not the ocean.
```

---

## Part 1 — Open and orient

1. From the **Main Menu**, go to **File** and select **Open...**, then choose
   the PACE `.nc` granule. One netCDF file holds many variables, so WISER opens
   the **Subdataset Chooser** dialog and asks which one you want.
2. In **Subdataset Choice**, select `geophysical_data/Rrs`. The panel beside it
   shows that subdataset's **Dataset Name**, number of **Bands**, **Wavelength
   units**, **GeoTransform** and **Spatial Ref System**, so you can confirm you
   picked the cube and not a metadata array. Leave **Use Good Wavelength Bands**
   ticked, and click **OK**.
3. Click **Band chooser** on the Main Toolbar. With **RGB** selected, set
   **Red Band** to **Band 125: 660 nm**, **Green Band** to **Band 84: 555 nm**
   and **Blue Band** to **Band 39: 443 nm**, then click **OK**.
4. Click **Stretch builder**, select **Linear Stretch**, and click
   **2.5% linear**.
5. Water is dark, so the default stretch will be dominated by cloud and land.
   Still in the **Stretch builder**, type a smaller value into each channel's
   **Maximum** box and click that channel's **Apply** button, so the stretch is
   computed on the water alone. Tick **Apply minimum/maximum values across all
   channels** at the bottom to set all three at once. See {doc}`Display and
   Contrast Stretch <../../user-content/display-and-stretch>`.

:::{figure} ../../_static/tutorials/lab_pace_truecolour.png
:width: 100%
:align: center
:alt: PACE true color over coastal water, showing blue open ocean, a turbid coastal band with visible eddies, and black gaps where cloud prevented a retrieval
:::

Most of the frame is black, and that is the ordinary condition for ocean color
at Level 2. About one pixel in six of this granule carries a retrieval; the
rest was cloud, sun glint or otherwise rejected, and the processing left it
empty. Expect to work in the clear part of a scene rather than across all of
it. What survives shows deep blue open water on the left, a turbid coastal band
threaded with eddies and filaments, and a plume at the bottom.

**Deliverable 1:** a stretched true-color image in which water structure —
fronts, plumes, blooms — is visible, plus a note on the stretch limits used.

---

## Part 2 — Four water types, four spectra

Draw ROIs ({doc}`Tutorial 3 <../03-regions-of-interest>`) over:

1. **Clear offshore water** — dark, blue
2. **A bloom** — green, or turquoise if it is a coccolithophore
3. **A sediment plume** — brown, usually near a river mouth
4. **Transitional water** — between the first two, which is what most of a
   coastal scene actually is

Collect all four mean spectra.

### Read them

**Clear water** is highest in the blue and falls steeply through green and red.
Almost all of the signal is molecular scattering.

**A bloom** peaks in the green near 550 nm, with a trough near 443 nm
(chlorophyll-a) and a second trough near 675 nm, and often a small bump at
**685 nm**: chlorophyll fluorescence, light re-emitted by the cells.

**Sediment** is high everywhere and rises towards the red, with pigment
features weak or absent.

**Transitional water** keeps the blue peak but loses the steep falloff: green
stays elevated and a little red survives. It is the most common thing in a
coastal scene and the hardest to assign, because it is a mixture rather than a
type.

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

The vertical scale runs to about 0.03 sr⁻¹ here. Remote-sensing reflectance is
a few percent of what the instrument measured before atmospheric correction. If your numbers are in the thousands, you are reading stored
integers rather than reflectance; see the note at the top of this lab.

**Deliverable 2:** the four mean spectra on one labeled plot, each diagnostic
feature annotated.

---

## Part 3 — Chlorophyll and fluorescence

### Blue-green ratio

The classical chlorophyll algorithm, in band math:

```text
b443 / b555
```

High ratio → clear water; low → more chlorophyll. It is a proxy, not a
concentration, and in coastal water it cannot tell the two apart: chlorophyll
and CDOM both absorb blue, so both push the ratio the same direction. That is
the limitation Part 3's other two indices are built to work around.

:::{figure} ../../_static/tutorials/lab_pace_bandmath.png
:width: 100%
:align: center
:alt: The band math dialog with the expression blue divided by green, its two variables bound to the 443 and 555 nm bands
:::

Open **Tools ▸ Band math...**, type the expression into **Expression:**, and
press **Enter**. In the **Variable bindings:** table, leave **Type** as **Image
Band** and scroll right to **Variable Assignments**. Bind `blue` to **Band 39:
443 nm** and `green` to **Band 84: 555 nm** of your `Rrs` dataset. Type a
**Result name (optional):**, then click **OK**.

The band numbers above are for the 172-band AOP product. If yours differ, open
the **Band chooser** and read the numbers off the dropdown, which lists every
band as `Band N: wavelength`.

:::{figure} ../../_static/tutorials/lab_pace_ratio.png
:width: 100%
:align: center
:alt: The blue-green ratio map, bright over clear offshore water and dark through the coastal plume and the cloud gaps alike
:::

Bright values are clear, blue-dominated water; dark values are where
chlorophyll and sediment have taken the blue out. A pixel with no retrieval
carries no value and renders at the same
end of the color scale as a genuinely low ratio, so cloud gaps and productive
water look alike here. Compare against the true-color image before calling any
of it a bloom.

### Fluorescence line height (FLH)

The 685 nm bump above a baseline between its shoulders:

```text
b685 - (0.5 * b665 + 0.5 * b710)
```

This is a difference rather than a ratio, and that is the point: sediment and
CDOM vary smoothly across 665, 685 and 710 nm, so whatever they add to the
baseline is largely subtracted back out. A narrow feature at 685 nm survives;
a smooth slope does not.

The granule's own `nflh` variable is the mission's version of this. Open it
beside your result and compare.

### Cyanobacteria index

Phycocyanin absorbs near 620 nm, and among the phytoplankton common in these
waters it is mostly cyanobacteria that carry it. So a dip at 620 nm that the
wavelengths either side of it do not share is worth looking for. The expression
has the same shape as the one above, a feature measured against a baseline
built from its two shoulders:

```text
1 - b620 / (0.5 * b600 + 0.5 * b650)
```

A positive value means 620 nm came back darker than its shoulders predict,
which may point to a cyanobacterial bloom rather than, say, a diatom one. Treat
it as a hint rather than an identification. Other things darken that part of the
spectrum too, and water sampled from a boat is what would settle it.

Display each index with a sequential colormap and a tight stretch.

**Deliverable 3:** the three index maps, and an explanation of where the
blue-green ratio and the fluorescence line height disagree.

---

## Part 4 — Unmix the water

1. Collect your four ROI mean spectra from Part 2, if you have not already
   ({doc}`Tutorial 3 <../03-regions-of-interest>`).
2. From the **Main Menu**, go to **Tools ▸ Data Analysis ▸ Linear Unmixing**.
3. Set **Input Dataset** to your `Rrs` dataset.
4. Click **Add Collected Spectrum** once per endmember to load your four
   spectra into the **Endmembers** list.
5. Tick **Sum to Unity**, since you chose endmembers meant to span the scene's
   water, then click **OK**.
6. When it finishes, switch to the result and read the **RMSE** band first.

```{admonition} Interpretation
:class: note
A high residual marks water your four endmembers do not describe. That could be
a fifth optical type you did not sample, or cloud shadow, or sun glint. Reading
RMSE before the abundance maps stops you trusting an abundance number in a
place where the model never fit.
```

**Deliverable 4:** abundance maps for the four components, the RMSE map, and a
short account of where the model breaks down.

---

## Part 5 — Cluster the optical types

1. From the **Main Menu**, go to **Tools ▸ Data Analysis ▸ K-means**.
2. Set **Input Dataset** to your `Rrs` dataset and **K clusters** to 5 or 6.
3. Expand **Advanced Options** and set **Random Seed** to a fixed number so the
   run is reproducible ({doc}`Tutorial 5 <../05-classification>`).
4. Click **OK**, then wait for the run to finish in the activity monitor.
5. Reopen the **K-means Dialog** and click **View Centroids**. This opens
   **K-Means — Past Runs**, a table of your runs; click **View** on the row for
   this one. Each cluster's mean spectrum is then plotted together, which is how
   you work out what each cluster is.
6. Compare the cluster map against the index maps you built in Part 3.

Optical water-type classification is used operationally to decide **which
algorithm to apply where** — a chlorophyll retrieval tuned for open ocean gives
nonsense in a river plume.

**Deliverable 5:** a labeled optical water-type map with each class named from
its centroid spectrum.

---

## Questions to answer

1. Your blue-green ratio map and your fluorescence map disagree over the plume,
   even though both were built from the same pixels. One divides two bands, the
   other subtracts a baseline. Why would sediment affect those two differently?
2. The granule comes with its own `nflh` band. Open it next to the fluorescence
   map you made in Part 3. Do they broadly agree? If not, what would you try
   changing first?
3. Open `l2_flags` and look at how much of the scene has no retrieval at all.
   Does that change how you would describe the classes you made in Part 5?
4. In Part 1, WISER applied the file's scale factor and offset for you. That is
   what puts your Part 2 spectra in the range you plotted them on, a few
   hundredths of a sr⁻¹. Without it they would come back in the tens of
   thousands. What might someone conclude about this water if they did not
   notice?
5. PACE pixels are about 1 km across. Compare a spectrum from the middle of the
   plume with one a few pixels offshore. How sharp is the edge between them, and
   how much of the plume do you think you are actually seeing?

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
