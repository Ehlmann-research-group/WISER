# Lab F — Close-Range Imaging of Manufactured Materials

- **Field:** materials science, industrial inspection, cultural heritage, laboratory spectroscopy
- **Instrument:** any bench or close-range imaging spectrometer
- **Prerequisites:** {doc}`Tutorials 1–5 <../index>`

```{admonition} You do not need to download anything for this lab
:class: note
It runs entirely on a scene already in your WISER checkout, which makes it the
safe choice when network access or disk space cannot be relied on. Bring your
own cube instead if you have one.
```

---

## Why this lab exists

Nothing in WISER assumes an aircraft or a satellite. A cube is a cube: *x*, *y*
and wavelength. Bench-scale imaging spectrometers sort plastics for recycling,
verify pharmaceutical tablets, detect bruising in fruit, authenticate pigments
in paintings, and inspect electronics, and the methods in the tutorials apply
unchanged.

This lab uses a scene already in your checkout, so you can run all of it now,
with no download.

**Data:** `src/test_utils/test_datasets/circuit_4_100_150_um.hdr` — a 150 × 150
pixel, 4-band close-range scene of a printed circuit board at **0.525, 0.635,
0.740 and 1.200 µm**. Note the units: micrometres, not nanometres. WISER reads
them from the header and labels the axes accordingly.

---

## Part 1 — Open and look

1. **File ▸ Open...** → `circuit_4_100_150_um.hdr`.
2. Turn on all four panes, **Zoom to fit**, and apply a **2.5% linear**
   stretch.

:::{figure} ../../_static/tutorials/lab_board_rgb.png
:width: 90%
:align: center
:alt: The circuit-board scene in WISER with all panes visible
:::

The board's structure is all there: solder mask, exposed pads, plated
through-holes, silkscreen legend, and the dark epoxy bodies of the components.

3. Click each material in turn and collect its spectrum
   ({doc}`Tutorial 2 <../02-spectra>`), recolouring each from the list.

:::{figure} ../../_static/tutorials/lab_board_spectra.png
:width: 60%
:align: center
:alt: Collected spectra from three points on the circuit board
:::

Four bands make a coarse spectrum, but the separations are already clear:
surfaces that look similar differ in how their reflectance falls towards
1.2 µm. On a 100-plus-band bench instrument each becomes a rich, identifiable
signature.

**Deliverable 1:** labeled spectra for at least three materials on the board.

---

## Part 2 — Separate materials by index

The normalized-difference trick that maps vegetation
({doc}`Tutorial 4 <../04-band-math-ndvi>`) works on any two bands that respond
differently to the materials you care about.

**Tools ▸ Band math...**:

```text
(b1200 - b740) / (b1200 + b740)
```

Bind `b1200` to band 3 and `b740` to band 2. The result separates surfaces by
how their reflectance changes into the near-infrared, which distinguishes
metallised from polymer areas far more cleanly than any single band.

Display with a diverging colormap, set the stretch, and compare against the
true-color image in a **1 × 2 grid**.

**Deliverable 2:** the index map, with a note on which physical difference it
keys on.

---

## Part 3 — Classify

1. Run **K-means** with K = 4 or 5 and a fixed seed
   ({doc}`Tutorial 5 <../05-classification>`).
2. Display with a categorical colormap.
3. Click **View Centroids** and name each cluster from its spectrum.

You now have an unsupervised material map. In an inspection context this is the
basis for anomaly detection: build the class map from a known-good board, then
flag pixels on later boards whose spectra fall far from every centroid.

**Deliverable 3:** a labeled material map with each cluster identified.

---

## Part 4 — Build class signatures

1. Draw ROIs over each material — solder mask, exposed pad, component body,
   silkscreen ({doc}`Tutorial 3 <../03-regions-of-interest>`).
2. Collect each ROI's mean spectrum.
3. **Export all spectra in ROI...** for each class.

That export is a labeled training set: every pixel's spectrum with a class
label attached, ready for a classifier in Python or R. Building it in WISER,
where you can see which pixels you are labelling, is usually faster and more
reliable than scripting it blind.

**Deliverable 4:** an exported per-class spectrum file, and the mean spectrum
of each class on one plot.

---

## Taking this to your own instrument

The bundled scene is 4 bands and a unit-test fixture. Your own bench system
will give you hundreds. What to get right:

**Calibrate to reflectance.** Image a white reference panel (Spectralon or
similar) and a dark frame under the same illumination and exposure, then
compute

$$R = \frac{S - D}{W - D}$$

with band math, binding `S`, `W` and `D` as image cubes. Everything
spectroscopic depends on this step. Raw digital numbers carry your lamp's
spectrum and your sensor's response, and no reference library will match them.

**Watch the geometry.** Close-range imaging has strong, uneven illumination and
specular highlights — a shiny solder pad can saturate at one angle and read
near-zero at another. Use diffuse illumination, and prefer
{doc}`SAM <../07-detection>` and continuum removal, both of which discount
brightness, over methods that depend on absolute level.

**Save your own library.** Image known reference materials, collect their
spectra, and export them as an ENVI spectral library. Then SAM and SFF work
against *your* materials under *your* optics, which beats any published library
for the specific question you are asking.

**Smooth before differentiating.** Bench spectra of dark materials are noisy.
The **Savitzky–Golay filter** (right-click the image ▸ **Filters ▸
Savitzky–Golay Filter...**) smooths along the spectral axis while preserving
band shape and depth far better than a moving average — see
{doc}`Filters <../../user-content/filters>`.

---

## Questions to answer

1. Why must a bench cube be converted to reflectance before you compare it with
   a spectral library?
2. A specular highlight saturates a copper pad in three bands. What does that
   do to a SAM result for that pixel, and to a linear-unmixing result?
3. You want to distinguish two visually identical black plastics. What would
   you need from your instrument that a 4-band system cannot give you?
4. Why is a library you measured yourself often better than a published one for
   an inspection task?

---

## Related applications to try

| Field | Scene | What to look for |
|---|---|---|
| **Recycling** | Mixed plastic flakes | C–H overtones, 1600–1800 nm and 2200–2400 nm; separates PET / HDPE / PP / PS |
| **Food quality** | Fruit surface | Water at 970 nm and 1450 nm; bruising shows before it is visible |
| **Pharmaceutical** | Tablets | API and excipient distribution; blend uniformity |
| **Cultural heritage** | Painting or manuscript | Pigment identification; underdrawing in the NIR; retouching |
| **Forensics** | Documents, fibers | Ink discrimination where inks are visually identical |
| **Soil science** | Core or sample tray | Organic carbon, clay mineralogy, moisture |
