# Labs

Six worked analyzes to build a lab from. Each carries one field's question
through a real dataset, with the band indices, pixel coordinates and expressions
written out, so you can run it as printed or change it knowing what every number
in it does.

Treat them as baselines rather than finished assignments. The parts, the
deliverables and the questions at the end of each one are a starting set: swap
the scene for a region your students work on, drop a part that does not fit the
course, or keep the structure and change the science. All but Lab F need a
dataset you download first, and each opens with what it needs.

```{list-table}
:header-rows: 1
:widths: 6 30 28 36

* -
  - Lab
  - Field
  - Data
* - A
  - {doc}`Urban Vegetation and Materials with AVIRIS-NG <lab-aviris-ng-urban>`
  - Urban ecology, remote-sensing methods
  - AVIRIS-NG — 551 MB, no account
* - B
  - {doc}`Mineral Mapping at Cuprite <lab-cuprite-minerals>`
  - Economic geology, alteration mapping
  - AVIRIS-Classic — no account
* - C
  - {doc}`Martian Mineralogy with CRISM <lab-mars-crism>`
  - Planetary science, astrobiology
  - MRO/CRISM MTRDR — no account
* - D
  - {doc}`Surface Mineralogy with EMIT <lab-emit-dust>`
  - Earth system science, climate forcing
  - EMIT L2A — Earthdata Login
* - E
  - {doc}`Phytoplankton and Coastal Water with PACE <lab-pace-phytoplankton>`
  - Biological oceanography, water quality
  - PACE/OCI L2 — Earthdata Login
* - F
  - {doc}`Close-Range Materials Imaging <lab-materials-imaging>`
  - Materials, inspection, cultural heritage
  - Ships with WISER
```

```{admonition} For instructors
:class: note
Nothing here is fixed. These scenes were chosen because the data is public and
the answers are checkable against published work, not because they are the right
subject for your course. The same sequence of steps runs on a scene from your own
field site.

For planning: **Lab F** runs entirely on bundled data, which makes it the safest
choice when you cannot rely on students having network access or disk space.
**Labs A, B and C** need no account of any kind. **D** and **E** need a free NASA
Earthdata Login, worth having students create in advance.
```

```{admonition} Where the screenshots come from
:class: note
Labs A, B, C, E and F are illustrated with figures captured by driving WISER
through exactly the steps described, on exactly the data named, including the
band indices, pixel coordinates and band-math expressions in the text.

Lab D specifies the same level of detail but ships no screenshots yet, so
capture your own as you work.
```

## What carries over to your own scene

Change the data and the science and this part does not, which makes it the
sequence worth keeping in whatever you build.

1. **Check what you are opening.** Reflectance or radiance, Level 1 or Level 2,
   map-projected or not. Half the mistakes in imaging spectroscopy are made
   before any analysis starts.
2. **Look at spectra before running anything.** Every tool in WISER ranks
   pixels against a hypothesis you supply. If you cannot see the feature in a
   spectrum, a detection map will not create it.
3. **Set the stretch on every computed product.** Index and score images have
   no reason to fill a display range sensibly, and one edge pixel can flatten
   everything else. Lab A shows what this looks like when it goes wrong.
4. **Read the diagnostic output, not just the classification.** The Spectral
   Angle Mapper's angle image, Spectral Feature Fitting's fit error, the
   Mixture-Tuned Matched Filter's infeasibility and the unmixing residual are
   where you find out whether the answer is any good.
5. **State your uncertainties.** Mixed pixels, grain size, illumination,
   residual atmosphere, threshold choice. Every lab asks for this explicitly.

---

```{toctree}
:hidden:

lab-aviris-ng-urban
lab-cuprite-minerals
lab-mars-crism
lab-emit-dust
lab-pace-phytoplankton
lab-materials-imaging
```
