# Labs

Six applied workflows, each sized for a 2–3 hour session. All but Lab F need
a dataset you download first; each lab opens with what it needs.

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

## Choosing one

| If you want | Start with |
|---|---|
| The best-supported route in, and a full 425-band cube end to end | **Lab A** |
| No download at all | **Lab F**, which runs on data already in your checkout |
| The canonical imaging-spectroscopy exercise | **Lab B**. Cuprite is where most methods in this field were first demonstrated, so your results have a literature to check against |
| Something other than Earth, still no account | **Lab C**. CRISM data are open, and the Jezero carbonate detection is one you can reproduce |
| A current mission and a live science question | **Lab D** or **Lab E** |

```{admonition} For instructors
:class: note
Each lab lists **deliverables** and **questions to answer**. **Lab F** runs
entirely on bundled data, which makes it the safest choice when you cannot rely
on students having network access or disk space. **Labs A, B and C** need no
account of any kind; **D** and **E** need a free NASA Earthdata Login, worth
having students create in advance.
```

```{admonition} Where the screenshots come from
:class: note
Labs A, B, C and F are illustrated with figures captured by driving WISER
through exactly the steps described, on exactly the data named, including the
band indices, pixel coordinates and band-math expressions in the text.

Labs D and E specify the same level of detail but ship no screenshots. Both
datasets sit behind an Earthdata Login, which the figure harness cannot
authenticate to, so capture your own as you work.
```

## What every lab expects you to do

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
