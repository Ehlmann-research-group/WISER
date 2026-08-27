# Applied Labs

Each lab takes one field's question to a real dataset, runs a full analysis in
WISER, and ends with a result you have to interpret. They assume you have
worked through the {doc}`Getting Started tutorials <../index>` — the labs
describe *what* to do and *why*, and link back to the tutorials for mechanics.

All six are sized for a 2–3 hour session and list deliverables and questions,
so they can be used directly as course labs.

```{list-table}
:header-rows: 1
:widths: 6 26 24 22 22

* -
  - Lab
  - Field
  - Instrument
  - Data
* - A
  - {doc}`Urban Vegetation and Materials with AVIRIS-NG <lab-aviris-ng-urban>`
  - Urban ecology, methods
  - AVIRIS-NG, 425 bands
  - 551 MB, no account
* - B
  - {doc}`Mineral Mapping at Cuprite <lab-cuprite-minerals>`
  - Economic geology
  - AVIRIS-Classic
  - ~600 MB, no account
* - C
  - {doc}`Martian Mineralogy with CRISM <lab-mars-crism>`
  - Planetary science
  - MRO/CRISM
  - Free, no account
* - D
  - {doc}`Surface Mineralogy with EMIT <lab-emit-dust>`
  - Earth system science
  - EMIT (ISS)
  - ~1.8 GB, Earthdata Login
* - E
  - {doc}`Phytoplankton with PACE <lab-pace-phytoplankton>`
  - Oceanography
  - PACE/OCI
  - Earthdata Login
* - F
  - {doc}`Close-Range Materials Imaging <lab-materials-imaging>`
  - Materials, inspection
  - Bench spectrometer
  - **Ships with WISER**
```

## Choosing a lab

- **Start with Lab A.** One download, no account, a scene you can look up, and
  every figure in it was produced by running the steps. It is also the only lab
  that walks through a full 425-band cube end to end.
- **No downloads at all** → **Lab F**, which uses data already in your
  checkout.
- **The canonical imaging-spectroscopy exercise** → **Lab B**. Cuprite is where
  most methods in this field were first demonstrated, so your results have a
  large literature to check against.
- **Something other than Earth, still no account** → **Lab C**. CRISM data are
  open, and the Jezero carbonate detection is a real result you can reproduce.
- **A current mission and a live science question** → **Lab D** or **Lab E**.

```{admonition} Where the screenshots come from
:class: note
Labs A and F are illustrated with figures captured by driving WISER through
exactly the steps described, on the data named. Labs B–E specify the same level
of detail but ship no screenshots, because their datasets are large downloads
the documentation build does not carry — capture your own as you work.
```

## What every lab expects you to do

1. **Check what you are opening.** Reflectance or radiance, Level 1 or Level 2,
   map-projected or not. Half the mistakes in imaging spectroscopy are made
   before any analysis starts.
2. **Look at spectra before running anything.** Every tool in WISER ranks
   pixels against a hypothesis you supply. If you cannot see the feature in a
   spectrum, a detection map will not create it.
3. **Set the stretch on every computed product.** Index and score images have
   no reason to fill a display range sensibly; one edge pixel can flatten
   everything else. Lab A shows what this looks like when it goes wrong.
4. **Read the diagnostic output, not just the classification.** The SAM angle
   image, the SFF RMSE, MTMF infeasibility, the unmixing residual — that is
   where you find out whether the answer is any good.
5. **State your uncertainties.** Mixed pixels, grain size, illumination,
   residual atmosphere, threshold choice. Every lab asks for this explicitly.

```{toctree}
:hidden:

lab-aviris-ng-urban
lab-cuprite-minerals
lab-mars-crism
lab-emit-dust
lab-pace-phytoplankton
lab-materials-imaging
```
