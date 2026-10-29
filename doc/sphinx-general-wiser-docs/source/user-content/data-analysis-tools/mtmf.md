# Mixture-Tuned Matched Filter (MTMF)

MTMF is a **target-detection** tool: given a reference spectrum for a material
of interest, it scores every pixel for how strongly that material is present,
without needing spectra for any of the other ("background") materials in the
scene.

## How it works

MTMF combines a *matched filter* with a *mixture-tuning* feasibility check.

1. **Noise-whiten the scene (MNF).** The cube is first run through a full
   [MNF transform](mnf.md), which rotates the data so noise is uniform in all
   directions. The matched filter is far more reliable in this whitened space.
   The target spectrum is projected into the same MNF space.
2. **Matched filter.** For each pixel, MTMF computes a score measuring how far
   the pixel lies along the direction of the target, relative to the
   background/noise distribution. Because this is done in MNF space, the noise
   covariance is the identity and the background covariance is diagonal (the MNF
   eigenvalues). The score behaves like an **abundance estimate**: near 0 for
   background/noise, near 1 for a pure target pixel. This score map is the
   tool's first output.
3. **Mixture tuning (infeasibility).** MTMF also measures each
   pixel's *infeasibility*: how much its spectrum deviates from a physically
   plausible mixture of background and target. A high matched-filter score is
   only a true detection when its infeasibility is low; this step is what
   suppresses the false positives that a plain matched filter produces.

Bad bands are excluded automatically, and nodata pixels become `NaN`. Each
target produces two float32 datasets:

- `MTMF [target]: <source>` — the matched-filter score; higher values indicate
  a stronger match.
- `IF [target]: <source>` — the infeasibility. Trust a high score only where
  the same pixel's infeasibility is low.

## Using the tool

:::{figure} ../../_static/tutorials/t7_mtmf_dialog.png
:width: 75%
:align: center
:alt: The MTMF dialog
:::


1. **Input** — leave the type as **Image Cube** and choose the dataset to
   analyze. (Spectrum input is not currently supported.)
2. **Noise method** — choose how background noise is estimated:
   - **Image Cube Based** — estimate noise from the scene itself via the
     shift-difference method; pick a direction (Down/Up/Left/Right) in the
     row below.
   - **Dark Image Based** — supply a separate dark/noise dataset (same bands
     as the input) in the value box.
   - **ROI Based** — supply a region of interest (ROI) and the dataset to take
     it from.
3. **Target** — choose the reference spectrum to detect.
4. Click **OK**. The run proceeds in the background.

## See also

- {doc}`Tutorial 7 — Finding a Known Material <../../tutorials/07-detection>`
