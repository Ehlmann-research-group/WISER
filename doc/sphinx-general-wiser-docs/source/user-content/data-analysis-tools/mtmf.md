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
   the pixel lies along the direction of the target, relative to the background/noise
   distribution (Note that because this is done in MNF, the background distribution is
   the identity matrix). The score behaves like an **abundance estimate**: near 0 for
   background/noise, near 1 for a pure target pixel. This score map is the tool's
   output.
3. **Mixture tuning (infeasibility).** Internally MTMF also measures each
   pixel's *infeasibility* — how much its spectrum deviates from a physically
   plausible mixture of background and target. A high matched-filter score is
   only a true detection when its infeasibility is low; this step is what
   suppresses the false positives that a plain matched filter produces.

Bad bands are excluded automatically, and nodata pixels become `NaN`. Each
target produces one float32 score image named `MTMF [target]: <source>`; higher
values indicate a stronger match.

## Using the tool

1. **Input** — leave the type as **Image Cube** and choose the dataset to
   analyze. (Spectrum input is not currently supported.)
2. **Noise method** — choose how background noise is estimated:
   - **Image Cube Based** — estimate noise from the scene itself via the
     shift-difference method; pick a direction (Down/Up/Left/Right) in the
     row below.
   - **Dark Image Based** — supply a separate dark/noise dataset (same bands
     as the input) in the value box.
   - **ROI Based** — supply a region of interest (ROI) and a dataset where
     the ROI will be pulled from
3. **Target** — choose the reference spectrum to detect.
4. Click **OK**. The run proceeds in the background.

<!-- PICTURE: Screenshot of the MTMF dialog labeling the Input type + Input
     dataset row, the Noise method type + value row, the Shift Diff Method
     (direction) row, the Target row, and the OK / Cancel buttons. -->

<!-- PICTURE: The dialog in "Image Cube Based" mode, highlighting the
     Shift Difference direction dropdown that becomes relevant. -->

<!-- PICTURE: Example matched-filter score image, with high-scoring (likely
     target) pixels standing out brightly against the darker background. -->
