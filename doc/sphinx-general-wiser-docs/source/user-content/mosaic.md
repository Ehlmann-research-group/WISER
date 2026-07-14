# Mosaic

The **Mosaic** tool combines several overlapping georeferenced images into a
single image on one shared output grid. You add the scenes you want, stack them
in the order you want them to appear, preview the result, and export a
full-resolution ENVI file.

Typical uses:

- Stitching adjacent flight lines or tiles into one continuous image.
- Layering a higher-resolution scene on top of a broader background scene.
- Bringing scenes with different coordinate reference systems onto one common
  projection.

Open it from the menu bar: **Tools → Mosaic**. The window is non-modal, so you
can keep working in the main WISER window (opening more files, changing display
bands) while it is open.

## Before you start

Scenes are picked from the datasets **already loaded in WISER**, not from a file
picker. Load every image you want to mosaic first (**File → Open**), then open
the Mosaic tool.

Each scene should be georeferenced — it needs a coordinate reference system and
real map coordinates. You can still add a scene that isn't, but it will sit in
the list disabled until you georeference it; see
[Pending scenes](#pending-scenes-greyed-out).

## The mosaic window

The window has two parts:

- **The preview canvas** (left) — an interactive map showing the composited
  scenes, their outlines, and where they overlap.
- **The controls** (right) — Add scene, the scene stack, output resolution,
  target CRS, resampling, band metadata, and Export.

### Moving around the preview

| Action | How |
|--------|-----|
| Pan | Middle-click and drag |
| Zoom | Mouse wheel (zooms toward the cursor) |
| Frame one scene | Right-click the scene in the **Scenes** list → **Zoom to Scene** |

The preview frames the whole mosaic once, the first time there is something to
show. After that it never moves the camera on you — adding a scene will not
yank your view away from where you were looking.

### What the overlay colors mean

- **Green outlines** — each visible scene's footprint (the outline of its valid
  pixels, not just its rectangle).
- **Dashed box** — the bounding box of the whole mosaic; this is the extent that
  will be exported.
- **Magenta / purple shading** — where a scene is *covered* by another scene
  above it in the stack. This is where the stacking order is actually deciding
  which pixels you see.

The overlay is purely a visual aid. It marks overlaps; it does not change which
pixels win.

## Adding scenes

1. Pick a dataset from the drop-down in the **Add scene** box.
2. Click **Add Scene…**.

A progress dialog appears while WISER prepares the scene. This blocks only the
mosaic window — the rest of WISER stays usable — and you can cancel it.

Your original file is never modified. WISER works on its own temporary copies
for the whole session.

Two things will stop a scene from being added at all:

- **It's already in the mosaic.** A dataset can only be added once.
- **Its band count doesn't match.** Every scene in a mosaic must have the same
  number of bands as the first scene you added.

Anything else — a missing data-ignore value, a different pixel size, a different
CRS, a different data type — is fine. Those are handled for you.

```{note}
The preview shows each scene using the bands **currently displayed for it in the
main WISER window**, with a contrast stretch applied per scene to account for outliers. If you want a
different band combination in the preview, change the displayed bands in the
main view *before* adding the scene. This only affects how the preview looks —
the exported file always contains all the bands, with their original values.
```

## The scene stack

The **Scenes** list is the stacking order, shown **top-most first**. The scene
at the top of the list is the one drawn on top, and it wins wherever scenes
overlap.

- **Reorder** — drag a row up or down. The preview restacks immediately.
- **Hide / show** — untick or tick a scene's checkbox. Hidden scenes are left
  out of the preview and the export, but stay in the list.
- **Remove** — select a row and click **Remove Selected**.

Reordering and hiding are instant: they re-draw from what is already loaded and
never re-read the imagery.

### Pending scenes (greyed out)

A scene that WISER cannot place on the mosaic's grid is kept in the list as a
**pending** scene: greyed out, marked with a warning icon, and excluded from the
preview, the grid, and the export. Hover over it to see why. There are two
reasons:

- **It has no CRS** — the image isn't georeferenced.
- **Its CRS isn't compatible** with the mosaic's target CRS — WISER can't
  transform it into the projection the mosaic is using.

This is deliberate: you can assemble your whole working set up front, including
scenes you still intend to register, instead of hitting a dead end at Add
Scene. Fix a pending scene by right-clicking it → **Georeference…** (below).
Once it has a compatible CRS it goes live on its own and the mosaic rebuilds to
include it.

## Fixing a misplaced scene

If a scene is georeferenced but lands in the wrong place relative to the others,
you don't have to leave the mosaic. Right-click it in the **Scenes** list and
choose **Georeference…**.

This opens the Georeferencer with that scene already locked in as the target.
Place your control points or pick a reference, click **Run Warp**, and the
corrected scene is swapped straight back into the mosaic at its original
position in the stack, so you can see whether the fix worked.

- **Save to Mosaic** keeps the corrected scene.
- **Cancel** puts the original scene back, unchanged.

You can re-run the warp as many times as you like; each attempt replaces the
last rather than piling up. The correction lives for the mosaic session and is
baked into the exported file — it does not overwrite your original dataset on
disk.

## Target CRS

Every scene is reprojected onto one shared **target CRS**. The **first scene you
add sets it automatically**, and every scene you add afterward is reprojected
onto that CRS.

To use a different projection, click **Choose Target CRS…**. The dialog shows
each scene and its current CRS, and lets you pick the target from any scene's
CRS, a built-in preset (WGS 84, Web Mercator, NAD83 / UTM 15N), a CRS you made
with the CRS Creator, or an authority code you type in (for example `EPSG` +
`4326`).

Changing the target CRS re-evaluates every scene: any scene that can not be
transformed into the new CRS becomes [pending](#pending-scenes-greyed-out), and
any pending scene that *can* reach it goes live. If the new CRS leaves no usable
scenes at all, WISER warns you that the preview is empty but still applies your
choice. If the change pushes the mosaic off-screen, the preview reframes itself
onto the scenes.

## Output settings

These control the file you export. Set them before you click Export.

### Output Spatial Resolution

The pixel size of the exported mosaic, in the target CRS's units.

| Mode | Pixel size used |
|------|-----------------|
| **Top scene** (default) | The pixel size of the scene at the top of the stack |
| **Highest (finest)** | The smallest pixel size among the visible scenes |
| **Lowest (coarsest)** | The largest pixel size among the visible scenes |
| **Average** | The mean pixel size across the visible scenes |
| **Custom…** | An X and Y pixel size you type in |

### Resampling

How pixel values are computed when a scene is reprojected onto the output grid.

- **Nearest Neighbor** (default) — copies the closest source pixel. Values are
  preserved exactly. Use this for anything you plan to analyze quantitatively,
  such as classification or spectral analysis.
- **Bilinear** and **Cubic Convolution** — interpolate between pixels, giving a
  smoother-looking image but **inventing values that were never measured**.
  WISER warns you when you pick one.

### Band metadata

Choose which scene's band metadata — wavelengths, band names, bad-band list —
gets written onto the exported file. It defaults to the top scene. This is
labeling only: it never changes how many bands the output has, or the pixel
values.

## Exporting

Click **Export / Finish…**, choose an output path, and WISER composites every
visible scene at full resolution and writes an **ENVI** file (`.img` plus a
`.hdr`).

What you get:

- Every visible scene, reprojected onto the target CRS at your chosen output
  resolution.
- Overlaps resolved strictly by the stacking order — the top scene wins. Values
  are never blended or averaged.
- Original pixel values (with Nearest Neighbor), with invalid and outside-the-
  footprint areas written as the mosaic's no-data value, so a hole in an upper
  scene lets the scene beneath show through.
- The chosen scene's band metadata written into the `.hdr`, so the file re-opens
  in WISER with the right spectral labels.

Export runs in the background with a progress bar and can be cancelled. Large
mosaics are streamed to disk piece by piece, so a mosaic bigger than your RAM is
not a problem.

A few things worth knowing before you click it:

- **Pending scenes are skipped.** If any exist, WISER asks you to confirm before
  continuing; they simply won't be in the output.
- **Hidden scenes are skipped.** Unticked scenes are not exported.
- **Metadata edits made after adding a scene are not picked up.** A mosaic scene
  is a snapshot taken when you added it. If you edit a dataset's metadata (say,
  its wavelengths or data-ignore value) in the main WISER window afterward, the
  export uses the values from when the scene was added, and WISER warns you which
  scenes have drifted. To pick up the edit, remove the scene and add it again.
- **The result is not loaded back into WISER.** Open the exported file with
  **File → Open** if you want to look at it.

## Troubleshooting

**A scene is greyed out with a warning icon.** It's
[pending](#pending-scenes-greyed-out) — hover it to see whether it has no CRS or
an incompatible one, then right-click → **Georeference…**.

**The preview is blank.** Either every scene is pending or hidden, or the camera
is parked away from the scenes. Right-click a live scene → **Zoom to Scene** to
jump back to it.

**A scene won't add at all.** Check the band count — it has to match the first
scene in the mosaic — and check it isn't already in the list.

**A scene is in the wrong place.** Right-click it → **Georeference…** and warp
it in place; see [Fixing a misplaced scene](#fixing-a-misplaced-scene).

**Changing the output resolution does nothing to the preview.** That's expected
— it only sizes the exported file.

**The exported values look interpolated.** Set **Resampling** back to
**Nearest Neighbor** and export again.
