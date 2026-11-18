# Rendering Pipeline

This page documents how WISER turns a loaded raster dataset into the pixels you
see on screen: how display bands are chosen, how contrast stretches are built
and applied, how results are cached, and how all of this is driven by
`RasterView`. It is the entry point for three closely related subsystems, each
documented in depth on its own page:

- [Band Chooser](band-chooser.md) — selecting which dataset band(s) map to red,
  green, blue, or grayscale.
- [Stretch Builder](stretch-builder.md) — building and applying the per-channel
  contrast stretches that map raw values to display brightness.
- [Data Caching](data-caching.md) — the three-tier cache that avoids recomputing
  band data, rendered images, and histograms.

For how `RasterView` fits into the larger display system (panes, zoom, linked
scrolling, coordinate systems), see [Viewport System](viewport-system.md). This
page covers only the *data → image* path.

---

## Overview

`RasterView` owns the rendering pipeline. Its core method,
`update_display_image()` (`src/wiser/gui/rasterview.py`), is the
single place where a `RasterDataSet`, a set of display bands, a list of
stretches, and an optional colormap are combined into the `QPixmap` shown on
screen.

The three inputs are held as attributes on `RasterView` and changed through
setters that invalidate cached state and re-render:

| Attribute | Set by | Source subsystem |
|-----------|--------|------------------|
| `_display_bands` | `set_display_bands()` | [Band Chooser](band-chooser.md) |
| `_stretches` | `set_stretches()` | [Stretch Builder](stretch-builder.md) |
| `_colormap` | `set_display_bands()` | [Band Chooser](band-chooser.md) |

Everything else — normalized band data, the final image, and histograms used by
the Stretch Builder — flows through the shared [data cache](data-caching.md).

---

## The Pipeline, End to End

```{mermaid}
flowchart TD
    DS["RasterDataSet"]
    BANDS["_display_bands<br/>(1 or 3 indices)"]
    NORM["get_band_data_normalized()<br/>float values in [0, 1]"]
    JOINT{"Joint stretch?<br/>(_detect_joint_stretch)"}
    PERBAND["Per-band path:<br/>make_channel_image()<br/>conditioner then stretch"]
    JOINTPATH["Joint path:<br/>_render_joint_channels()<br/>apply_multi() on (H,W,3)"]
    U8["per-channel uint8<br/>(0..255)"]
    RGB["make_rgb_image() /<br/>make_grayscale_image()<br/>packed uint32 ARGB"]
    QIMG["QImage / QPixmap<br/>(_image_pixmap)"]

    CC[("Computation cache<br/>normalized bands")]
    RC[("Render cache<br/>final uint32 image")]

    DS --> NORM
    BANDS --> NORM
    NORM -.->|lookup / store| CC
    NORM --> JOINT
    JOINT -->|no| PERBAND
    JOINT -->|yes| JOINTPATH
    PERBAND --> U8
    JOINTPATH --> U8
    U8 --> RGB
    RGB -.->|lookup / store| RC
    RGB --> QIMG
```

### Stage-by-stage

| Stage | Responsible code | Cache used |
|-------|------------------|------------|
| Check for a fully-rendered image | `update_display_image()` | Render cache |
| Read + normalize each band to `[0, 1]` | `RasterDataSet.get_band_data_normalized()` (`src/wiser/raster/dataset.py`) | Computation cache |
| Decide per-band vs joint stretch | `RasterView._detect_joint_stretch()` | — |
| Apply stretch, scale to `uint8` | `make_channel_image_strips()` / `_render_joint_channels()` | Joint result cache (in-view) |
| Pack channels into ARGB `uint32` | `make_rgb_image_strips()` / `make_grayscale_image()` | — |
| Store rendered image | `update_display_image()` | Render cache |
| Wrap as `QImage` → `QPixmap` | `update_display_image()` | — |

The render cache is checked first: if a `uint32` image already exists for the
exact `(dataset, bands, stretches, colormap)` combination, all of the
intermediate work is skipped. See [Data Caching](data-caching.md) for the cache
keys and an important known issue affecting whether these lookups currently
hit.

---

## Per-band vs Joint Stretches

Most stretches operate on one band at a time (`StretchBase.apply()` mutates a
2-D array in place). A few — notably the **decorrelation stretch** — are
*joint*: they must see all three display bands together. `RasterView`
distinguishes the two cases with `_detect_joint_stretch()`, which inspects each
channel's stretch and returns a joint stretch only if **every** channel agrees
on the same one (otherwise it logs a warning and falls back to per-band).

- **Per-band path** — for each dirty channel, `make_channel_image_strips()`
  applies the optional conditioner then the main stretch, clips to `[0, 1]`,
  and scales to `uint8`, a row strip at a time.
- **Joint path** — `_render_joint_channels()` gathers all bands into one
  `(H, W, 3)` float32 buffer, applies per-band conditioners, runs
  `apply_multi()` once, then clips and scales each channel. The joint result is
  memoized in `RasterView._joint_render_cache` so re-renders that don't change
  the bands or conditioners can skip the expensive transform.

See [Stretch Builder](stretch-builder.md) for the stretch model and the
conditioner/stretch composition that this detection relies on.

---

## How Changes Reach the Screen

The Band Chooser and Stretch Builder are dialogs; they never touch `RasterView`
directly. Changes flow through `ApplicationState` (the state holder) and the
`App` class (the signal broker), then down to every `RasterView` showing the
affected dataset. This mirrors the broker pattern described in
[Viewport System](viewport-system.md).

```{mermaid}
sequenceDiagram
    participant SB as StretchBuilderDialog
    participant App as App (broker)
    participant AS as ApplicationState
    participant RP as RasterPane
    participant RV as RasterView

    Note over SB,RV: Stretch changed
    SB->>App: stretch_changed(ds_id, bands, stretches)
    App->>AS: set_stretches(ds_id, bands, stretches)
    AS->>RP: stretch_changed(ds_id, bands)
    RP->>AS: get_stretches(ds_id, bands)
    RP->>RV: set_stretches(stretches)
    RV->>RV: update_display_image()

    Note over SB,RV: Bands / colormap changed (global)
    Note right of RP: BandChooserDialog returns, RasterPane emits display_bands_change
    RP->>App: display_bands_change(ds_id, bands, colormap, is_global)
    App->>RP: set_display_bands() on every pane
    RP->>RV: set_display_bands(bands, stretches, colormap)
    RV->>RV: update_display_image()
```

**Stretch change path** — `StretchBuilderDialog.stretch_changed`
→ `App._on_stretch_changed` → `ApplicationState.set_stretches` (stores the
stretch per `(ds_id, band_index)` and re-emits `stretch_changed`)
→ `RasterPane._on_stretch_changed` → `RasterView.set_stretches`.

**Band/colormap change path** — `BandChooserDialog` returns from `exec_()`
→ `RasterPane._on_band_chooser` emits `display_bands_change`
→ `App._on_display_bands_change` calls `set_display_bands` on all panes (when
global) → `RasterView.set_display_bands`. A non-global change is applied only to
the originating pane.

Both setters null out `RasterView._joint_render_cache` and call
`update_display_image()`, so the next paint reflects the change.

---

## Progress, Cancel, and the GUI Thread

```{warning}
`update_display_image()` runs **synchronously on the GUI thread**, and it runs
**once per pane**: the context pane, main view, and zoom pane each hold a
`RasterView` that renders the same change. On large images a stretch change or
a dataset open still takes seconds of GUI-thread time.
```

"Opening a dataset is slow" is mostly a *rendering* cost rather than a read
cost. `loader.load_from_file()` probes drivers and reads metadata, not pixels;
the band read, data-ignore masking, normalization, stretch, and compositing all
happen here, driven off the `dataset_added` signal.

What keeps that time from looking like a hang:

- **Strips.** The per-channel stretch (`make_channel_image_strips()`) and the
  RGB packing (`make_rgb_image_strips()`) run in horizontal row strips into
  preallocated arrays. `strip_rows(width)` sizes a strip so its float32 working
  copy of a band is about `RENDER_STRIP_BYTES` (8 MiB). The numba or Python
  kernel is chosen by the whole band's size, so the output is identical to a
  whole-band render. The joint path is not row-separable; it stays whole-band
  and reports one step per stage.
- **One progress dialog.** `ApplicationState.set_stretches()` and
  `ApplicationState.add_dataset()` run their signal emit inside
  `render_progress()` (`src/wiser/gui/render_progress.py`). Every
  `RasterView` render under it reports its band reads, strips, joint stages and
  pixmap as steps to one `QProgressDialog`, which appears only after the work
  has taken 500 ms (`MINIMUM_DURATION_MS`), so small images never show it. Once
  shown, the dialog processes events at each step, which keeps the application
  repainting and the Cancel button live.
- **All-or-nothing cancel for a stretch change.** A cancel is checked between
  steps. Each `RasterView` records its render state before its first change
  under the progress; on cancel every touched view is restored, including panes
  that had already finished, the render-cache entries those renders added are
  removed, `set_stretches()` puts back the previous stretches in
  `ApplicationState`, and its `stretch_change_cancelled` signal makes the
  stretch builder reload its controls from them. No pane shows a partial image.
- **Dataset open shows progress without Cancel.** Opening a dataset always
  displays it.
- **A band-chooser change has no dialog.** `set_display_bands()` renders in
  strips like the others but outside any `render_progress()`, so it reports no
  progress and cannot be cancelled.

A pane holds its previous image until the new one is complete, so during a
render peak memory is roughly one extra display image per pane.

The `QImage` / `QPixmap` tail cannot leave the GUI thread. Moving the NumPy work
off-thread would need the seam cut between the arrays and the Qt handoff, with
stale renders discarded on arrival.

The render cache does not currently store new entries (see
[Data Caching](data-caching.md) and issue #772), so this path runs uncached
today.

Related: #758 (moving the NumPy portion off-thread) and #69 (LoD-pyramid
rendering, which would reduce the cost enough to change what the right fix is).

---

## Where to Read Next

- [Band Chooser](band-chooser.md) — `_display_bands` and `_colormap`.
- [Stretch Builder](stretch-builder.md) — `_stretches`, the `StretchBase`
  hierarchy, conditioners, and the joint decorrelation path.
- [Data Caching](data-caching.md) — the render, computation, and histogram
  caches, their keys, lifecycle, and a known issue.
- [Viewport System](viewport-system.md) — zoom, pan, coordinate systems, and how
  `RasterView` sits inside the pane hierarchy.
