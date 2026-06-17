# Stretch Builder

A *contrast stretch* maps raw band values onto display brightness. The Stretch
Builder is the dialog where the user designs those mappings per channel, and the
`StretchBase` hierarchy is the data model that captures them and applies them
during rendering. This page covers both halves and the joint multi-band path
used by the decorrelation stretch.

This builds on the [Rendering Pipeline](rendering-pipeline.md) overview; read
that first for how stretches fit into the larger data → image flow.

---

## Overview

Stretches always operate on **normalized** data — float values in `[0, 1]`
produced by `RasterDataSet.get_band_data_normalized()`. Keeping every stretch in
the same normalized domain is what makes them composable and dataset-agnostic.

- **GUI:** `StretchBuilderDialog`, `ChannelStretchWidget`, `StretchConfigWidget`
  (`src/wiser/gui/stretch_builder.py`)
- **Model:** `StretchBase` and subclasses
  (`src/wiser/raster/stretch.py`)
- **Applied by:** `RasterView.update_display_image()` /
  `make_channel_image()` / `_render_joint_channels()`
  (`src/wiser/gui/rasterview.py`)

A stretch can be built from two parts: an optional **conditioner** (a nonlinear
pre-shaping step such as square-root or log) followed by a **main stretch**
(linear, histogram-equalize, or decorrelation). The two are bound together with
`StretchComposite`.

---

## The Stretch Model

**File:** `src/wiser/raster/stretch.py`

Every stretch derives from `StretchBase` and shares one contract:

| Member | Purpose |
|--------|---------|
| `apply(a)` | Mutate the 2-D normalized array `a` in place. The per-band path. |
| `requires_all_bands()` | `True` only for joint transforms (decorrelation). Tells the renderer to use the multi-band path. |
| `apply_multi(bands)` | Joint entry point: mutate an `(H, W, N)` float32 stack in place. Called only when `requires_all_bands()` is `True`. |
| `get_stretches()` | Decompose into `[first, second]` — used by the renderer and joint detection. A simple stretch returns `[self, None]`. |
| `get_hash_tuple()` / `__hash__` / `__eq__` | Value identity, used as part of [render-cache](data-caching.md) keys. |

```{mermaid}
classDiagram
    direction TB

    class StretchBase {
        stretch.py
        +apply(a)
        +requires_all_bands() bool
        +apply_multi(bands)
        +get_stretches()
        +get_hash_tuple()
    }
    class StretchLinear {
        +_lower, _upper
        +_slope, _offset
        +set_bounds(lower, upper)
    }
    class StretchHistEqualize {
        +_cdf, _histo_edges
    }
    class StretchSquareRoot {
        conditioner: sqrt(a)
    }
    class StretchLog2 {
        conditioner: log2(a+1)
    }
    class StretchDecorrelation {
        requires_all_bands = True
        +apply_multi(bands)
    }
    class StretchComposite {
        +_first, _second
        +apply(a) = first then second
    }

    StretchBase <|-- StretchLinear
    StretchBase <|-- StretchHistEqualize
    StretchBase <|-- StretchSquareRoot
    StretchBase <|-- StretchLog2
    StretchBase <|-- StretchDecorrelation
    StretchComposite o-- StretchBase : first (conditioner)
    StretchComposite o-- StretchBase : second (main stretch)
```

> **Note:** `StretchComposite` is *not* a subclass of `StretchBase`; it is a
> wrapper that holds two stretches and exposes the same `apply()` /
> `get_stretches()` / `get_hash_tuple()` methods (duck typing). Its
> `get_stretches()` returns `[first, second]`, which is exactly what the renderer
> and joint-stretch detection inspect.

### The stretch types

| Class | Kind | What it does |
|-------|------|--------------|
| `StretchLinear(lower, upper)` | main | Linear remap of `[lower, upper]` → `[0, 1]`, then clip. Stores precomputed `_slope`/`_offset`. Bounds are in normalized `[0, 1]` units. |
| `StretchHistEqualize(bins, edges)` | main | Histogram equalization via the CDF of the supplied histogram (`np.interp`). |
| `StretchSquareRoot()` | conditioner | `sqrt(a)` — brightens shadows. |
| `StretchLog2()` | conditioner | `log2(a + 1)` — strong compression of bright values. |
| `StretchDecorrelation()` | joint main | Cross-band decorrelation; `requires_all_bands()` is `True`. A stateless marker — the math lives in `apply_multi` / `decor_numba`. |

Most stretches have a sibling `...UsingNumba` jitclass (e.g.
`StretchLinearUsingNumba`) selected at runtime for large arrays — see the Numba
Dispatch section below.

### Conditioner + stretch composition

The renderer never assumes a single stretch. For each channel it calls
`stretch.get_stretches()` and receives `[first, second]`:

- A bare stretch returns `[self, None]`.
- A conditioned stretch is a `StretchComposite(conditioner, main)` and returns
  `[conditioner, main]`.

`make_channel_image()` then applies `first` then `second`, clips to `[0, 1]`,
and scales to `uint8`. This is why a square-root or log conditioner can be
combined freely with any main stretch.

---

## The Stretch Builder Dialog

**File:** `src/wiser/gui/stretch_builder.py`

**Purpose:** Interactive UI for designing the stretch on each display channel
and emitting the result.

| Class | Role |
|-------|------|
| `StretchBuilderDialog` | Top-level dialog; coordinates one `ChannelStretchWidget` per channel, slider linking, and emits `stretch_changed`. |
| `ChannelStretchWidget` | Per-channel histogram display with draggable low/high bounds. |
| `StretchConfigWidget` | Stretch-type and conditioner radio selection, plus quick 2.5% / 5% linear presets. |

`StretchType` and `ConditionerType` (`src/wiser/raster/stretch.py`)
enumerate the UI choices: `NO_STRETCH`, `LINEAR_STRETCH`, `EQUALIZE_STRETCH`,
`DECORRELATION_STRETCH`; and `NO_CONDITIONER`, `SQRT_CONDITIONER`,
`LOG_CONDITIONER`.

**Lifecycle:** the dialog is shown via `show(dataset, display_bands, stretches)`.
It loads each band's normalized data, computes (or looks up) a histogram from the
[histogram cache](data-caching.md), and seeds the controls from any existing
stretches. As the user drags bounds or changes type/conditioner, it constructs
fresh stretch objects and emits:

```python
stretch_changed = Signal(int, tuple, list)   # (dataset_id, display_bands, stretches)
```

Emissions are gated by an `_enable_stretch_changed_events` flag so that
programmatic setup (loading existing state, linking sliders) does not fire
spurious updates.

For how `stretch_changed` reaches every `RasterView`, see the "How Changes Reach
the Screen" section of the [Rendering Pipeline](rendering-pipeline.md).
In short: `App` stores the stretches in `ApplicationState` (keyed per
`(ds_id, band_index)`) and re-emits a state-level `stretch_changed`, which
`RasterPane` turns into `RasterView.set_stretches()` calls.

---

## Applying Stretches During Rendering

### Per-band path

For each dirty channel, `RasterView`'s `update_display_image()` calls:

```python
stretches = self._stretches[i].get_stretches()   # [conditioner, main]
new_data = make_channel_image(band_data, stretches[0], stretches[1])
```

`make_channel_image()` applies the two stretches in order, clips to `[0, 1]`,
and returns `uint8`.

### Joint path (decorrelation)

Some stretches are inherently cross-band. `RasterView._detect_joint_stretch()`
walks the three channels and, for each, splits its stretch into
`(conditioner, main)` and checks `main.requires_all_bands()`. If **all three**
channels agree on the same joint stretch (by value equality), the renderer takes
the joint path; if they disagree, it logs a warning and falls back to per-band.

`_render_joint_channels()` then runs in three phases
(`src/wiser/gui/rasterview.py`):

1. **Gather + condition** — collect all bands into one `(H, W, 3)` float32
   buffer, applying each channel's conditioner in place. Masked (data-ignore)
   pixels are zeroed so conditioners stay in their valid domain.
2. **Joint compute (cached)** — if `_joint_render_cache` matches the current
   `(dataset_id, bands, conditioner_signature)` key, reuse it; otherwise call
   `joint_stretch.apply_multi(bands)` and cache the result.
3. **Scale + restore** — clip each channel to `[0, 1]`, scale to `uint8`, and
   restore masks.

`StretchDecorrelation.apply_multi()` delegates the heavy math to `decor_numba`
(`src/wiser/raster/decorrelation_stretch.py`),
then per-band-normalizes the result back into `[0, 1]`.

```{mermaid}
flowchart TD
    DETECT{"_detect_joint_stretch()<br/>all 3 channels agree?"}
    PB["Per-band:<br/>make_channel_image() x3"]
    G["Phase 1: gather (H,W,3)<br/>+ per-band conditioners"]
    JC{"_joint_render_cache<br/>matches key?"}
    REUSE["reuse cached joint result"]
    COMPUTE["apply_multi() (decor_numba)<br/>+ cache result"]
    S["Phase 3: clip, scale uint8,<br/>restore masks"]

    DETECT -->|no / disagree| PB
    DETECT -->|yes| G
    G --> JC
    JC -->|hit| REUSE
    JC -->|miss| COMPUTE
    REUSE --> S
    COMPUTE --> S
```

---

## Numba Dispatch

The pixel-level helpers (`make_channel_image`, `make_rgb_image`) and the stretch
classes each have a pure-Python implementation and a numba-compiled variant. The
numba version is used only when an array exceeds `ARRAY_NUMBA_THRESHOLD`
(`src/wiser/raster/utils.py`); below that, JIT warm-up costs
outweigh the benefit and the Python path is used. The numba wrappers also
convert `float64` inputs to `float32` for speed and memory. `StretchDecorrelation`
has no jitclass variant (a jitclass cannot hold the Python references it needs);
its numba acceleration lives inside `decor_numba` instead.
