# Band Chooser

The Band Chooser is the dialog that lets the user pick which dataset band(s) are
displayed and how: three bands for RGB, or one band for grayscale (optionally
colorized with a matplotlib colormap). This page covers the dialog itself, the
automatic band-selection helpers it relies on, and how a selection propagates
into the [rendering pipeline](rendering-pipeline.md).

---

## Overview

A raster dataset can have anywhere from one to hundreds of bands. The renderer,
however, only ever shows **1 band (grayscale)** or **3 bands (RGB)**. The Band
Chooser is the UI that maps dataset bands onto those display slots.

- **GUI:** `BandChooserDialog` (`src/wiser/gui/band_chooser.py`)
- **Auto-selection helpers:** `find_truecolor_bands()` / `find_display_bands()`
  (`src/wiser/raster/dataset.py`)
- **Consumed by:** `RasterView.set_display_bands()`
  (`src/wiser/gui/rasterview.py`)

---

## `BandChooserDialog`

**File:** `src/wiser/gui/band_chooser.py`

**Purpose:** A modal `QDialog` for choosing display bands and colormap for one
dataset. It is purely a chooser — it reads the dataset to populate its controls
and reports the user's choice back to the caller; it does not apply anything
itself.

**Constructor:**

```python
BandChooserDialog(
    app_state: ApplicationState,
    dataset: RasterDataSet,
    display_bands: List[int],     # current bands: 1-tuple or 3-tuple
    colormap: Optional[str] = None,
    can_apply_global: bool = True,
    parent=None,
)
```

On construction the dialog:

- Populates the red/green/blue/gray band combo boxes from
  `dataset.band_list()`, labeling each entry with `dataset.get_band_label(i)`
  (which includes a `(bad)` marker for bad bands) and right-aligning the text
  when the dataset `has_wavelengths()`.
- Fills the colormap combo box from `matplotlib.pyplot.colormaps()`.
- Selects RGB or grayscale mode based on the length of `display_bands` and shows
  the matching page of the config stack.
- Configures the "choose defaults" / "choose visible-light" buttons based on
  dataset capabilities (`_configure_buttons()`).

**Public API (read by the caller after `exec_()`):**

| Method | Returns | Description |
|--------|---------|-------------|
| `get_display_bands()` | `tuple` | `(r, g, b)` in RGB mode, or `(gray,)` in grayscale mode (combo-box indices) |
| `get_colormap_name()` | `str \| None` | Colormap name if grayscale + colormap enabled, else `None` |
| `use_colormap()` | `bool` | True when grayscale mode and the "use colormap" box is checked |
| `apply_globally()` | `bool` | Whether the change should apply to all panes (the "apply globally" checkbox) |

**Controls:**
- RGB vs grayscale mode toggle and the corresponding band combo boxes.
- Colormap selection and a live preview gradient strip
  (`_on_grayscale_choose_colormap`, connected to the combo box's `activated`
  signal, samples the selected colormap at 256 points into a 256×24 `QImage`
  shown on `lbl_colormap_display`).
- The "apply globally" checkbox (disabled when `can_apply_global=False`).

**Does not control:**
- Applying the selection (the caller does this — see the Propagation section
  below).
- Stretches or rendering.
- Which dataset is shown.

---

## Automatic Band Selection

Two module-level helpers in `src/wiser/raster/dataset.py` decide
sensible defaults so the user rarely has to choose bands manually.

### `find_truecolor_bands(dataset, red, green, blue)`

Returns a `(red_index, green_index, blue_index)` triple of the bands closest to
the given visible-light wavelengths, or `None` if the dataset has no wavelength
metadata or a match cannot be found for any channel. The Band Chooser calls this
via `_get_truecolor_bands()`, passing the configured red/green/blue wavelengths
from app config (`general.red_wavelength_nm`, etc.).

### `find_display_bands(dataset, red, green, blue)`

Picks the bands to display when a dataset is first shown, in priority order:

1. The dataset's own `default_display_bands()` (may be a 3-tuple or 1-tuple).
2. Otherwise, `find_truecolor_bands()` if wavelengths are available.
3. Otherwise, `(0, 1, 2)` if the dataset has ≥ 3 bands, else `(0,)` (grayscale).

These helpers are also wired to the dialog's buttons: "choose visible-light
bands" calls `find_truecolor_bands()`, and "choose defaults" reads
`dataset.default_display_bands()`.

---

## Propagation

The key thing to understand is that **`BandChooserDialog` is passive**: it never
calls into the `RasterPane` or `RasterView`, and it changes no application state.
It is a modal data-collection widget. `RasterPane._on_band_chooser()`
(`src/wiser/gui/rasterpane.py`) constructs it, blocks on `exec_()`, and then —
once the user clicks OK — **pulls** the user's choices back out of the dialog
with its getters and decides what to do with them.

```{mermaid}
sequenceDiagram
    participant U as User
    participant RP as RasterPane
    participant BC as BandChooserDialog
    participant App as App (broker)
    participant RV as RasterView

    U->>RP: click "Band Chooser"
    RP->>BC: construct + exec_() (modal, blocks)
    U->>BC: pick bands / colormap, click OK
    Note over BC: passive - only records the choice in its own widgets, no app state
    BC-->>RP: exec_() returns QDialog.Accepted
    RP->>BC: get_display_bands() / get_colormap_name() / apply_globally()
    BC-->>RP: bands, colormap, is_global
    alt is_global is True (Apply globally checked)
        RP->>App: display_bands_change(ds_id, bands, colormap, is_global)
        App->>RP: set_display_bands(ds_id, bands, colormap) on every pane
        RP->>RV: set_display_bands(bands, stretches, colormap)
    else local (this pane only)
        RP->>RV: set_display_bands(bands, colormap=colormap)
    end
    RV->>RV: update_display_image()
```

You're right that the dashed return line is mostly getters — that is the whole
point. The dialog holds no application state, so nothing it does affects the pane
until the pane reads these values after the modal closes:

| Getter | Returns | Role |
|--------|---------|------|
| `get_display_bands()` | new band tuple (1 or 3 indices) | the new data to apply |
| `get_colormap_name()` | colormap name or `None` | the new data to apply |
| `apply_globally()` | `bool` | **routing decision only** — not data |

So `apply_globally()` is the only value that changes the pane's *behavior*
(global vs local); the other two carry the new bands/colormap. The pane then
routes them (`_on_band_chooser`, `src/wiser/gui/rasterpane.py`, the
`if dialog.exec_() == QDialog.Accepted:` block):

- **Global** (`apply_globally()` is `True`): emit
  `display_bands_change(ds_id, bands, colormap, is_global)`. `App` receives it and
  calls `set_display_bands` on every pane (context, main, zoom) so they stay in
  sync.
- **Local**: call this pane's own `set_display_bands(ds_id, bands, colormap)`
  directly, skipping the broadcast.

Both routes converge on `RasterPane.set_display_bands(ds_id, bands, colormap)`,
which loops over the pane's rasterviews and, for each one showing this dataset,
calls `RasterView.set_display_bands(bands, stretches, colormap)` — forwarding the
exact `bands` tuple that `BandChooserDialog.get_display_bands()` returned. (The
`stretches` are looked up from `ApplicationState` by the pane at this point so the
view only re-renders once.)

The same `BandChooserDialog` is reused elsewhere (e.g.
`similarity_transform_pane.py`, `geo_reference_pane.py`, and a plugin-facing
instance held by `ApplicationState`).

---

## `RasterView.set_display_bands()`

**File:** `src/wiser/gui/rasterview.py`

This is the method that connects the Band Chooser to an actual render. As traced
above, the `display_bands` argument it receives is exactly the tuple the user
chose in the dialog (`BandChooserDialog.get_display_bands()` →
`RasterPane.set_display_bands()` → here); `colormap` is
`BandChooserDialog.get_colormap_name()`, and `stretches` are the ones the pane
looked up from `ApplicationState`. The dialog itself is never referenced here —
by the time this runs, its values have already been extracted and passed in.

Its key behavior is a
**dirty-flag optimization**: it compares the old and new band tuples and builds
an `ImageColors` flag (`RED`, `GREEN`, `BLUE`, or all) describing which channels
actually changed, so `update_display_image()` only regenerates those channels.

```python
def set_display_bands(self, display_bands, stretches=None, colormap=None):
    # ... validate length is 1 or 3 ...
    changed = ImageColors.NONE
    # ... set RED/GREEN/BLUE flags for each band index that differs ...
    self._display_bands = display_bands
    self._colormap = colormap
    self._joint_render_cache = None   # band selection feeds joint stretch input
    self.update_display_image(colors=changed)
```

`ImageColors` (`src/wiser/gui/rasterview.py`) is an `IntFlag`
with `NONE=0`, `RED=1`, `GREEN=2`, `BLUE=4`, `RGB=7`. Switching the number of
display bands (RGB ↔ grayscale) or changing the single grayscale band marks all
channels dirty.

### RGB vs grayscale rendering

- **RGB (3 bands):** each channel is stretched and scaled independently
  (`make_channel_image`), then the three `uint8` channels are packed into a
  `uint32` ARGB image by `make_rgb_image`.
- **Grayscale (1 band):** the single stretched channel is duplicated across R, G,
  B, then `make_grayscale_image` either packs it as gray or, if a colormap is
  set, maps each `uint8` value through a 256-entry colormap lookup table.

For how the stretch portion of this works (per-band vs joint), see the
[Stretch Builder](stretch-builder.md). For how results are cached, see
[Data Caching](data-caching.md).
