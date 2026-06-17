# Spectrum Plot System

This page documents the internals of WISER's spectrum-plot subsystem: the plot
widgets that render spectra, the Region-of-Interest (ROI) model that defines
*which* pixels a spectrum is drawn from, how a spectrum is actually *computed*
from raster data, and how those pieces are wired together through application
state. It is a guide for developers reading, debugging, or extending any part of
this pipeline. Please keep it up to date.

## Overview

Three cooperating layers turn raster pixels into a plotted line:

- **Model layer** (`src/wiser/raster/`) — `RegionOfInterest` and `Selection`
  describe a set of pixels; the functions in `spectrum.py` aggregate those
  pixels' band values into a single spectrum; the `Spectrum` class hierarchy
  wraps the result behind a common, lazily-evaluated interface.
- **Application state** (`src/wiser/gui/app_state.py`) — `ApplicationState`
  owns the "active spectrum", the list of "collected" spectra, and loaded
  spectral libraries. It broadcasts changes through Qt signals. No widget holds
  a direct reference to another; they communicate only through these signals.
- **GUI layer** (`src/wiser/gui/spectrum_plot.py`) — `SpectrumPlotGeneric` is a
  reusable matplotlib-backed plot widget; `SpectrumPlot` subclasses it and binds
  it to `ApplicationState`.

The central idea: **ROIs and pixel clicks produce `Spectrum` objects; app state
broadcasts them via signals; the plot widget renders them with matplotlib.**

```{mermaid}
flowchart TB
    subgraph Model["Model layer (wiser/raster)"]
        ROI["RegionOfInterest / Selection"]
        CALC["spectrum.py calc_* functions"]
        SPEC["Spectrum objects"]
    end
    subgraph State["App state"]
        AS["ApplicationState (signals)"]
    end
    subgraph GUI["GUI layer (wiser/gui)"]
        PLOT["SpectrumPlot / SpectrumPlotGeneric"]
    end

    ROI --> CALC --> SPEC
    SPEC --> AS
    AS -- "signals" --> PLOT
```

---

## Plot widget class hierarchy

All of the following live in
`src/wiser/gui/spectrum_plot.py`. The two widget
classes are `SpectrumPlotGeneric` (the reusable base) and `SpectrumPlot` (the
app-bound subclass); the rest are small helpers each responsible for one drawing
or UI concern.

```{mermaid}
classDiagram
    direction TB

    class QWidget["QWidget (Qt)"]

    class SpectrumPlotGeneric {
        spectrum_plot.py :492
        +_collected_spectra : List~Spectrum~
        +_spectrum_display_info : Dict
        +closed : Signal
        +add_collected_spectrum()
        +set_title() / set_legend()
        +set_x_range() / set_y_range()
        +_add_spectrum_to_plot()
        +_draw_spectra()
    }

    class SpectrumPlot {
        spectrum_plot.py :1679
        +_dataset : RasterDataSet
        +_on_active_spectrum_changed()
        +_on_collected_spectra_changed()
        +_on_spectral_library_added()
        +_on_collect_spectrum()
    }

    QWidget <|-- SpectrumPlotGeneric : subclass
    SpectrumPlotGeneric <|-- SpectrumPlot : subclass
```

The helper classes the plot composes:

```{mermaid}
classDiagram
    direction TB

    class SpectrumPlotGeneric {
        spectrum_plot.py :492
    }
    class SpectrumPlotCanvas {
        spectrum_plot.py :124
        matplotlib FigureCanvas + context menu
    }
    class SpectrumDisplayInfo {
        spectrum_plot.py :151
        one matplotlib Line2D per spectrum
        +generate_plot()
        +remove_plot()
    }
    class SpectrumPointDisplayInfo {
        spectrum_plot.py :281
        selected-point crosshair + label
        +generate_plot()
    }
    class SpectrumPlotDatasetChooser {
        spectrum_plot.py :427
        toolbar dataset dropdown
    }

    SpectrumPlotGeneric --> SpectrumPlotCanvas : contains
    SpectrumPlotGeneric --> SpectrumDisplayInfo : one per spectrum
    SpectrumPlotGeneric --> SpectrumPointDisplayInfo : zero or one (selection)
    SpectrumPlot --> SpectrumPlotDatasetChooser : contains (toolbar)
```

**Plotting library.** The widget uses **matplotlib** with the `Qt5Agg` backend.
Each visible spectrum is a matplotlib `Line2D` managed by a `SpectrumDisplayInfo`
(`:151`); the X-axis shows wavelengths when the spectra carry them, and falls
back to band index otherwise. Display configuration (title, legend placement,
font sizes, axis ranges, tick intervals, selection-marker style, default
area-average mode) is driven by `SpectrumPlotConfigDialog`
(`src/wiser/gui/spectrum_plot_config.py:15`).

---

## `SpectrumPlot` vs `SpectrumPlotGeneric`

This is the most important distinction in the subsystem, and the most commonly
misunderstood.

A common misconception is that `SpectrumPlotGeneric` is fully decoupled from
application state. It is **not** — *both* classes receive an `app_state` in their
constructor. The real difference is the **data source for collected spectra**, as
the in-code comment at `spectrum_plot.py:557-559` states:

> This class gets its collected spectra from a list while the child class
> `SpectrumPlot` gets its collected spectrum from app_state.

- `SpectrumPlotGeneric` reads from its own `_collected_spectra: List[Spectrum]`,
  populated **manually** by calling `add_collected_spectrum(spectrum)` (`:1366`).
  It connects to *no* app-state signals.
- `SpectrumPlot` connects to `ApplicationState` signals in its constructor
  (`:1694-1700`) and **syncs automatically**: the active spectrum via
  `_on_active_spectrum_changed` (`:1901`), the collected list via
  `_on_collected_spectra_changed` (`:1959`), and spectral libraries via
  `_on_spectral_library_added` (`:1835`) / `_on_spectral_library_removed`.

| Aspect | `SpectrumPlotGeneric` | `SpectrumPlot` |
| --- | --- | --- |
| Collected-spectra source | own `_collected_spectra` list | `app_state.get_collected_spectra()` via signals |
| Active spectrum | none | synced from `app_state.get_active_spectrum()` |
| App-state signal wiring | none | `active_spectrum_changed`, `collected_spectra_changed`, `spectral_library_added/removed`, `dataset_removed` |
| Dataset-chooser toolbar | no | yes (`SpectrumPlotDatasetChooser`) |
| Collect-spectrum / load-library actions | no | yes |
| Spectral libraries | not shown | loaded, shown, hidden |
| Plot rendering, axis/tick/font/legend config, mouse selection | yes (defined here) | inherited |

### When to use which

- **Use `SpectrumPlotGeneric`** when you have your own set of `Spectrum` objects
  and want a plot window without app-state integration — e.g. a Tools-menu
  plugin or a results dialog that compares a target against library matches.
- **Use `SpectrumPlot`** for the main application's live spectrum window, where
  the plot must follow the user's active spectrum, collected spectra, and loaded
  libraries automatically.

### Using `SpectrumPlotGeneric`

Instantiate it directly, add `Spectrum` objects, and apply any display config via
the setters. A real example is the spectral-computation results dialog in
`src/wiser/gui/generic_spectral_tool.py` (around `:897`):

```python
plot_widget = SpectrumPlotGeneric(
    app_state=self._app_state,
    parent=self.parent(),
)
layout.addWidget(plot_widget)

# Push spectra in directly — no signals involved.
plot_widget.add_collected_spectrum(target)
for rec in rows[:5]:
    plot_widget.add_collected_spectrum(rec["ref_obj"])
```

Configuration setters available from the base class include `set_title()`,
`set_legend()`, `set_x_range()` / `set_y_range()`, `set_x_label()` /
`set_y_label()`, the tick-interval setters, and `set_font_size()` /
`set_font_name()`.

---

## Region of Interest (ROI) model

An ROI is a *composite* of one or more `Selection`s, so a single named region can
mix rectangles, polygons, and loose pixels. The aggregation in
`get_all_pixels()` deduplicates across selections, so overlapping selections
never double-count a pixel.

```{mermaid}
classDiagram
    direction TB

    class RegionOfInterest {
        raster/roi.py :12
        +_selections : List~Selection~
        +name / color / id
        +get_all_pixels() Set
        +get_bounding_box()
    }

    class Selection {
        raster/selection.py :31
        +get_type() SelectionType
        +get_all_pixels()
    }

    class SelectionType {
        <<enumeration>>
        SINGLE_PIXEL
        MULTI_PIXEL
        RECTANGLE
        POLYGON
        PREDICATE
    }

    RegionOfInterest --> Selection : contains many
    Selection --> SelectionType : tagged by
```

Each concrete `Selection` subtype implements `get_all_pixels()` to yield its
pixel coordinates; `RegionOfInterest.get_all_pixels()` unions them into a single
deduplicated set of `(x, y)` tuples.

---

## Spectrum class hierarchy

A `Spectrum` is the common, lazily-evaluated wrapper that the plot consumes. The
abstract base defines the interface (`get_spectrum()`, `get_wavelengths()`,
`get_bad_bands()`, `num_bands()`, color/name/id). Dataset-derived spectra defer
the actual computation until the first `get_spectrum()` call and cache the result.

```{mermaid}
classDiagram
    direction TB

    class Spectrum {
        raster/spectrum.py :175
        <<abstract>>
        +get_spectrum() ndarray
        +get_wavelengths()
        +get_bad_bands()
        +num_bands()
    }

    class RasterDataSetSpectrum {
        raster/spectrum.py :487
        <<abstract>>
        +_spectrum (cached, lazy)
        +_calculate_spectrum()
        +_reset_internal_state()
    }

    class ROIAverageSpectrum {
        raster/spectrum.py :706
        +set_avg_mode()
    }

    class SpectrumAtPoint {
        raster/spectrum.py :629
        single pixel / area average
    }

    class NumPyArraySpectrum {
        raster/spectrum.py :336
        wraps a precomputed array
    }

    Spectrum <|-- RasterDataSetSpectrum : subclass
    Spectrum <|-- NumPyArraySpectrum : subclass
    RasterDataSetSpectrum <|-- ROIAverageSpectrum : subclass
    RasterDataSetSpectrum <|-- SpectrumAtPoint : subclass
```

**Lazy evaluation.** `RasterDataSetSpectrum.get_spectrum()` computes `_spectrum`
only on first access by calling the subclass's `_calculate_spectrum()`, then
caches it. Changing configuration (e.g. `ROIAverageSpectrum.set_avg_mode()` or a
`SpectrumAtPoint` area change) calls `_reset_internal_state()`, which invalidates
the cache so the next `get_spectrum()` recomputes. This means constructing a
spectrum is cheap; the cost is paid the first time it is actually plotted.

For example, `ROIAverageSpectrum._calculate_spectrum()` (`:733`) simply delegates
to the calculation pipeline below:

```python
def _calculate_spectrum(self):
    self._spectrum = calc_roi_spectrum(self._dataset, self._roi, self._avg_mode)
```

---

## How spectra are calculated

The aggregation math lives in
`src/wiser/raster/spectrum.py`. A spectrum over multiple
pixels is the **per-band mean or median** across those pixels, selected by
`SpectrumAverageMode` (`MEAN` or `MEDIAN`). Both reductions are **NaN-aware**
(`np.nanmean` / `np.nanmedian`, `:96-108`), so bad bands and data-ignore values
(stored as NaN) are excluded without any explicit masking.

The performance-sensitive part is *reading* the pixels. Accessing a raster pixel
by pixel is slow, so `calc_spectrum_fast` (`:50`, wrapped by `calc_roi_spectrum`
at `:163`) converts the ROI into a small number of rectangular blocks and reads
each block in one call:

```{mermaid}
flowchart TB
    A["ROI"] --> B["create_raster_from_roi() — binary mask"]
    B --> C["RLE rectangle-pack along X axis"]
    B --> D["RLE rectangle-pack along Y axis"]
    C --> E["pick the axis with fewer rectangles"]
    D --> E
    E --> F["read each block: get_all_bands_at_rect()"]
    F --> G["stack pixel spectra"]
    G --> H["np.nanmean / np.nanmedian per band"]
    H --> I["1-D spectrum array"]
```

The run-length-encoding (RLE) rectangle packing is implemented in
`src/wiser/raster/roi_utils.py`
(`raster_to_combined_rectangles_x_axis` / `_y_axis`); the code packs along both
axes and keeps whichever produces fewer rectangles. For a plain rectangular
region there is a simpler single-block path, `calc_rect_spectrum` (`:113`), which
reads the whole rectangle at once and reduces over the spatial dimensions.

> **Threading note.** These calculations run synchronously on the Qt event loop.
> For very large ROIs the first `get_spectrum()` can briefly block the UI; there
> is no background worker for spectrum computation today.

---

## End-to-end data flow

Putting it together, here is the path from "user asks for an ROI's average
spectrum" to "a line appears on the plot":

```{mermaid}
flowchart TB
    A["User: 'Show ROI average spectrum' (RasterPane)"]
    B["_on_show_roi_avg_spectrum() — rasterpane.py:1792"]
    C["build ROIAverageSpectrum(dataset, roi)"]
    D["app_state.set_active_spectrum() — app_state.py:776"]
    E["emit active_spectrum_changed"]
    F["SpectrumPlot._on_active_spectrum_changed() — :1901"]
    G["spectrum.get_spectrum() — lazy calc_roi_spectrum()"]
    H["SpectrumDisplayInfo.generate_plot()"]
    I["matplotlib Line2D drawn (wavelengths or band index)"]

    A --> B --> C --> D --> E --> F --> G --> H --> I
```

The **pixel-click path** is analogous: clicking a pixel in a raster view builds a
`SpectrumAtPoint` (single pixel, or an area average around it) and likewise calls
`set_active_spectrum()`, so it flows through the same signal and rendering steps.
"Collecting" the active spectrum (`_on_collect_spectrum`, `:1947`) promotes it
into the collected list, which emits `collected_spectra_changed` and adds a
second, persistent line.

---

## Key signals reference

`SpectrumPlot` listens to these `ApplicationState` signals
(`src/wiser/gui/app_state.py`):

| Signal | Fired when | Plot handler |
| --- | --- | --- |
| `active_spectrum_changed` | active spectrum set/cleared (`set_active_spectrum`, `:776`) | `_on_active_spectrum_changed` (`:1901`) |
| `collected_spectra_changed` | a spectrum is added to / removed from the collected list | `_on_collected_spectra_changed` (`:1959`) |
| `spectral_library_added` | a spectral library is loaded | `_on_spectral_library_added` (`:1835`) |
| `spectral_library_removed` | a spectral library is unloaded | `_on_spectral_library_removed` (`:1868`) |
| `dataset_removed` | a dataset is closed | `_on_dataset_removed` (`:1751`) |

The plot widget itself emits one signal: `closed` (`spectrum_plot.py:497`),
fired when the plot window is closed so the rest of the app can react.
