# Viewport System

This page documents the internal architecture of WISER's image-display system: how the three visible panes (Context, Main, Zoom) are built, how they interact, and what each class is and is not responsible for.

## Overview

The viewport system consists of two parallel class hierarchies that work together:

- **Display hierarchy** — low-level Qt widgets that render a single raster image with pan/zoom.
  `ImageWidget` → `ImageScrollArea` → `RasterView` → `TiledRasterView`

- **Pane hierarchy** — higher-level container widgets that manage one or more `RasterView` instances and expose Qt signals to the rest of the application.
  `RasterPane` (base) → `ContextPane` / `ZoomPane` / `MainViewWidget`

`RasterPane` **contains** `TiledRasterView` instances in an MxN grid. The `App` class (`app.py`) wires the panes together by connecting their signals to slots — panes themselves have no direct references to each other.

---

## Class Hierarchy

```{mermaid}
classDiagram
    direction TB

    class QWidget["QWidget (Qt)"]
    class QScrollArea["QScrollArea (Qt)"]

    class ImageWidget {
        rasterview.py
        +paintEvent()
        +mousePressEvent()
        +mouseMoveEvent()
        +keyPressEvent()
    }

    class ImageScrollArea {
        rasterview.py
        +scrollContentsBy()
        +wheelEvent()
        +_propagate_scroll : bool
    }

    class RasterView {
        rasterview.py
        +_raster_data : RasterDataSet
        +_scale_factor : float
        +_image_pixmap : QPixmap
        +set_raster_data()
        +scale_image()
        +get_visible_region()
        +image_coord_to_raster_coord()
        +raster_coord_to_image_coord()
    }

    class TiledRasterView {
        rasterpane.py
        +_position : tuple[int,int]
        +_toolbar : QToolBar
        +show_toolbar()
        +hide_toolbar()
    }

    class RasterPane {
        rasterpane.py
        +_rasterviews : dict
        +_num_views : tuple[int,int]
        +visibility_change : Signal
        +viewport_change : Signal
        +click_pixel : Signal
        +display_bands_change : Signal
        +roi_selection_changed : Signal
        +views_changed : Signal
        +show_dataset()
        +set_scale()
        +set_pixel_highlight()
        +set_viewport_highlight()
    }

    class ContextPane {
        context_pane.py
        +_ds_id : int
        +_act_fit_to_window : QAction
        +_update_image_scale()
    }

    class ZoomPane {
        zoom_pane.py
        +_zoom_in_scale()
        +_zoom_out_scale()
    }

    class MainViewWidget {
        main_view.py
        +_link_view_state : GeographicLinkState
        +_link_view_scrolling : bool
        +_init_rasterviews()
        +_sync_scroll_state()
    }

    QWidget <|-- ImageWidget : subclass
    QScrollArea <|-- ImageScrollArea : subclass
    QWidget <|-- RasterView : subclass
    RasterView <|-- TiledRasterView : subclass
    QWidget <|-- RasterPane : subclass
    RasterPane <|-- ContextPane : subclass
    RasterPane <|-- ZoomPane : subclass
    RasterPane <|-- MainViewWidget : subclass

    RasterView --> ImageWidget : contains
    RasterView --> ImageScrollArea : contains
    RasterPane --> TiledRasterView : contains (MxN grid)
```

---

## Class Responsibilities

### ImageWidget

**File:** `src/wiser/gui/rasterview.py`

**Purpose:** The innermost Qt widget — paints the image `QPixmap` on screen and forwards user-input events upward.

**Controls:**
- Rendering the current `QPixmap` via `paintEvent`
- Middle-mouse-button pan (translates the scroll area)
- Forwarding mouse, keyboard, and context-menu events through the `forward` dictionary

**Does not control:**
- What image is drawn (set externally via `RasterView`)
- Scrollbars or zoom level
- Data loading or stretch computation

---

### ImageScrollArea

**File:** `src/wiser/gui/rasterview.py`

**Purpose:** Manages the scroll area that wraps `ImageWidget`, handles `Ctrl+scroll` zoom, and optionally propagates scroll events to other linked views.

**Controls:**
- Horizontal and vertical scrollbar state
- Intercepting `Ctrl+wheel` and converting it to a zoom call on the parent `RasterView`
- The `_propagate_scroll` flag that prevents infinite scroll-sync loops between linked views

**Does not control:**
- What zoom level results from `Ctrl+wheel` (delegated to `RasterView.scale_image_around_point`)
- Image content
- Signal emission

---

### RasterView

**File:** `src/wiser/gui/rasterview.py`

**Purpose:** The core single-viewport widget. Owns the full rendering pipeline from raw data through stretch application to a displayed `QPixmap`, and manages zoom, scroll, and coordinate conversions.

**Controls:**
- Loading and caching the rendered `QPixmap` (`update_display_image`)
- Zoom scale factor (`_scale_factor`), accessible via `get_scale()` / `scale_image()`
- Coordinate conversion between raster, image, and display spaces
- Reporting the currently visible region of the dataset (`get_visible_region`)
- Scrollbar synchronization for linked-view scrolling (`get_scrollbar_state` / `set_scrollbar_state`)

**Does not control:**
- Qt signal emission — all events are forwarded to the parent `RasterPane` via the `forward` dict
- Drawing highlights (viewport boxes, pixel crosshairs, ROI overlays) — done by `RasterPane`
- Multi-view layout decisions

**Key attributes:**

| Attribute | Type | Description |
|-----------|------|-------------|
| `_raster_data` | `RasterDataSet` | Dataset currently displayed |
| `_display_bands` | `tuple` | Band indices to render (1 or 3 elements) |
| `_stretches` | `list[StretchBase]` | Per-band contrast stretch objects |
| `_scale_factor` | `float` | Current zoom (1.0 = 100%) |
| `_image_pixmap` | `QPixmap` | Cached rendered image |
| `_colormap` | `str \| None` | Colormap name for single-band display |

---

### TiledRasterView

**File:** `src/wiser/gui/rasterpane.py`

**Purpose:** Extends `RasterView` with a per-view toolbar and a grid position `(row, col)`, making it suitable for use inside `RasterPane`'s MxN view grid.

**Controls:**
- Per-view toolbar visibility (`show_toolbar` / `hide_toolbar`)
- Keeping the dataset-chooser combo box in sync when the displayed dataset changes

**Does not control:**
- Rendering, zoom, or coordinate conversion (all in `RasterView`)
- Layout or grid decisions (managed by `RasterPane`)
- Signal emission

---

### RasterPane

**File:** `src/wiser/gui/rasterpane.py`

**Purpose:** Base class for all three visible panes. Owns the grid of `TiledRasterView` instances, the main toolbar, all Qt signals, highlight drawing, and ROI tools.

**Controls:**
- The MxN grid of `TiledRasterView` objects (`_rasterviews` dict keyed by `(row, col)`)
- All Qt signal definitions (see Signal Reference section below)
- Receiving forwarded events from child `RasterView` instances and re-emitting them as signals.
  Each `TiledRasterView` is constructed with the following `forward` dict pointing back to `RasterPane` handlers:

  ```python
  forward = {
      "mousePressEvent":   self._onRasterMousePress,
      "mouseReleaseEvent": self._onRasterMouseRelease,
      "mouseMoveEvent":    self._onRasterMouseMove,
      "keyPressEvent":     self._onRasterKeyPress,
      "keyReleaseEvent":   self._onRasterKeyRelease,
      "contextMenuEvent":  self._onRasterContextMenu,
      "paintEvent":        self._afterRasterPaint,    # overlay drawing hook
      "scrollContentsBy":  self._afterRasterScroll,   # emits viewport_change
  }
  ```
- Drawing overlays on top of the raster image: viewport-highlight boxes, pixel crosshairs, ROI polygons
- ROI creation, editing, deletion, and export operations
- Toolbar setup (zoom combo box, band chooser, stretch builder, selection tools)

**Does not control:**
- Subclass-specific behaviors (overridden by `ContextPane`, `ZoomPane`, `MainViewWidget`)
- Inter-pane communication (wired in `App`)
- What datasets are loaded (managed by `ApplicationState`)

---

### ContextPane

**File:** `src/wiser/gui/context_pane.py`

**Purpose:** Overview/minimap pane that shows the full extent of the dataset at a zoomed-out scale and draws colored highlight boxes indicating the visible regions of other panes.

**Controls:**
- Automatically fitting the image to the widget size (`_act_fit_to_window` toggle)
- Dataset lock mode: either "use clicked dataset" (follows what was last clicked in any pane) or locked to a specific dataset ID (`_ds_id`)
- Rendering viewport-highlight boxes for the main view and/or zoom pane

**Does not control:**
- Scrolling (the image always fits the window; there is no free scrolling)
- Which dataset the main view or zoom pane display
- Signal wiring

---

### ZoomPane

**File:** `src/wiser/gui/zoom_pane.py`

**Purpose:** High-magnification pane (1–16×, default 8×) for precise pixel-level inspection.

**Controls:**
- Magnification range: 1–16× (integer steps only, unlike the main view's ×1.25 multiplier)
- Zoom increment behavior: `_zoom_in_scale` adds 1, `_zoom_out_scale` subtracts 1

**Does not control:**
- Which pixel is centered (set by `App` in response to main-view clicks via `set_pixel_highlight`)
- Signal wiring

---

### MainViewWidget

**File:** `src/wiser/gui/main_view.py`

**Purpose:** The primary image viewer. Supports multi-view grid layouts (1×1, 1×2, 2×1, 2×2, or custom MxN) and linked scrolling between views with optional geographic coordinate transformation.

**Controls:**
- View layout grid (`_init_rasterviews`, `_on_split_views`)
- Linked-scrolling mode: `NO_LINK`, `PIXEL` (same-dimension sync), or `SPATIAL` (geographic coordinate transform)
- Zoom range: 0.25×–16× in ×1.25 increments
- Synchronizing scroll positions across views when linked (`_sync_scroll_state`)
- ROI/selection tools (rectangle, polygon, multi-pixel)

**Does not control:**
- What the context pane or zoom pane display (set by `App` view signals that depend on what is clicked in MainView)
- Signal wiring (done in `App`)

---

## Signal Reference

All six signals are defined on `RasterPane` and inherited by `ContextPane`, `ZoomPane`, and `MainViewWidget`.

| Signal | Arguments | When emitted |
|--------|-----------|--------------|
| `visibility_change` | `bool` | Pane is shown (`True`) or hidden (`False`) |
| `views_changed` | `tuple (rows, cols)` | Multi-view layout changes |
| `display_bands_change` | `int, tuple, object, bool` | Display bands or colormap changes; the `bool` is `True` if the change should propagate to all panes |
| `click_pixel` | `tuple (row, col), QPoint` | User clicks a pixel; `QPoint` is in dataset coordinates |
| `roi_selection_changed` | `ROI, Selection \| None` | An ROI selection is added, changed, or deleted |
| `viewport_change` | `tuple (row, col) \| None` | Scroll position changes; `None` means all views |

---

## Signal Flow

The `App` class acts as the signal broker — panes never call each other directly.

Each scenario below reads top-to-bottom: a pane emits a signal, the named
`App` handler runs, and the handler calls methods back on the other panes.
`App` sits in the center lane because every signal passes through it.

```{mermaid}
sequenceDiagram
    participant CP as ContextPane
    participant Main as MainViewWidget
    participant App as App (broker)
    participant Zoom as ZoomPane
    participant Plot as Spectrum Plot

    Note over CP,Plot: Main view scrolled
    Main->>App: viewport_change<br/>(_on_mainview_viewport_change)
    App->>CP: set_viewport_highlight()

    Note over CP,Plot: Zoom pane scrolled
    Zoom->>App: viewport_change<br/>(_on_zoom_viewport_change)
    App->>Main: set_viewport_highlight()

    Note over CP,Plot: Pixel clicked in main view
    Main->>App: click_pixel<br/>(_on_mainview_raster_pixel_select)
    App->>Zoom: set_pixel_highlight() + center on pixel
    App->>CP: show_dataset()
    App->>Plot: push pixel spectrum

    Note over CP,Plot: Pixel clicked in context pane
    CP->>App: click_pixel<br/>(_on_context_raster_pixel_select)
    App->>Main: make_point_visible()

    Note over CP,Plot: Pixel clicked in zoom pane
    Zoom->>App: click_pixel<br/>(_on_zoom_raster_pixel_select)
    App->>Main: set_pixel_highlight() (respects link mode)
    App->>Plot: push pixel spectrum

    Note over CP,Plot: Bands / colormap changed (global)
    Main->>App: display_bands_change(global=True)<br/>(_on_display_bands_change)
    App->>Main: set_display_bands()
    App->>Zoom: set_display_bands()
    App->>CP: set_display_bands()
```

### Interaction flows in plain language

**Main view scrolled** — `viewport_change` → `App._on_mainview_viewport_change` → `ContextPane.set_viewport_highlight()` updates the colored box in the context pane showing where the main view is looking.

**Zoom pane scrolled** — `viewport_change` → `App._on_zoom_viewport_change` → `MainViewWidget.set_viewport_highlight()` updates the small box drawn on the main view showing where the zoom pane is looking.

**User clicks a pixel in the main view** — `click_pixel` → `App._on_mainview_raster_pixel_select` → centers and highlights the pixel in the zoom pane, updates the context pane dataset, and pushes the pixel spectrum to the spectrum plot.

**User clicks a pixel in the context pane** — `click_pixel` → `App._on_context_raster_pixel_select` → `MainViewWidget.make_point_visible()` scrolls the main view so the clicked location is visible.

**User clicks a pixel in the zoom pane** — `click_pixel` → `App._on_zoom_raster_pixel_select` → highlights the pixel in the main view (respecting linked-scrolling mode) and pushes the spectrum.

**Band or colormap changes in any pane** — `display_bands_change(global=True)` → `App._on_display_bands_change` → propagates the change to all three panes so they stay in sync.

---

## Coordinate Systems

Three coordinate spaces are used throughout the codebase. Conversion methods live on `RasterView`.

| Space | Description | Unit |
|-------|-------------|------|
| **Raster / dataset** | Raw pixel indices in the loaded `RasterDataSet` | pixel (col, row) |
| **Image** | Screen pixels as if zoom = 1.0 (one screen pixel per dataset pixel) | pixel |
| **Display** | Actual on-screen pixels at the current zoom level | pixel |

**Conversion methods on `RasterView`:**

- `image_coord_to_raster_coord(pos)` — image → raster
- `raster_coord_to_image_coord(coord)` — raster → image
- Image ↔ display: multiply or divide by `_scale_factor`

```{mermaid}
flowchart LR
    R["Raster coords\n(dataset pixels)"]
    I["Image coords\n(at zoom = 1.0)"]
    D["Display coords\n(on screen)"]

    R -- "raster_coord_to_image_coord()" --> I
    I -- "image_coord_to_raster_coord()" --> R
    I -- "× scale_factor" --> D
    D -- "÷ scale_factor" --> I
```

---

## Supporting Infrastructure

### Event forwarding via `forward` dict

`RasterView` is constructed with an optional `forward` dictionary mapping event names (e.g., `"mousePressEvent"`) to callables. When `ImageWidget` or `ImageScrollArea` fires an event, `RasterView` looks it up in `forward` and calls the corresponding handler on the parent `RasterPane`. This decouples the low-level widgets from the pane without requiring additional subclasses.

### Rendering cache

`RasterView` caches the rendered `QPixmap` (`_image_pixmap`) and per-band stretched arrays (`_display_data`). The cache is only invalidated and regenerated when `update_display_image()` is called explicitly — for example, when bands, stretches, or the dataset change. Scrolling and zooming reuse the cached pixmap.

### Task delegates

`RasterPane` uses a `_task_delegate` attribute for modal, multi-step operations such as drawing a new ROI selection or editing an existing geometry. While a task delegate is active, other mouse events are routed to it rather than processed as normal clicks.
