# Data Caching

Rendering a raster is expensive: reading and normalizing bands, applying
stretches, and packing pixels all cost real time, and the same work is requested
repeatedly as the user scrolls, switches panes, or reopens dialogs. WISER avoids
recomputation with a three-tier in-memory cache. This page documents what each
tier stores, how keys and eviction work, the cache lifecycle, and how storing on
a miss keeps the running size and eviction in step.

This is the caching half of the [Rendering Pipeline](rendering-pipeline.md).

---

## Overview

All caching lives in `src/wiser/raster/data_cache.py`. A single
`DataCache` object is created once at startup and holds three caches:

| Cache | Class | Stores | Default capacity |
|-------|-------|--------|------------------|
| Render | `RenderCache` | Final packed `uint32` ARGB images | 10 GB |
| Computation | `ComputationCache` | Raw and normalized band arrays, full image arrays | 10 GB |
| Histogram | `HistogramCache` | `(bins, edges)` tuples for the Stretch Builder | 100 MB |

All three subclass a generic `Cache` base that provides size-bounded storage,
integer hash keys, and per-dataset bulk eviction.

```{mermaid}
classDiagram
    direction TB

    class Cache {
        data_cache.py
        +_capacity : int
        +_size : int
        +_cache : OrderedDict
        +_key_lookup_table : dict
        +add_cache_item(key, value) bool
        +get_cache_item(key)
        +in_cache(key) bool
        +remove_cache_item(key)
        +clear_keys_from_partial(partial_key)
        +_evict()
        +get_cache_key(*args)*
        +get_partial_key(dataset)*
    }
    class RenderCache {
        key = hash(dataset, *bands, *stretches, colormap)
    }
    class ComputationCache {
        key = hash(dataset, band_index, normalized)
    }
    class HistogramCache {
        key = hash(dataset, band, stretch_type, conditioner_type, min, max)
        stores (bins, edges) tuples
    }
    class DataCache {
        +get_render_cache()
        +get_computation_cache()
        +get_histogram_cache()
    }

    Cache <|-- RenderCache
    Cache <|-- ComputationCache
    Cache <|-- HistogramCache
    DataCache o-- RenderCache
    DataCache o-- ComputationCache
    DataCache o-- HistogramCache
```

---

## The Base `Cache`

**File:** `src/wiser/raster/data_cache.py`

**Purpose:** A size-bounded key→array store backed by an `OrderedDict`.

**Controls:**
- **Capacity by bytes** — `_capacity` (bytes) and a running `_size`. When adding
  an item would exceed capacity, `_evict()` removes items.
- **FIFO eviction** — `_evict()` pops from the front of the `OrderedDict`
  (`popitem(last=False)`), i.e. oldest-inserted first, until back within
  capacity. (This is insertion-order FIFO, not true LRU — reads do not refresh
  an item's position.)
- **Integer hash keys** — `get_cache_key(*args)` is abstract; each subclass
  hashes the inputs that uniquely identify a cached value.
- **Per-dataset bulk eviction** — every `get_cache_key` also records the key
  under a *partial key* (`get_partial_key(dataset)` = `hash(dataset)`) in
  `_key_lookup_table`. `clear_keys_from_partial(partial_key)` then removes every
  entry belonging to one dataset in a single call.

**Does not control:**
- What gets cached or when (callers decide).
- Thread safety (see the Threading section below).

### Cache keys at a glance

| Cache | `get_cache_key(...)` hashes |
|-------|-----------------------------|
| `RenderCache` | `(dataset, *band_tuple, *stretches, colormap)` |
| `ComputationCache` | `(dataset, band_index, normalized)` |
| `HistogramCache` | `(dataset, band_index, stretch_type, conditioner_type, min_bound, max_bound)` |

Because stretch objects implement `__hash__`/`__eq__` by value (see
[Stretch Builder](stretch-builder.md)), two equal stretches map to the same
render-cache entry.

### `HistogramCache` differences

`HistogramCache` stores a **tuple** `(bins, edges)` rather than a single array,
so it overrides `add_cache_item`, `_evict`, `clear_cache`, and
`remove_cache_item` to sum the `nbytes` of both tuple elements. Its default
capacity is only 100 MB because histograms are tiny relative to images.

---

## Lifecycle

```{mermaid}
flowchart LR
    APP["App.__init__<br/>DataCache()"] --> AS["ApplicationState<br/>.set_data_cache()"]
    AS --> LOAD["loader.py<br/>RasterDataSet(impl, data_cache)"]
    LOAD --> USE["dataset / rasterview<br/>read + write caches"]
    USE --> RM["ApplicationState.remove_dataset()<br/>clear_keys_from_partial()"]
```

1. **Creation** — `App.__init__` constructs one `DataCache()` and hands it to
   `ApplicationState.set_data_cache()` (`src/wiser/gui/app.py`).
2. **Attachment** — when a dataset is loaded, the `RasterDataSet` is constructed
   with that cache (`src/wiser/raster/loader.py`), reachable via
   `dataset.get_cache()`.
3. **Population (computation cache)** — `get_band_data()`,
   `get_band_data_normalized()`, and `get_image_data()`
   (`src/wiser/raster/dataset.py`) check the computation cache, and
   on a miss read from the underlying `RasterDataImpl` (GDAL/PDS/NumPy),
   normalize, and store.
4. **Population (render cache)** — `RasterView.update_display_image()` checks the
   render cache before stretching, and stores the packed `uint32` image after.
5. **Population (histogram cache)** — the Stretch Builder looks up/stores
   histograms keyed by dataset, band, stretch type, conditioner, and bounds.
6. **Invalidation** — `ApplicationState.remove_dataset()` removes the dataset's
   computation entry and calls `render_cache.clear_keys_from_partial(...)` to
   drop all its rendered images. Within a view, changing bands or stretches nulls
   `RasterView._joint_render_cache` (see [Stretch Builder](stretch-builder.md));
   the render cache itself is keyed by bands+stretches, so a new combination
   naturally maps to a different entry.

### `BandStats` — a separate, smaller cache

Independently of `DataCache`, each `RasterDataSet` keeps a per-band
`BandStats` (min/max) cache in `_cached_band_stats`
(`src/wiser/raster/dataset.py`). It is populated the first time a
band is read and is what lets normalization reuse min/max without rescanning the
array.

---

## Threading

The caches use a plain `OrderedDict` with **no locks**. This is safe only
because all cache access happens on the Qt main (GUI) thread — rendering, data
reads, and dialog interactions are all synchronous on that thread. Any future
background loading or worker-thread rendering would need to add synchronization
around `add_cache_item` / `get_cache_item` / `_evict`, since `OrderedDict`
compound operations are not atomic.

---

## Lookup / Miss / Store Flow

```{mermaid}
flowchart TD
    REQ["caller needs value<br/>(image / band / histogram)"]
    KEY["get_cache_key(...)<br/>also records partial key"]
    IN{"in_cache(key)?"}
    HIT["get_cache_item(key)"]
    MISS["compute value<br/>(read/normalize/stretch)"]
    ADD["add_cache_item(key, value)<br/>evict if over capacity"]
    OUT["use value"]

    REQ --> KEY --> IN
    IN -->|yes| HIT --> OUT
    IN -->|no| MISS --> ADD --> OUT
```

---

## Storing on a Miss

`Cache.add_cache_item` stores the entry whether or not the key is already
present. That is what lets the render and computation caches serve hits at all:
callers only reach it *after* a lookup missed, so the key is never already there
and an insert-only-if-present guard would drop every new entry.

Two details keep `_size` honest, and eviction with it:

- **A replacement subtracts before it adds.** Re-storing a key that is already
  in the cache gives back the old value's `nbytes` before adding the new
  value's, so an overwritten key is counted once rather than twice. The
  refreshed entry moves to the back of the `OrderedDict`, so FIFO eviction
  treats it as the newest item rather than the oldest.
- **Eviction runs after the insert.** `_evict()` is called once the new item is
  in and `_size` includes it, so the loop frees enough room for the incoming
  value instead of measuring a size that does not yet count it. A value larger
  than the entire capacity is refused up front — `add_cache_item` returns
  `False` and nothing is evicted on its behalf.

`src/tests/test_data_cache.py` covers the miss/store/hit cycle for the render
and computation caches, the size arithmetic across replacement and removal, and
FIFO eviction at capacity.
