# Data Caching

Rendering a raster is expensive: reading and normalizing bands, applying
stretches, and packing pixels all cost real time, and the same work is requested
repeatedly as the user scrolls, switches panes, or reopens dialogs. WISER avoids
recomputation with a three-tier in-memory cache. This page documents what each
tier stores, how keys and eviction work, the cache lifecycle, and a known issue
to be aware of.

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

## Known Issue: `Cache.add_cache_item` Guard

```{warning}
The base `Cache.add_cache_item`
(`src/wiser/raster/data_cache.py`) currently guards its insert
with `if key in self._cache:` and returns `False` otherwise:

​```python
def add_cache_item(self, key, value) -> bool:
    if key in self._cache:        # <-- only updates EXISTING keys
        ...
        self._cache[key] = value
        self._size += value.nbytes
        return True
    return False                  # <-- a brand-new key is never stored
​```

Callers add an item only **after** a cache miss (`in_cache(key)` returned
`False`), so the key is never already present — which means new entries are
never stored. As written, the **render cache and computation cache do not
actually serve hits**: every render recomputes from scratch. The condition is
almost certainly meant to be `if key not in self._cache:`.

`HistogramCache` overrides `add_cache_item` *without* this guard, so the
histogram cache works as intended.

This page documents the *intended* design (the caches are meant to store on
miss and serve on subsequent requests). The note is here so readers aren't
misled by the current code; fixing the guard is a separate change.
```
