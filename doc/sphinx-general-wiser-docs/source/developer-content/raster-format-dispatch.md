# Raster Format Dispatch

When WISER is handed a file path, something has to decide *which* of the raster
formats it knows about should open it. That decision is **format dispatch**, and
it lives in `src/wiser/raster/format_registry.py` and
`RasterDataLoader.load_from_file`.

This page documents how a path becomes a `RasterDataSet`, and — the part most
developers come here for — [how to add support for a new file
type](#adding-a-new-format).

---

## Overview

Dispatch is deliberately split into three stages, so that the cheap decisions
happen before the expensive ones and nothing irreversible happens while WISER is
still guessing.

| Stage | What it does | Cost |
|-------|--------------|------|
| **Order** | Rank the registered formats for this path, most likely first. | String comparison only. |
| **Identify** | Ask each candidate, in order, whether it recognizes the file. | A few bytes read per candidate. No handles opened. |
| **Open** | Open the file with the format that won. | One file handle, once. |

```{mermaid}
flowchart TB
    Path["load_from_file(path)"]
    Override{"format= given?"}
    One["Use that format only.<br/>Fail loudly if it doesn't work."]
    Order["candidates_for(path)<br/>extension-claimants first, then<br/>everything else, then the catch-all"]
    Ident["identify() down the list"]
    Yes{"Confidence.YES?"}
    Win["Winner"]
    Maybe["Highest-priority MAYBE"]
    Open["try_load_file() on the winner"]
    Build["FormatSpec.loader ->  RasterDataSet(s)"]

    Path --> Override
    Override -- yes --> One --> Open
    Override -- no --> Order --> Ident --> Yes
    Yes -- "stop immediately" --> Win --> Open
    Yes -- "none were certain" --> Maybe --> Open
    Open --> Build
```

### Why not just try every format?

That is what WISER used to do, and it caused three problems worth remembering,
because each one is a rule the current design exists to enforce:

- **The winner was whichever format was tried last.** The old loop had no
  `break`, so every format was attempted and the last success overwrote the
  earlier ones. Which implementation opened a file therefore depended on the
  declaration order of a dict.
- **Every successful attempt opened a file handle**, and all but one were
  discarded.
- **Identification could block on the user.** The NetCDF sub-dataset dialog ran
  inside the attempt, so it could be shown for a file that a later format then
  claimed, throwing away what the user chose.

Hence: exactly one format wins, `identify()` never opens or prompts, and only
the winner is opened.

---

## The registry

Every format WISER can open is one `FormatSpec` entry in `RASTER_FORMATS`:

```python
FormatSpec(
    name="ENVI",                                    # the override token
    impl=ENVI_GDALRasterDataImpl,                   # class that opens it
    extensions=frozenset({"", ".hdr", ".img", ".dat"}),
    priority=70,
    loader=load_normal_dataset,
    interactive_step=False,
)
```

| Field | Meaning |
|-------|---------|
| `name` | Stable identifier. Used by the `format=` override and **written into project files**, so renaming one is a breaking change. |
| `impl` | The `RasterDataImpl` subclass that opens this format. |
| `extensions` | Extensions this format claims. An **ordering hint only** (see below). |
| `priority` | Higher is tried earlier. Breaks ties between equally confident formats. |
| `loader` | How to turn the opened impl into datasets. Usually `load_normal_dataset`. |
| `interactive_step` | True if opening can block on a dialog. Informational today; marks what must stay on the main thread when loading moves to a worker. |
| `is_fallback` | True only for the GDAL catch-all, which always sorts last. |

### Extensions order, they do not gate

This is the single most important thing to understand about the registry.

A format is **never excluded** for failing to claim the path's extension. The
extension only decides *where in the queue* a format sits. `candidates_for()`
returns *every* registered format for *every* path — extension-claimants first
by priority, then everything else by priority, then the catch-all.

The reason is that extensions in this domain are genuinely ambiguous:

| Extension | Could be |
|-----------|----------|
| `.img` | ENVI raster **or** PDS3 |
| `.hdr` | ENVI raster header **or** ENVI spectral library |
| `.xml` | PDS4 label **or** any other XML |
| `.lbl` | PDS3 label |
| *(none)* | ENVI data file |

A strict extension-to-format map would be wrong on real data immediately, and
would make a file with an unusual extension unloadable even when WISER could
read it perfectly well. Ordering gives the speed and determinism of extension
dispatch without the brittleness.

---

## `identify()`

`RasterDataImpl.identify(path)` returns a `Confidence`:

| Value | Meaning | Effect on the search |
|-------|---------|----------------------|
| `YES` | Positive content match. | Wins immediately; the search stops. |
| `MAYBE` | Plausible, no positive signal. | Kept as a fallback if nothing is certain. |
| `NO` | Positively excluded. | Dropped. |

The base implementation returns `MAYBE` — "no opinion" — so an implementation
that doesn't override it stays in the running but never outranks a format that
is actually sure.

`identify()` must be cheap and side-effect-free. Specifically it must **not**:

- open a lasting handle or allocate significant memory,
- show a dialog or otherwise block on the user,
- raise for an unreadable or missing file — return `NO` instead.

### It delegates to GDAL

For GDAL-backed formats, `identify()` does **not** hand-roll magic-byte checks.
It calls `gdal.IdentifyDriverEx`, which runs the driver's own `Identify()`
without opening the dataset:

```python
@classmethod
def identify(cls, path: str) -> Confidence:
    drv = gdal.IdentifyDriverEx(
        cls.get_load_filename(path), allowed_drivers=list(cls.get_gdal_drivers())
    )
    return Confidence.YES if drv is not None else Confidence.NO
```

GDAL already maintains the magic-byte and label checks for every driver it
ships; duplicating them in WISER would only create something to drift.

```{note}
The Python binding keyword is `allowed_drivers`, not `allowedDrivers`, and
passing `None` raises rather than meaning "any driver". Omit the argument if you
want every driver considered.
```

### `priority` still matters

It is reasonable to ask why priority matters if the first `YES` wins. Because
`identify()` is not a perfect oracle: an `.img` with a PDS label sitting beside
an ENVI header will get a confident yes from **both** formats. `identify()`
decides where the walk *stops*; `priority` decides the order it *walks*, and
therefore which of two equally confident formats is asked first.

---

## Sidecar files: `DATA_EXTENSIONS` and `SIDECAR_EXTENSIONS`

Some formats are a header file plus a separately-named data file, and the user
may legitimately select either one. The file-open dialog offers `*.hdr` and
`*.tfw` filters, so this is a normal thing to happen, not an edge case.

Two class attributes describe that relationship:

```python
class ENVI_GDALRasterDataImpl(GDALRasterDataImpl):
    DATA_EXTENSIONS = ("", ".img", ".dat")   # the raster itself, in resolution order
    SIDECAR_EXTENSIONS = (".hdr",)           # what the user might click instead
```

`DATA_EXTENSIONS` does **two** jobs:

1. **Resolution.** `get_load_filename("scene.hdr")` tries `scene`, then
   `scene.img`, then `scene.dat`, and returns the first that exists. This runs
   before `identify()` — GDAL will not identify the header file itself, so
   skipping it would make WISER reject a dataset the user validly selected.
2. **Exclusion.** `identify()` returns `NO` for a path whose own extension is
   not in the list. This matters because GDAL's ENVI driver claims *any* file
   with a same-stem `.hdr` beside it — without the restriction, an unrelated
   `scene.tif` sitting next to `scene.hdr` would be identified as ENVI.

Set `DATA_EXTENSIONS = None` (the default) for a self-describing single-file
format, which imposes no extension restriction. That is the right choice when
the driver's own content check is strong and extensions vary in the wild — PDS3
and NetCDF both do this.

---

(adding-a-new-format)=
## Adding a new format

Adding support for a new file type is normally two steps.

### 1. Write the implementation

Subclass `GDALRasterDataImpl` in `src/wiser/raster/dataset_impl.py` and declare
which GDAL drivers it uses:

```python
class MyFormat_GDALRasterDataImpl(GDALRasterDataImpl):
    GDAL_DRIVERS = ["MYDRIVER"]

    # Only if the format has a header/data split, or you need to stop it
    # claiming unrelated files that share a basename:
    # DATA_EXTENSIONS = (".dat",)
    # SIDECAR_EXTENSIONS = (".hdr",)

    @classmethod
    def try_load_file(cls, path: str, **kwargs) -> ["MyFormat_GDALRasterDataImpl"]:
        gdal.UseExceptions()
        gdal_dataset = gdal.OpenEx(
            cls.get_load_filename(path),
            nOpenFlags=gdalconst.OF_READONLY | gdalconst.OF_VERBOSE_ERROR,
            allowed_drivers=cls.GDAL_DRIVERS,
        )
        if gdal_dataset is None:
            raise ValueError(f"Unable to open file: {path}")
        return [cls(gdal_dataset)]
```

Declaring `GDAL_DRIVERS` is all that is needed to get a working `identify()` —
the inherited implementation uses it. Pass the same list to `OpenEx` so that
identification and opening can never disagree about which driver is in play.

If the usable driver set can only be determined at run time (JPEG2000 support
varies by GDAL build), override `get_gdal_drivers()` instead of setting
`GDAL_DRIVERS`.

### 2. Register it

Add one `FormatSpec` to `RASTER_FORMATS` in `format_registry.py`:

```python
FormatSpec(
    name="MyFormat",
    impl=MyFormat_GDALRasterDataImpl,
    extensions=frozenset({".myf"}),
    priority=45,
),
```

Priorities are spaced by 5 so a format can be slotted between two existing ones
without renumbering. Choose one relative to formats it might be confused with:
if your format and an existing one can both open the same file, the higher
priority is asked first.

That is the whole registration. There is no second table to update — the spec
carries the impl, the ordering, and the dataset-construction strategy together.

### Non-GDAL formats

If the format is not read through GDAL — the `pdr` planetary-data library, for
instance — subclass `RasterDataImpl` directly and write `identify()` by hand.
Keep to the contract: cheap, no handles, no dialogs, `NO` rather than an
exception on unreadable input.

### If opening needs to ask the user something

Set `interactive_step=True` on the spec and give it a `loader` that runs the
dialog, as `load_FITS_dataset` does. Respect the `interactive` keyword in
`try_load_file`: when it is `False` — project restore, tests, headless use —
choose a sensible default instead of prompting.

### Checklist

- [ ] `GDAL_DRIVERS` declared (or `identify()` written by hand)
- [ ] Same driver list used by `try_load_file`
- [ ] `DATA_EXTENSIONS` set if the format has sidecars or a greedy driver
- [ ] `FormatSpec` added to `RASTER_FORMATS` with a considered `priority`
- [ ] `interactive=False` produces a non-prompting default
- [ ] File-open dialog filter added in `show_open_file_dialog` if users should be
      able to force this format (see below)
- [ ] Tests in `src/tests/test_format_registry.py`

---

## Overrides

Detection is only used when nobody has told WISER what the file is. Three
callers can state the format outright, and all three funnel into the same
`format=` parameter on `load_from_file`:

| Source | Where |
|--------|-------|
| Programmatic callers | `load_from_file(path, format="ENVI")` |
| The user's file-dialog filter | `show_open_file_dialog` in `src/wiser/gui/app.py` |
| A saved project | `format` field in the dataset manifest entry |

An override tries **only** that format and raises if it fails, rather than
quietly falling back to guessing — a wrong override is a mistake worth
surfacing, not one worth hiding.

### The file-open dialog

Filters are declared as `(text, format_name)` pairs. `QFileDialog` returns which
filter the user selected, and that selection is treated as a statement of what
the file is:

```python
supported_formats = [
    (self.tr("All supported files (...)"), None),   # None = detect
    (self.tr("ENVI raster files (*.img *.hdr *.dat)"), "ENVI"),
    ...
]
```

A filter that cannot name a single format — "All supported files", "Try luck
with GDAL" — maps to `None`, meaning ordinary detection. When adding a format,
add a filter for it only if a user could plausibly need to force it.

### Project files

`dataset_to_pyrep` records the format that opened each referenced dataset, and
the load path passes it back as the override. Without this, a project would
re-detect on every open, and a registry change or a new file appearing beside
the data could silently resolve the same path to a different implementation than
the one in use when the project was saved.

Both directions degrade gracefully: a manifest with no `format` field (written
by an older WISER) falls back to detection, and a `format` naming something no
longer registered logs a warning and does the same.

---

## Future work

**Splitting the interactive step out of `try_load_file`.** The NetCDF
sub-dataset dialog still runs inside `try_load_file`. Because only the winning
format is ever opened, it can no longer be shown speculatively or have its
result discarded — but it does mean the GUI step and the I/O are still in one
call, with no seam between them.

Separating them into `open()` (worker-thread safe) and `configure()` (main
thread only) is what would allow dataset loading to move off the UI thread:
identify and open on a worker, then marshal only `configure` back to the main
thread for the formats whose spec sets `interactive_step`. The registry already
records which formats those are.
