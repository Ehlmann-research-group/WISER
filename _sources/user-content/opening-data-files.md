# Opening Data Files

WISER reads several raster formats natively and can fall back to any format GDAL
supports. Most of the time you can pick a file and WISER works out the rest.

This page covers the cases where it is worth being deliberate — because **the
file type you choose in the Open dialog matters**, and so does *which* file you
pick when a dataset is spread across more than one.

---

## The file type you choose matters

The **file type** dropdown at the bottom of the Open dialog is not only a way of
hiding files you are not interested in. It also tells WISER **what the file is**.

- Leaving it on **All supported files** asks WISER to work the format out for
  itself. This is the right choice almost always.
- Choosing a **specific type** — "ENVI raster files", "NetCDF raster files",
  "PDS3 raster files", and so on — tells WISER to open the file *as that format*
  and nothing else.

### Why this matters

Some files can be read by more than one format. A `.img` file might be an ENVI
raster or a PDS3 product; an `.xml` file might be a PDS4 label or something else
entirely. WISER inspects the contents rather than trusting the extension, and it
is right the overwhelming majority of the time.

But when a file is genuinely ambiguous — or when it is unusual enough that
automatic detection picks a format that reads it, just not the way you want —
picking the type explicitly settles it.

```{tip}
If a file opens but the data looks wrong — wrong number of bands, missing
wavelengths, unexpected scaling — try opening it again with the file type set
explicitly. Automatic detection may have chosen a format that *can* read the
file but does not understand all of its metadata.
```

### Forcing a type will report failure rather than guess again

Choosing a specific file type is treated as a statement of fact. If WISER cannot
read the file as that format, it reports an error rather than quietly trying
something else.

That is deliberate: if you have said a file is NetCDF and it is not, having it
silently open as something else would hide the problem rather than tell you
about it. Switch back to **All supported files** to let WISER decide.

---

## Which file to pick for multi-file datasets

Several formats store one dataset across more than one file. You may select
either the header or the data file — WISER finds the other.

### ENVI

An ENVI dataset is a **header** (`.hdr`) plus a **data file**, which may have no
extension at all, or `.img`, or `.dat`:

```
scene.hdr     <- the header
scene.dat     <- the data (or "scene", or "scene.img")
```

Select either one. Both files must sit in the same directory and share a name;
WISER cannot open the header on its own.

```{note}
If you have an ENVI header whose data file is missing or has been renamed to
something unexpected, WISER reports which filenames it looked for. Renaming the
data file to match the header — same name, one of the extensions above — is
usually enough to fix it.
```

### GeoTIFF

A GeoTIFF is normally a single `.tif` or `.tiff` file. Some are accompanied by a
`.tfw` world file carrying the georeferencing. Select either the image or the
world file.

### PDS

PDS3 products may have a **detached label** (`.lbl`) beside the data, or an
**attached label** inside a single `.img`. Select the label when there is one.
PDS4 products use an `.xml` label — select that.

---

## Formats WISER opens

| Format | Typical files |
|--------|---------------|
| ENVI raster | `*.img`, `*.hdr`, `*.dat`, or no extension |
| TIFF / GeoTIFF | `*.tif`, `*.tiff`, `*.tfw` |
| NetCDF | `*.nc` |
| JPEG 2000 | `*.JP2` |
| PDS3 | `*.PDS`, `*.img`, `*.lbl` |
| PDS4 | `*.xml` |
| FITS | `*.fits`, `*.fit`, `*.fts` |
| ASCII Grid | `*.asc` |
| ENVI spectral library | `*.sli`, `*.hdr` |
| Anything else GDAL reads | HDF4/5, GRIB, COG, and more |

If your file is not in this list, try **Try luck with GDAL** in the file-type
dropdown. GDAL supports a great many formats beyond those WISER handles
specially, and this option asks it to open the file with whatever driver fits.
Data loaded this way may carry less metadata — wavelengths and band names in
particular — than a natively supported format.

---

## Files containing several datasets

Some files hold more than one dataset. A NetCDF file, for example, may contain
several variables, each a separate raster.

When you open such a file, WISER asks which one you want before loading. Pick
the sub-dataset you are interested in and it is loaded as its own dataset in the
session.

If you save your session as a project, the specific sub-dataset you chose is
recorded, so reopening the project restores exactly what you had rather than
asking again.

---

## Troubleshooting

**"Couldn't load file ... unsupported format"**
: No format WISER knows could read the file. Check the file is complete and not
  truncated. For ENVI data, check its `.hdr` is present and beside it. As a last
  resort try **Try luck with GDAL**.

**"Can't find the raster file corresponding to ..."**
: You selected an ENVI header or a `.tfw` world file, but the matching data file
  is not beside it. The message lists the filenames WISER looked for.

**The file opens, but as the wrong kind of data**
: Open it again with the file type set explicitly instead of **All supported
  files**.

**A file that used to open no longer does**
: Check whether files have been added or renamed alongside it. A stray header
  file with the same base name as an unrelated raster can change how a directory
  is interpreted.
