# Band Math

Band math evaluates an arithmetic expression over your data and adds the result
as a new dataset. It is how you compute an index, build a mask, ratio two
scenes, or apply any per-pixel formula WISER has no dedicated tool for.

Open it with **Tools ▸ Band math...**.

:::{figure} ../_static/tutorials/t4_bandmath_dialog.png
:width: 90%
:align: center
:alt: The band math dialog with an NDVI expression, two bound variables and the operator help panel
:::

For a worked example see
{doc}`Tutorial 4 <../tutorials/04-band-math-ndvi>`; for the same thing on a
425-band cube, {doc}`Lab A <../tutorials/labs/lab-aviris-ng-urban>`.

---

## How a run works

1. **Type an expression** and press Enter.
2. WISER parses it. Every name it does not recognise as a function becomes a
   **variable** and gets a row in **Variable bindings**.
3. **Bind each variable** — choose its type, then the dataset, band or spectrum
   it refers to.
4. Give the output a **Result name**.
5. Click **OK**.

Above the table, WISER reports the **result type and size** — for example
`Result: Image Band, 150x150 (87.9KB)`. Read it before running anything large.

**Toggle Help** opens a reference panel listing every operator, beside the
expression you are writing.

---

## Operators

| Category | Operators |
|---|---|
| Arithmetic | `+` `-` `*` `/` |
| Power and root | `**` — `x ** 2` squares, `x ** 0.5` is a square root |
| Negation | unary `-` |
| Comparison | `==` `!=` `<` `>` `<=` `>=` |
| Grouping | `( )` |

Standard precedence applies: `3 * 2 + 4` is 10, `3 * (2 + 4)` is 18.

Comparisons return **1 where the test passes and 0 where it fails**, which is
how you build masks:

```text
ndvi > 0.35
```

Multiply masks for a logical AND:

```text
(ndvi > 0.35) * (band4 < 0.2)
```

## Built-in functions

```text
sin  cos  tan  arcsin  arccos  arctan  arctan2  dotprod
```

```{admonition} That is the complete list
:class: warning
There is **no** built-in `sqrt`, `log`, `exp`, `abs`, `min` or `max`. Use
`** 0.5` for a square root. For anything else, write a
{doc}`band-math plugin <../extending-wiser/bandmath_plugins>` — a plugin
registers new functions in this dialog without rebuilding WISER, and is the
supported way to extend the language.
```

`dotprod` takes the dot product of two operands — useful for projecting a cube
onto a spectrum.

## Variables

Names must start with a letter or underscore and may contain letters, digits
and underscores. **Names are case-insensitive**: `NIR`, `nir` and `Nir` are the
same variable.

Each variable binds to one of:

| Type | Binds to | Shape |
|---|---|---|
| **Image** | A whole dataset | `[band][y][x]` |
| **Image Band** | One band of a dataset | `[y][x]` |
| **Spectrum** | A collected spectrum or a library entry | `[band]` |

Numeric literals may be used directly and need no binding.

Types combine as you would expect — a cube times a number is a cube; a cube
divided by a spectrum divides every pixel's spectrum band-by-band. WISER checks
compatibility when it parses and tells you before running if shapes do not
match.

---

## Saved expressions

- **Save expression** — adds it to the **Saved expressions** dropdown, for this
  session and later ones
- **Save to file... / Load from file...** — writes the saved set out or reads
  one in, so a group can share a common library of indices

Saved expressions are also stored in a {doc}`project <projects>`.

---

## Batch processing

Tick **Enable Batch Processing** to apply one expression to **every raster in a
folder**.

1. Set an **input folder**. Variables can then be bound to *batch* types —
   `Image Cube Batch` or `Image Band Batch` — which resolve against each file
   in turn.
2. For a batch band variable, pick the band by **index** or by **wavelength**
   with a tolerance. Choose wavelength when the files come from different
   sensors or have different band counts.
3. Set an **output folder** and a **result suffix**.
4. Choose whether results are **loaded into WISER** as well as written to disk
   — for a large batch you usually do not want a hundred datasets open.
5. **Create batch job**, then start it. Jobs run with progress bars and can be
   cancelled; per-file errors are collected and shown by **View errors**.

Variables can be mixed: bind one to a batch folder and another to a single
fixed dataset, for example to ratio every scene in a folder against one
reference.

---

## Memory

```{admonition} Band math is not streamed for every case
:class: note
Band math loads its operands into memory. WISER shows the expected result size
before you run and chunks large image-cube expressions where it can, but a
full-size flight line with several intermediates can still exhaust available
RAM. Check the reported size, subset spatially before you start
(**Save as...**, Dimensions tab), and prefer **Image Band** operands to whole
**Image** operands when the expression only needs a few bands.
```

---

## Common expressions

| Purpose | Expression | Notes |
|---|---|---|
| Normalised difference (NDVI, NDWI, NDSI…) | `(a - b) / (a + b)` | Bind `a` and `b` to the two bands |
| Simple ratio | `a / b` | Iron oxide: red / blue |
| Band depth | `1 - c / (0.5 * s1 + 0.5 * s2)` | `c` the band centre, `s1`/`s2` the shoulders |
| Mask | `a > 0.35` | 1 where true, 0 where false |
| Combined mask | `(a > 0.35) * (b < 0.2)` | Logical AND |
| Reflectance from raw | `(s - d) / (w - d)` | Sample, dark, white-reference cubes |
| Ratio to a reference spectrum | `cube / spec` | `spec` bound as a Spectrum |

---

## See also

- {doc}`Tutorial 4 — Band Math: Mapping Vegetation <../tutorials/04-band-math-ndvi>`
- {doc}`Band-Math Plugins <../extending-wiser/bandmath_plugins>` — add your own
  functions
- {doc}`Band Math Internals <../developer-content/bandmath-internals>`
