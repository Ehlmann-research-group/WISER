# Filters and Smoothing

WISER has four filters for suppressing noise in a cube. They are on the raster
view's context menu, under **Filters**: right-click the image and choose the
one you want. There is no Tools-menu equivalent.

| Filter | Smooths along | Preserves |
|---|---|---|
| **Savitzky–Golay** | Wavelength | Absorption band shape and depth |
| **Mean** | Your choice of axis | Nothing in particular — fastest |
| **Median** | Your choice of axis | Edges; removes isolated spikes |
| **Gaussian** | Your choice of axis | Smooth structure; well-behaved weighting |

Each produces a **new dataset**; the input is untouched. Runs proceed in the
background — track them in the Activity Monitor.

```{admonition} Filter for a reason
:class: important
Smoothing throws information away. Filter when you have a specific noise
problem — a spectrum you cannot read, salt-and-pepper speckle, striping — not
as a routine first step. In particular, do not smooth before
{doc}`MNF <data-analysis-tools/mnf>`: MNF's noise estimate assumes the noise it
finds is the sensor's, and pre-smoothing invalidates that.
```

---

## Savitzky–Golay filter

The Savitzky–Golay filter fits a low-order polynomial to a sliding window of
each pixel's spectrum by least squares and replaces the centre value with the
fitted one. Because it fits a curve rather than averaging, it removes noise
while leaving the **position, shape and depth** of absorption bands close to
intact — exactly what a moving average destroys.

This is the standard smoothing step in spectroscopy, and the right choice
whenever you intend to measure a band afterwards.

Right-click the image and choose **Filters ▸ Savitzky–Golay Filter...**.

:::{figure} ../_static/tutorials/t9_savgol_dialog.png
:width: 45%
:align: center
:alt: The Savitzky-Golay filter dialog with window length, polynomial order and dataset
:::

| Setting | Meaning |
|---|---|
| **Window Length (odd)** | Number of bands in the fitting window. Must be odd, and greater than the polynomial order. Larger smooths more. |
| **Polynomial Order** | Degree of the fitted polynomial. 2 or 3 is usual. |
| **Choose Dataset** | The cube to filter. |

**Choosing the parameters.** Start with a window of 5–9 bands and order 2 or 3.
The window should be **narrower than the narrowest feature you care about** — a
21-band window will flatten a 15-band absorption. Raising the order preserves
more curvature at the cost of removing less noise.

The filter runs along the **spectral** axis only.

---

## Mean, median and Gaussian smoothing

These three share one dialog, reached by right-clicking the image and choosing
**Filters ▸ Mean / Median / Gaussian Smoothing Filter...**.

:::{figure} ../_static/tutorials/t9_smooth_dialog.png
:width: 50%
:align: center
:alt: The smoothing filter dialog showing axis, mode, constant value, sigma and radius
:::

| Setting | Meaning |
|---|---|
| **Input Dataset** | The cube to filter |
| **Axis** | **Spectral** — along wavelength, per pixel. **Spatial** — within each band, across the image. |
| **Mode** | Edge handling: `reflect`, `constant`, `nearest`, `mirror`, `wrap` |
| **Constant Value (cval)** | The value used outside the edge when **Mode** is `constant` |
| **Sigma** | Gaussian only — standard deviation of the kernel |
| **Radius / Truncate** | Gaussian only — where to cut the kernel off, as a radius in samples or a multiple of sigma |
| **Size** | Mean and median only — the window width |

WISER remembers the parameters you typed for each axis separately, so switching
between **Spectral** and **Spatial** does not lose what you entered for the
other.

### Which one

- **Mean** — cheapest. Blurs edges and is pulled around by outliers.
- **Median** — removes isolated spikes and dead pixels without blurring
  boundaries. The right choice for salt-and-pepper noise, and for a spectrum
  with a few bad bands you have not flagged.
- **Gaussian** — weights nearby samples more than distant ones, giving a
  smoother result than a mean filter at the same width and no ringing. The
  default choice for general spatial smoothing.

### Spectral or spatial

- **Spectral** smoothing reduces noise in each pixel's spectrum, which is what
  you want before reading a shallow absorption band. Prefer **Savitzky–Golay**
  here — same job, far less damage to band shape.
- **Spatial** smoothing trades resolution for signal-to-noise: each band's
  image gets cleaner and every pixel's spectrum becomes a mixture of its
  neighbours'. Reasonable on a coarse scene where you want regional patterns;
  destructive when you care about small targets.

```{tip}
Averaging in the {doc}`Spectrum Plot <spectra-and-libraries>` — the **Number of
pixels to average** setting, mean or median over an *n* × *n* box — gives you
the effect of spatial smoothing on the spectra you are looking at, without
producing a whole new dataset. Use it when you only want to *read* a spectrum
more clearly.
```

---

## See also

- {doc}`Minimum Noise Fraction <data-analysis-tools/mnf>` — separates signal
  from noise rather than blurring both
- {doc}`Tutorial 2 — Reading Spectra <../tutorials/02-spectra>`
- {doc}`Lab F — Close-Range Materials Imaging <../tutorials/labs/lab-materials-imaging>` —
  where Savitzky–Golay earns its place
