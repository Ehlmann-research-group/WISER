# Interface Overview

A tour of the main window and the settings that apply across every tool.

```{tip}
To *use* these controls rather than read about them, start with
{doc}`Tutorial 1 — Your First Scene <../tutorials/01-first-look>`, which drives
the panes, band chooser and contrast stretch on real data.
```

## The main window

When WISER starts, no data is loaded and both display areas read **(no data)**:

:::{figure} ../_static/tutorials/t1_empty.png
:width: 80%
:align: center
:alt: WISER at startup with no data loaded
:::

The WISER interface provides multiple panes for displaying raster data at
varying levels of magnification. The Context Pane starts out on the left side
of the UI, and shows the raster data "scaled out," so that either one or both
dimensions are fully visible within the pane. The primary viewing area is
called the Main Window, providing more detailed interactions with raster data,
possibly scaled up to as much as 1600%. In the above screenshot, no raster
data is loaded yet, so these areas display "(no data)".

Across the top of the Main Window is the Main Toolbar, which provides various
tools to work with raster data:

:::{figure} ../_static/images/main_toolbar.png
:width: 70%
:align: center
:alt: The WISER main toolbar
:::

The buttons marked "Display Toggles" will show and hide specific tools for
interacting with spectral data. These buttons are as follows:

:::{figure} ../_static/images/display_toggles.png
:width: 40%
:align: center
:alt: Display toggle buttons in the main toolbar
:::

These tools are described in subsequent sections.

**NOTE:** WISER can also be extended with custom functionality through its
plugin API — see the [Extending WISER](../extending-wiser/index) section of
this documentation.

## WISER Configuration

WISER provides a configuration panel for specifying common configuration across
the various tools. You can access these properties through the WISER menubar.
For example, on macOS you can access "WISER" -> "Preferences" to show this
dialog:

:::{figure} ../_static/images/wiser_config.png
:width: 45%
:align: center
:alt: WISER preferences dialog
:::

These settings are saved on disk so that they don't need to be specified every
time. Some additional details are given in the following sections.

### Appearance and Color Scheme

WISER can be shown in a light or a dark color scheme. The **Color scheme**
setting, in the **Appearance** group at the top of the configuration dialog,
controls this:

:::{figure} ../_static/images/color_scheme_config.png
:width: 45%
:align: center
:alt: The Color scheme drop-down in the Appearance group of the WISER configuration dialog
:::

There are three choices:

*   **System** (the default) — WISER follows your operating system's light/dark
    setting, and automatically updates if you change your OS theme.
*   **Light** — always use the light color scheme, regardless of the OS setting.
*   **Dark** — always use the dark color scheme, regardless of the OS setting.

WISER adjusts the window colors, the toolbar icons and the selection highlight
so that everything stays legible in either scheme. The startup window shown while
WISER loads follows the same setting. The change is applied when you click
**OK**, and is remembered between sessions.

:::{figure} ../_static/images/color_scheme_light.png
:width: 80%
:align: center
:alt: WISER shown in the light color scheme
:::

:::{figure} ../_static/images/color_scheme_dark.png
:width: 80%
:align: center
:alt: WISER shown in the dark color scheme, with light-tinted toolbar icons
:::

### WISER Crash and Error Reporting

WISER is capable of sending crash reports to an online service called
[BugSnag](https://www.bugsnag.com). This option is **off** by default, but
it is very helpful if you turn this feature on so that application errors and
crashes can be identified and addressed automatically. No personally
identifying information is sent to BugSnag, but some users may still not want
to leave such a feature on.

### Wavelengths for Red/Green/Blue Colors

For spectral data sets that include visible-light frequencies, WISER is able to
automatically choose "true-color" bands that are close to the frequencies of
red, green and blue light. However, different data sets and instruments may
require tweaking of what is considered "red", "green" or "blue". Thus, WISER
allows the user to configure these values.

---

## Viewing an Image

Here is WISER after loading AVIRIS data over the Caltech campus, with every
pane shown:

:::{figure} ../_static/tutorials/t1_all_panes.png
:width: 80%
:align: center
:alt: WISER displaying AVIRIS Caltech data with all panes visible and annotated
:::

In this image, all of the different tools have been shown using the display
toggle buttons in the main toolbar:  the context pane, the main window and the
zoom pane, as well as the spectral plot window and the dataset information
window.
All of these components are dockable, and can be moved or resized within the
WISER user interface. They can also be undocked from the UI, so that they
appear as separate windows. Arrange WISER's user interface however you like it
best!
As the snapshot indicates, the area visible in the zoom pane is indicated in the
main window. Correspondingly, the area visible in the main window is indicated
in the context window.  _(Tip:  The color of this viewport highlight can be
changed in the WISER configuration dialog.)_  Mouse-clicks or scrolling within
the various display windows will update the other windows.
Mouse clicks within the main or zoom windows will update the spectrum plot window
with the pixel's spectrum.

---

## Toolbars

Every raster display area has its own toolbar. The main toolbar carries all of
these; the context and zoom panes carry the subset that applies to them.

:::{figure} ../_static/images/main_display_buttons.png
:width: 30%
:align: center
:alt: The dataset toolbar buttons
:::

| Control | Does |
|---|---|
| **Dataset chooser** | Switches which loaded dataset the pane shows |
| **Band chooser** | RGB or grayscale, which bands, and the colormap — see {doc}`Display and Contrast Stretch <display-and-stretch>` |
| **Contrast stretch** | Opens the stretch builder for the displayed bands |
| **Grid** | Splits the main window into panels; the dataset, band and stretch controls then move above each panel |
| **Link** | Ties panning and zooming across panels. Requires every open dataset to have the same width and height |
| **Zoom in / out / to fit / to 100%** | Navigation. The percentage box takes a value directly |
| **Create ROI** and **selection tools** | See {doc}`Regions of Interest <regions-of-interest>` |
| **ROI dropdown** | Chooses which ROI a new selection is added to |

---

## The status bar

The bar along the bottom reports, left to right:

- A **message area** — WISER's running commentary, and the instructions for
  whichever selection tool is active. Read it when a shape is not behaving.
- The **display values** for each colour channel at the pixel under the cursor
- The **pixel coordinate**, as `(x, y)`
- The **ground coordinate**, when the dataset is georeferenced

The two buttons at the far right open the **Dataset Info** pane and the
**Activity Monitor**.

---

## The Activity Monitor

Analyses, filters and band-math jobs run in the background so the interface
stays usable. The **Activity Monitor** is where you watch them.

It lists **Active Tasks** with a progress bar and a **cancel** button each, and
keeps a collapsible **Finished Tasks** list below. Each row names the task and
the parameters it was given — the dataset, the component count, the expression
— which makes it a usable record of what produced which output.

Failed tasks stay in the list with their error message, so a run that produced
no dataset can be diagnosed rather than guessed at.

Open it from the button at the right of the status bar.

---

## The Dataset Info pane

The **Dataset Info** display toggle shows the header metadata of every loaded
dataset: dimensions, band count, data type, wavelengths, bad-band list, data
ignore value, coordinate reference system and description.

Check it whenever a tool behaves unexpectedly. Most surprises — a missing
wavelength axis, an unset data ignore value, bands you did not know were
flagged — are visible here.

---

## Editing a dataset

Right-click an image and choose **Edit dataset...** to change a dataset's
metadata **in place**, without writing a new file: its name, description, data
ignore value, wavelengths, bad-band list and default display bands.

This is the fix for a file whose header is wrong or incomplete — a cube with no
wavelengths, or one whose nodata value was never declared, so WISER treats
`-9999` as a real measurement.

To write the change to disk use **Save as...** — see
{doc}`Saving and Exporting <saving-and-exporting>`.

---

## Where to go next

- {doc}`Tutorial 1 — Your First Scene <../tutorials/01-first-look>` — these
  controls used on real data
- {doc}`Opening Data Files <opening-data-files>` — formats and troubleshooting
- {doc}`Display and Contrast Stretch <display-and-stretch>` — bands, colormaps,
  grid view, stretches
- {doc}`Extending WISER <../extending-wiser/index>` — add your own tools to
  these menus
