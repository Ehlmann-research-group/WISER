# WISER Overview

The goal of WISER is to provide an intuitive and configurable user interface
that supports many different workflows and styles of interaction.  When WISER
is started, the UI looks like this:

<img class="img_center" src="../_static/images/wiser_start.png" width="80%">

The WISER interface provides multiple panes for displaying raster data at
varying levels of magnification.  The Context Pane starts out on the left side
of the UI, and shows the raster data "scaled out," so that either one or both
dimensions are fully visible within the pane.  The primary viewing area is
called the Main Window, providing more detailed interactions with raster data,
possibly scaled up to as much as 1600%.  In the above screenshot, no raster
data is loaded yet, so these areas display "(no data)".

Across the top of the Main Window is the Main Toolbar, which provides various
tools to work with raster data:

<img class="img_center" src="../_static/images/main_toolbar.png" width="70%">

The buttons marked "Display Toggles" will show and hide specific tools for
interacting with spectral data.  These buttons are as follows:

<img class="img_center" src="../_static/images/display_toggles.png" width="40%">

These tools are described in subsequent sections.

**NOTE:** WISER can also be extended with custom functionality through its
plugin API — see the [Extending WISER](../extending-wiser/index) section of
this documentation.

## WISER Configuration

WISER provides a configuration panel for specifying common configuration across
the various tools.  You can access these properties through the WISER menubar.
For example, on macOS you can access "WISER" -> "Preferences" to show this
dialog:

<img class="img_center" src="../_static/images/wiser_config.png" width="30%">

These settings are saved on disk so that they don't need to be specified every
time.  Some additional details are given in the following sections.

### WISER Crash and Error Reporting

WISER is capable of sending crash reports to an online service called
[BugSnag](https://www.bugsnag.com).  This option is **off** by default, but
it is very helpful if you turn this feature on so that application errors and
crashes can be identified and addressed automatically.  No personally
identifying information is sent to BugSnag, but some users may still not want
to leave such a feature on.

### Wavelengths for Red/Green/Blue Colors

For spectral data sets that include visible-light frequencies, WISER is able to
automatically choose "true-color" bands that are close to the frequencies of
red, green and blue light.  However, different data sets and instruments may
require tweaking of what is considered "red", "green" or "blue".  Thus, WISER
allows the user to configure these values.
