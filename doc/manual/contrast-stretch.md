# Contrast Stretch

WISER supports manipulating the contrast stretch of data sets being displayed.
This is often very helpful to bring out important details in the data.

Before discussing how stretch may be applied, it is important to understand what
WISER must do to display data.  The band data being displayed may be of many
different data types; floating point or integer data, of various bit widths
(e.g. 8, 16, 32 or 64 bits).  Therefore, the data itself may cover many
different ranges of values.  The Workbench must map each band's values to an
integer color value in the range [0, 255].  If the Workbench is showing image
data in RGB mode, this must occur for each color channel; if the Workbench is
using grayscale then only one band is being used and it is mapped to a single
[0, 255] value.

Note that raster data sets may specify "data ignore values" - values that should
explicitly be ignored by tools working with the data.  The Workbench filters out
such values before any of this processing occurs; they are represented as NaNs,
and all stretch calculations are implemented to ignore NaN values throughout.

Finally, note that the calculations for displaying raster data are not used
when plotting spectra; spectrum calculations use the raster data directly.

## Specifying Minimum and Maximum Limits on Band Data

The Contrast Stretch UI provides users the ability to specify minimum and
maximum limits on band data.  This can be useful when raster data doesn't
specify a "data ignore value," or when raster data is computed and has a large
range of possible values, but only a small range of "useful" or "interesting"
values.

The min/max limits are applied before any histograms are computed in the
Contrast Stretch UI.  Note that values outside of the min/max limits are simply
filtered out; they are not clamped to the min/max limits.  This is particularly
important for N% linear stretches and histogram-equalization stretches; values
outside of the min/max limits are completely ignored in the configuration of
these stretches.

## Overview of Display Calculations

The Workbench follows this approach for displaying band data.  This process is
followed for each color channel being displayed.

*   The band data is normalized to either 32-bit or 64-bit floating point values
    in the range [0.0, 1.0].

    Note that this normalization _does not_ include the application of min/max
    limits; those limits are only used in the calculation of histograms in the
    Contrast Stretch UI.  (TODO:  Will this result in accuracy issues for e.g.
    data sets with a very large range of input values, and a very small range of
    values the user wants to visualize?  Need to think about this!)

    Normalized values are usually 32-bit floating point; they will only be
    64-bit floating point if the input data is also 64-bit floating point.

*   Apply any user-specified conditioner to the data.  Conditioners are
    discussed below, but they are functions that consume normalized data and
    produce normalized data.

*   Apply any user-specified stretch to the data.  Again, this mapping consumes
    normalized data and produces normalized data.

*   The final floating point values are in the range [0.0, 1.0], so they are
    multiplied by 255.0 and then cast to an 8-bit unsigned integer to yield a
    color intensity.

## Stretch Types

WISER supports three stretch types:

*   100% linear stretch
*   Linear stretch
*   Histogram equalization stretch

If WISER is displaying an RGB image, all color channels will use the same
stretch type.  It is not possible to configure WISER to display different color
channels of the same image with different stretch types.  (Obviously, the
parameters for each channel may be different, as each channel's data
distribution will be different.)

### 100% Linear Stretch

The "100% linear stretch" type causes the Workbench to follow a very
straightforward, simple mapping of band data to color values in the range
[0, 255].  The specific mapping depends on the type of the input data.

1.  The minimum and maximum data value for each band is determined.

2.  Values in each band are mapped to the range [0.0, 1.0] with a simple
    calculation:  (value - band_min) / (band_max - band_min).  This is typically
    represented as a 32-bit floating point value, but may be 64-bit floating
    point if the initial data was 64-bit floating point.

3.  Values are then scaled to integers in the range [0, 255], with 0.0 mapping
    to 0, and 1.0 mapping to 255.

This approach is called "100% linear stretch" because it is just the "linear
stretch" method, with low and high bounds taken from the band's minimum and
maximum values.

### Linear Stretch

The "linear stretch" type causes WISER to map band values in a range
[stretch_low, stretch_high] to floating-point values in the range [0.0, 1.0].

*   Values less than stretch_low are mapped to 0.0.
*   Values greater than stretch_high are mapped to 1.0.
*   Values between stretch_low and stretch_high are linearly mapped to a value
    in the range [0.0, 1.0].

These intermediate values are then mapped to a color value in the range
[0, 255].

WISER supports arbitrary values for the stretch_low and stretch_high values,
and each color channel will specify its own values for stretch_low and
stretch_high.  The Contrast Stretch UI calculates and displays a histogram of
the band data for each channel, to aid the user in selecting appropriate
stretch_low / stretch_high values.  In addition, the UI provides the ability to
apply a 2.5% or 5% linear stretch across all channels.

For an N% linear stretch, WISER will choose each channel's stretch_low and
stretch_high such that (N/2)% of the low values will be excluded, and (N/2)%
of the high values will be excluded.  This is computed from the band's
histogram, and is therefore approximate.  (As mentioned earlier, this histogram
is computed after the min/max limits have been used to filter the band's data.)

### Histogram Equalization Stretch

The "histogram equalization stretch" type causes WISER to map the normalized
band values to floating point values in the range [0.0, 1.0] such that the
density of the output values is uniform across this range.  This is computed
from the band's histogram.  (As mentioned earlier, this histogram is computed
after the min/max limits have been used to filter the band's data.)

## Conditioners

Conditioners can be useful when the input data's distribution needs to be
modified; for example, to bring out details in low intensity values and to mute
variations in high intensity values.  WISER supports three conditioner options:

*   No conditioner
*   Square root conditioner
*   Logarithmic conditioner

All conditioners consume normalized data, and produce normalized data.  In the
case of "no conditioner," this can be thought of as the identity function, but
the implementation simply does nothing.

It may be noted that sqrt(x) for x in [0.0, 1.0] already produces values only
in the range [0.0, 1.0].  This is what WISER does for square root conditioning.

In the case of logarithmic conditioner, WISER uses the function
log<sub>2</sub>(x + 1.0); when given values in the range [0.0, 1.0], this will
only produce values in the range [0.0, 1.0].
