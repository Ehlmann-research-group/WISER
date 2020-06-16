# Multiple Views in the Main Image Window

The main image window is capable of showing multiple views, either different
images of the same samples, or images of different samples.  The views may
optionally be linked so that scrolling in one view will automatically update the
scroll position in the other views.

The primary constraint for multiple views in the main image window is that they
will all use the same zoom level.  It is not possible to have different views
with different scroll levels.

When linked scrolling is turned on, the views will all automatically update to
match the scroll settings of the top left view.

# Main Window / Zoom Window Interactions

The interactions between the main image window and the zoom window become more
complex when multiple raster views are being displayed.

When the user clicks pixels in the main window, the zoom window is updated to
show both the dataset and the click location.  Since different views may be
showing different datasets, the zoom window will switch to displaying the same
dataset that the clicked view was showing.

When the user clicks pixels in the zoom window, the main window is updated to
display the click location.  Since all views in the main window may not be
showing the same data set in the zoom window, the following behaviors are
followed:

*   If linked scrolling is enabled, all views in the main window will be updated
    to show the click location.
*   If linked scrolling is disabled, only the views in the main window that are
    showing the same dataset will be updated to show the click location; other
    views will not be updated (and any previously-showing click location will be
    discarded).

These behaviors are also followed when the user clicks in the zoom window.
