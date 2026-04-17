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
dataset that the clicked view was showing. However, if the views are linked, the
zoom window will not switch to display the different dataset.

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

# TODO:  Confusing Scenarios

Scenario:  different "unrelated" spectral data sets

*   Main image window showing two views with different datasets (e.g. oman1 and
    oman2 spectral data).  Linked scrolling is OFF.
*   Context window showing one of these data sets.

*   What should the context window viewport highlight indicate?  Should it
    always be drawn?
    *   Issue:  The viewport reported from the unrelated dataset doesn't really
        mean anything.
    *   Fix:    We only show the viewport highlight in the context pane if the
        matching dataset is open in the main window.

*   Click in main window; zoom pane should switch to the clicked-on data set,
    and show the appropriate spectrum in the plot.

*   Click in zoom pane.  What should happen in the main image window?
    *   We should only show highlighted pixel in views with the same data set.
        If the views are linked, we should show highlighted pixel in all views.

    *   Same questions for viewport highlight.  (This is the same issue as with
        the main window and the context window.) ANSWER: The viewport highlight
        should only show up in the main window raster view's that have the same
        dataset as the zoom pane's rasterview. If they are linked, the viewport
        highlight should show up in all.

Scenario:  different "related" data sets over the same spatial area

*   Main image window showing two views with related datasets (e.g. oman1
    spectral data and oman1 mineral map).  Linked scrolling is ON.
*   Context window showing one of these data sets.

*   Click in main window.

    *   Context window viewport highlight is easy; all main window views are
        showing the same area.

    *   Should zoom pane switch to the clicked-on data set? Currently, zoom pane does not.

*   Click in zoom pane.  What should happen in the main image window?
    *   Show highlighted pixel in all views.
    *   Show viewport highlight in all views.
