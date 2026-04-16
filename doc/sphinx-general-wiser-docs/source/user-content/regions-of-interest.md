# Regions of Interest

WISER supports the creation of Regions of Interest (ROI) on a raster data set.
Regions of Interest may be created in both the main window and the zoom window.
Once a ROI has been created, the average spectrum over the ROI may be plotted,
and the spectra of all pixels in the ROI may be exported as an ASCII file.

Here are the Region of Interest tools in the main toolbar:

<img class="img_center" src="../_static/images/roi_tools_annotated.png" width=400>

The first button allows a new Region of Interest to be created; a dialog allows
the user to enter basic details about the Region of Interest. It is recommended
to use a different color for each Region of Interest to avoid confusion.

<img class="img_center" src="../_static/images/roi_create.png" width=400>

Once a Region of Interest is created, _selections_ may be added to the ROI.
The right button allows users to create rectangle, polygon, and point-set
selections, which will then be added to the current Region of Interest. The
ROI that the selection is added to may be changed with the drop-down combobox
in the toolbar.

<img class="img_center" src="../_static/images/roi_add_selection.png" width=400>

> Tip:  The status bar at the bottom of the UI provides instructions about how
> to create each kind of selection.

Here is the UI state after two Regions of Interest have been created - one named
"grass" and the other named "solar panels". Note that the "solar panels" ROI
is comprised of multiple overlapping rectangle selections (this could also be
done with a single polygon selection).  **It is not a problem to have
overlapping selections in a Region of Interest;** each pixel in the ROI will
only be used once by operations on the ROI.

<img class="img_center" src="../_static/images/roi_add_selection_3.png" width=400>

Once a Region of Interest has been created, right-clicking in the ROI's
selections will pop up a context menu providing various operations with the ROI.

* The ROI's information or display color may be edited
* Individual selections in the ROI may be edited or deleted, or the entire
  ROI may be deleted
* The average spectrum of the ROI may be displayed in the spectrum plot
  window
* The spectra of every pixel in the ROI may be exported as an ASCII file
* Import and export ROI's as .geojson files

<img class="img_center" src="../_static/images/rois.png" width=600>
