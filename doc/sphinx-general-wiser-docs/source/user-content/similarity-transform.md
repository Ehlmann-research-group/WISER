---
title: Similarity Transform
---

# Similarity Transform

The similarity transform tool allows you to do two things: rotate/scale an
image and translate an image's coordinate system.

## Translating Coordinate System

This section refers to the tool with the title 'Translate Coordinate Sys' in
the Similarity Transform Dialog window.

The way we translate the coordinate system is by changing the upper-left
coordinates of the geotransform. If you do not know what a geotransform is,
[this link](https://gdal.org/en/stable/tutorials/geotransforms_tut.html) is a
useful reference. From this link, you will see that we change GT(0) and GT(3).
This translates the geotransform for your coordinate system.

GT(0) corresponds to Lon/East and GT(3) corresponds to Lat/North. When the
'Translate Lat/North' and 'Translate Lon/East' fields are 0, the 'New Upper
Left Pixel Lat/North' field will equal GT(3) and the 'New Upper Left Pixel
Lon/East' field will equal GT(0). By entering values into the 'Translate
Lat/North' and 'Translate Lon/East' editable fields, you will change the new
upper left coordinate of your geo transform.

You can click on any pixel in the raster dataset and see the new spatial
coordinate it will have. This information is under the raster image. Note that
this feature only works with datasets that have geographic information.
