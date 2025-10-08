import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from PySide2.QtCore import *

from osgeo import gdal, gdalconst, ogr

from .roi import RegionOfInterest
from .selection import (
    SelectionType,
    SinglePixelSelection,
    MultiPixelSelection,
    RectangleSelection,
    PolygonSelection,
)


logger = logging.getLogger(__name__)


def export_roi_list_to_geojson_file(
    roi_list: List[RegionOfInterest], filename, pretty=False
):
    """
    This function exports a list of Regions of Interest into a GeoJSON file.
    """

    driver = gdal.GetDriverByName("GeoJSON")
    dataset = driver.Create(filename, 0, 0)
    layer = dataset.CreateLayer("ROIs")
    # layer.CreateField(ogr.FieldDefn('raster_width', ogr.OFTInteger))
    # layer.CreateField(ogr.FieldDefn('raster_height', ogr.OFTInteger))

    for roi in roi_list:
        feature = roi_to_ogr_feature(roi)
        layer.CreateFeature(feature)

    # This causes the dataset to be written back to disk.
    layer = None
    dataset = None
    driver = None

    if pretty:
        with open(filename) as f:
            data = json.load(f)

        with open(filename, "w") as f:
            json.dump(data, f, indent=2)


def export_roi_to_geojson_file(roi: RegionOfInterest, filename, pretty=False):
    """
    This function exports a single Region of Interest into a GeoJSON file.
    """

    feature = roi_to_ogr_feature(roi)
    json_str = feature.ExportToJson()

    with open(filename, "w") as f:
        if pretty:
            data = json.loads(json_str)
            json.dump(data, f, indent=2)
        else:
            f.write(json_str)


def roi_to_ogr_feature(roi: RegionOfInterest) -> ogr.Feature:
    """
    This function converts a Region of Interest into a GDAL/OGR Feature object.
    """
    selections = roi.get_selections()

    # The feature records the metadata of the ROI as well as its geometry.

    feature_def = ogr.FeatureDefn("ROI")
    feature_def.AddFieldDefn(ogr.FieldDefn("name"))
    feature_def.AddFieldDefn(ogr.FieldDefn("color"))
    feature_def.AddFieldDefn(ogr.FieldDefn("description"))
    feature_def.AddFieldDefn(ogr.FieldDefn("raster_width", ogr.OFTInteger))
    feature_def.AddFieldDefn(ogr.FieldDefn("raster_height", ogr.OFTInteger))
    feature = ogr.Feature(feature_def)

    # Set the ROI's metadata on the feature.

    feature["name"] = roi.get_name()
    feature["color"] = roi.get_color()
    feature["description"] = roi.get_description()

    # Build the geometry of the ROI now.  We try to be clever by only creating
    # a top-level group if we know we have more than one selection in the ROI.
    # If we have only one selection, we don't need the top-level group.

    geom = None
    if len(selections) > 1:
        geom = ogr.Geometry(ogr.wkbGeometryCollection)

    for sel in selections:
        sel_type = sel.get_type()

        if sel_type == SelectionType.SINGLE_PIXEL:
            point = ogr.Geometry(ogr.wkbPoint)
            p = sel.get_pixel()
            point.AddPoint(p.x(), p.y())

            if geom is not None:
                geom.AddGeometry(point)
            else:
                geom = point

        elif sel_type == SelectionType.MULTI_PIXEL:
            multi_point = ogr.Geometry(ogr.wkbMultiPoint)

            for p in sel.get_pixels():
                point = ogr.Geometry(ogr.wkbPoint)
                point.AddPoint(p.x(), p.y())
                multi_point.AddGeometry(point)

            if geom is not None:
                geom.AddGeometry(multi_point)
            else:
                geom = multi_point

        elif sel_type == SelectionType.RECTANGLE:
            # OGR doesn't have a specific type for rectangles.  Should be fun.

            p1 = sel.get_top_left()
            p2 = sel.get_bottom_right()

            ring = ogr.Geometry(ogr.wkbLinearRing)
            ring.AddPoint(p1.x(), p1.y())
            ring.AddPoint(p2.x(), p1.y())
            ring.AddPoint(p2.x(), p2.y())
            ring.AddPoint(p1.x(), p2.y())

            poly = ogr.Geometry(ogr.wkbPolygon)
            poly.AddGeometry(ring)

            if geom is not None:
                geom.AddGeometry(poly)
            else:
                geom = poly

        elif sel_type == SelectionType.POLYGON:
            ring = ogr.Geometry(ogr.wkbLinearRing)
            for p in sel.get_points():
                ring.AddPoint(p.x(), p.y())

            poly = ogr.Geometry(ogr.wkbPolygon)
            poly.AddGeometry(ring)

            if geom is not None:
                geom.AddGeometry(poly)
            else:
                geom = poly

        else:
            raise ValueError(
                f"ROI contains a selection of type {sel_type}, "
                + "but WISER doesn't know how to export this."
            )

    geom.FlattenTo2D()
    feature.SetGeometry(geom)

    return feature


def import_geojson_file_to_rois(filename) -> List[RegionOfInterest]:
    logger.info(f"Opening dataset from file {filename}")
    dataset = gdal.OpenEx(
        filename,
        nOpenFlags=gdalconst.OF_READONLY | gdalconst.OF_VERBOSE_ERROR,
        allowed_drivers=["GeoJSON"],
    )

    rois = []
    while True:
        (feature, layer) = dataset.GetNextFeature()
        if feature is None:
            break

        logger.info(f"Feature:  {feature}\tLayer:  {layer}")

        roi = ogr_feature_to_roi(feature)
        rois.append(roi)

    return rois


def ogr_feature_to_roi(feature: ogr.Feature) -> RegionOfInterest:
    """
    This function converts a GDAL/OGR Feature object into a Region of Interest.
    """

    # Try to fetch the ROI's custom attributes off of the OGR Feature.  If this
    # fails, no biggie - we will auto-generate them.

    def try_get_attr(feature, attr, default=None):
        # This is basically the dict.get() function, but we are working against
        # GDAL Feature objects, and they don't provide get().
        try:
            return feature[attr]
        except:
            return default

    name = try_get_attr(feature, "name", "unnamed")
    color = try_get_attr(feature, "color")
    description = try_get_attr(feature, "description")

    # Get all leaf Geometry objects in the Feature; each will become a ROI
    # selection.

    def all_leaf_geometries(geom):
        # A helper function to find all leaf-geometries in the OGR Feature.
        all = []
        if geom.GetGeometryType() == ogr.wkbGeometryCollection:
            logger.debug(f"Looking for child geometries in geometry:  {geom}")
            for i in range(geom.GetGeometryCount()):
                child_geom = geom.GetGeometryRef(i)
                all.extend(all_leaf_geometries(child_geom))
        else:
            logger.debug(f"Found leaf geometry:  {geom}")
            all.append(geom)

        return all

    geometry = feature.geometry()
    geometry.FlattenTo2D()
    leaf_geoms = all_leaf_geometries(geometry)

    roi = RegionOfInterest(name, color)
    roi.set_description(description)
    for geom in leaf_geoms:
        geom_type = geom.GetGeometryType()

        logger.debug(f"Converting geometry {geom} of type {geom_type}")

        if geom_type == ogr.wkbPoint:
            # Single-point selection
            point = QPoint(geom.GetX(), geom.GetY())
            sel = SinglePixelSelection(point)
            roi.add_selection(sel)

        elif geom_type == ogr.wkbMultiPoint:
            # Multi-point selection

            # Get the points in the multi-point geometry so we can store them.
            # GDAL/OGR puts multi-point points into sub-geometries in the parent
            # geometry.

            points = []
            for i in range(geom.GetGeometryCount()):
                child_geom = geom.GetGeometryRef(i)
                # child_type = child_geom.GetGeometryType() # Should be ogr.wkbPoint

                point = QPoint(child_geom.GetX(), child_geom.GetY())
                points.append(point)

            sel = MultiPixelSelection(points)
            roi.add_selection(sel)

        elif geom_type == ogr.wkbPolygon:
            # Either a polygon or a rectangle.

            # Get the points in the polygon so we can analyze them.  GDAL/OGR
            # puts polygon points into a linear-ring sub-geometry, so we have to
            # pull that out.

            if geom.GetGeometryCount() > 1:
                raise ValueError(
                    "WISER doesn't know how to handle polygons "
                    + "with multiple sub-geometries."
                )

            geom = geom.GetGeometryRef(0)

            points = []
            for i in range(geom.GetPointCount()):
                point = QPoint(geom.GetX(i), geom.GetY(i))
                points.append(point)

            rect = get_rectangle_from_points(points)
            if rect:
                sel = RectangleSelection(rect[0], rect[1])
            else:
                sel = PolygonSelection(points)

            roi.add_selection(sel)

        else:
            raise ValueError(
                "Feature contains a geometry of type "
                + f"{geom_type}, but WISER doesn't know how to import this."
            )

    return roi


def get_rectangle_from_points(points):
    """
    Given a list of points, this function determines if the points form a
    rectangle.  If so, the function returns two opposite corners of the
    rectangle; if not, it returns ``None``.
    """

    if len(points) != 4:
        return None

    if (
        points[0].x() == points[1].x()
        and points[2].x() == points[3].x()
        and points[1].y() == points[2].y()
        and points[3].y() == points[0].y()
    ):
        return (points[0], points[2])

    elif (
        points[0].y() == points[1].y()
        and points[2].y() == points[3].y()
        and points[1].x() == points[2].x()
        and points[3].x() == points[0].x()
    ):
        return (points[0], points[2])

    else:
        return None
