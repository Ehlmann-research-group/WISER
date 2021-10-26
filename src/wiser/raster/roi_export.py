import json
from typing import Any, Dict, List, Optional, Tuple

from osgeo import gdal, ogr

from .roi import RegionOfInterest
from .selection import SelectionType


def export_roi_list_to_geojson_file(roi_list: List[RegionOfInterest], filename,
        pretty=False):
    '''
    This function exports a list of Regions of Interest into a GeoJSON file.
    '''

    driver = gdal.GetDriverByName('GeoJSON')
    dataset = driver.Create(filename, 0, 0)
    layer = dataset.CreateLayer('ROIs')
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

        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)


def export_roi_to_geojson_file(roi: RegionOfInterest, filename, pretty=False):
    '''
    This function exports a single Region of Interest into a GeoJSON file.
    '''

    feature = roi_to_ogr_feature(roi)
    json_str = feature.ExportToJson()

    with open(filename, 'w') as f:
        if pretty:
            data = json.loads(json_str)
            json.dump(data, f, indent=2)
        else:
            f.write(json_str)


def roi_to_ogr_feature(roi: RegionOfInterest) -> ogr.Feature:
    '''
    This function converts a Region of Interest into a GDAL/OGR Feature object.
    '''
    selections = roi.get_selections()

    # The feature records the metadata of the ROI as well as its geometry.

    feature_def = ogr.FeatureDefn('ROI')
    feature_def.AddFieldDefn(ogr.FieldDefn('name'))
    feature_def.AddFieldDefn(ogr.FieldDefn('color'))
    feature_def.AddFieldDefn(ogr.FieldDefn('description'))
    feature_def.AddFieldDefn(ogr.FieldDefn('raster_width', ogr.OFTInteger))
    feature_def.AddFieldDefn(ogr.FieldDefn('raster_height', ogr.OFTInteger))
    feature = ogr.Feature(feature_def)

    # Set the ROI's metadata on the feature.

    feature['name'] = roi.get_name()
    feature['color'] = roi.get_color()
    feature['description'] = roi.get_description()

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
            raise ValueError(f'ROI contains a selection of type {sel_type},' +
                'but WISER doesn\'t know how to export this.')

    feature.SetGeometry(geom)

    return feature


'''
def import_geojson_file_to_rois(filename) -> List[RegionOfInterest]:
    dataset = gdal.OpenEx(filename,
        nOpenFlags=gdalconst.OF_READONLY | gdalconst.OF_VERBOSE_ERROR,
        allowed_drivers=['GeoJSON'])

    layer =
'''
