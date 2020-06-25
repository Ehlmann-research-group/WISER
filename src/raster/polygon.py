from typing import List, Set, Tuple

from osgeo import gdal, gdalconst, ogr
import numpy as np


class RasterizedPolygon:
    '''
    This class holds the result of rasterizing a polygon.  The rasterized
    version of the polygon can be retrieved as a NumPy byte-array, or as a
    Python set containing all coordinates that are inside the rasterized
    polygon.

    Rasterized polygons can also be pretty-printed to the console, because it
    just looks cool.
    '''

    def __init__(self, boundary_points, pixel_array):
        self._boundary_points = boundary_points
        self._pixel_array = pixel_array

        xs = [p[0] for p in boundary_points]
        ys = [p[1] for p in boundary_points]
        self._x_min = min(xs)
        self._x_max = max(xs)
        self._y_min = min(ys)
        self._y_max = max(ys)

    def get_x_bounds(self) -> Tuple[int, int]:
        return (self._x_min, self._x_max)

    def get_y_bounds(self) -> Tuple[int, int]:
        return (self._y_min, self._y_max)

    def get_array(self) -> np.ndarray:
        '''
        Returns the rasterized polygon as a NumPy array of bytes, where 0 means
        "outside" and 1 means "inside".  The array is only as large as is
        required to cover the full extent of the polygon.
        '''
        return self._pixel_array

    def get_set(self) -> Set[Tuple[int, int]]:
        '''
        Returns the rasterized polygon as a set of 2-tuples specifying all
        pixels that are inside the polygon.  The pixel coordinates

        The set is regenerated every time this function is called.
        '''
        all_coords = set()

        (height, width) = self._pixel_array.shape
        assert width  == (self._x_max - self._x_min)
        assert height == (self._y_max - self._y_min)

        for y in range(height):
            for x in range(width):
                if self._pixel_array[y][x] != 0:
                    coord = (x + self._x_min, y + self._y_min)
                    all_coords.add(coord)

        return all_coords

    def pprint(self):
        (height, width) = self._pixel_array.shape
        assert width  == (self._x_max - self._x_min)
        assert height == (self._y_max - self._y_min)

        for y in range(height):
            for x in range(width):
                if self._pixel_array[y][x] != 0:
                    ch = '\u2588'
                else:
                    ch = '\u00b7'

                print(' ' + ch, end='')

            print()


def make_polygon_geometry(points: List[Tuple[int, int]]) -> ogr.Geometry:
    '''
    This helper function takes a list of point coordinates specifying the
    boundary of a polygon, and generates an OGR Geometry object of
    type ogr.wkbPolygon to represent the polygon.

    The winding order of the points does not matter; OGR will still figure out
    what is "inside" and "outside" properly.
    '''
    if len(points) < 3:
        raise ValueError('must specify at least 3 points')

    # Create ring
    ring = ogr.Geometry(ogr.wkbLinearRing)
    for p in points:
        ring.AddPoint(p[0], p[1])

    # Create polygon
    poly = ogr.Geometry(ogr.wkbPolygon)
    poly.AddGeometry(ring)

    return poly


def rasterize_polygon(polygon_points: List[Tuple[int, int]]) -> RasterizedPolygon:
    '''
    Given a list of points specifying a polygon, this function will rasterize
    the polygon, returning an object that contains the rasterized version.

    Pixels will be set to 1 if they are on the rendering path between two
    points, or if the pixel's center falls within the polygon's boundary.
    '''

    # Generate an OGR polygon Geometry object from the input points.  This
    # operation is insensitive to the winding order of the points.
    polygon = make_polygon_geometry(polygon_points)

    # Make an OGR in-memory data source, to feed to the GDAL rasterize operation
    source_ds = ogr.GetDriverByName('Memory').CreateDataSource('')
    source_layer = source_ds.CreateLayer('', geom_type=ogr.wkbPolygon)

    feature = ogr.Feature(source_layer.GetLayerDefn())
    feature.SetGeometry(polygon)
    # feature.SetField("id", 1)
    source_layer.CreateFeature(feature)
    # source_srs = source_layer.GetSpatialRef()

    # These values are floating point
    x_min, x_max, y_min, y_max = source_layer.GetExtent()
    # print(f'x_min={x_min}\tx_max={x_max}\t\ty_min={y_min}\ty_max={y_max}')

    # Make a GDAL in-memory data source for the polygon to be rasterized into
    x_res = int(x_max - x_min)
    y_res = int(y_max - y_min)
    # print(f'x_res={x_res}\ty_res={y_res}')
    target_ds = gdal.GetDriverByName('MEM').Create('', x_res, y_res, eType=gdal.GDT_Byte)
    # target_ds.SetGeoTransform((x_min, 1, 0, y_max, 0, -1))
    target_ds.SetGeoTransform((x_min, 1, 0, y_min, 0, 1))
    band = target_ds.GetRasterBand(1)
    # band.SetNoDataValue(NoData_value)

    # Rasterize the polygon into the target layer
    gdal.RasterizeLayer(target_ds, [1], source_layer, burn_values=[1])

    # Read the result as a NumPy array
    array = band.ReadAsArray()

    return RasterizedPolygon(polygon_points, array)


if __name__ == '__main__':
    points = [(15,25), (25, 50), (45, 15), (30, 10)]
    poly = make_polygon_geometry(points)
    print(poly.ExportToWkt())

    rpoly = rasterize_polygon(points)
    rpoly.pprint()

    points.reverse()
    poly = make_polygon_geometry(points)
    print(poly.ExportToWkt())

    rpoly = rasterize_polygon(points)
    rpoly.pprint()

    points = [(10,10), (10,20), (20,20), (20,10)]
    poly = make_polygon_geometry(points)
    print(poly.ExportToWkt())

    rpoly = rasterize_polygon(points)
    rpoly.pprint()
