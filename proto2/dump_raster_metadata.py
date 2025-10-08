import sys

from osgeo import gdal, gdalconst, gdal_array, osr

# ruff: noqa: E501


def pixel_coord_to_geo_coord(pixel_coord, geo_transform):
    (pixel_x, pixel_y) = pixel_coord
    geo_x = geo_transform[0] + pixel_x * geo_transform[1] + pixel_y * geo_transform[2]
    geo_y = geo_transform[3] + pixel_x * geo_transform[4] + pixel_y * geo_transform[5]
    return (geo_x, geo_y)


def geo_coord_to_angular_coord(geo_coord, spatial_ref):
    (geo_x, geo_y) = geo_coord
    ang_spatial_ref = spatial_ref.CloneGeogCS()
    coord_xform = osr.CoordinateTransformation(spatial_ref, ang_spatial_ref)
    return coord_xform.TransformPoint(geo_x, geo_y)


def dump_coords(pixel_coord, geo_transform, spatial_ref):
    if geo_transform:
        geo_coord = pixel_coord_to_geo_coord(pixel_coord, geo_transform)

        if spatial_ref:
            ang_coord = geo_coord_to_angular_coord(geo_coord, spatial_ref)
            print(f"Pixel {pixel_coord} = Linear {geo_coord} = Angular {ang_coord}")

        else:
            print(f"Pixel {pixel_coord} = Linear {geo_coord}")


def dump_raster_metadata(path):
    print(f"Opening {path}...")
    gdal.UseExceptions()
    ds = gdal.OpenEx(path, nOpenFlags=gdalconst.OF_READONLY | gdalconst.OF_VERBOSE_ERROR)

    print()
    print(f"File list:  {ds.GetFileList()}")

    print()
    print(f"RasterXSize = {ds.RasterXSize}")
    print(f"RasterYSize = {ds.RasterYSize}")
    print(f"RasterCount = {ds.RasterCount}")

    print()
    print(f'Dataset description:  "{ds.GetDescription()}"')
    print("Dataset metadata domains:")
    for domain in ds.GetMetadataDomainList():
        print(f' * Domain "{domain}":  {ds.GetMetadata_Dict(domain)}')

    geo_transform = ds.GetGeoTransform()
    spatial_ref = ds.GetSpatialRef()

    print()
    print(f"Projection = {ds.GetProjection()}")
    print(f"GeoTransform = {geo_transform}")
    print(f"SpatialRef = {spatial_ref}")

    if geo_transform is not None:
        print()

        ang_units = ""
        lin_units = ""
        if spatial_ref is not None:
            ang_units = spatial_ref.GetAngularUnitsName()
            lin_units = spatial_ref.GetLinearUnitsName()
            print(f"Angular units = {ang_units}, Linear units = {lin_units}")

        if geo_transform[2] == 0 and geo_transform[4] == 0:
            print(
                f"North-up image.  Pixel size is {geo_transform[1]} {lin_units} x {geo_transform[5]} {lin_units}."
            )

        print()
        # print(f'Top-left of image at {pixel_coord_to_geo_coord(0, 0, geo_transform)}')
        dump_coords((0, 0), geo_transform, spatial_ref)
        # print(f'Bottom-right of image at {pixel_coord_to_geo_coord(ds.RasterXSize, ds.RasterYSize, geo_transform)}')
        dump_coords((ds.RasterXSize, ds.RasterYSize), geo_transform, spatial_ref)

    for i in range(1, ds.RasterCount + 1):
        b = ds.GetRasterBand(i)
        print()
        print(f"Band {i}:")
        print(f"  XSize:  {b.XSize}\tYSize:  {b.YSize}")
        print(
            f"  DataType:  {b.DataType} (GDAL), {gdal_array.GDALTypeCodeToNumericTypeCode(b.DataType)} (NumPy)"
        )
        print(f"  NoData:  {b.GetNoDataValue()}")
        print(f'  Description:  "{b.GetDescription()}"')
        print(f"  Unit Type:  {b.GetUnitType()}")
        print(f"  Raster category names:  {b.GetRasterCategoryNames()}")
        print("  Metadata domains:")
        domain_list = b.GetMetadataDomainList()
        if domain_list:
            for domain in domain_list:
                print(f'   * Domain "{domain}":  {b.GetMetadata_Dict(domain)}')


if __name__ == "__main__":
    for path in sys.argv[1:]:
        dump_raster_metadata(path)
        print()
