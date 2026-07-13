"""
Qt-free GDAL warp engine for the georeferencer (WISER#684).

This module holds all the GDAL/OSR logic that used to live inside
``GeoReferencerDialog``: building the warp/transformer options from the chosen transform
type, computing per-GCP residuals for the interactive preview, and producing the final
warped GeoTIFF. Keeping it Qt-free (GDAL + WISER raster helpers only, mirroring
:mod:`wiser.raster.mosaic_export`) means the warp math is unit-testable without a running
app and can run on a background thread.

The three entry points are:

- :func:`build_warp_kwargs` -- turn a resample algorithm + transform type + output SRS into
  the ``gdal.Warp`` kwargs and the matching ``gdal.Transformer`` options.
- :func:`compute_residuals` -- the fast interactive residual calc: warp a 1x1 placeholder
  purely to drive GDAL's transformer, and return per-GCP pixel residuals.
- :func:`warp_dataset_to_path` -- the full multi-band output warp, streamed in RAM-bounded
  band chunks with progress reporting and cooperative cancellation.

Callers pass plain data (``gdal.GCP`` lists, ``osr.SpatialReference`` objects) and a
:class:`~wiser.utils.progress.ProgressReporter`; these functions never touch Qt, show
dialogs, or log. Errors are raised; progress and exceptions are the only channels out.
"""

from enum import Enum
from typing import List, Optional, Tuple

import numpy as np

from osgeo import gdal, osr, gdal_array

from wiser.raster.dataset import RasterDataSet
from wiser.raster.dataset_impl import GDALRasterDataImpl
from wiser.raster.utils import (
    copy_metadata_to_gdal_dataset,
    numpy_dtype_to_gdal_export_types,
    set_data_ignore_of_gdal_dataset,
)
from wiser.bandmath.utils import write_raster_to_dataset
from wiser.bandmath.builtins.constants import MAX_RAM_BYTES
from wiser.utils.progress import ProgressReporter


# All of GDAL's GRA_* resampling algorithms, e.g. GRA_NearestNeighbour, GRA_Bilinear,
# GRA_Cubic, GRA_CubicSpline, GRA_Lanczos, ...
RESAMPLE_ALGORITHMS = {name: getattr(gdal, name) for name in dir(gdal) if name.startswith("GRA_")}


class TRANSFORM_TYPES(Enum):
    POLY_1 = "Affine (Polynomial 1)"
    POLY_2 = "Polynomial 2"
    POLY_3 = "Polynomial 3"
    TPS = "Thin Plate Spline (TPS)"


min_points_per_transform = {
    TRANSFORM_TYPES.POLY_1: 3,
    TRANSFORM_TYPES.POLY_2: 6,
    TRANSFORM_TYPES.POLY_3: 10,
    TRANSFORM_TYPES.TPS: 10,
}


def build_warp_kwargs(
    resample_alg,
    transform_type: TRANSFORM_TYPES,
    output_srs: osr.SpatialReference,
) -> Tuple[dict, List[str]]:
    """
    Build the ``gdal.Warp`` kwargs and the matching ``gdal.Transformer`` options for the
    chosen transform.

    ``output_srs`` is expected to already have its axis-mapping strategy set (typically
    ``OAMS_TRADITIONAL_GIS_ORDER``) by the caller. Returns ``(warp_kwargs,
    transformer_options)``.
    """
    warp_kwargs = {
        "copyMetadata": True,
        "resampleAlg": resample_alg,
        "dstSRS": output_srs,
    }

    transformer_options = [f"DST_SRS={output_srs.ExportToWkt()}"]

    if transform_type == TRANSFORM_TYPES.TPS:
        warp_kwargs["tps"] = True
        transformer_options += ["METHOD=GCP_TPS", "MAX_GCP_ORDER=-1"]
    elif transform_type == TRANSFORM_TYPES.POLY_1:
        warp_kwargs["polynomialOrder"] = 1
        transformer_options += ["METHOD=GCP_POLYNOMIAL", "MAX_GCP_ORDER=1"]
    elif transform_type == TRANSFORM_TYPES.POLY_2:
        warp_kwargs["polynomialOrder"] = 2
        transformer_options += ["METHOD=GCP_POLYNOMIAL", "MAX_GCP_ORDER=2"]
    elif transform_type == TRANSFORM_TYPES.POLY_3:
        warp_kwargs["polynomialOrder"] = 3
        transformer_options += ["METHOD=GCP_POLYNOMIAL", "MAX_GCP_ORDER=3"]
    else:
        raise RuntimeError(f"Unknown transform type: {transform_type}")

    warp_kwargs["transformerOptions"] = transformer_options
    return warp_kwargs, transformer_options


def compute_residuals(
    gcps: List[gdal.GCP],
    ref_srs: osr.SpatialReference,
    output_srs: osr.SpatialReference,
    warp_kwargs: dict,
    transformer_options: List[str],
) -> List[Tuple[float, float]]:
    """
    Compute per-GCP residuals (in target-pixel units) for the chosen transform.

    Does **not** warp the real image: it builds a 1x1 placeholder dataset purely to drive
    GDAL's transformer, then measures how far each GCP's transformed position lands from
    its reference coordinate.

    ``ref_srs`` and ``output_srs`` must already have their axis-mapping strategy set
    (``OAMS_TRADITIONAL_GIS_ORDER``). Returns a list of ``(residual_x, residual_y)`` in the
    same order as ``gcps``. Raises on GDAL failure (e.g. a degenerate GCP configuration).
    """
    gdal.UseExceptions()

    ref_projection = ref_srs.ExportToWkt()
    assert (
        ref_projection is not None and ref_srs is not None
    ), f"ref_srs ({ref_srs}) or ref_project ({ref_projection}) is None!"

    # We need a temporary gdal dataset so we can calculate the residuals
    # without making a massive data object
    place_holder_arr = np.zeros((1, 1), np.uint8)
    temp_gdal_ds: gdal.Dataset = gdal_array.OpenNumPyArray(place_holder_arr, True)
    temp_gdal_ds.SetSpatialRef(ref_srs)
    temp_gdal_ds.SetGCPs(gcps, ref_projection)

    # This will be used to transform the pixel coordinate to the
    # output srs coordinate that has undergone the spatial warping
    tr_pixel_to_output_srs: gdal.Transformer = gdal.Transformer(temp_gdal_ds, None, transformer_options)

    # Sneak peek into the transformed dataset's geo transform so we can use them later
    # when calculating residuals. We don't just use the geotransform here to calculate
    # residuals because transformed_gt goes from the warped datasets pixel coordinates
    # to the warped dataset's spatial coordinates. Our GCPs are just in the target
    # dataset's pixel coordinates and reference srs's spatial coordinates.
    warp_options = gdal.WarpOptions(**warp_kwargs)
    warp_save_path = f"/vsimem/temp_band_{0}"
    place_holder_arr = np.zeros((1, 1), np.uint8)
    temp_gdal_ds = gdal_array.OpenNumPyArray(place_holder_arr, True)
    temp_gdal_ds.SetGCPs(gcps, ref_projection)
    transformed_ds: gdal.Dataset = gdal.Warp(warp_save_path, temp_gdal_ds, options=warp_options)
    transformed_gt = transformed_ds.GetGeoTransform()

    # The output_srs here is the target spatial reference system (srs). In case
    # the target srs is different from the reference srs, then we need a way to
    # go between them.
    tr_output_srs_to_ref_srs = osr.CoordinateTransformation(output_srs, ref_srs)

    residuals: List[Tuple[float, float]] = []
    # The general flow of calculating the residuals is to go from our target
    # dataset's pixel coordinate (gcp.GCPPixel, gcp.GCPLine) to our output
    # srs coordinates. Then we go from the output srs coordinate to the
    # reference srs coordinate. We then get the difference in reference srs
    # coordinates from the original GCP to get the spatial error. Then to make
    # it a pixel error we divide by the width/height per pixel of the warped
    # dataset's geo transform.
    for gcp in gcps:
        # These coordinates could get back to us in either lat/lon, lon/lat,
        # or north/easting, easting/north
        (
            ok,
            (output_spatial_x, output_spatial_y, z),
        ) = tr_pixel_to_output_srs.TransformPoint(False, gcp.GCPPixel, gcp.GCPLine)

        # Since we use OAMS_TRADITIONAL_GIS_ORDER on both reference and output CRS, we don't need
        # to swap the spatial coordinates
        ref_spatial_coord = tr_output_srs_to_ref_srs.TransformPoint(output_spatial_x, output_spatial_y, 0)

        ref_spatial_x, ref_spatial_y = (
            ref_spatial_coord[0],
            ref_spatial_coord[1],
        )

        error_spatial_x = gcp.GCPX - ref_spatial_x
        error_spatial_y = gcp.GCPY - ref_spatial_y

        error_raster_x = error_spatial_x / transformed_gt[1]
        error_raster_y = error_spatial_y / transformed_gt[5]

        residuals.append((error_raster_x, error_raster_y))

    tr_pixel_to_output_srs = None
    tr_output_srs_to_ref_srs = None
    temp_gdal_ds = None
    gdal.Unlink(warp_save_path)

    return residuals


def warp_dataset_to_path(
    target_dataset: RasterDataSet,
    gcps: List[gdal.GCP],
    warp_kwargs: dict,
    ref_srs: osr.SpatialReference,
    save_path: str,
    *,
    progress: Optional[ProgressReporter] = None,
) -> str:
    """
    Warp ``target_dataset`` onto real-world coordinates using ``gcps`` and write the result
    as a GeoTIFF at ``save_path``. Returns ``save_path``.

    Because hyperspectral cubes can be very large, the band data is processed in RAM-bounded
    chunks. Three cases (matching the original in-dialog implementation):

    - target backed by a GDAL dataset -> Translate to a VRT and warp the whole thing;
    - a numpy-backed target that fits in RAM -> warp the whole array;
    - otherwise -> warp band chunks and write incrementally, reporting ``progress`` and
      calling :meth:`ProgressReporter.raise_if_cancelled` between chunks.

    ``ref_srs`` supplies the destination projection attached to the GCPs. The caller is
    responsible for validating inputs (save path, GCP count, target selected) before
    calling.
    """
    if progress is None:
        progress = ProgressReporter()

    gdal.UseExceptions()

    target_dataset_impl = target_dataset.get_impl()
    ref_projection = ref_srs.ExportToWkt()

    driver: gdal.Driver = gdal.GetDriverByName("GTiff")
    gdal_dtype, _ = numpy_dtype_to_gdal_export_types(target_dataset.get_elem_type())

    output_dataset: gdal.Dataset = None
    output_gt = None

    # We warp one band of the input dataset to a virtual memory file, so we can create the
    # correct output data size. Then we create the output dataset with width and height
    # equal to the warp, but correct number of bands. Then we override each band in the
    # created array and flush cache.
    warp_options = gdal.WarpOptions(**warp_kwargs)
    warp_save_path = f"/vsimem/temp_band_{0}"
    band_arr = target_dataset.get_band_data(0)
    temp_gdal_ds: gdal.Dataset = gdal_array.OpenNumPyArray(band_arr, True)
    # Make sure dataset has no spatial information that could mess with warping
    temp_gdal_ds.SetGCPs(gcps, ref_projection)
    transformed_ds: gdal.Dataset = gdal.Warp(warp_save_path, temp_gdal_ds, options=warp_options)

    width = transformed_ds.RasterXSize
    height = transformed_ds.RasterYSize
    output_size = (width, height)
    output_bytes = width * height * target_dataset.num_bands() * target_dataset.get_elem_type().itemsize

    gdal.Unlink(warp_save_path)

    progress.raise_if_cancelled()
    progress.report(0, target_dataset.num_bands(), "Warping...")

    ratio = MAX_RAM_BYTES / output_bytes
    if isinstance(target_dataset_impl, GDALRasterDataImpl):
        # Saving the full gdal dataset
        target_gdal_dataset = target_dataset_impl.gdal_dataset
        temp_vrt_path = "/vsimem/ref.vrt"
        if target_dataset.get_data_ignore_value() is not None:
            translate_opts = gdal.TranslateOptions(
                format="VRT",
                noData=target_dataset.get_data_ignore_value(),
            )
        else:
            translate_opts = gdal.TranslateOptions(
                format="VRT",
            )
        temp_gdal_ds = gdal.Translate(temp_vrt_path, target_gdal_dataset, options=translate_opts)
        # Make sure dataset has no spatial information that could mess with warping
        temp_gdal_ds.SetGCPs(gcps, ref_projection)
        warp_options = gdal.WarpOptions(**warp_kwargs)
        output_dataset = gdal.Warp(save_path, temp_gdal_ds, options=warp_options)
        progress.report(target_dataset.num_bands(), target_dataset.num_bands(), "Warping...")
    elif not isinstance(target_dataset_impl, GDALRasterDataImpl) and ratio > 1.0:
        # Saving the full object array
        warp_options = gdal.WarpOptions(**warp_kwargs)
        dataset_arr = target_dataset.get_image_data()
        temp_gdal_ds = gdal_array.OpenNumPyArray(dataset_arr, True)
        # Make sure dataset has no spatial information that could mess with warping
        temp_gdal_ds.SetGCPs(gcps, ref_projection)
        set_data_ignore_of_gdal_dataset(temp_gdal_ds, target_dataset)
        output_dataset = gdal.Warp(save_path, temp_gdal_ds, options=warp_options)
        output_dataset.FlushCache()
        progress.report(target_dataset.num_bands(), target_dataset.num_bands(), "Warping...")
    else:
        # Saving incrementally using the numpy dataset
        num_bands_per = int(ratio * target_dataset.num_bands())
        for band_index in range(0, target_dataset.num_bands(), num_bands_per):
            progress.raise_if_cancelled()
            band_list_index = [
                band
                for band in range(band_index, band_index + num_bands_per)
                if band < target_dataset.num_bands()
            ]
            warp_options = gdal.WarpOptions(**warp_kwargs)
            warp_save_path = f"/vsimem/temp_band_{min(band_list_index)}_to_{max(band_list_index)}"

            band_arr = target_dataset.get_multiple_band_data(band_list_index)
            temp_gdal_ds = gdal_array.OpenNumPyArray(band_arr, True)
            # Make sure dataset has no spatial information that could mess with warping
            temp_gdal_ds.SetGCPs(gcps, ref_projection)
            set_data_ignore_of_gdal_dataset(temp_gdal_ds, target_dataset)
            transformed_ds = gdal.Warp(warp_save_path, temp_gdal_ds, options=warp_options)

            width = transformed_ds.RasterXSize
            height = transformed_ds.RasterYSize
            assert (
                width == output_size[0] and height == output_size[1]
            ), "Width and/or height of warped band does not equal a previous warped band"

            if output_dataset is None:
                output_dataset = driver.Create(
                    save_path,
                    width,
                    height,
                    target_dataset.num_bands(),
                    gdal_dtype,
                )
                output_gt = transformed_ds.GetGeoTransform()

            write_raster_to_dataset(
                output_dataset,
                band_list_index,
                transformed_ds.ReadAsArray(),
                gdal_dtype,
            )

            gdal.Unlink(warp_save_path)
            transformed_ds = None
            progress.report(
                min(band_index + num_bands_per, target_dataset.num_bands()),
                target_dataset.num_bands(),
                "Warping...",
            )
        output_dataset.SetGeoTransform(output_gt)
        output_dataset.SetSpatialRef(ref_srs)

    if output_dataset is None:
        raise RuntimeError("gdal.Warp failed to produce a transformed dataset.")

    copy_metadata_to_gdal_dataset(output_dataset, target_dataset)
    gt = output_dataset.GetGeoTransform()
    if gt is None:
        raise RuntimeError("Failed to retrieve geotransform from the transformed dataset.")

    output_dataset.FlushCache()
    output_dataset = None

    progress.report(target_dataset.num_bands(), target_dataset.num_bands(), "Done warping!")
    return save_path
