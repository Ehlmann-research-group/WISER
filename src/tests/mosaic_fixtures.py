"""
Reusable test fixtures for the Seamless Mosaic feature.

These small, overlapping, georeferenced scene builders are shared across the
mosaic test suite (materialization adapter here, plus scene ingestion, common
grid/CRS resolution, compositor, and export in the follow-on issues). Keeping
them in one place avoids each test re-inventing georeferenced scenes.

Builders return :class:`~wiser.raster.dataset.RasterDataSet` objects:

* :func:`make_numpy_scene` -- an in-RAM ``NumPyRasterDataImpl``-backed scene.
* :func:`make_gtiff_scene` -- a disk-backed GeoTIFF scene (GDAL impl).
* :func:`make_overlapping_scene_pair` -- two scenes whose footprints partially
  overlap, one NumPy-backed and one GeoTIFF-backed.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
from osgeo import gdal, gdal_array, osr

from wiser.raster.dataset import RasterDataSet
from wiser.raster.dataset_impl import GTiff_GDALRasterDataImpl, NumPyRasterDataImpl

# Default UTM zone 11N (EPSG:32611) - a metric CRS that makes overlap geometry
# easy to reason about in meters.
DEFAULT_EPSG = 32611


def wkt_for_epsg(epsg: int) -> str:
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(epsg)
    return srs.ExportToWkt()


def _build_cube(num_bands: int, height: int, width: int, dtype: np.dtype, base: float) -> np.ndarray:
    """Deterministic [band][y][x] cube so reads are easy to assert against."""
    n = num_bands * height * width
    arr = (np.arange(n, dtype=np.float64) + base).reshape(num_bands, height, width)
    return arr.astype(dtype)


def _wavelength_band_info(wavelengths: Sequence[float], units: str) -> List[dict]:
    info: List[dict] = []
    for i, wl in enumerate(wavelengths):
        info.append(
            {
                "index": i,
                "description": f"Band {i}",
                "wavelength_name": f"{wl} {units}",
                "wavelength_str": str(wl),
                "wavelength_units": units,
            }
        )
    return info


def make_numpy_scene(
    *,
    width: int = 8,
    height: int = 6,
    num_bands: int = 3,
    dtype: np.dtype = np.dtype(np.float32),
    origin: Tuple[float, float] = (300000.0, 4000000.0),
    pixel_size: Tuple[float, float] = (10.0, -10.0),
    epsg: int = DEFAULT_EPSG,
    nodata: Optional[float] = -9999.0,
    wavelengths: Optional[Sequence[float]] = None,
    wavelength_units: str = "nm",
    base_value: float = 0.0,
) -> RasterDataSet:
    """
    Build an in-RAM, georeferenced ``NumPyRasterDataImpl``-backed scene.

    The geotransform is north-up: ``(origin_x, px_w, 0, origin_y, 0, px_h)``
    (``px_h`` is typically negative). Spatial reference, nodata, and optional
    per-band wavelength metadata are stamped onto the ``RasterDataSet`` object.
    """
    cube = _build_cube(num_bands, height, width, np.dtype(dtype), base_value)
    dataset = RasterDataSet(NumPyRasterDataImpl(cube))

    geo_transform = (origin[0], pixel_size[0], 0.0, origin[1], 0.0, pixel_size[1])
    dataset._set_geo_transform(geo_transform)
    dataset._set_wkt(wkt_for_epsg(epsg))

    if nodata is not None:
        dataset.set_data_ignore_value(nodata)

    if wavelengths is not None:
        if len(wavelengths) != num_bands:
            raise ValueError("wavelengths must have one entry per band")
        dataset.set_band_list(_wavelength_band_info(wavelengths, wavelength_units))

    return dataset


def write_gtiff(
    dest_path: Path,
    *,
    width: int = 8,
    height: int = 6,
    num_bands: int = 3,
    dtype: np.dtype = np.dtype(np.float32),
    origin: Tuple[float, float] = (300100.0, 3999900.0),
    pixel_size: Tuple[float, float] = (10.0, -10.0),
    epsg: int = DEFAULT_EPSG,
    nodata: Optional[float] = -9999.0,
    base_value: float = 1000.0,
) -> np.ndarray:
    """
    Write a small GeoTIFF to ``dest_path`` and return its [band][y][x] cube.

    Used both to seed disk-backed scenes and to assert that on-disk metadata is
    correctly overridden by a wrapping ``RasterDataSet``.
    """
    cube = _build_cube(num_bands, height, width, np.dtype(dtype), base_value)
    gdal_type = gdal_array.NumericTypeCodeToGDALTypeCode(np.dtype(dtype))

    driver = gdal.GetDriverByName("GTiff")
    ds = driver.Create(str(dest_path), width, height, num_bands, gdal_type)
    ds.SetGeoTransform((origin[0], pixel_size[0], 0.0, origin[1], 0.0, pixel_size[1]))
    ds.SetProjection(wkt_for_epsg(epsg))
    for b in range(num_bands):
        band = ds.GetRasterBand(b + 1)
        if nodata is not None:
            band.SetNoDataValue(float(nodata))
        band.WriteArray(cube[b])
    ds.FlushCache()
    ds = None
    return cube


def make_gtiff_scene(dest_path: Path, **kwargs) -> RasterDataSet:
    """
    Write a small GeoTIFF to ``dest_path`` and load it as a GDAL-backed
    ``RasterDataSet``. Accepts the same keyword arguments as :func:`write_gtiff`.
    """
    write_gtiff(dest_path, **kwargs)
    impls = GTiff_GDALRasterDataImpl.try_load_file(str(dest_path), interactive=False)
    if not impls:
        raise RuntimeError(f"Failed to load GeoTIFF scene from {dest_path}")
    return RasterDataSet(impls[0], data_cache=None)


def make_overlapping_scene_pair(tmp_dir: Path) -> Tuple[RasterDataSet, RasterDataSet]:
    """
    Return two georeferenced scenes (one NumPy-backed, one GeoTIFF-backed) in
    the same CRS whose footprints partially overlap.

    Scene A origin (300000, 4000000); Scene B origin shifted by +40m east and
    -40m south (4 pixels at 10 m), so the 8x6 footprints overlap but are not
    identical.
    """
    scene_a = make_numpy_scene(origin=(300000.0, 4000000.0), base_value=0.0)
    scene_b = make_gtiff_scene(
        tmp_dir / "scene_b.tif",
        origin=(300040.0, 3999960.0),
        base_value=1000.0,
    )
    return scene_a, scene_b
