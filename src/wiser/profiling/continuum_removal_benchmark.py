import os
import sys

import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from wiser.raster.loader import RasterDataLoader
from wiser.gui.permanent_plugins.continuum_removal_plugin import continuum_removal_image_numba, continuum_removal_image
from wiser.profiling.benchmarks import profile_function

def profile_continuum_removal_image_numba():
        current_dir = os.path.dirname(os.path.abspath(__file__))

        # target_path = os.path.normpath(os.path.join(current_dir, "..", "..","test_utils", "test_datasets", "caltech_4_100_150_nm"))
        target_path = os.path.normpath(os.path.join("C:\\Users\\jgarc\\OneDrive\\Documents\\Data\\ang20171108t184227_corr_v2p13_subset_bil.hdr"))
        loader = RasterDataLoader()
        dataset = loader.load_from_file(target_path)[0]
        image_data = dataset.get_image_data()
        if isinstance(image_data, np.ma.MaskedArray):
            image_data = image_data.data
        if image_data.dtype != np.float32:
            image_data = image_data.astype(np.float32)
        band_info = dataset.band_list()
        x_axis = np.array([band['wavelength'].value for band in band_info], dtype=np.float32)
        rows = np.int32(dataset.get_height())
        cols = np.int32(dataset.get_width())
        bands = np.int32(dataset.num_bands())
        print(f"x_axis: {x_axis}")
        result = profile_function("output/continuum_removal_image_numba_500MB_try2.txt", continuum_removal_image_numba, image_data, x_axis, rows, cols, bands)
        return result

def profile_continuum_removal_image():
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # target_path = os.path.normpath(os.path.join(current_dir, "..", "..","test_utils", "test_datasets", "caltech_4_100_150_nm"))
    target_path = os.path.normpath(os.path.join("C:\\Users\\jgarc\\OneDrive\\Documents\\Data\\ang20171108t184227_corr_v2p13_subset_bil.hdr"))
    loader = RasterDataLoader()
    dataset = loader.load_from_file(target_path)[0]
    image_data = dataset.get_image_data()
    if isinstance(image_data, np.ma.MaskedArray):
        image_data = image_data.data
    band_info = dataset.band_list()
    x_axis = np.array([band['wavelength'].value for band in band_info], dtype=np.float32)
    rows = np.int32(dataset.get_height())
    cols = np.int32(dataset.get_width())
    bands = np.int32(dataset.num_bands())
    print(f"x_axis: {x_axis}")
    result = profile_function("output/continuum_removal_image_500MB_try2.txt", continuum_removal_image, image_data, x_axis, rows, cols, bands)
    return result

if __name__ == "__main__":
    numba_result = profile_continuum_removal_image_numba()
    image_result = profile_continuum_removal_image()
    print(f"equal? {np.allclose(numba_result, image_result)}")
    print(f"numba_result: {numba_result[0:10,0:10,0:10]}")
    print(f"image_result: {image_result[0:10,0:10,0:10]}")