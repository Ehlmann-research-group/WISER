import os
import sys

import statistics as stats
import timeit
import numpy as np
from typing import Callable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from wiser.raster.loader import RasterDataLoader
from wiser.gui.permanent_plugins.continuum_removal_plugin import continuum_removal_image_numba, continuum_removal_image, continuum_removal, continuum_removal_numba
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
    result = profile_function("output/continuum_removal_image_500MB_try2.txt", continuum_removal_image, image_data, x_axis, rows, cols, bands)
    return result


def profile_continuum_removal_spectrum_numba():
        current_dir = os.path.dirname(os.path.abspath(__file__))

        # target_path = os.path.normpath(os.path.join(current_dir, "..", "..","test_utils", "test_datasets", "caltech_4_100_150_nm"))
        target_path = os.path.normpath(os.path.join("C:\\Users\\jgarc\\OneDrive\\Documents\\Data\\ang20171108t184227_corr_v2p13_subset_bil.hdr"))
        loader = RasterDataLoader()
        dataset = loader.load_from_file(target_path)[0]
        rows = np.int32(dataset.get_height())
        cols = np.int32(dataset.get_width())
        bands = np.int32(dataset.num_bands())
        spectrum = dataset.get_all_bands_at(rows//2, cols//2)
        if isinstance(spectrum, np.ma.MaskedArray):
            spectrum = spectrum.data
        if spectrum.dtype != np.float32:
            spectrum = spectrum.astype(np.float32)
        band_info = dataset.band_list()
        x_axis = np.array([band['wavelength'].value for band in band_info], dtype=np.float32)
        # result = profile_function("output/continuum_removal_spectrum_numba.txt", continuum_removal_numba, spectrum, x_axis)
        result = continuum_removal_numba(spectrum, x_axis)
        return result

def profile_continuum_spectrum_image():
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # target_path = os.path.normpath(os.path.join(current_dir, "..", "..","test_utils", "test_datasets", "caltech_4_100_150_nm"))
    target_path = os.path.normpath(os.path.join("C:\\Users\\jgarc\\OneDrive\\Documents\\Data\\ang20171108t184227_corr_v2p13_subset_bil.hdr"))
    loader = RasterDataLoader()
    dataset = loader.load_from_file(target_path)[0]
    rows = np.int32(dataset.get_height())
    cols = np.int32(dataset.get_width())
    spectrum = dataset.get_all_bands_at(rows//2, cols//2)
    if isinstance(spectrum, np.ma.MaskedArray):
        spectrum = spectrum.data
    if spectrum.dtype != np.float32:
        spectrum = spectrum.astype(np.float32)
    band_info = dataset.band_list()
    x_axis = np.array([band['wavelength'].value for band in band_info], dtype=np.float32)
    # result = profile_function("output/continuum_removal_spectrum.txt", continuum_removal, spectrum, x_axis)
    result = continuum_removal(spectrum, x_axis)
    return result

def _prepare_inputs():
    """
    Replicates your setup exactly once (IO/data prep not included in timing).
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # target_path = os.path.normpath(os.path.join(current_dir, "..", "..","test_utils", "test_datasets", "caltech_4_100_150_nm"))
    target_path = os.path.normpath(
        os.path.join("C:\\Users\\jgarc\\OneDrive\\Documents\\Data\\ang20171108t184227_corr_v2p13_subset_bil.hdr")
    )

    loader = RasterDataLoader()
    dataset = loader.load_from_file(target_path)[0]

    rows = np.int32(dataset.get_height())
    cols = np.int32(dataset.get_width())

    spectrum = dataset.get_all_bands_at(rows // 2, cols // 2)
    if isinstance(spectrum, np.ma.MaskedArray):
        spectrum = spectrum.data
    if spectrum.dtype != np.float32:
        spectrum = spectrum.astype(np.float32)

    band_info = dataset.band_list()
    x_axis = np.array([band["wavelength"].value for band in band_info], dtype=np.float32)
    return spectrum, x_axis

def time_continuum_removal(repeats: int = 50, warmups: int = 5, func: Callable = continuum_removal):
    """
    Times only the continuum_removal(spectrum, x_axis) call.

    Parameters
    ----------
    repeats : int
        How many timing runs to collect (each run executes 'number' iterations).
    warmups : int
        Warm-up runs to stabilize caches/JIT before measuring.

    Returns
    -------
    dict with statistics and raw timings.
    """
    spectrum, x_axis = _prepare_inputs()

    # Warm up (especially helpful if continuum_removal uses Numba or allocs)
    for _ in range(warmups):
        _ = func(spectrum, x_axis)

    # Build timer that ONLY calls the function (data already prepared)
    t = timeit.Timer(lambda: func(spectrum, x_axis))

    # Let timeit pick a reasonable inner-loop count for ~0.2s
    number, _ = t.autorange()

    # Gather timing samples
    samples = t.repeat(repeat=repeats, number=number)  # seconds per 'number' calls

    # Convert to per-call seconds for readability
    per_call = [s / number for s in samples]

    # Stats
    per_call_sorted = sorted(per_call)
    result = {
        "number_per_run": number,
        "repeats": repeats,
        "per_call_seconds": per_call,
        "min_s": min(per_call),
        "mean_s": stats.mean(per_call),
        "stdev_s": stats.pstdev(per_call) if len(per_call) > 1 else 0.0,
        "p50_s": per_call_sorted[len(per_call_sorted) // 2],
        "p90_s": per_call_sorted[int(0.9 * (len(per_call_sorted) - 1))],
    }

    # Pretty print
    print("=== continuum_removal timing (per call) ===")
    print(f"repeats={repeats}, number per run={number}")
    print(f"min   : {result['min_s']:.6f} s")
    print(f"mean  : {result['mean_s']:.6f} s  (stdev {result['stdev_s']:.6f} s)")
    print(f"p50   : {result['p50_s']:.6f} s")
    print(f"p90   : {result['p90_s']:.6f} s")
    return result

if __name__ == "__main__":

    # For timinng numba vs no numba on single spectrum
    print(f"=======Numba Timing=======")
    time_continuum_removal(func=continuum_removal_numba)
    print(f"=======No Numba Timing=======")
    time_continuum_removal(func=continuum_removal)

    # # For profiling numba
    # numba_result = profile_continuum_removal_spectrum_numba()
    # image_result = profile_continuum_spectrum_image()
    # print(f"equal? {np.allclose(numba_result, image_result)}")
    # print(f"numba_result: {numba_result[0:10,0:10,0:10]}")
    # print(f"image_result: {image_result[0:10,0:10,0:10]}")