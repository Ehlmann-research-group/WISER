import sys
sys.path.append("C:\\Users\\jgarc\\OneDrive\\Documents\\Schmidt-Code\\WISER\\src\\wiser")
sys.path.append("C:\\Users\\jgarc\\OneDrive\\Documents\\Schmidt-Code\\WISER\\src")
from wiser.raster.dataset import RasterDataSet
from wiser.raster.loader import RasterDataLoader
from wiser.raster.roi import RegionOfInterest
from wiser.raster.selection import RectangleSelection
from wiser.raster.spectrum import calc_roi_spectrum
# from PySide2.QtCore import *
from wiser.bandmath.types import BandMathValue, VariableType
from wiser.bandmath.builtins import OperatorAdd, OperatorCompare, OperatorDivide, OperatorMultiply, OperatorPower, OperatorSubtract, OperatorUnaryNegate
from wiser.gui.app import DataVisualizerApp

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *
from PySide2 import QtConcurrent
from typing import Any, Dict, List, Optional

from wiser.gui.rasterview import normalize_ndarray, StretchBase
from wiser.raster.stretch import StretchLog2, StretchHistEqualize, StretchComposite, StretchLinear, StretchSquareRoot

import cProfile
import pstats
import numpy as np
import time

def normalize_array_by_dim_1(array: np.ndarray, minval=None, maxval=None) -> np.ndarray:
    '''
    Normalize each 2D slice of the specified 3D array along the first dimension (b),
    using vectorized operations. If minval and maxval are provided as 1D arrays of 
    length `b`, they will be used for each respective slice. Each (x, y) slice is 
    normalized independently. NaN values are left unaffected.
    '''
    
    # If minval and maxval are not provided, compute them for each 2D slice
    if minval is None:
        minval = np.nanmin(array, axis=(1, 2))  # Shape (b,)

    if maxval is None:
        maxval = np.nanmax(array, axis=(1, 2))  # Shape (b,)

    # Reshape minval and maxval to (b, 1, 1) to enable broadcasting across (x, y)
    minval = minval[:, np.newaxis, np.newaxis]
    maxval = maxval[:, np.newaxis, np.newaxis]

    # Avoid division by zero by ensuring maxval > minval
    denominator = (maxval - minval)
    denominator[denominator == 0] = 1  # To handle cases where maxval == minval

    # Normalize the array (keeping NaN values unaffected)
    normalized_array = (array - minval) / denominator

    return normalized_array

def make_channel_image(dataset: RasterDataSet, band: int, stretch: StretchBase = None) -> np.ndarray:
    '''
    Given a raster data set, band index, and optional contrast stretch object,
    this function generates color channel data into a NumPy array.  Elements in
    the output array will be in the range [0, 255].
    '''

    # Time the extraction of raw band data and statistics
    start_time = time.perf_counter()
    raw_data = dataset.get_band_data(band)
    stats = dataset.get_band_stats(band)
    end_time = time.perf_counter()
    # print(f"MCI: Time for getting band data and stats: {end_time - start_time:.6f} seconds")

    # Time the normalization of raw band data
    start_time = time.perf_counter()
    band_data = normalize_ndarray(raw_data,
        minval=stats.get_min(), maxval=stats.get_max())
    end_time = time.perf_counter()
    # print(f"MCI: Time for normalizing band data: {end_time - start_time:.6f} seconds")

    # Time applying the stretch, if provided
    if stretch is not None:
        start_time = time.perf_counter()
        # print(f"MCI: Number of bytes band data: {band_data.nbytes}")
        stretch.apply(band_data)
        end_time = time.perf_counter()
        # print(f"MCI: Time for applying stretch: {end_time - start_time:.6f} seconds")

    # Time clipping the data
    start_time = time.perf_counter()
    np.clip(band_data, 0.0, 1.0, out=band_data)
    end_time = time.perf_counter()
    # print(f"MCI: Time for clipping band data: {end_time - start_time:.6f} seconds")

    # Time converting the data into color channel
    start_time = time.perf_counter()
    channel_data = (band_data * 255.0).astype(np.uint32)
    end_time = time.perf_counter()
    # print(f"MCI: Time for converting to color channel: {end_time - start_time:.6f} seconds")

    return channel_data

def make_channel_image_many_bands(dataset: RasterDataSet, band_list: List[int], stretch: StretchBase = None) -> np.ndarray:
    
    start_time = time.perf_counter()
    raw_bands = dataset.get_multiple_band_data(band_list)
    end_time = time.perf_counter()
    # print(f"MCI: Time for getting band data and stats: {end_time - start_time:.6f} seconds")
    
    start_time = time.perf_counter()
    band_data = normalize_array_by_dim_1(raw_bands)
    end_time = time.perf_counter()
    # print(f"MCI: Time for normalizing band data: {end_time - start_time:.6f} seconds")

    if stretch is not None:
        start_time = time.perf_counter()
        # print(f"MCI: Number of bytes band data: {band_data.nbytes}")
        stretch.apply(band_data)
        end_time = time.perf_counter()
        # print(f"MCI: Time for applying stretch: {end_time - start_time:.6f} seconds")

    start_time = time.perf_counter()
    np.clip(band_data, 0.0, 1.0, out=band_data)
    end_time = time.perf_counter()
    # print(f"MCI: Time for clipping band data: {end_time - start_time:.6f} seconds")

    # Time converting the data into color channel
    start_time = time.perf_counter()
    channel_data = (band_data * 255.0).astype(np.uint32)
    end_time = time.perf_counter()
    # print(f"MCI: Time for converting to color channel: {end_time - start_time:.6f} seconds")

    return channel_data

def profile(dataset_path: str):
    app = QApplication([])

    #========================================================================
    # WISER Application Initialization


    # Set the initial window size to be 70% of the screen size.
    wiser_ui = DataVisualizerApp()
    screen_size = app.screens()[0].size()
    wiser_ui.resize(screen_size * 0.7)
    wiser_ui.show()
    app_state = wiser_ui._app_state
    loader = RasterDataLoader()
    dataset1 = loader.load_from_file(dataset_path)

    app_state.add_dataset(dataset1)
    # profiler = cProfile.Profile()
    # profiler.enable()
    # print('================Enabled Profile================')
    wiser_ui._main_view.get_stretch_builder()._stretch_config._ui.rb_cond_none.setChecked(True)
    wiser_ui._main_view.get_stretch_builder()._stretch_config._ui.rb_cond_sqrt.setChecked(True)
    wiser_ui._main_view.get_stretch_builder()._on_conditioner_type_changed()

    # profiler.disable()
    # print('================Disabled Profile================')
    # # Save the profiling stats to a file
    # with open("output/bandmath_add_stats.txt", "w") as f:
    #     ps = pstats.Stats(profiler, stream=f)
    #     ps.sort_stats("tottime") 
    #     ps.print_stats()
    # sys.exit(app.exec_())

def time_function(N, func, *args, **kwargs):
    '''
    Run the specified function N times and record the execution time of each run.
    Returns the mean and standard deviation of the recorded times.
    
    Parameters:
    - func: The function to be executed.
    - N: The number of times to run the function.
    - *args: Positional arguments to be passed to the function.
    - **kwargs: Keyword arguments to be passed to the function.
    
    Returns:
    - mean_time: The mean of the execution times.
    - std_time: The standard deviation of the execution times.
    '''
    
    times = []
    
    # Run the function N times and record execution time
    for _ in range(N):
        start_time = time.perf_counter()
        func(*args, **kwargs)
        end_time = time.perf_counter()
        
        elapsed_time = end_time - start_time
        times.append(elapsed_time)
    
    # Convert the list of times to a numpy array for calculating mean and std
    times_array = np.array(times)
    mean_time = np.mean(times_array)
    std_time = np.std(times_array)
    
    return mean_time, std_time

def update_display_image_loop(dataset, stretch):
    time_1 = time.perf_counter()
    for i in range(3):
        make_channel_image(dataset, i+1, stretch)
    time_2 = time.perf_counter()
    print(f"For loop iter: {i} count since last: {time_2 - time_1:0.02f}")

def update_display_image_vec(dataset, stretch):
    band_list = list(range(3))
    time_1 = time.perf_counter()
    make_channel_image_many_bands(dataset, band_list, stretch)
    time_2 = time.perf_counter()
    print(f"Total: {time_2 - time_1:0.02f}")

def compare_both_methods(dataset, stretch, N=10):
    mean_loop, std_loop = time_function(N, update_display_image_loop, \
                                        dataset, stretch)

    mean_vec, std_vec =  time_function(N, update_display_image_vec, \
                                        dataset, stretch)

    print(f"Old method:\n \
            \t Mean: {mean_loop} \
            \t Std: {std_loop}")
    
    print(f"New method:\n \
            \t Mean: {mean_vec} \
            \t Std: {std_vec}")


def profile2(dataset_path: str):
    loader = RasterDataLoader()
    dataset = loader.load_from_file(dataset_path)
    log_stretch = StretchLog2()

    # compare_both_methods(dataset, log_stretch, N=100)
    
    QtConcurrent.run(compare_both_methods, dataset, log_stretch, N=100)
    # update_display_image_vec(dataset, log_stretch)
    # update_display_image_loop(dataset, log_stretch)
    # profiler = cProfile.Profile()
    # profiler.enable()
    # print('================Enabled Profile================')
    # make_channel_image(dataset, 1, log_stretch)
    # profiler.disable()
    # print('================Disabled Profile================')
    # # Save the profiling stats to a file
    # with open("output/hist_log_stats.txt", "w") as f:
    #     ps = pstats.Stats(profiler, stream=f)
    #     ps.sort_stats("tottime") 
    #     ps.print_stats()


if __name__ == '__main__':
    '''
    It is okay for this profile to use an image that is not incredibly big because the sampler profile takes a long time to run
    '''
    # dataset_path = "c:\\Users\\jgarc\\OneDrive\\Documents\\Data\\ang20171108t184227_corr_v2p13_subset_bil.hdr"
    # dataset_path = 'C:\\Users\\jgarc\\OneDrive\\Documents\\Data\\RhinoLeft_2016_07_28_12_56_01_SWIRcalib_atmcorr.hdr'
    large_bands = "C:\\Users\\jgarc\\OneDrive\\Documents\\Data\\C5705B-00003Z-01_2018_07_28_14_18_38_VNIRcalib.hdr"
    # dataset_path = "C:\\Users\\jgarc\\OneDrive\\Documents\\Data\\Task1.1_SlowBandMath_10gb\\ang20171108t184227_corr_v2p13_subset_bil_expanded_bands_by_40.hdr"
    # dataset_path = "C:\\Users\\jgarc\\OneDrive\\Documents\\Data\\Task1.1_SlowBandMath_10gb\\ang20171108t184227_corr_v2p13_subset_bil_increased_bands_by_80.hdr"

    profile2(large_bands)
    # profile_cube_band(dataset_path)
    # profile_cube_spectrum(dataset_path)
    print('Done with profiling')
