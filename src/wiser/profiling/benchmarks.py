# ruff: noqa: E402
# ruff: noqa: E501
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from wiser.raster.dataset import RasterDataSet
from wiser.raster.loader import RasterDataLoader
from wiser.raster.spectrum import calc_roi_spectrum
from wiser.raster.roi import RegionOfInterest
from wiser.raster.selection import (
    Selection,
    RectangleSelection,
    SinglePixelSelection,
    PolygonSelection,
    MultiPixelSelection,
)

# from PySide2.QtCore import *
from wiser.bandmath.types import VariableType
from wiser.bandmath.analyzer import get_bandmath_expr_info
import cProfile
import pstats
import time
from typing import Dict, Any, Tuple
from wiser import bandmath
import numpy as np

from wiser.profiling.bandmath_benchmark import stress_test_benchmark, test_both_methods

from wiser.gui.app_state import ApplicationState
from wiser.gui.rasterpane import RasterPane
from wiser.gui.app import DataVisualizerApp

import logging
import traceback

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

app = QApplication([])  # Initialize the QApplication


def get_hdr_files(folder_path):
    """
    Helper function for getting all of the hdr files in a folder
    """
    if isinstance(folder_path, str):
        hdr_files = []
        # Walk through the directory
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.endswith(".hdr"):
                    # Get absolute file path and append to the list
                    hdr_files.append(os.path.join(os.path.abspath(root), file))
        return hdr_files
    return folder_path


# Set up logging to track crashes
logging.basicConfig(filename="output/app_test.log", level=logging.DEBUG)


def benchmark_function(dataset_paths, function_to_test, N=3, output_file="output/benchmark_results.txt"):
    """
    Benchmarks the provided function on a list of dataset paths.

    Parameters:
    - dataset_paths: List of strings, paths to the .hdr files.
    - function_to_test: Function, the function to benchmark. It should take a single argument: dataset_path.
    - N: Integer, the number of times to run the function on each dataset.
    - output_file: String, the file to write the benchmark results to.

    Outputs:
    - Prints the timing results to the console.
    - Writes the timing results to the specified output file.
    """

    # Ensure the directory for the log file exists
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Clear the file contents at the start
    with open(output_file, "w"):
        pass  # Opening in 'w' mode clears the file

    # Create a separate logger for the benchmark function
    logger = logging.getLogger("benchmark_logger")
    logger.setLevel(logging.INFO)

    # Remove any existing handlers from the logger
    if logger.hasHandlers():
        logger.handlers.clear()

    # Create a file handler for the output file
    fh = logging.FileHandler(output_file)
    fh.setLevel(logging.INFO)

    # Optionally, create a console handler if you want logs to also appear on the console
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # Create a formatter and set it for both handlers
    formatter = logging.Formatter("%(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)

    print(f"Running: {function_to_test.__name__}")
    for dataset_path in dataset_paths:
        if not os.path.isfile(dataset_path):
            print(f"File not found: {dataset_path}")
            logging.info(f"File not found: {dataset_path}")
            continue

        times = []
        print(f"\nBenchmarking on dataset: {dataset_path}")
        logger.info(f"\nBenchmarking on dataset: {dataset_path}")

        for i in range(N):
            start_time = time.time()
            try:
                function_to_test(dataset_path)
            except Exception as e:
                print(f"Error during function execution: {e}")
                logger.error(f"Error during function execution: {e}")
                break  # Exit the loop if the function fails
            end_time = time.time()
            elapsed_time = end_time - start_time
            times.append(elapsed_time)
            print(f"Run {i+1}/{N}: {elapsed_time:.4f} seconds")
            logger.info(f"Run {i+1}/{N}: {elapsed_time:.4f} seconds")

        if times:
            mean_time = np.mean(times)
            std_time = np.std(times)
            print(f"Mean time over {N} runs: {mean_time:.4f} seconds")
            print(f"Standard deviation: {std_time:.4f} seconds")
            logger.info(f"Mean time over {N} runs: {mean_time:.4f} seconds")
            logger.info(f"Standard deviation: {std_time:.4f} seconds")
        else:
            print("No valid runs were recorded.")
            logger.info("No valid runs were recorded.")


def profile_function(profile_outpath, func, *args, **kwargs):
    profiler = cProfile.Profile()

    profiler.enable()  # Start profiling
    print("================Enabled Profile================")

    # Call the function with its arguments and keyword arguments
    result = func(*args, **kwargs)

    profiler.disable()  # Stop profiling
    print("================Disabled Profile================")

    with open(profile_outpath, "w+") as f:
        ps = pstats.Stats(profiler, stream=f)
        ps.sort_stats("tottime")
        ps.print_stats()

    print("Done with profiling")

    return result


def run_function_in_ui(dataset_path, func):
    wiser_ui = None

    try:
        # Set up the GUI
        wiser_ui = DataVisualizerApp()
        wiser_ui.show()

        loader = RasterDataLoader()
        dataset = loader.load_from_file(path=dataset_path, data_cache=wiser_ui._data_cache)[0]

        # Create an application state, no need to pass the app here
        app_state = wiser_ui._app_state

        # raster_pane = RasterPane(app_state)
        app_state.add_dataset(dataset)

        func(dataset, wiser_ui, app_state)

        # This should happen X milliseconds after the above stuff runs
        QTimer.singleShot(100, app.quit)
        # Run the application event loop
        app.exec_()

    except Exception as e:
        logging.error(f"Application crashed: {e}")
        traceback.print_exc()

    finally:
        if wiser_ui:
            wiser_ui.close()


def open_and_display_dataset(dataset_path):
    def func(dataset: RasterDataSet, wiser_ui: DataVisualizerApp, app_state: ApplicationState):
        return

    run_function_in_ui(dataset_path, func)


def use_stretch_builder_linear_gui(dataset_path: str):
    def func(dataset: RasterDataSet, wiser_ui: DataVisualizerApp, app_state: ApplicationState):
        wiser_ui._main_view._on_stretch_builder()
        stretch_builder = wiser_ui._main_view._stretch_builder
        stretch_config = stretch_builder._stretch_config

        stretch_config._ui.button_linear_2_5.click()

    run_function_in_ui(dataset_path, func)


def use_stretch_builder_equalize_gui(dataset_path: str):
    def func(dataset: RasterDataSet, wiser_ui: DataVisualizerApp, app_state: ApplicationState):
        wiser_ui._main_view._on_stretch_builder()
        stretch_builder = wiser_ui._main_view._stretch_builder
        stretch_config = stretch_builder._stretch_config

        stretch_config._ui.rb_stretch_equalize.click()

    run_function_in_ui(dataset_path, func)


def use_stretch_builder_sqrt_gui(dataset_path: str):
    def func(dataset: RasterDataSet, wiser_ui: DataVisualizerApp, app_state: ApplicationState):
        wiser_ui._main_view._on_stretch_builder()
        stretch_builder = wiser_ui._main_view._stretch_builder
        stretch_config = stretch_builder._stretch_config

        stretch_config._ui.rb_cond_sqrt.click()

    run_function_in_ui(dataset_path, func)


def use_stretch_builder_log2_gui(dataset_path: str):
    def func(dataset: RasterDataSet, wiser_ui: DataVisualizerApp, app_state: ApplicationState):
        wiser_ui._main_view._on_stretch_builder()
        stretch_builder = wiser_ui._main_view._stretch_builder
        stretch_config = stretch_builder._stretch_config

        stretch_config._ui.rb_cond_log.click()

    run_function_in_ui(dataset_path, func)


def calculate_roi_average_spectrum_rectangle(dataset_path):
    def func(dataset: RasterDataSet, wiser_ui: DataVisualizerApp, app_state: ApplicationState):
        raster_width = dataset.get_width()
        raster_height = dataset.get_height()

        roi_one_tenth = RegionOfInterest(name="roi_one_tenth")
        roi_one_tenth.add_selection(
            RectangleSelection(QPoint(0, 0), QPoint(int(raster_width / 10), int(raster_height / 10)))
        )

        # Create an application state, no need to pass the app here
        app_state = wiser_ui._app_state

        app_state.add_dataset(dataset)
        app_state.add_roi(roi_one_tenth)

        main_view = wiser_ui._main_view
        wiser_ui._main_view._on_show_roi_avg_spectrum(roi_one_tenth, main_view._rasterviews[(0, 0)])

    run_function_in_ui(dataset_path, func)


def calculate_roi_average_spectrum_polygon(dataset_path):
    def func(dataset: RasterDataSet, wiser_ui: DataVisualizerApp, app_state: ApplicationState):
        raster_width = dataset.get_width()
        raster_height = dataset.get_height()

        roi = RegionOfInterest(name="roi_polygon")
        roi.add_selection(
            PolygonSelection(
                [
                    QPoint(0, 0),
                    QPoint(int(raster_width / 3), int(raster_height / 3)),
                    QPoint(int(raster_width / 5), int(raster_height / 5)),
                ]
            )
        )

        # Create an application state, no need to pass the app here
        app_state = wiser_ui._app_state

        app_state.add_dataset(dataset)
        app_state.add_roi(roi)

        main_view = wiser_ui._main_view
        wiser_ui._main_view._on_show_roi_avg_spectrum(roi, main_view._rasterviews[(0, 0)])

    run_function_in_ui(dataset_path, func)


def calculate_roi_average_spectrum_multipix(dataset_path):
    def func(dataset: RasterDataSet, wiser_ui: DataVisualizerApp, app_state: ApplicationState):
        raster_width = dataset.get_width()
        raster_height = dataset.get_height()

        roi = RegionOfInterest(name="roi_multipixel")
        pixel_list = [QPoint(int(raster_width / i), int(raster_height / i)) for i in range(50)]
        roi.add_selection(MultiPixelSelection(pixel_list))

        # Create an application state, no need to pass the app here
        app_state = wiser_ui._app_state

        app_state.add_dataset(dataset)
        app_state.add_roi(roi)

        main_view = wiser_ui._main_view
        wiser_ui._main_view._on_show_roi_avg_spectrum(roi, main_view._rasterviews[(0, 0)])

    run_function_in_ui(dataset_path, func)


if __name__ == "__main__":
    """
    How to use this file:
        1. Create aboslute paths to the .hdr files of your datasets as you see below
        2. Put those paths into the list 'dataset_list' as variables
        3. Use one of the functions currently made or make your own and then pass that function
        into benchmark_function as a parameter with the list of datasets and output file path
            3.5. tHE  output file path should write to the output folder as is done below
        4. If you add a new function, incremment total_func by one and add succ_func+=1 to the try block as is below
        5. Lastly, in a terminal that has python run `python .\benchmarks.py` while in the parent folder
        of this file
    """
    dataset_500mb = (
        "C:\\Users\\jgarc\\OneDrive\\Documents\\Data\\ang20171108t184227_corr_v2p13_subset_bil.hdr"
    )
    dataset_900mb = (
        "C:\\Users\\jgarc\\OneDrive\\Documents\\Data\\RhinoLeft_2016_07_28_12_56_01_SWIRcalib_atmcorr.hdr"
    )
    dataset_6B_3bands = "C:\\Users\\jgarc\\OneDrive\\Documents\\Data\\Task1.3_Slow_Histogram_calc_2Gb_img_or_mosaic\\Gale_MSL_HiRISE_Color_Mosaic_warp.tif.hdr"
    dataset_6GB = "C:\\Users\\jgarc\\OneDrive\\Documents\\Data\\Benchmarks\\RhinoLeft_2016_07_28_12_56_01_SWIRcalib_atmcorr_expanded_lines_and_samples_2.hdr"
    dataset_15gb = (
        "C:\\Users\\jgarc\\OneDrive\\Documents\\Data\\C5705B-00003Z-01_2018_07_28_14_18_38_VNIRcalib.hdr"
    )
    dataset_20GB = "C:\\Users\\jgarc\\OneDrive\\Documents\\Data\\Task1.1_SlowBandMath_10gb\\ang20171108t184227_corr_v2p13_subset_bil_expanded_bands_by_40.hdr"
    dataset_list = [dataset_15gb]
    benchmark_folder = "C:\\Users\jgarc\\OneDrive\\Documents\\Data\\Benchmarks"
    N = 1

    # calculate_roi_average_spectrum(dataset_900mb)
    # use_stretch_builder(dataset_900mb)

    open_and_display_dataset_funcs = [open_and_display_dataset]
    use_stretch_builder_funcs = [
        use_stretch_builder_linear_gui,
        use_stretch_builder_equalize_gui,
        use_stretch_builder_sqrt_gui,
        use_stretch_builder_log2_gui,
    ]
    roi_avg_funcs = [
        calculate_roi_average_spectrum_rectangle,
        calculate_roi_average_spectrum_polygon,
        calculate_roi_average_spectrum_multipix,
    ]

    succ_func = 0
    total_func = len(open_and_display_dataset_funcs) + len(use_stretch_builder_funcs) + len(roi_avg_funcs)
    # try:
    #     print("Running open_and_display_dataset functions...")
    #     for open_and_display_dataset_func in open_and_display_dataset_funcs:
    #         benchmark_function([dataset_6B_3bands], open_and_display_dataset_func, \
    #                         output_file='output/display_dataset_results.txt', N=2)
    #         succ_func +=1
    # except BaseException as e:
    #     print(f"Opening and displaying dataset failed with: \n {e}")

    try:
        print("Running use_stretch_builder functions...")
        for stretch_builder_func in use_stretch_builder_funcs:
            benchmark_function(
                dataset_list,
                stretch_builder_func,
                output_file=f"output/stretch_builder_results_{stretch_builder_func.__name__}.txt",
            )
            succ_func += 1
    except Exception as e:
        print(f"Error in use_stretch_builder: {e}")

    try:
        print("Running calculate_roi_average_spectrum functions...")
        for roi_avg_func in roi_avg_funcs:
            benchmark_function(
                dataset_list,
                roi_avg_func,
                output_file=f"output/roi_avg_results_{roi_avg_func.__name__}.txt",
            )
            succ_func += 1
    except Exception as e:
        print(f"Error in calculate_roi_average_spectrum: {e}")

    # try:
    #     print("Running stress test benchmark...")
    #     stress_test_benchmark(dataset_6GB, dataset_500mb, dataset_6GB, use_both_methods=False, N=N, \
    #                         output_file='output/stress_test_benchmark.txt')
    #     succ_func +=1
    # except Exception as e:
    #     print(f"Error in stress_test_benchmark: {e}")

    print(f"Functions successful: {succ_func} / {total_func}")
    sys.exit(1)
