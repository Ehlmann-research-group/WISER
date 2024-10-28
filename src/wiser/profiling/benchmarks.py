import sys
import os
sys.path.append("C:\\Users\\jgarc\\OneDrive\\Documents\\Schmidt-Code\\WISER\\src\\wiser")
sys.path.append("C:\\Users\\jgarc\\OneDrive\\Documents\\Schmidt-Code\\WISER\\src")
from wiser.raster.dataset import RasterDataSet
from wiser.raster.loader import RasterDataLoader
from wiser.raster.spectrum import calc_roi_spectrum
from wiser.raster.roi import RegionOfInterest
from wiser.raster.selection import Selection, RectangleSelection, SinglePixelSelection, PolygonSelection, MultiPixelSelection

# from PySide2.QtCore import *
from wiser.bandmath.types import VariableType
from wiser.bandmath.analyzer import get_bandmath_expr_info
import cProfile
import pstats
import time
from typing import Dict, Any, Tuple
from wiser import bandmath
import numpy as np


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
    '''
    Helper function for getting all of the hdr files in a folder
    '''
    if isinstance(folder_path, str):
        hdr_files = []
        # Walk through the directory
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.endswith('.hdr'):
                    # Get absolute file path and append to the list
                    hdr_files.append(os.path.join(os.path.abspath(root), file))
        return hdr_files
    return folder_path

def benchmark_function(dataset_paths, function_to_test, N=1, output_file='output/benchmark_results.txt'):
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
    # Set up logging to write to a file
    logging.basicConfig(filename=output_file, level=logging.INFO, filemode='w')
    
    for dataset_path in dataset_paths:
        if not os.path.isfile(dataset_path):
            print(f"File not found: {dataset_path}")
            logging.info(f"File not found: {dataset_path}")
            continue
        
        times = []
        print(f"\nBenchmarking on dataset: {dataset_path}")
        logging.info(f"\nBenchmarking on dataset: {dataset_path}")
        
        for i in range(N):
            start_time = time.time()
            try:
                function_to_test(dataset_path)
            except Exception as e:
                print(f"Error during function execution: {e}")
                logging.error(f"Error during function execution: {e}")
                break  # Exit the loop if the function fails
            end_time = time.time()
            elapsed_time = end_time - start_time
            times.append(elapsed_time)
            print(f"Run {i+1}/{N}: {elapsed_time:.4f} seconds")
            logging.info(f"Run {i+1}/{N}: {elapsed_time:.4f} seconds")
        
        if times:
            mean_time = np.mean(times)
            std_time = np.std(times)
            print(f"Mean time over {N} runs: {mean_time:.4f} seconds")
            print(f"Standard deviation: {std_time:.4f} seconds")
            logging.info(f"Mean time over {N} runs: {mean_time:.4f} seconds")
            logging.info(f"Standard deviation: {std_time:.4f} seconds")
        else:
            print("No valid runs were recorded.")
            logging.info("No valid runs were recorded.")

def profile_function(profile_outpath, func, *args, **kwargs):
    profiler = cProfile.Profile()
    
    profiler.enable()  # Start profiling
    print('================Enabled Profile================')
    
    # Call the function with its arguments and keyword arguments
    result = func(*args, **kwargs)
    
    profiler.disable()  # Stop profiling
    print('================Disabled Profile================')

    with open(profile_outpath, "w+") as f:
        ps = pstats.Stats(profiler, stream=f)
        ps.sort_stats("tottime")
        ps.print_stats()

    print('Done with profiling')
    
    return result

# Set up logging to track crashes
logging.basicConfig(filename="output/app_test.log", level=logging.DEBUG)

def run_function_in_ui(dataset_path, func):
    wiser_ui = None

    try:
        # Set up the GUI
        wiser_ui = DataVisualizerApp()
        wiser_ui.show()

        loader = RasterDataLoader()
        dataset = loader.load_from_file(dataset_path)

        # Create an application state, no need to pass the app here
        app_state = wiser_ui._app_state

        # raster_pane = RasterPane(app_state)
        app_state.add_dataset(dataset)

        func(dataset, wiser_ui, app_state)

        # This should happen X milliseconds after the ahove stuff runs
        QTimer.singleShot(100, app.quit)
        # Run the application event loop
        app.exec_()

    except Exception as e:
        logging.error(f"Application crashed: {e}")
        traceback.print_exc()

    finally:
        if wiser_ui:
            wiser_ui.close()
        # ensure_qapp_closes(app)

def use_stretch_builder(dataset_path: str):
    def func(dataset: RasterDataSet, wiser_ui: DataVisualizerApp, app_state: ApplicationState):
        wiser_ui._main_view._on_stretch_builder()
        stretch_builder = wiser_ui._main_view._stretch_builder
        stretch_config = stretch_builder._stretch_config
        
        # Set histogram equalization (expensive)
        stretch_config._ui.rb_stretch_equalize.setChecked(True)
        stretch_config._on_stretch_radio_button(True)

        # Set log conditioner (expensive) 
        stretch_config._ui.rb_cond_log.setChecked(True)
        stretch_config._on_conditioner_radio_button(True)
    run_function_in_ui(dataset_path, func)

def open_and_display_dataset(dataset_path):
    def func(dataset: RasterDataSet, wiser_ui: DataVisualizerApp, app_state: ApplicationState):
        return
    run_function_in_ui(dataset_path, func)

def calculate_roi_average_spectrum(dataset_path):
    def func(dataset: RasterDataSet, wiser_ui: DataVisualizerApp, app_state: ApplicationState):
        raster_width = dataset.get_width()
        raster_height = dataset.get_height()

        roi_one_tenth = RegionOfInterest(name="roi_one_tenth")
        roi_one_tenth.add_selection(RectangleSelection(QPoint(0, 0), \
                                                    QPoint(int(raster_width/10), int(raster_height/10))))

        # Create an application state, no need to pass the app here
        app_state = wiser_ui._app_state

        app_state.add_dataset(dataset)
        app_state.add_roi(roi_one_tenth)

        main_view = wiser_ui._main_view
        wiser_ui._main_view._on_show_roi_avg_spectrum(roi_one_tenth, \
                                                        main_view._rasterviews[(0,0)])
    run_function_in_ui(dataset_path, func)

if __name__ == '__main__':
    '''
    How to use this file:
        1. Create aboslute paths to the .hdr files of your datasets as you see below
        2. If you are using the function test_both_methods function then you can either pass
        in an absolute path to a folder where you want to run all of your benchmarking or you can
        pass in a list of the file paths like in the variable dataset_list. The function will go through
        each of these files and ensure both methods get the same answer for all the equations in 
        equation_dict. Feel free to only use the equations you need. 
        3. If you are using stress_test_benchmark, then you will need to pass in 3 paths to datasets.
        The first path is to a dataset with very large bands. The second path is to a normal sized dataset
        (which is roughly defined as one that fits into RAM). The third path is to a large dataset (which 
        is roughly defined as one that does not fit into RAM).
        4. Look at the commented sections below to see examples
        5. Lastly, in a terminal that has python run `python .\bandmath_benchmark.py` while in the parent folder
        of this file
    '''
    dataset_500mb = "C:\\Users\\jgarc\\OneDrive\\Documents\\Data\\ang20171108t184227_corr_v2p13_subset_bil.hdr"
    dataset_900mb = 'C:\\Users\\jgarc\\OneDrive\\Documents\\Data\\RhinoLeft_2016_07_28_12_56_01_SWIRcalib_atmcorr.hdr'
    dataset_6B_3bands = "C:\\Users\\jgarc\\OneDrive\\Documents\\Data\\Task1.3_Slow_Histogram_calc_2Gb_img_or_mosaic\\Gale_MSL_HiRISE_Color_Mosaic_warp.tif.hdr"
    dataset_6GB = "C:\\Users\\jgarc\\OneDrive\\Documents\\Data\\Benchmarks\\RhinoLeft_2016_07_28_12_56_01_SWIRcalib_atmcorr_expanded_lines_and_samples_2.hdr"
    dataset_20GB = "C:\\Users\\jgarc\\OneDrive\\Documents\\Data\\Task1.1_SlowBandMath_10gb\\ang20171108t184227_corr_v2p13_subset_bil_expanded_bands_by_40.hdr"
    dataset_list = [dataset_900mb]
    benchmark_folder = 'C:\\Users\jgarc\\OneDrive\\Documents\\Data\\Benchmarks'
    N = 1

    # calculate_roi_average_spectrum(dataset_900mb)
    # use_stretch_builder(dataset_900mb)
    succ_func = 0
    total_func = 3
    try:
        print("Running open_and_display_dataset...")
        benchmark_function(dataset_list, open_and_display_dataset, \
                           output_file='output/display_dataset_results.txt')
        succ_func +=1 
    except BaseException as e:
        print(f"Opening and displaying dataset failed with: \n {e}")
        sys.exit(1)

    try:
        print("Running use_stretch_builder...")
        benchmark_function(dataset_list, use_stretch_builder, \
                           output_file='output/stretch_builder_results.txt')
        succ_func +=1 
    except Exception as e:
        print(f"Error in use_stretch_builder: {e}")
        sys.exit(1) 

    try:
        print("Running calculate_roi_average_spectrum...")
        benchmark_function(dataset_list, calculate_roi_average_spectrum, \
                           output_file='output/roi_avg_results.txt')
        succ_func +=1 
    except Exception as e:
        print(f"Error in calculate_roi_average_spectrum: {e}")
        sys.exit(1) 

    print(f"Functions successful: {succ_func} / {total_func}")
