import sys
import os
sys.path.append("C:\\Users\\jgarc\\OneDrive\\Documents\\Schmidt-Code\\WISER\\src\\wiser")
sys.path.append("C:\\Users\\jgarc\\OneDrive\\Documents\\Schmidt-Code\\WISER\\src")
from wiser.raster.dataset import RasterDataSet
from wiser.raster.loader import RasterDataLoader
from wiser.raster.roi import RegionOfInterest
from wiser.raster.selection import RectangleSelection
from wiser.raster.spectrum import calc_roi_spectrum
# from PySide2.QtCore import *
from wiser.bandmath.types import VariableType
from wiser.bandmath.analyzer import get_bandmath_expr_info
import cProfile
import pstats
import time
from typing import List, Dict, Any, Tuple
from wiser import bandmath
import numpy as np
import logging

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

def measure_bandmath_time(equation: str, variables: Dict[str, Tuple[VariableType, Any]], use_old_method = False):
    '''
    A helper function for measuring the time it takes to call eval_bandmath_expr
    '''
    expr_info = get_bandmath_expr_info(equation,
        variables, {})
    result_name = 'test_result'

    start_time = time.perf_counter()
    (_, result_dataset) = bandmath.eval_bandmath_expr(equation, expr_info, result_name,
        variables, {}, use_old_method)
    end_time = time.perf_counter()
    return end_time-start_time, result_dataset

def get_nan_count(arr: np.ndarray):
    '''
    A useful helper function for counting the number of nans in an array
    '''
    nan_count = np.isnan(arr).sum()
    return nan_count

def setup_logger(output_file):
    """Sets up the logger to output to both the console and a file."""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Create a file handler
    file_handler = logging.FileHandler(output_file)
    file_handler.setLevel(logging.INFO)

    # Create a console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Set the format for both handlers
    formatter = logging.Formatter('%(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

def stress_test_benchmark(large_band_dataset_path: str, normal_image_cube_path: str,
                          large_image_cube_path: str, use_both_methods = False, \
                            use_old_method = False, N = 1, output_file='output/bandmath_benchmark.txt'):
    '''
    Used to stress test the evaluator algorithm in 4 main scenarios that would commonly be used
    in WISER. Result mean and standard deviation of values are printed to the screen. You can
    choose to either use both methods, or just one of the methods to test. 
    '''
    logger = setup_logger(output_file)

    keyA = "A"
    keyB1 = "B-Normal Cube/Spectrum"
    keyB2 = "B2-Large Cube/Spectrum"
    keyC1 = "C-Normal Dotprod"
    keyC2 = "C-Large Dotprod"
    keyD1 = "D-Normal Atm Correction"
    keyD2 = "D-Large Atm Correction"

    stressing_equation_dict = {
        # keyA: '1.0 - 2.0*b1/(b2+b3)',
        # keyB1: 'nd / (ns*(ns>0.2)+ns<0.2)',
        keyB2: 'ld / (ls*(ls>0.2)+ls<0.2)',
        # keyC1: 'dotprod(nd, ns)',
        keyC2: 'dotprod(ld, ls)',
        # keyD1: 'nd*(2.718)**0.2', #Atmospheric correction
        keyD2: 'ld*(2.718)**0.2', 
    }

    loader = RasterDataLoader()
    large_band_dataset = loader.load_from_file(large_band_dataset_path)
    normal_image_cube = loader.load_from_file(normal_image_cube_path)
    large_image_cube = loader.load_from_file(large_image_cube_path)
    b1 = large_band_dataset.get_band_data(0)
    b2 = large_band_dataset.get_band_data(1)
    b3 = large_band_dataset.get_band_data(2)
    normal_spectrum = normal_image_cube.get_all_bands_at(normal_image_cube.get_width()/2, \
                                                         normal_image_cube.get_height()/2)
    large_spectrum = large_image_cube.get_all_bands_at(large_image_cube.get_width()/2, \
                                                       large_image_cube.get_height()/2)
    
    variables = {'b1':(VariableType.IMAGE_BAND, b1),
                'b2':(VariableType.IMAGE_BAND, b2),
                'b3':(VariableType.IMAGE_BAND, b3),
                'ns':(VariableType.SPECTRUM, normal_spectrum),
                'ls':(VariableType.SPECTRUM, large_spectrum),
                'nd':(VariableType.IMAGE_CUBE, normal_image_cube),
                'ld':(VariableType.IMAGE_CUBE, large_image_cube)}

    oper_file_time_dict = {}

    results_old_method = {
        keyA: None,
        keyB1: None,
        keyB2: None,
        keyC1: None,
        keyC2: None,
        keyD1: None,
        keyD2: None,
    }

    results_new_method = {
        keyA: None,
        keyB1: None,
        keyB2: None,
        keyC1: None,
        keyC2: None,
        keyD1: None,
        keyD2: None,
    }
    oper_file_time_dict_both = {}

    file_times = []
    for key, value in stressing_equation_dict.items():
        oper_times = []
        oper_times_new_method = []
        oper_times_old_method = []
        print(f"Operations: {key}")
        print(f"Equation: {value}")
        for iter in range(N):
            # print(f"iter: {iter}")
            time_outer = None
            if use_both_methods:
                # Whichever method goes first has to load the data from disk into memory so will take slower.
                # To somewhat account for this, increase N and get rid of the first observation
            
                print(f"New method calculating!")
                time_new_method, result_new_method = measure_bandmath_time(value, variables, use_old_method=False)
                print(f"New method done calculating!")
                if results_new_method[key] is None:
                    results_new_method[key] = result_new_method
                print(f"time_new_method: {time_new_method}")
                
                print(f"Old method calculating!")
                time_old_method, result_old_method = measure_bandmath_time(value, variables, use_old_method=True)
                print(f"Old method done calculating!")
                if results_old_method[key] is None:
                    results_old_method[key] = result_old_method
                print(f"time_old_method: {time_old_method}")

                oper_times_new_method.append(time_new_method)
                oper_times_old_method.append(time_old_method)
                time_outer = time_new_method
            elif use_old_method:
                print(f"Old method calculating!")
                time, _ = measure_bandmath_time(value, variables, use_old_method=use_old_method)
                print(f"Old method done calculating!")
                time_outer = time
            else:
                print(f"New method calculating!")
                time, _ = measure_bandmath_time(value, variables)
                print(f"New method done!")
                time_outer = time
            oper_times.append(time_outer)
        oper_file_time_dict[key] = oper_times
        oper_file_time_dict_both[key] = {}
        oper_file_time_dict_both[key]["new"] = oper_times_new_method
        oper_file_time_dict_both[key]["old"] = oper_times_old_method
        file_times += (oper_times)
        
    print("==========Oper File Time Benchmarks==========")
    logger.info("==========Oper File Time Benchmarks==========")
    if use_both_methods:
        for oper_file in oper_file_time_dict_both:
            new_times = oper_file_time_dict_both[oper_file]['new']
            old_times = oper_file_time_dict_both[oper_file]['old']
            print(f"{oper_file}: \n \
                  New times: \n \
                  \t Mean: {np.mean(new_times):.6f} \t Std: {np.std(new_times):.6f} \n \
                  Old times: \n \
                  \t Mean: {np.mean(old_times):.6f} \t Std: {np.std(old_times):.6f}")
            logger.info(f"{oper_file}: \n \
                  New times: \n \
                  \t Mean: {np.mean(new_times):.6f} \t Std: {np.std(new_times):.6f} \n \
                  Old times: \n \
                  \t Mean: {np.mean(old_times):.6f} \t Std: {np.std(old_times):.6f}")
    else:
        for oper_file in oper_file_time_dict.keys():
            print(f"{oper_file}:\n \
                    \t Mean: {np.mean(oper_file_time_dict[oper_file])} \n \
                    \t Std: {np.std(oper_file_time_dict[oper_file])}")
            logger.info(f"{oper_file}:\n \
                    \t Mean: {np.mean(oper_file_time_dict[oper_file])} \n \
                    \t Std: {np.std(oper_file_time_dict[oper_file])}")
    

def test_both_methods(hdr_paths, N=1):
    '''
    This function is to test if both evaluator methods give the same answer. Uncomment the commented
    sections below if you want to peak at the values that are not the same between the methods. 
    '''
    loader = RasterDataLoader()

    key_plus_1 = "+"
    key_mult = "*"
    key_div = "/"
    key_minus = "-"
    key_neg = "neg"
    key_less_than = "<"
    key_combo_1 = "/-*+"
    key_combo_2 = "-+<*"
    key_exponent = "**"
    key_formula = "formula"

    equation_dict = {
        key_plus_1: '(a+c)+((c+a)+((e+f)+(g+h)))',
        key_mult: '(a*b)*(c*d)',
        key_div: '(a/b)/(c/d)',
        key_minus: '(a-b)-(c-d)',
        key_neg: 'a+b',
        key_less_than: '((a-b)-d)<c',
        key_combo_1: '(a/b)-(c*d)+a',
        key_combo_2: "(((a-b)+d)<c)*a",
        key_exponent: "a**b+a**(0.5)",
        key_formula: "0.5*(1-(b/(0.4*k+0.6*l)))+0.5"
    }

    # Replacing keys in results_old_method
    results_old_method = {
        key_plus_1: None,
        key_mult: None,
        key_div: None,
        key_minus: None,
        key_neg: None,
        key_less_than: None,
        key_combo_1: None,
        key_combo_2: None,
        key_exponent: None,
        key_formula: None,
    }

    # Replacing keys in results_new_method
    results_new_method = {
        key_plus_1: None,
        key_mult: None,
        key_div: None,
        key_minus: None,
        key_neg: None,
        key_less_than: None,
        key_combo_1: None,
        key_combo_2: None,
        key_exponent: None,
        key_formula: None,
    }

    total_close = 0
    total_comparisons = 0
    hdr_files = get_hdr_files(hdr_paths)
    for hdr_file in hdr_files:
        base_name = os.path.basename(hdr_file)
        print(f"Going through file: {base_name}")
        dataset = loader.load_from_file(hdr_file)
        band = dataset.get_band_data(0)
        band2 = dataset.get_band_data(1)
        band3 = dataset.get_band_data(2)
        spectrum = dataset.get_all_bands_at(100, 100)
        spectrum2 = dataset.get_all_bands_at(120, 120)
        spectrum3 = dataset.get_all_bands_at(140, 140)
        variables = {'a':(VariableType.IMAGE_CUBE, dataset),
                    'c':(VariableType.IMAGE_CUBE, dataset),
                    'b':(VariableType.IMAGE_BAND, band),
                    'd':(VariableType.SPECTRUM, spectrum),
                    'e':(VariableType.IMAGE_CUBE, dataset),
                    'f':(VariableType.IMAGE_CUBE, dataset),
                    'g':(VariableType.IMAGE_CUBE, dataset),
                    'h':(VariableType.IMAGE_CUBE, dataset),
                    'i':(VariableType.SPECTRUM, spectrum2),
                    'j':(VariableType.SPECTRUM, spectrum3),
                    'k':(VariableType.IMAGE_BAND, band2),
                    'l':(VariableType.IMAGE_BAND, band3)}
    
        for key, value in equation_dict.items():
            print(f"Operation: {key}")
            print(f"Equation: {value}")
            print(f"New method calculating")
            _, result_new_method = measure_bandmath_time(value, variables, use_old_method=False)
            print(f"New method done calculating")
            if results_new_method[key] is None:
                results_new_method[key] = result_new_method
            print(f"Old method calculating!")
            _, result_old_method = measure_bandmath_time(value, variables, use_old_method=True)
            print(f"Old method done calculating!")
            if isinstance(result_new_method, RasterDataSet):
                arr_new_method = result_new_method.get_image_data()
            elif isinstance(result_new_method, np.ndarray):
                arr_new_method = result_new_method
            arr_old_method = result_old_method
            original_arr = dataset.get_image_data()
            print(f"np.assertequal: {np.allclose(arr_new_method, arr_old_method, equal_nan=True)}")
            if results_old_method[key] is None:
                results_old_method[key] = result_old_method
            are_close = np.allclose(arr_new_method, arr_old_method, rtol=1e-4, equal_nan=True)
            print(f"Are the arrays close: {are_close}")
            if are_close:
                total_close += 1
            total_comparisons += 1
            # Find elements that are not close
            not_close = ~np.isclose(arr_new_method, arr_old_method, rtol=1e-4)
            # Print indices and values that are not close
            amt_not_close = 0

            # if not are_close:
                # with open('output/close_indices.txt', 'w') as f:
                #     f.write("Pairs of values that are not close:\n")
                #     for index in np.argwhere(not_close):
                #         # Unpack all dimensions dynamically, uncomment this if results do not match
                #         index_str = ", ".join(map(str, index))
                #         mask_value = arr_old_method.mask[tuple(index)]
                #         # Write to the file instead of printing
                #         if not np.isnan(arr_new_method[tuple(index)]) and not np.isnan(arr_old_method[tuple(index)]):
                #             f.write(f"arr_new_method[{index_str}] = {arr_new_method[tuple(index)]}, "
                #                     f"arr_old_method[{index_str}] = {arr_old_method[tuple(index)]}, mask = {mask_value}\n")
            # else:
            #     print("All values are close within the given tolerance.")
            print(f"Amount not close: \n {amt_not_close}")
            print(f"Amount nan in each: \n new method: {get_nan_count(arr_new_method)} " +
                  f"\n old method: {get_nan_count(arr_old_method)}")
            print(f"Mean of original array: {np.mean(original_arr)}")
            print(f"Mean of new method: {np.nanmean(arr_new_method)}")
            print(f"Mean of old method: {np.nanmean(arr_old_method)}")
            assert np.allclose(arr_new_method, arr_old_method, equal_nan=True)
    print(f"The total fraction close are: {total_close} / {total_comparisons}")
    return results_new_method, results_old_method

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
    dataset_500mb = 'c:\\Users\\jgarc\\OneDrive\\Documents\\Data\\ang20171108t184227_corr_v2p13_subset_bil.hdr'
    dataset_500mb_copy = "c:\\Users\\jgarc\\OneDrive\\Documents\\Data\\caltech-pic-copy.hdr"
    dataset_900mb = 'C:\\Users\\jgarc\\OneDrive\\Documents\\Data\\RhinoLeft_2016_07_28_12_56_01_SWIRcalib_atmcorr.hdr'
    dataset_6B_3bands = "C:\\Users\\jgarc\\OneDrive\\Documents\\Data\\Task1.3_Slow_Histogram_calc_2Gb_img_or_mosaic\Gale_MSL_HiRISE_Color_Mosaic_warp.tif.hdr"
    dataset_6GB = "C:\\Users\\jgarc\\OneDrive\\Documents\\Data\\Benchmarks\\RhinoLeft_2016_07_28_12_56_01_SWIRcalib_atmcorr_expanded_lines_and_samples_2.hdr"
    dataset_15gb = "C:\\Users\\jgarc\\OneDrive\\Documents\\Data\\C5705B-00003Z-01_2018_07_28_14_18_38_VNIRcalib.hdr"
    dataset_20GB = "C:\\Users\\jgarc\\OneDrive\\Documents\\Data\\Task1.1_SlowBandMath_10gb\\ang20171108t184227_corr_v2p13_subset_bil_expanded_bands_by_40.hdr"
    dataset_list = [dataset_500mb]
    benchmark_folder = 'C:\\Users\jgarc\\OneDrive\\Documents\\Data\\Benchmarks'
    N = 1

    '''
    How to use test_both_methods with a list of datasets
    '''
    # test_both_methods(dataset_list)
    
    '''
    How to use test_both_methods with a folder that contain's .hdr paths
    '''
    # test_both_methods(benchmark_folder)

    '''
    How to use stress_test_benchmark
    '''
    stress_test_benchmark(dataset_15gb, dataset_500mb, dataset_6GB, use_both_methods=True, N=N)

    '''
    How to use the profiler for test_both_methods
    '''
    # profile_outpath = f"output/bandmath_test_both_N-{N}.txt"
    # profile_function(profile_outpath, test_both_methods, dataset_list)
    
    '''
    How to use the profiler for stress_test_benchmark
    '''
    # profile_outpath = f"output/bandmath_stress_test_N-{N}.txt"
    # profile_function(profile_outpath, stress_test_benchmark, dataset_500mb, dataset_500mb, dataset_500mb)
