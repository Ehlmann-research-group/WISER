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
from wiser.bandmath.types import BandMathValue, VariableType
from wiser.bandmath.builtins import OperatorAdd, OperatorCompare, OperatorDivide, OperatorMultiply, OperatorPower, OperatorSubtract, OperatorUnaryNegate
from wiser.bandmath.analyzer import get_bandmath_expr_info
import cProfile
import pstats
import time
from typing import List, Dict, Any, Tuple
from wiser import bandmath
import numpy as np

def get_hdr_files(folder_path):
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

def calc_func_speed(hdr_paths: List[str], func, roi: RegionOfInterest):
    '''calculates the speed that it takes for an average roi to be calculated for all the hdrs passed in'''
    loader = RasterDataLoader()
    times = []
    for hdr_path in hdr_paths:
        print(f"Running the file: {hdr_path}")
        dataset = loader.load_from_file(hdr_path)
        start_time = time.time()
        func(dataset, roi)
        end_time = time.time()
        times.append(end_time-start_time)
    return times

def benchmark_addition(hdr_paths: str):
    equation = '(a+b)+(c+d)'
    
    hdr_files = get_hdr_files(hdr_paths)
    loader = RasterDataLoader()
    times = []
    for hdr_file in hdr_files:
        print(f"Going through file: {os.path.basename(hdr_file)}")
        dataset = loader.load_from_file(hdr_file)
        band = dataset.get_band_data(0)
        spectrum = dataset.get_all_bands_at(100, 100)

        variables = {'a':(VariableType.IMAGE_CUBE, dataset),
                'c':(VariableType.IMAGE_CUBE, dataset),
                'b':(VariableType.IMAGE_BAND, band),
                'd':(VariableType.SPECTRUM, spectrum)}

        expr_info = get_bandmath_expr_info(equation,
            variables, {})
        result_name = 'test_result'

        start_time = time.time()
        (result_type, result_dataset) = bandmath.eval_bandmath_expr(equation, expr_info, result_name,
            variables, {})
        end_time = time.time()
        times.append(end_time-start_time)
    return times

def measure_bandmath_time(equation: str, variables: Dict[str, Tuple[VariableType, Any]], use_old_method = False):
    expr_info = get_bandmath_expr_info(equation,
        variables, {})
    result_name = 'test_result'

    start_time = time.time()
    (result_type, result_dataset) = bandmath.eval_bandmath_expr(equation, expr_info, result_name,
        variables, {}, use_old_method)
    end_time = time.time()
    return end_time-start_time, result_dataset


def stress_test_benchmark(hdr_paths: str, use_both_methods = False, use_old_method = False, N = 1):
    equation_dict = {
        "+": '(a+b)+(c+d)'
        # "*": '(a*b)*(c*d)',
        # "/": '(a/b)/(c/d)',
        # "-": '(a-b)-(c-d)',
        # "<": '((a-b)-d)<c'
    }
    stressing_equation_dict = {
        "A": '1.0 - 2.0*a/(c+e)',
        "B": 'a / (d*(d>0.2)+d<0.2)',
        "C": 'dotprod(a, d)',
        "D": 'a*(2.718)**0.2', #Atmospheric correction
        # "+": '(a+b)+((c+d)+(e+f)+(g+h))',
        # "*": '(a*b)*(c*d)',
        # "/": '(a/b)/(c/d)',
        # "-": '(a-b)-(c-d)',
        # "neg": '-a+1',
        # "<": '((a-b)-d)<c',
        # "/-*+" : '(a/b)-(c*d)+a',
        # "--<*": "(((a-b)-d)<c)*a",
        # "**": "a**b-(a**0.5)",
        # "formula": "0.5*(1-(b/(0.4*i+0.6*j)))+0.5"
    }

    oper_file_time_dict = {}

    file_time_dict = {}
    hdr_files = get_hdr_files(hdr_paths)
    loader = RasterDataLoader()
    results_old_method = {
        "+": None,
        "*": None,
        "/": None,
        "-": None,
        "<": None 
    }

    results_new_method = {
            "+": None,
            "*": None,
            "/": None,
            "-": None,
            "<": None 
    }
    for hdr_file in hdr_files:
        base_name = os.path.basename(hdr_file)
        print(f"Going through file: {base_name}")
        dataset = loader.load_from_file(hdr_file)
        band1 = dataset.get_band_data(0)
        band2 = dataset.get_band_data(1)
        band3 = dataset.get_band_data(2)
        spectrum1 = dataset.get_all_bands_at(100, 100)
        spectrum2 = dataset.get_all_bands_at(120, 120)
        spectrum3 = dataset.get_all_bands_at(140, 140)
        variables = {'a':(VariableType.IMAGE_CUBE, dataset),
                    'c':(VariableType.IMAGE_CUBE, dataset),
                    'b':(VariableType.IMAGE_BAND, band1),
                    'd':(VariableType.SPECTRUM, spectrum1),
                    'e':(VariableType.IMAGE_CUBE, dataset),
                    'f':(VariableType.IMAGE_CUBE, dataset),
                    'g':(VariableType.IMAGE_CUBE, dataset),
                    'h':(VariableType.IMAGE_CUBE, dataset),
                    'i':(VariableType.SPECTRUM, spectrum2),
                    'j':(VariableType.SPECTRUM, spectrum3),
                    'k':(VariableType.SPECTRUM, band2),
                    'l':(VariableType.SPECTRUM, band3)}
        file_times = []
        file_times_new_method = []
        file_times_old_method = []
        for key, value in stressing_equation_dict.items():
            oper_times = []
            # oper_times_new_method = []
            # oper_times_old_method = []
            print(f"operations: {key}")
            print(f"equation: {value}")
            for _ in range(N):
                # print(f"iter: {_}")
                time_outer = None
                if use_both_methods:
                    print(f"results new method calculating")
                    time_new_method, result_new_method = measure_bandmath_time(value, variables, use_old_method=False)
                    if results_new_method[key] is None:
                        results_new_method[key] = result_new_method
                    print(f"results old method calculating!")
                    time_old_method, result_old_method = measure_bandmath_time(value, variables, use_old_method=True)
                    print(f"results old method done!")
                    # print(f"key: {key}")
                    # print(f"results_old_method[key]: {results_old_method[key]}")
                    if results_old_method[key] is None:
                        results_old_method[key] = result_old_method
                    # oper_times_new_method.append(time_new_method)
                    # oper_times_old_method.append(time_old_method)
                    time_outer = time_new_method
                elif use_old_method:
                    print(f"results old method calculating!")
                    time, result = measure_bandmath_time(value, variables, use_old_method=use_old_method)
                    print(f"results old method done!")
                    time_outer = time
                else:
                    print(f"results new method calculating!")
                    time, result = measure_bandmath_time(value, variables)
                    print(f"results new method done!")
                    time_outer = time

                oper_times.append(time_outer)
            oper_file_time_dict[f"{key}: {base_name}"] = oper_times
            file_times += (oper_times)
        file_time_dict[base_name] = file_times
    
    print(f"oper_file_time_dict: {oper_file_time_dict}")
    print(f"file_time_dict: {file_time_dict}")

    print("==========File Time Benchmarks==========")
    for file in file_time_dict.keys():
        print(f"{file}:\n \
                \t Mean: {np.mean(file_time_dict[file])} \n \
                \t Std: {np.std(file_time_dict[file])}")
        
    print("==========Oper File Time Benchmarks==========")
    for oper_file in oper_file_time_dict.keys():
        print(f"{oper_file}:\n \
                \t Mean: {np.mean(oper_file_time_dict[oper_file])} \n \
                \t Std: {np.std(oper_file_time_dict[oper_file])}")
    
    if use_both_methods:
        print("Using both methods")
        for key, value in results_new_method.items():
            result_new_method = value
            result_old_method = results_old_method[key]

            result_new_value = result_new_method.get_image_data()
            result_old_value = result_old_method
            # result_new_value = result_new_method.reshape(result_new_method.get_shape())
            # result_old_value = result_old_value.reshape(result_old_method.get_shape())
            # print(f"OPERATION: {key}")
            # print(f"result_new_value.shape) {result_new_value.shape} ?==? {result_old_value.shape} (result_old_value.shape")
            # print(f"all close? {np.allclose(result_new_value, result_old_value, rtol=1e-4)}")
            # print(f"new_value: {result_new_value[:,100:110,100:110]}")
            # print(f"old_value: {result_old_value[:,100:110,100:110]}")
            assert(result_new_value.shape == result_old_value.shape)
            # np.testing.assert_allclose(result_new_value, result_old_value)

def benchmark_all_bandmath(hdr_paths: List[str], use_both_methods = False, use_old_method = False, N = 1):
    equation_dict = {
        "+": '(a+b)+(c+d)'
        # "*": '(a*b)*(c*d)',
        # "/": '(a/b)/(c/d)',
        # "-": '(a-b)-(c-d)',
        # "<": '((a-b)-d)<c'
    }
    equation_dict = {
        "+": '(a+b)+((c+d)+(e+f)+(g+h))',
        # "*": '(a*b)*(c*d)',
        # "/": '(a/b)/(c/d)',
        # "-": '(a-b)-(c-d)',
        # "neg": '-a+1',
        # "<": '((a-b)-d)<c',
        # "/-*+" : '(a/b)-(c*d)+a',
        # "--<*": "(((a-b)-d)<c)*a",
        # "**": "a**b-(a**0.5)",
        # "formula": "0.5*(1-(b/(0.4*i+0.6*j)))+0.5"
    }

    oper_file_time_dict = {}

    file_time_dict = {}
    hdr_files = get_hdr_files(hdr_paths)
    loader = RasterDataLoader()
    results_old_method = {
        "+": None,
        "*": None,
        "/": None,
        "-": None,
        "<": None 
    }

    results_new_method = {
            "+": None,
            "*": None,
            "/": None,
            "-": None,
            "<": None 
    }
    oper_file_time_dict_both = {}
    for hdr_file in hdr_files:
        base_name = os.path.basename(hdr_file)
        print(f"Going through file: {base_name}")
        dataset = loader.load_from_file(hdr_file)
        band = dataset.get_band_data(0)
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
                    'j':(VariableType.SPECTRUM, spectrum3)}
        file_times = []
        file_times_new_method = []
        file_times_old_method = []
        for key, value in equation_dict.items():
            oper_times = []
            oper_times_new_method = []
            oper_times_old_method = []
            print(f"operations: {key}")
            print(f"equation: {value}")
            for _ in range(N):
                # print(f"iter: {_}")
                time_outer = None
                if use_both_methods:
                    print(f"results new method calculating")
                    time_new_method, result_new_method = measure_bandmath_time(value, variables, use_old_method=False)
                    if results_new_method[key] is None:
                        results_new_method[key] = result_new_method
                    print(f"results old method calculating!")
                    time_old_method, result_old_method = measure_bandmath_time(value, variables, use_old_method=True)
                    print(f"results old method done!")
                    # print(f"key: {key}")
                    # print(f"results_old_method[key]: {results_old_method[key]}")
                    if results_old_method[key] is None:
                        results_old_method[key] = result_old_method
                    oper_times_new_method.append(time_new_method)
                    oper_times_old_method.append(time_old_method)
                    time_outer = time_new_method
                elif use_old_method:
                    print(f"results old method calculating!")
                    time, result = measure_bandmath_time(value, variables, use_old_method=use_old_method)
                    print(f"results old method done!")
                    time_outer = time
                else:
                    print(f"results new method calculating!")
                    time, result = measure_bandmath_time(value, variables)
                    print(f"results new method done!")
                    time_outer = time

                oper_times.append(time_outer)
            oper_file_time_dict[f"{key}: {base_name}"] = oper_times
            oper_file_time_dict_both[f"{key}: {base_name}"] = {}
            oper_file_time_dict_both[f"{key}: {base_name}"]["new"] = oper_times_new_method
            oper_file_time_dict_both[f"{key}: {base_name}"]["old"] = oper_times_old_method
            file_times += (oper_times)
        file_time_dict[base_name] = file_times
    
    print(f"oper_file_time_dict: {oper_file_time_dict}")
    print(f"file_time_dict: {file_time_dict}")

    print("==========File Time Benchmarks==========")
    for file in file_time_dict.keys():
        print(f"{file}:\n \
                \t Mean: {np.mean(file_time_dict[file])} \n \
                \t Std: {np.std(file_time_dict[file])}")
        
    print("==========Oper File Time Benchmarks==========")
    for oper_file in oper_file_time_dict.keys():
        print(f"{oper_file}:\n \
                \t Mean: {np.mean(oper_file_time_dict[oper_file])} \n \
                \t Std: {np.std(oper_file_time_dict[oper_file])}")
    
    if use_both_methods:
        print("Using both methods")
        for key, value in results_new_method.items():
            if value is not None:
                result_new_method = value
                result_old_method = results_old_method[key]

                print(f"Type of result new method: {type(result_new_method)}")
                result_new_value = result_new_method.get_image_data()
                result_old_value = result_old_method
                # result_new_value = result_new_method.reshape(result_new_method.get_shape())
                # result_old_value = result_old_value.reshape(result_old_method.get_shape())
                # print(f"OPERATION: {key}")
                # print(f"result_new_value.shape) {result_new_value.shape} ?==? {result_old_value.shape} (result_old_value.shape")
                # print(f"all close? {np.allclose(result_new_value, result_old_value, rtol=1e-4)}")
                # print(f"new_value: {result_new_value[:,100:110,100:110]}")
                # print(f"old_value: {result_old_value[:,100:110,100:110]}")
                assert(result_new_value.shape == result_old_value.shape)
                # np.testing.assert_allclose(result_new_value, result_old_value)
        for oper_file in oper_file_time_dict_both:
            new_times = oper_file_time_dict_both[oper_file]['new']
            old_times = oper_file_time_dict_both[oper_file]['old']
            print(f"{oper_file}: \n \
                  New times: \n \
                  \t Mean: {np.mean(new_times):.6f} \t Std: {np.std(new_times):.6f} \n \
                  Old times: \n \
                  \t Mean: {np.mean(old_times):.6f} \t Std: {np.std(old_times):.6f}")


def get_nan_count(arr: np.ndarray):
    nan_count = np.isnan(arr).sum()
    return nan_count

def test_both_methods(hdr_paths, N=1):
    loader = RasterDataLoader()
    equation_dict = {
        # "+": '(a+c)+((c+a)+((e+f)+(g+h)))'
        # "+": '(a+c)+((c+a)+(e+f)+(g+h))',
        # "*": '(a*b)*(c*d)',
        # "/": '(a/b)/(c/d)',
        # "-": '(a-b)-(c-d)',
        # "neg": '-a+1',
        # "<": '((a-b)-d)<c',
        # "/-*+" : '(a/b)-(c*d)+a',
        "--<*": "(((a-b)-d)<c)*a",
        # "**": "a**b-(a**0.5)",
        # "formula": "0.5*(1-(b/(0.4*i+0.6*j)))+0.5"
    }
    results_old_method = {
        "+": None,
        "*": None,
        "/": None,
        "-": None,
        "neg": None,
        "<": None,
        "/-*+": None,
        "--<*": None,
        "**": None,
        "formula": None
    }

    results_new_method = {
        "+": None,
        "*": None,
        "/": None,
        "-": None,
        "neg": None,
        "<": None,
        "/-*+": None,
        "--<*": None,
        "**": None,
        "formula": None
    }

    caltech1 = 'c:\\Users\\jgarc\\OneDrive\\Documents\\Data\\ang20171108t184227_corr_v2p13_subset_bil.hdr'
    caltech2 = "c:\\Users\\jgarc\\OneDrive\\Documents\\Data\\caltech-pic-copy.hdr"
    caltech1_dataset = loader.load_from_file(caltech1)
    caltech2_dataset = loader.load_from_file(caltech2)

    hdr_files = get_hdr_files(hdr_paths)
    for hdr_file in hdr_files:
        base_name = os.path.basename(hdr_file)
        print(f"Going through file: {base_name}")
        dataset = loader.load_from_file(hdr_file)
        print(f"Bad bands: {len(dataset._bad_bands)}")
        print(f"Shape: {dataset.get_shape()}")
        band = dataset.get_band_data(0)
        spectrum = dataset.get_all_bands_at(100, 100)
        spectrum2 = dataset.get_all_bands_at(120, 120)
        spectrum3 = dataset.get_all_bands_at(140, 140)
        variables = {'a':(VariableType.IMAGE_CUBE, dataset),
                    'c':(VariableType.IMAGE_CUBE, caltech2_dataset),
                    'b':(VariableType.IMAGE_BAND, band),
                    'd':(VariableType.SPECTRUM, spectrum),
                    'e':(VariableType.IMAGE_CUBE, dataset),
                    'f':(VariableType.IMAGE_CUBE, dataset),
                    'g':(VariableType.IMAGE_CUBE, dataset),
                    'h':(VariableType.IMAGE_CUBE, dataset),
                    'i':(VariableType.SPECTRUM, spectrum2),
                    'j':(VariableType.SPECTRUM, spectrum3)}
    
        for key, value in equation_dict.items():
            # oper_times_new_method = []
            # oper_times_old_method = []
            print(f"operations: {key}")
            print(f"equation: {value}")
            # print(f"iter: {_}")
            time_outer = None
            print(f"results new method calculating")
            time_new_method, result_new_method = measure_bandmath_time(value, variables, use_old_method=False)
            if results_new_method[key] is None:
                results_new_method[key] = result_new_method
            print(f"results old method calculating!")
            time_old_method, result_old_method = measure_bandmath_time(value, variables, use_old_method=True)
            arr_new_method = result_new_method.get_image_data()
            arr_old_method = result_old_method
            original_arr = dataset.get_image_data()
            # print(f"type of arr_new_method: {arr_new_method}")
            # print(f"type of arr_old_method: {arr_old_method}")
            print(f"np.assertequal: {np.allclose(arr_new_method, arr_old_method)}")
            print(f"arr_new_shape: {arr_new_method.shape}")
            print(f"arr_old_shape: {arr_old_method.shape}")
            print(f"arr_new_method[69, 82, 573] = {arr_new_method[69, 82, 573]}")
            print(f"arr_old_method[69, 82, 573] = {arr_old_method[69, 82, 573]}")
            print(f"results old method done!")
            # print(f"key: {key}")
            # print(f"results_old_method[key]: {results_old_method[key]}")
            if results_old_method[key] is None:
                results_old_method[key] = result_old_method
            # oper_times_new_method.append(time_new_method)
            # oper_times_old_method.append(time_old_method)
            # Use np.allclose to check if arrays are close
            # Use np.allclose to check if arrays are close
            are_close = np.allclose(arr_new_method, arr_old_method, rtol=1e-4, equal_nan=True)
            print(f"Are the arrays close: {are_close}")

            # Find elements that are not close
            not_close = ~np.isclose(arr_new_method, arr_old_method, rtol=1e-4)

            # Print indices and values that are not close
            amt_not_close = 0
            if np.any(not_close):
                print("Pairs of values that are not close:")
                for index in np.argwhere(not_close):
                    # # Unpack all dimensions dynamically
                    # index_str = ", ".join(map(str, index))
                    # print(f"arr_new_method[{index_str}] = {arr_new_method[tuple(index)]}, arr_old_method[{index_str}] = {arr_old_method[tuple(index)]}")
                    if amt_not_close == 0:
                        print(f"original arr[10:11,100:105,100:101] = \n {original_arr[tuple(index)]}")
                        print(f"arr_new_method[10:11,100:105,100:101] = \n {arr_new_method[tuple(index)]}")
                        print(f"arr_old_method[10:11,100:105,100:101] = \n {arr_old_method[tuple(index)]}")
                        
                        # assert arr_old_method[tuple(index)] == 2 * original_arr[tuple(index)]
                    # print(f" new method : {arr_new_method[tuple(index)]}")
                    amt_not_close += 1
            else:
                print("All values are close within the given tolerance.")
            print(f"Amount not close: \n {amt_not_close}")
            print(f"Amount nan in each: \n new method: {get_nan_count(arr_new_method)} " +
                  f"\n old method: {get_nan_count(arr_old_method)}")
            print(f"mean of original array: {np.mean(original_arr)}")
            print(f"mean of new method: {np.nanmean(arr_new_method)}")
            print(f"mean of old method: {np.nanmean(arr_old_method)}")
            print()
            assert np.isclose(np.nanmean(arr_new_method), np.mean(arr_old_method))
            assert np.allclose(arr_new_method, arr_old_method, equal_nan=True)
    return results_new_method, results_old_method


if __name__ == '__main__':
    '''
    It is okay for this profile to use an image that is not incredibly big because the sampler profile takes a long time to run
    '''
    dataset_500mb = 'c:\\Users\\jgarc\\OneDrive\\Documents\\Data\\ang20171108t184227_corr_v2p13_subset_bil.hdr'
    dataset_500mb_copy = "c:\\Users\\jgarc\\OneDrive\\Documents\\Data\\caltech-pic-copy.hdr"
    dataset_900mb = 'C:\\Users\\jgarc\\OneDrive\\Documents\\Data\\RhinoLeft_2016_07_28_12_56_01_SWIRcalib_atmcorr.hdr'
    dataset_6GB = "C:\\Users\\jgarc\\OneDrive\\Documents\\Data\\Benchmarks\\RhinoLeft_2016_07_28_12_56_01_SWIRcalib_atmcorr_expanded_lines_and_samples_2.hdr"
    dataset_15gb = "C:\\Users\\jgarc\\OneDrive\\Documents\\Data\\C5705B-00003Z-01_2018_07_28_14_18_38_VNIRcalib.hdr"
    dataset_20GB = "C:\\Users\\jgarc\\OneDrive\\Documents\\Data\\Task1.1_SlowBandMath_10gb\\ang20171108t184227_corr_v2p13_subset_bil_expanded_bands_by_40.hdr"
    dataset_list = [dataset_500mb]
    benchmark_folder = 'C:\\Users\jgarc\\OneDrive\\Documents\\Data\\Benchmarks'
    N = 1

    test_both_methods(dataset_list)
    # benchmark_all_bandmath(dataset_list, use_both_methods=False, N=N)
    # # benchmark_addition(dataset_list)
    # use_old_method = False
    # profiler = cProfile.Profile()
    # profiler.enable()
    # print('================Enabled Profile================')
    # benchmark_all_bandmath(dataset_list, use_both_methods=False, use_old_method=use_old_method, N=N)
    # profiler.disable()
    # print('================Disabled Profile================')
    # print('Done with profiling')

    # # Save the profiling stats to a file
    # with open(f"output/bandmath_menmark_old_method_{use_old_method}_N-{N}_20GB_future_random.txt", "w+") as f:
    #     ps = pstats.Stats(profiler, stream=f)
    #     ps.sort_stats("tottime")
    #     ps.print_stats()
'''
My method, not out of core memory
==========File Time Benchmarks==========
ang20171108t184227_corr_v2p13_subset_bil.hdr:
                         Mean: 7.542485070228577
                         Std: 2.4022795338605065
RhinoLeft_2016_07_28_12_56_01_SWIRcalib_atmcorr.hdr:
                         Mean: 17.648072028160094
                         Std: 13.978789891880394
==========Oper File Time Benchmarks==========
+: ang20171108t184227_corr_v2p13_subset_bil.hdr:
                         Mean: 8.612649321556091
                         Std: 1.7267318964004517
*: ang20171108t184227_corr_v2p13_subset_bil.hdr:
                         Mean: 6.710284352302551
                         Std: 0.1203080415725708
/: ang20171108t184227_corr_v2p13_subset_bil.hdr:
                         Mean: 11.458780169487
                         Std: 0.20197069644927979
-: ang20171108t184227_corr_v2p13_subset_bil.hdr:
                         Mean: 5.507996201515198
                         Std: 0.008822321891784668
<: ang20171108t184227_corr_v2p13_subset_bil.hdr:
                         Mean: 5.4227153062820435
                         Std: 0.10021007061004639
+: RhinoLeft_2016_07_28_12_56_01_SWIRcalib_atmcorr.hdr:
                         Mean: 36.11972641944885
                         Std: 22.562442541122437
*: RhinoLeft_2016_07_28_12_56_01_SWIRcalib_atmcorr.hdr:
                         Mean: 13.411945700645447
                         Std: 0.1275874376296997
/: RhinoLeft_2016_07_28_12_56_01_SWIRcalib_atmcorr.hdr:
                         Mean: 16.497625827789307
                         Std: 2.4480559825897217
-: RhinoLeft_2016_07_28_12_56_01_SWIRcalib_atmcorr.hdr:
                         Mean: 9.286758184432983
                         Std: 0.726743221282959
<: RhinoLeft_2016_07_28_12_56_01_SWIRcalib_atmcorr.hdr:
                         Mean: 12.924304008483887
                         Std: 2.955629587173462

Previous method, not out of core memory:
==========File Time Benchmarks==========
ang20171108t184227_corr_v2p13_subset_bil.hdr:
                         Mean: 0.024712491035461425
                         Std: 0.008103873437199982
RhinoLeft_2016_07_28_12_56_01_SWIRcalib_atmcorr.hdr:
                         Mean: 0.020456504821777344
                         Std: 0.0030501536120402482
==========Oper File Time Benchmarks==========
+: ang20171108t184227_corr_v2p13_subset_bil.hdr:
                         Mean: 0.03227221965789795
                         Std: 0.005271792411804199
*: ang20171108t184227_corr_v2p13_subset_bil.hdr:
                         Mean: 0.021501660346984863
                         Std: 0.004502654075622559
/: ang20171108t184227_corr_v2p13_subset_bil.hdr:
                         Mean: 0.03377687931060791
                         Std: 0.006221890449523926
-: ang20171108t184227_corr_v2p13_subset_bil.hdr:
                         Mean: 0.01800692081451416
                         Std: 0.0009940862655639648
<: ang20171108t184227_corr_v2p13_subset_bil.hdr:
                         Mean: 0.018004775047302246
                         Std: 0.0010088682174682617
+: RhinoLeft_2016_07_28_12_56_01_SWIRcalib_atmcorr.hdr:
                         Mean: 0.02176344394683838
                         Std: 0.0022345781326293945
*: RhinoLeft_2016_07_28_12_56_01_SWIRcalib_atmcorr.hdr:
                         Mean: 0.018503189086914062
                         Std: 0.0004966259002685547
/: RhinoLeft_2016_07_28_12_56_01_SWIRcalib_atmcorr.hdr:
                         Mean: 0.018006563186645508
                         Std: 5.4836273193359375e-06
-: RhinoLeft_2016_07_28_12_56_01_SWIRcalib_atmcorr.hdr:
                         Mean: 0.020508766174316406
                         Std: 0.0004928112030029297
<: RhinoLeft_2016_07_28_12_56_01_SWIRcalib_atmcorr.hdr:
                         Mean: 0.023500561714172363
                         Std: 0.004498839378356934

My method, not out of core
==========File Time Benchmarks==========
ang20171108t184227_corr_v2p13_subset_bil.hdr:
                         Mean: 6.877469959259034
                         Std: 2.6801841444979897
==========Oper File Time Benchmarks==========
+: ang20171108t184227_corr_v2p13_subset_bil.hdr:
                         Mean: 5.1696594715118405
                         Std: 0.4378767794464763
*: ang20171108t184227_corr_v2p13_subset_bil.hdr:
                         Mean: 5.110408473014831
                         Std: 0.1849520397889482
/: ang20171108t184227_corr_v2p13_subset_bil.hdr:
                         Mean: 12.023122239112855
                         Std: 0.6437421777592739
-: ang20171108t184227_corr_v2p13_subset_bil.hdr:
                         Mean: 6.22693452835083
                         Std: 0.8877227970632332
<: ang20171108t184227_corr_v2p13_subset_bil.hdr:
                         Mean: 5.85722508430481
                         Std: 0.7124463910432807

Old method not out of core:
==========File Time Benchmarks==========
ang20171108t184227_corr_v2p13_subset_bil.hdr:
                         Mean: 3.7085764360427858
                         Std: 2.1897731021880555
==========Oper File Time Benchmarks==========
+: ang20171108t184227_corr_v2p13_subset_bil.hdr:
                         Mean: 2.6810461282730103
                         Std: 0.35526421572213307
*: ang20171108t184227_corr_v2p13_subset_bil.hdr:
                         Mean: 2.5091066360473633
                         Std: 0.07491073803559681
/: ang20171108t184227_corr_v2p13_subset_bil.hdr:
                         Mean: 8.068095517158508
                         Std: 0.24763072493266286
-: ang20171108t184227_corr_v2p13_subset_bil.hdr:
                         Mean: 2.6754376888275146
                         Std: 0.0556550159616684
<: ang20171108t184227_corr_v2p13_subset_bil.hdr:
                         Mean: 2.6091962099075316
                         Std: 0.05764504523467106
Done with profiling

My method, out of core memory:
==========File Time Benchmarks==========
ang20171108t184227_corr_v2p13_subset_bil_expanded_bands_by_40.hdr:
                         Mean: 614.4646313905716
                         Std: 528.3629144705588
==========Oper File Time Benchmarks==========
+: ang20171108t184227_corr_v2p13_subset_bil_expanded_bands_by_40.hdr:
                         Mean: 386.13062584400177
                         Std: 20.1355322599411
*: ang20171108t184227_corr_v2p13_subset_bil_expanded_bands_by_40.hdr:
                         Mean: 377.27920615673065
                         Std: 19.014050126075745
/: ang20171108t184227_corr_v2p13_subset_bil_expanded_bands_by_40.hdr:
                         Mean: 626.469718337059
                         Std: 0.33723437786102295
-: ang20171108t184227_corr_v2p13_subset_bil_expanded_bands_by_40.hdr:
                         Mean: 405.63125586509705
                         Std: 8.66338849067688
<: ang20171108t184227_corr_v2p13_subset_bil_expanded_bands_by_40.hdr:
                         Mean: 1276.8123507499695
                         Std: 896.7392926216125
'''