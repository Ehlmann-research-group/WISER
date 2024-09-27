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

def benchmark_all_bandmath(hdr_paths: str, use_both_methods = False, use_old_method = False):
    equation_dict = {
        "+": '(a+b)+(c+d)',
        # "*": '(a*b)*(c*d)',
        # "/": '(a/b)/(c/d)',
        # "-": '(a-b)-(c-d)',
        # "<": '((a-b)-d)<c'
    }

    oper_file_time_dict = {}

    file_time_dict = {}
    hdr_files = get_hdr_files(hdr_paths)
    loader = RasterDataLoader()
    N = 10
    for hdr_file in hdr_files:
        base_name = os.path.basename(hdr_file)
        print(f"Going through file: {base_name}")
        dataset = loader.load_from_file(hdr_file)
        band = dataset.get_band_data(0)
        spectrum = dataset.get_all_bands_at(100, 100)
        variables = {'a':(VariableType.IMAGE_CUBE, dataset),
                    'c':(VariableType.IMAGE_CUBE, dataset),
                    'b':(VariableType.IMAGE_BAND, band),
                    'd':(VariableType.SPECTRUM, spectrum)}
        file_times = []
        file_times_new_method = []
        file_times_old_method = []
        for key, value in equation_dict.items():
            oper_times = []
            # oper_times_new_method = []
            # oper_times_old_method = []
            print(f"operations: {key}")
            print(f"equation: {value}")
            for _ in range(N):
                print(f"iter: {_}")
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
            

if __name__ == '__main__':
    '''
    It is okay for this profile to use an image that is not incredibly big because the sampler profile takes a long time to run
    '''
    dataset_500mb = 'c:\\Users\\jgarc\\OneDrive\\Documents\\Data\\ang20171108t184227_corr_v2p13_subset_bil.hdr'
    dataset_900mb = 'C:\\Users\\jgarc\\OneDrive\\Documents\\Data\\RhinoLeft_2016_07_28_12_56_01_SWIRcalib_atmcorr.hdr'
    dataset_20GB = "C:\\Users\\jgarc\\OneDrive\\Documents\\Data\\Task1.1_SlowBandMath_10gb\\ang20171108t184227_corr_v2p13_subset_bil_expanded_bands_by_40.hdr"
    dataset_list = [dataset_500mb]
    benchmark_folder = 'C:\\Users\jgarc\\OneDrive\\Documents\\Data\\Benchmarks'
    
    # benchmark_addition(dataset_list)
    benchmark_all_bandmath(dataset_list, use_both_methods=False, use_old_method=False)
    print('Done with profiling')

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
                         Mean: 7.553817027807236
                         Std: 2.1903186266950794
==========Oper File Time Benchmarks==========
*: ang20171108t184227_corr_v2p13_subset_bil.hdr:
                         Mean: 5.738137745857239
                         Std: 0.5395997628516592
/: ang20171108t184227_corr_v2p13_subset_bil.hdr:
                         Mean: 11.153445172309876
                         Std: 0.38876079517795353
-: ang20171108t184227_corr_v2p13_subset_bil.hdr:
                         Mean: 6.615255737304688
                         Std: 0.6748620085989032
<: ang20171108t184227_corr_v2p13_subset_bil.hdr:
                         Mean: 6.708429455757141
                         Std: 0.6653192305938764

Old method not out of core:
==========File Time Benchmarks==========
ang20171108t184227_corr_v2p13_subset_bil.hdr:
                         Mean: 3.6751036739349363
                         Std: 2.1291596821606737
==========Oper File Time Benchmarks==========
+: ang20171108t184227_corr_v2p13_subset_bil.hdr:
                         Mean: 2.6323692321777346
                         Std: 0.30737778636724084
*: ang20171108t184227_corr_v2p13_subset_bil.hdr:
                         Mean: 2.481278729438782
                         Std: 0.038514671852865896
/: ang20171108t184227_corr_v2p13_subset_bil.hdr:
                         Mean: 7.919213986396789
                         Std: 0.13509954030408053
-: ang20171108t184227_corr_v2p13_subset_bil.hdr:
                         Mean: 2.6504650831222536
                         Std: 0.047461862172893304
<: ang20171108t184227_corr_v2p13_subset_bil.hdr:
                         Mean: 2.6921913385391236
                         Std: 0.09565251545364144
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