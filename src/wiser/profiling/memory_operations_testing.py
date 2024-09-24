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
import cProfile
import pstats
import numpy as np
import time
from wiser.gui.bandmath_dialog import BandMathDialog

def profile(dataset_path: str):
    
    loader = RasterDataLoader()
    dataset1 = loader.load_from_file(dataset_path)
    dataset2 = loader.load_from_file(dataset_path)

    lhs = BandMathValue(VariableType.IMAGE_CUBE, dataset1)
    rhs = BandMathValue(VariableType.IMAGE_CUBE, dataset2)

    # profiler = cProfile.Profile()
    # profiler.enable()
    # print('================Enabled Profile================')
    res = OperatorAdd().apply([lhs, rhs])
    # profiler.disable()
    # print('================Disabled Profile================')
    # # Save the profiling stats to a file
    # with open("output/bandmath_add_stats.txt", "w") as f:
    #     ps = pstats.Stats(profiler, stream=f)
    #     ps.sort_stats("tottime") 
    #     ps.print_stats()

def profile_cube_band(dataset_path: str):
    
    loader = RasterDataLoader()
    dataset1 = loader.load_from_file(dataset_path)
    dataset2 = loader.load_from_file(dataset_path)

    lhs = BandMathValue(VariableType.IMAGE_CUBE, dataset1)
    rhs = BandMathValue(VariableType.IMAGE_BAND, dataset2.get_band_data(0))

    # profiler = cProfile.Profile()
    # profiler.enable()
    # print('================Enabled Profile================')
    res = OperatorAdd().apply([lhs, rhs])
    # profiler.disable()
    # print('================Disabled Profile================')
    # # Save the profiling stats to a file
    # with open("output/bandmath_add_stats.txt", "w") as f:
    #     ps = pstats.Stats(profiler, stream=f)
    #     ps.sort_stats("tottime") 
    #     ps.print_stats()

def profile_cube_spectrum(dataset_path: str):
    
    loader = RasterDataLoader()
    dataset1 = loader.load_from_file(dataset_path)
    dataset2 = loader.load_from_file(dataset_path)

    lhs = BandMathValue(VariableType.IMAGE_CUBE, dataset1)
    rhs = BandMathValue(VariableType.SPECTRUM, dataset2.get_all_bands_at(50, 50))

    # profiler = cProfile.Profile()
    # profiler.enable()
    # print('================Enabled Profile================')
    res = OperatorAdd().apply([lhs, rhs])
    # profiler.disable()
    # print('================Disabled Profile================')
    # # Save the profiling stats to a file
    # with open("output/bandmath_add_stats.txt", "w") as f:
    #     ps = pstats.Stats(profiler, stream=f)
    #     ps.sort_stats("tottime") 
    #     ps.print_stats()

def testing_data_loading(dataset_path: str):
    
    loader = RasterDataLoader()
    dataset1 = loader.load_from_file(dataset_path)
    # dataset2 = loader.load_from_file(dataset_path)

    print(f"INTERLEAVE: {dataset1.get_interleave()}")
    samples_max = dataset1._impl.gdal_dataset.RasterXSize
    lines_max = dataset1._impl.gdal_dataset.RasterYSize
    bands_max = dataset1.num_bands()

    # x is samples, y is lines
    print("samples: ", samples_max)
    print("lines: ", lines_max)
    print("bands: ", bands_max)
    print(dataset1.get_shape())

    # BSQ
    max_ram_bytes = 500000000
    data_bytes = 4
    usable_slots = max_ram_bytes / data_bytes
    usable_bands = int(np.floor(usable_slots/(lines_max*samples_max)))
    partial_band_list = [b for b in range(0, usable_bands)]
    print(f"BSQ data amount: {usable_bands*lines_max*samples_max*data_bytes}")
    start = time.time()
    arr = dataset1.get_custom_array(partial_band_list, 0, samples_max, 0, lines_max)
    end = time.time()
    print(f"Total read time for BSQ like reading: {end-start}")

    # BIL
    max_ram_bytes = 500000000
    data_bytes = 4
    usable_slots = max_ram_bytes / data_bytes
    useable_lines = int(np.floor(usable_slots/(bands_max*samples_max)))
    full_band_list = [b for b in range(0, bands_max)]
    print(f"BIL data amount: {useable_lines*bands_max*samples_max*data_bytes}")
    start = time.time()
    arr = dataset1.get_custom_array(full_band_list, 0, samples_max, 0, useable_lines)
    end = time.time()
    print(f"Total read time for BIL like reading: {end-start}")

    # BIP
    max_ram_bytes = 500000000
    data_bytes = 4
    usable_slots = max_ram_bytes / data_bytes
    usable_samples = int(np.floor(usable_slots/(bands_max*lines_max)))
    full_band_list = [b for b in range(0, bands_max)]
    print(f"BIP data amount: {usable_samples*lines_max*bands_max*data_bytes}")
    start = time.time()
    arr = dataset1.get_custom_array(full_band_list, 0, usable_samples, 0, lines_max)
    end = time.time()
    print(f"Total read time for BIP like reading: {end-start}")

    blocksize = dataset1._impl.gdal_dataset.GetRasterBand(1).GetBlockSize()
    print(f"Block size: {blocksize}")

def test_data_loading_numpy(dataset_path: str):
    
    loader = RasterDataLoader()
    dataset1 = loader.load_from_file(dataset_path)
    # dataset2 = loader.load_from_file(dataset_path)
    filename = dataset1._impl.gdal_dataset.GetFileList()[0]
    bands, lines, samples = dataset1.get_shape()
    print(f"bands: {bands}")
    print(f"lines: {lines}")
    print(f"samples: {samples}")
    with open(filename, 'rb') as f:
        data = np.memmap(f, np.float32, 'r', offset=0, shape=(lines, bands, samples))

    print(f"INTERLEAVE: {dataset1.get_interleave()}")
    # samples_max = dataset1._impl.gdal_dataset.RasterXSize
    # lines_max = dataset1._impl.gdal_dataset.RasterYSize
    # bands_max = dataset1.num_bands()

    # x is samples, y is lines
    # print("samples: ", samples_max)
    # print("lines: ", lines_max)
    # print("bands: ", bands_max)
    print(dataset1.get_shape())

    # BSQ
    max_ram_bytes = 500000000
    data_bytes = 4
    usable_slots = max_ram_bytes / data_bytes
    usable_bands = int(np.floor(usable_slots/(lines*samples)))
    print(f"BSQ data amount: {usable_bands*lines*samples*data_bytes}")
    start = time.time()
    # arr = data[0:lines,0:usable_bands,0:samples]+1
    arr = np.array(data[0:lines,0:usable_bands,0:samples])
    end = time.time()
    print(f"Total read time for BSQ like reading: {end-start}")

    # BIL
    max_ram_bytes = 500000000
    data_bytes = 4
    usable_slots = max_ram_bytes / data_bytes
    useable_lines = int(np.floor(usable_slots/(bands*samples)))
    print(f"BIL data amount: {useable_lines*bands*samples*data_bytes}")
    start = time.time()
    # arr = data[0:useable_lines,0:bands,0:samples]+1
    arr = np.array(data[0:useable_lines,0:bands,0:samples])
    end = time.time()
    print(f"Total read time for BIL like reading: {end-start}")

    # BIP
    max_ram_bytes = 500000000
    data_bytes = 4
    usable_slots = max_ram_bytes / data_bytes
    usable_samples = int(np.floor(usable_slots/(bands*lines)))
    print(f"BIP data amount: {usable_samples*lines*bands*data_bytes}")
    start = time.time()
    # arr = data[0:lines,0:bands,0:usable_samples]+1
    arr = np.array(data[0:lines,0:bands,0:usable_samples])
    end = time.time()
    print(f"Total read time for BIP like reading: {end-start}")



    print(f"type(arr): {type(arr)}")
    print(f"type(arr): {type(data)}")
    blocksize = dataset1._impl.gdal_dataset.GetRasterBand(1).GetBlockSize()
    print(f"Block size: {blocksize}")

# def test_bandmath_expr(dataset_path: str):
    
#     loader = RasterDataLoader()
#     dataset1 = loader.load_from_file(dataset_path)
#     dataset2 = loader.load_from_file(dataset_path)
#     bandmath_dialog = BandMathDialog(None)
#     bandmath_expr = "a+b"
#     bandmath_dialog._ui.ledit_expression.setTExt(bandmath_expr)
#     bandmath_dialog._analyze_expr()
#     bandmath_dialog._ui.tbl_variables.setCellWidget(0, 1, )




if __name__ == '__main__':
    '''
    It is okay for this profile to use an image that is not incredibly big because the sampler profile takes a long time to run
    '''
    dataset_path = "c:\\Users\\jgarc\\OneDrive\\Documents\\Data\\ang20171108t184227_corr_v2p13_subset_bil.hdr"
    # dataset_path = 'C:\\Users\\jgarc\\OneDrive\\Documents\\Data\\RhinoLeft_2016_07_28_12_56_01_SWIRcalib_atmcorr.hdr'
    # dataset_path = "C:\\Users\\jgarc\\OneDrive\\Documents\\Data\\C5705B-00003Z-01_2018_07_28_14_18_38_VNIRcalib.hdr"
    # dataset_path = "C:\\Users\\jgarc\\OneDrive\\Documents\\Data\\Task1.1_SlowBandMath_10gb\\ang20171108t184227_corr_v2p13_subset_bil_expanded_bands_by_40.hdr"
    print(f"Filename: {os.path.basename(dataset_path)}")
    # profile(dataset_path)
    # profile_cube_band(dataset_path)
    # profile_cube_spectrum(dataset_path)

    # testing_data_loading(dataset_path)
    test_data_loading_numpy(dataset_path)
    print('Done with profiling')