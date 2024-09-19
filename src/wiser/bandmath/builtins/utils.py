from typing import Any, Dict, List, Optional

import numpy as np
import dask.array as da
import os

from wiser.bandmath import VariableType, BandMathValue, BandMathExprInfo
from wiser.bandmath.functions import BandMathFunction

from wiser.bandmath.utils import (
    reorder_args,
    check_image_cube_compatible, check_image_band_compatible, check_spectrum_compatible,
    make_image_cube_compatible, make_image_band_compatible, make_spectrum_compatible,
    make_image_cube_compatible_by_bands,
)

from .constants import MAX_RAM_BYTES, SCALAR_BYTES, TEMP_FOLDER_PATH

from wiser.raster.dataset_impl import InterleaveType
import time

def perform_oper(operation: str, lhs: BandMathValue, rhs: BandMathValue):

    file_path = os.path.join(TEMP_FOLDER_PATH, 'oper_testing_result.dat')
    if lhs.type == VariableType.IMAGE_CUBE:
        bands, lines, samples = lhs.get_shape()
        lhs_interleave_type = lhs.value.get_interleave()
        print(f"Metadata: ", lhs.value._impl.gdal_dataset.GetFileList())
        # filename = lhs.value._impl.gdal_dataset.GetFileList()[0]
        # with open(filename, 'rb') as f:
        #     data = np.memmap(f, np.float32, 'r', offset=0, shape=(lines, bands, samples))
        #     band1 = data[:,0,:]
        #     band1_gdal = lhs.value._impl.gdal_dataset.GetRasterBand(1).ReadAsArray()
        #     # print(f"band1.shape: {band1.shape}")
        #     # print(f"band1_gdal.shape: {band1_gdal.shape}")
        #     # print(f"band1: {band1}")
        #     # print(f"band1_gdal: {band1_gdal}")
        #     np.testing.assert_allclose(band1, band1_gdal)
            # print(data.shape)
        result_shape = None
        if lhs_interleave_type == InterleaveType.BSQ:
            result_shape = (bands, samples, lines)
        elif lhs_interleave_type == InterleaveType.BIL:
            result_shape = (lines, bands, samples)
        elif lhs_interleave_type == InterleaveType.BIP:
            result_shape = (samples, lines, bands)
        else:
            result_shape = (bands, samples, lines)
        print(f"lhs_interleave_type: {lhs_interleave_type}")
        print(f"bands: {bands}")
        print(f"lines: {lines}")
        print(f"samples: {samples}")
        print(f"result_shape: {result_shape}")
        # result_arr = da.from_array(np.memmap(file_path, \
        #                         dtype=lhs.value.get_elem_type(), mode='w+', shape=result_shape))
        result_arr = np.memmap(file_path, \
                                dtype=lhs.value.get_elem_type(), mode='w+', shape=result_shape)
        max_slots = MAX_RAM_BYTES / SCALAR_BYTES

        lhs_file = lhs.value._impl.gdal_dataset.GetFileList()[0]
        with open(lhs_file, 'rb') as f:
            lhs_raw_numpy = np.memmap(f, np.float32, 'r', offset=0, shape=result_shape)
        
        rhs_file = rhs.value._impl.gdal_dataset.GetFileList()[0]
        with open(rhs_file, 'rb') as f:
            rhs_raw_numpy = np.memmap(f, np.float32, 'r', offset=0, shape=result_shape)

        if rhs.type == VariableType.IMAGE_CUBE:
            rhs_interleave_type = rhs.value.get_interleave()
            if lhs_interleave_type == rhs_interleave_type:
                # 52 seconds on adding two 15GB image
                if lhs_interleave_type == InterleaveType.BIL:
                    print(f"Both types are {rhs_interleave_type}")
                    dlines = int(np.floor(max_slots/(bands*samples)))
                    full_band_list = [b for b in range(0, bands)]
                    start = time.time()
                    for line_start in range(0, lines, dlines):
                        line_end = line_start + dlines
                        if line_end > lines:
                            line_end = lines
                        print(f"Line start: {line_start} \t | Line end: {line_end}")
                        # Lie, i think ysize is exclusive. Do line_end-1 because line_end is 0 indexed, since ysize is also 0 indexed, we are good
                        # lhs_value = lhs.get_custom_array(full_band_list, 0, samples, line_start, line_end)
                        # rhs_value = rhs.get_custom_array(full_band_list, 0, samples, line_start, line_end)
                        lhs_value = lhs_raw_numpy[line_start:line_end, : , :]
                        rhs_value = rhs_raw_numpy[line_start:line_end, : , :]
                        # Think about just having hte output of the addition be rhs_value to save memory
                        print(f"Intermediate lhs_value.shape: {lhs_value.shape}")
                        print(f"Intermediate rhs_value.shape: {rhs_value.shape}")
                        print(f"result_arr[line_start:line_end, :, :].shape: {result_arr[line_start:line_end, :, :].shape}")
                        result_arr[line_start:line_end, :, :] = lhs_value + rhs_value
                        del lhs_value, rhs_value
                    end = time.time()
                    print(f"Took {end-start} seconds long!")
                elif lhs_interleave_type == InterleaveType.BIP:
                    dsamples = int(np.floor(max_slots(bands*lines)))
                    full_band_list = [b for b in range(0, bands)]
                    start = time.time()
                    for sample_start in range(0, samples, dsamples):
                        sample_end = sample_start + dsamples
                        if sample_end > samples:
                            sample_end = samples
                        lhs_value = lhs.get_custom_array(full_band_list, sample_start, sample_end, 0, lines)
                        rhs_value = rhs.get_custom_array(full_band_list, sample_start, sample_end, 0, lines)
                        result_arr[sample_start:sample_end, :, :] = np.add(lhs_value, rhs_value, out=rhs_value)
                        del lhs_value, rhs_value
                    end = time.time()
                    print(f"Took {end-start} seconds long!")
                # 78 seconds on adding two 20gb images 
                else: # lhs_interleave_type == InterleaveType.BSQ:
                    dbands = int(np.floor(max_slots/(lines*samples)))
                    start = time.time()
                    for band_start in range(0, bands, dbands):
                        if band_start + dbands > bands:
                            dbands = bands - band_start
                        print(f"Band start: {band_start} \t | Line end: {band_start+dbands}")
                        partial_band_list = [band_start + inc for inc in range(0, dbands)]
                        # lhs_value = lhs.get_custom_array(partial_band_list, 0, samples, 0, lines)
                        # rhs_value = rhs.get_custom_array(partial_band_list, 0, samples, 0, lines)
                        lhs_value = lhs_raw_numpy[band_start:band_start + dbands, : , :]
                        rhs_value = rhs_raw_numpy[band_start:band_start + dbands, : , :]
                        print(f"lhs_value: {lhs_value.shape}")
                        print(f"rhs_value: {rhs_value.shape}")
                        print(f"result_arr[band_start:band_start+dbands, :, :]: {result_arr[band_start:band_start+dbands, :, :].shape}")
                        result_arr[band_start:band_start+dbands, :, :] = lhs_value + rhs_value
                        del lhs_value, rhs_value
                    end = time.time()
                    print(f"Took {end-start} seconds long!")
                
            print(f"result_shape: {result_shape}")
            result_arr_view = np.reshape(result_arr, lhs.get_shape())
            print(result_arr_view.base is result_arr)
            return result_arr_view
        elif rhs.type == VariableType.IMAGE_BAND:
            rhs_arr = rhs.as_numpy_array()
            if lhs_interleave_type == InterleaveType.BSQ:
                dbands = int(np.floor(max_slots/(lines*samples)))
                for band_start in range(0, bands, dbands):
                    if band_start + dbands > bands:
                        dbands = bands - band_start
                    partial_band_list = [band_start + inc for inc in range(0, dbands)]
                    lhs_value = lhs.get_custom_array(partial_band_list, 0, samples, 0, lines)
                    rhs_value = make_image_cube_compatible_by_bands(rhs, lhs_value.shape, partial_band_list)
                    print(f"rhs_value.shape with IMAGE_BAND: {rhs_value.shape}")
                    result_arr[band_start:band_start+dbands, :, :] = lhs_value + rhs_value
                    del lhs_value, rhs_value
            elif lhs_interleave_type == InterleaveType.BIL:
                    print(f"Both types are {rhs_interleave_type}")
                    dlines = int(np.floor(max_slots/(bands*samples)))
                    full_band_list = [b for b in range(0, bands)]
                    start = time.time()
                    for line_start in range(0, lines, dlines):
                        line_end = line_start + dlines
                        if line_end > lines:
                            line_end = lines
                        print(f"Line start: {line_start} \t | Line end: {line_end}")
                        # Lie, i think ysize is exclusive. Do line_end-1 because line_end is 0 indexed, since ysize is also 0 indexed, we are good
                        lhs_value = lhs.get_custom_array(full_band_list, 0, samples, line_start, line_end)
                        rhs_value = rhs.get_custom_array(full_band_list, 0, samples, line_start, line_end)
                        # Think about just having hte output of the addition be rhs_value to save memory
                        print(f"Intermediate lhs_value.shape: {lhs_value.shape}")
                        print(f"Intermediate rhs_value.shape: {rhs_value.shape}")
                        print(f"result_arr[line_start:line_end, :, :].shape: {result_arr[line_start:line_end, :, :].shape}")
                        result_arr[line_start:line_end, :, :] = np.add(lhs_value, rhs_value, out=rhs_value)
                        del lhs_value, rhs_value

            # elif lhs_interleave_type == InterleaveType.BIL:


