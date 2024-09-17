from typing import Any, Dict, List, Optional

import dask.array
import numpy as np
import dask

from wiser.bandmath import VariableType, BandMathValue, BandMathExprInfo
from wiser.bandmath.functions import BandMathFunction

from wiser.bandmath.utils import (
    reorder_args,
    check_image_cube_compatible, check_image_band_compatible, check_spectrum_compatible,
    make_image_cube_compatible, make_image_band_compatible, make_spectrum_compatible,
    make_image_cube_compatible_portion,
)


class OperatorAdd(BandMathFunction):
    '''
    Binary addition operator.
    '''

    def _report_type_error(self, lhs_type, rhs_type):
        raise TypeError(f'Operands {lhs_type} and {rhs_type} not compatible for +')


    def analyze(self, infos: List[BandMathExprInfo],
            options: Dict[str, Any] = None) -> BandMathExprInfo:

        if len(infos) != 2:
            raise ValueError('Binary addition requires exactly two arguments')

        lhs = infos[0]
        rhs = infos[1]

        # Take care of the simple case first.
        if (lhs.result_type == VariableType.NUMBER and
            rhs.result_type == VariableType.NUMBER):
            return BandMathExprInfo(VariableType.NUMBER)

        # If we got here, we are comparing more complex data types.

        # Since addition is commutative, swap LHS and RHS based on the types,
        # to make the analysis logic easier

        (lhs, rhs) = reorder_args(lhs.result_type, rhs.result_type, lhs, rhs)

        # Analyze the input types to determine the result type
        # TODO(donnie):  This code assumes the result's element-type will be the
        #     same as the LHS element-type, but this is not guaranteed.  Best
        #     way to do this is to ask NumPy what the result element-type will
        #     be.

        if lhs.result_type == VariableType.IMAGE_CUBE:
            check_image_cube_compatible(rhs, lhs.shape)

            info = BandMathExprInfo(VariableType.IMAGE_CUBE)
            info.shape = lhs.shape
            info.elem_type = lhs.elem_type

            # TODO(donnie):  Check that metadata are compatible, and maybe
            #     generate warnings if they aren't.
            info.spatial_metadata_source = lhs.spatial_metadata_source
            info.spectral_metadata_source = lhs.spectral_metadata_source

            return info

        elif lhs.result_type == VariableType.IMAGE_BAND:
            check_image_band_compatible(rhs, lhs.shape)

            info = BandMathExprInfo(VariableType.IMAGE_BAND)
            info.shape = lhs.shape
            info.elem_type = lhs.elem_type

            # TODO(donnie):  Check that metadata are compatible, and maybe
            #     generate warnings if they aren't.
            info.spatial_metadata_source = lhs.spatial_metadata_source

            return info

        elif lhs.result_type == VariableType.SPECTRUM:
            check_spectrum_compatible(rhs, lhs.shape)

            info = BandMathExprInfo(VariableType.SPECTRUM)
            info.shape = lhs.shape
            info.elem_type = lhs.elem_type

            # TODO(donnie):  Check that metadata are compatible, and maybe
            #     generate warnings if they aren't.
            info.spectral_metadata_source = lhs.spectral_metadata_source

            return info

        self._report_type_error(lhs.result_type, rhs.result_type)


    def apply(self, args: List[BandMathValue]):
        '''
        Add the LHS and RHS and return the result.
        '''
        print("APPLY")
        if len(args) != 2:
            raise Exception('+ requires exactly two arguments')

        lhs = args[0]
        rhs = args[1]

        # Take care of the simple case first, where it's just two numbers.
        if lhs.type == VariableType.NUMBER and rhs.type == VariableType.NUMBER:
            return BandMathValue(VariableType.NUMBER, lhs.value + rhs.value)

        # Since addition is commutative, arrange the arguments to make the
        # calculation logic easier.
        (lhs, rhs) = reorder_args(lhs.type, rhs.type, lhs, rhs)

        # Do the addition computation.
        # 4 GB, 15 GB max
        # 2 GB, 9 GB max
        # 1 GB, 5.5 GB max
        import time
        if lhs.type == VariableType.IMAGE_CUBE:
            # Dimensions:  [band][y][x]
            start = time.time()
            bands, y, x = lhs.get_shape()
            # result_arr = dask.array.zeros(lhs.get_shape(), dtype=lhs.value.get_elem_type())
            result_arr = np.memmap('oper_add_result.dat', dtype=lhs.value.get_elem_type(), \
                                   mode='w+', shape=lhs.value.get_shape())
            print(f"RESULT_ARR shape: {result_arr.shape}")
            print(f"LHS shape: {lhs.value.get_shape()}")
            print(result_arr.shape == lhs.value.get_shape())
            print(lhs.value.get_shape())
            print(f"lhs.value.get_elem_type(): {lhs.value.get_elem_type()}")
            end = time.time()
            print(f"Get shape time: {end-start}")
            max_ram_bytes = 4000000000
            # max_ram_bytes = 1575206400
            max_bytes = max_ram_bytes/4
            go_by_band = True
            y_start = 0
            x_start = 0 
            dx = 0
            dy = 0
            if rhs.type == VariableType.IMAGE_CUBE:
                print("===========IMAGE_CUBE===========")
                start = time.time()
                dbands = int(np.floor(max_bytes / (x*y)))
                # sqrt_area = int(np.floor(np.sqrt(x_y_area)))
                # dx = sqrt_area
                # dy = sqrt_area
                # We get by bands because the data is in bsq format (bands are the first dimension so easier
                # to access)
                # Before when we didn't get by bands, adding 2 out of memory arrays would take 854.712 seconds
                # Now, when we get by bands it takes... 207.248 seconds
                for band_start in range(0, bands, dbands):
                    if band_start + dbands > bands:
                        dbands = bands - band_start
                    band_list = [band_start + inc for inc in range(0, dbands)]
                    print(f"band_list, min: {min(band_list)} | max {max(band_list)}")
                    # print(f"band_list: {band_list}")
                    lhs_value = lhs.as_numpy_array_chunk(rhs, True, band_list, 0, \
                                                            0, 0, 0, 0)
                    print(f"lhs_value.shape: {lhs_value.shape}")
                    rhs_value = make_image_cube_compatible_portion(rhs, lhs_value.shape, True, band_start, dbands, \
                                                            0, 0, 0, 0)
                    print(f"rhs_value.shape: {rhs_value.shape}")
                    print(f"lhs_value type: {type(lhs_value)}")
                    print(f"rhs_value type: {type(rhs_value)}")
                    print(f"band_start: {band_start}")
                    print(f"dbands: {dbands}")
                    print(f"num bytes, lhs: {lhs_value.nbytes} | rhs: {rhs_value.nbytes}")
                    result_arr[band_start:band_start+dbands,:,:] = lhs_value + rhs_value
                    del lhs_value, rhs_value
                end=time.time()
                print(f"Took {end-start} long!")


                # start = time.time()
                # x_y_area = max_bytes / bands
                # sqrt_area = int(np.floor(np.sqrt(x_y_area)))
                # dx = sqrt_area
                # dy = sqrt_area
                # # print(f"sqrt_area: {sqrt_area}")
                # for x_start in range(0, x, dx):
                #     for y_start in range(0, y, dy):
                #         if x_start + dx > x:
                #             dx = x-x_start
                #         if y_start + dy > y:
                #             dy = y-y_start
                #         lhs_value = lhs.as_numpy_array_chunk(rhs, False, 0, 0, \
                #                                              x_start, dx, y_start, dy)
                #         # print(f"lhs_value.shape: {lhs_value.shape}")
                #         # print(f"num bytes: {lhs_value.nbytes}")
                #         # print(f"num bytes of single value: {lhs_value[0,0,0].nbytes}")
                #         rhs_value = make_image_cube_compatible_portion(rhs, lhs_value.shape, True, 0, 0, \
                #                                              x_start, dx, y_start, dy)
                #         # print(f"lhs.shape: {lhs_value.shape}")
                #         # print(f"rhs.shape: {rhs_value.shape}")
                #         # print(f"result_arr.shape: {result_arr.shape}")
                #         # print(f"x_start: {x_start}")
                #         # print(f"y_start: {y_start}")
                #         # print(f"dx: {dx}")
                #         # print(f"dy: {dy}")
                #         partial_result = lhs_value + rhs_value
                #         result_arr[:, y_start:y_start+dy, x_start:x_start+dx] = partial_result
                #         dx = sqrt_area
                #         dy = sqrt_area
                #         del lhs_value, rhs_value, partial_result
                # end=time.time()
                # print(f"Took {end-start} long!")
            elif rhs.type == VariableType.IMAGE_BAND:
                print("===========IMAGE_BAND===========")
                start = time.time()
                dbands = int(np.floor(max_bytes / (x*y)))
                # sqrt_area = int(np.floor(np.sqrt(x_y_area)))
                # dx = sqrt_area
                # dy = sqrt_area
                for band_start in range(0, bands, dbands):
                    if band_start + dbands > bands:
                        dbands = bands - band_start
                    band_list = [band_start + inc for inc in range(0, dbands)]
                    print(f"band_list, min: {min(band_list)} | max {max(band_list)}")
                    lhs_value = lhs.as_numpy_array_chunk(rhs, True, band_list, 0, \
                                                            0, 0, 0, 0)
                    print(f"lhs_value.shape: {lhs_value.shape}")
                    rhs_value = make_image_cube_compatible_portion(rhs, lhs_value.shape, True, 0, 0, \
                                                            0, 0, 0, 0)
                    # print(f"rhs_value.shape: {rhs_value.shape}")
                    # print(f"lhs_value type: {type(lhs_value)}")
                    # print(f"rhs_value type: {type(rhs_value)}")
                    # print(f"band_start: {band_start}")
                    # print(f"dbands: {dbands}")
                    # print(f"num bytes, lhs: {lhs_value.nbytes} | rhs: {rhs_value.nbytes}")
                    result_arr[band_start:band_start+dbands,:,:] = lhs_value + rhs_value
                    del lhs_value, rhs_value
                end=time.time()
                print(f"Took {end-start} long!")
                # 15 gb max
            elif rhs.type == VariableType.SPECTRUM:
                # 490 seconds for 15GB, slow version was 1,209.249 seconds
                # when adding with bands it took 103.522 seconds
                print("===========SPECTRUM===========")
                start = time.time()
                dbands = int(np.floor(max_bytes / (x*y)))
                # sqrt_area = int(np.floor(np.sqrt(x_y_area)))
                # dx = sqrt_area
                # dy = sqrt_area
                for band_start in range(0, bands, dbands):
                    if band_start + dbands > bands:
                        dbands = bands - band_start
                    band_list = [band_start + inc for inc in range(0, dbands)]
                    print(f"band_list, min: {min(band_list)} | max {max(band_list)}")
                    lhs_value = lhs.as_numpy_array_chunk(rhs, True, band_list, 0, \
                                                            0, 0, 0, 0)
                    print(f"lhs_value.shape: {lhs_value.shape}")
                    rhs_value = make_image_cube_compatible_portion(rhs, lhs_value.shape, True, band_start, dbands, \
                                                            0, 0, 0, 0)
                    print(f"rhs_value.shape: {rhs_value.shape}")
                    print(f"lhs_value type: {type(lhs_value)}")
                    print(f"rhs_value type: {type(rhs_value)}")
                    print(f"band_start: {band_start}")
                    print(f"dbands: {dbands}")
                    print(f"num bytes, lhs: {lhs_value.nbytes} | rhs: {rhs_value.nbytes}")
                    result_arr[band_start:band_start+dbands,:,:] = lhs_value + rhs_value
                    del lhs_value, rhs_value
                end=time.time()
                print(f"Took {end-start} long!")
                # # 490 seconds for 15GB, slow version was 1209.249
                # print("===========SPECTRUM===========")
                # start = time.time()
                # x_y_area = max_bytes / bands
                # sqrt_area = int(np.floor(np.sqrt(x_y_area)))
                # dx = sqrt_area
                # dy = sqrt_area
                # for x_start in range(0, x, dx):
                #     for y_start in range(0, y, dy):
                #         if x_start + dx > x:
                #             dx = x-x_start
                #         if y_start + dy > y:
                #             dy = y-y_start
                #         lhs_value = lhs.as_numpy_array_chunk(rhs, False, 0, 0, \
                #                                              x_start, dx, y_start, dy)
                #         rhs_value = make_image_cube_compatible_portion(rhs, lhs_value.shape, True, 0, 0, \
                #                                              x_start, dx, y_start, dy)
                #         # print(f"lhs.shape: {lhs_value.shape}")
                #         # print(f"rhs.shape: {rhs_value.shape}")
                #         # print(f"result_arr.shape: {result_arr.shape}")
                #         # print(f"x_start: {x_start}")
                #         # print(f"y_start: {y_start}")
                #         # print(f"dx: {dx}")
                #         # print(f"dy: {dy}")
                #         # print(f"num bytes, lhs: {lhs_value.nbytes} | rhs: {rhs_value.nbytes}")
                #         result_arr[:, y_start:y_start+dy, x_start:x_start+dx] = lhs_value + rhs_value
                #         dx = sqrt_area
                #         dy = sqrt_area
                #         del lhs_value, rhs_value
                # end=time.time()
                # print(f"Took {end-start} long!")
                
            # Instead, we will have this return the lhs_value, 
            # the band number we are on, the band delta
            # the x we are on and the x delta, and the y we are on and the y delta 
            # lhs_value = lhs.as_numpy_array()
            # print(f"total bytes: {lhs_value.nbytes}")
            # assert lhs_value.ndim == 3
            
            # # Instead, we will pass in rhs, lhs_value.shape, and the deltas
            # # we will read rhs in by the deltas and confirm dimension and shape

            # # We will also have to make this work for image bands and spectrum
            # # We will do a similar thing
            # rhs_value = make_image_cube_compatible(rhs, lhs_value.shape)
            # # lhs_value_dask = dask.array.from_array(lhs_value)
            # # rhs_value_dask = dask.array.from_array(rhs_value)
            # # print(type(lhs_value_dask))
            # # print(type(rhs_value_dask))
            # # print(type(lhs_value[0,0,0]))
            # result_arr = np.zeros_like(lhs_value)
            # # We get the width and height of the window we will do the operation on. Assuming we have 4GB of RAM to play with
            # ram_amt = 4000000000
            # byte_amt = ram_amt/lhs_value[0,0,0].nbytes
            # area_left = byte_amt/lhs_value.shape[0]
            # sqrt_area = int(np.floor(np.sqrt(area_left)))
            # print("sqrt_area: ", sqrt_area)
            # for i in range(lhs_value.shape[1]):
            #     for j in range(lhs_value.shape[2]):
            #         dim1_start = i*sqrt_area
            #         dim2_start = j*sqrt_area
            #         if dim1_start >= lhs_value.shape[1] or dim2_start >= lhs_value.shape[2]:
            #             continue
            #         else:
            #             dim1_end = min(dim1_start+sqrt_area, lhs_value.shape[1])
            #             dim2_end = min(dim2_start+sqrt_area, lhs_value.shape[2])
            #             result_arr[:,dim1_start:dim1_end,dim2_start:dim2_end] = lhs_value[:,dim1_start:dim1_end,dim2_start:dim2_end] + rhs_value[:,dim1_start:dim1_end,dim2_start:dim2_end]

            # dim1_ratio = 1
            # dim2_ratio = 1
            # if lhs_value.shape[1] > lhs_value.shape[2]:
            #     dim1_ratio = np.floor(lhs_value.shape[1]/lhs_value.shape[2])
            # else:
            #     dim2_ratio = np.floor(lhs_value.shape[2]/lhs_value.shape[1])
            # print(dim1_ratio)
            # print(dim2_ratio)
            # print("shape: ", lhs_value.shape)
            # result_arr = lhs_value_dask + rhs_value_dask

            # The result array should have the same dimensions as the LHS input
            # array.
            assert result_arr.ndim == 3
            assert result_arr.shape == lhs.value.get_shape()
            result_arr.flush()
            return BandMathValue(VariableType.IMAGE_CUBE, result_arr)

        elif lhs.type == VariableType.IMAGE_BAND:
            # Dimensions:  [y][x]
            lhs_value = lhs.as_numpy_array()
            assert lhs_value.ndim == 2

            rhs_value = make_image_band_compatible(rhs, lhs_value.shape)
            result_arr = lhs_value + rhs_value

            # The result array should have the same dimensions as the LHS input
            # array.
            assert result_arr.ndim == 2
            assert result_arr.shape == lhs_value.shape
            return BandMathValue(VariableType.IMAGE_BAND, result_arr)

        elif lhs.type == VariableType.SPECTRUM:
            # Dimensions:  [band]
            lhs_value = lhs.as_numpy_array()
            assert lhs_value.ndim == 1

            rhs_value = make_spectrum_compatible(rhs, lhs_value.shape)
            result_arr = lhs_value + rhs_value

            # The result array should have the same dimensions as the LHS input
            # array.
            assert result_arr.ndim == 1
            assert result_arr.shape == lhs_value.shape
            return BandMathValue(VariableType.SPECTRUM, result_arr)

        # If we get here, we don't know how to add the two types.
        # Use args[0] and args[1] instead of lhs and rhs, since lhs/rhs may be
        # reversed from the original inputs.
        self._report_type_error(args[0].type, args[1].type)
