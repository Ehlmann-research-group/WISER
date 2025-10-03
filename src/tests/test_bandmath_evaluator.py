import tests.context
# import context

from astropy import units as u

import os
import sys

import unittest
import numpy as np
from typing import List, Tuple, Callable


from wiser import bandmath
from wiser.bandmath import VariableType
from wiser.bandmath.analyzer import get_bandmath_expr_info
from wiser.raster.dataset import RasterDataSet, RasterDataBand, RasterDataBatchBand, RasterDataDynamicBand
from wiser.raster.loader import RasterDataLoader
from wiser.raster.spectrum import SpectrumAtPoint

from wiser.raster.data_cache import DataCache
from wiser.raster.serializable import SerializedForm

from test_utils.test_model import WiserTestModel

from wiser.bandmath.utils import load_image_from_bandmath_result, load_band_from_bandmath_result


def make_image(bands, width, height):
    arr = np.zeros(shape=(bands, width, height))
    # TODO(donnie):  Initialize values?
    return arr

def make_band(width, height):
    arr = np.zeros(shape=(width, height))
    # TODO(donnie):  Initialize values?
    return arr

def make_spectrum(bands):
    arr = np.zeros(shape=bands)
    # TODO(donnie):  Initialize values?
    return arr

loader = RasterDataLoader()

class TestBandmathEvaluator(unittest.TestCase):
    '''
    Exercise code in the bandmath.evaluator module.
    '''

    def setUp(self):
        self.test_model = WiserTestModel()

    def tearDown(self):
        self.test_model.close_app()
        del self.test_model

    #===========================================================================
    # NUMBERS-ONLY TEST

    # def test_bandmath_numbers_only(self):
    #     ''' Test band-math involving only numbers. '''
    #     expr_info = get_bandmath_expr_info('(2 * 3 + 4) / 2',
    #         {}, {})
    #     result_name = 'test_result'

    #     cache = DataCache()

    #     process_manager = bandmath.eval_bandmath_expr(succeeded_callback=lambda _: None,
    #         status_callback=lambda _: None, error_callback=lambda _: None, 
    #         bandmath_expr='(2 * 3 + 4) / 2', expr_info=expr_info, result_name=result_name, cache=cache,
    #         variables={}, functions={})

    #     process_manager.get_task().wait()
    #     result = process_manager.get_task().get_result()

    #     for result_type, result_value, result_name, expr_info in result:
    #         self.assertEqual(result_type, VariableType.NUMBER)
    #         self.assertEqual(result_value, 5)

    # #===========================================================================
    # # SIMPLE ADDITION TESTS

    # def test_bandmath_add_image_band(self):
    #     ''' Test band-math involving adding an image and a band. '''
    #     img = make_image(2, 3, 4)
    #     img[0].fill(1.0)
    #     img[1].fill(2.0)

    #     band = make_band(3, 4)
    #     for y in range(band.shape[1]):
    #         for x in range(band.shape[0]):
    #             band[x, y] = 0.5 * x + 10 * y

    #     expr_info = get_bandmath_expr_info('image + band',
    #         {'image':(VariableType.IMAGE_CUBE, img),
    #          'band':(VariableType.IMAGE_BAND, band)}, {})
    #     result_name = 'test_result'
    
    #     cache = DataCache()

    #     process_manager = bandmath.eval_bandmath_expr(succeeded_callback=lambda _: None,
    #         status_callback=lambda _: None, error_callback=lambda _: None, 
    #         bandmath_expr='image + band', expr_info=expr_info, result_name=result_name, cache=cache,
    #         variables={'image':(VariableType.IMAGE_CUBE, img),
    #          'band':(VariableType.IMAGE_BAND, band)}, functions={})

    #     process_manager.get_task().wait()
    #     results = process_manager.get_task().get_result()

    #     for result_type, result, result_name, expr_info in results:
    #         assert result_type == RasterDataSet or result_type == VariableType.IMAGE_CUBE

    #         # Check if result_type is RasterDataSet or IMAGE_CUBE and handle accordingly
    #         if result_type == RasterDataSet:
    #             result_dataset = load_image_from_bandmath_result(result_type, result, result_name, None, expr_info, loader, None)
    #             result_arr = result_dataset.get_image_data()
    #             result_shape = result_dataset.get_shape()
    #         elif result_type == VariableType.IMAGE_CUBE:
    #             result_arr = result
    #             result_shape = result.shape
    #         else:
    #             self.fail(f"Unexpected result type: {result_type}")

    #         # Make sure output image has correct results
    #         self.assertEqual(result_shape, (2, 3, 4))
    #         for b in range(result_shape[0]):
    #             for y in range(result_shape[2]):
    #                 for x in range(result_shape[1]):
    #                     self.assertEqual(result_arr[b, x, y], (b + 1.0) + 0.5 * x + 10 * y)
    #         # Make sure input image didn't change

    #         for value in np.nditer(img[0]):
    #             self.assertEqual(value, 1.0)

    #         for value in np.nditer(img[1]):
    #             self.assertEqual(value, 2.0)

    # def test_bandmath_add_image_number(self):
    #     ''' Test band-math involving adding an image and a number. '''
    #     img = make_image(2, 4, 5)
    #     img.fill(2.5)

    #     expr_info = get_bandmath_expr_info('image + 0.5',
    #         {'image':(VariableType.IMAGE_CUBE, img)}, {})
    #     result_name = 'test_result'

    #     cache = DataCache()

    #     process_manager = bandmath.eval_bandmath_expr(succeeded_callback=lambda _: None,
    #         status_callback=lambda _: None, error_callback=lambda _: None, 
    #         bandmath_expr='image + 0.5', expr_info=expr_info, result_name=result_name, cache=cache,
    #         variables={'image':(VariableType.IMAGE_CUBE, img)}, functions={})

    #     process_manager.get_task().wait()
    #     result = process_manager.get_task().get_result()

    #     for result_type, result, result_name, expr_info in result:
    #         assert result_type == RasterDataSet or result_type == VariableType.IMAGE_CUBE
        
    #         # Check if result_type is RasterDataSet or IMAGE_CUBE and handle accordingly
    #         if result_type == RasterDataSet:
    #             result_dataset = load_image_from_bandmath_result(result_type, result, result_name, None, expr_info, loader, None)
    #             result_arr = result_dataset.get_image_data()
    #         elif result_type == VariableType.IMAGE_CUBE:
    #             result_arr = result
    #         else:
    #             self.fail(f"Unexpected result type: {result_type}")

    #         # Make sure output image has correct results
    #         for value in np.nditer(result_arr):
    #             self.assertEqual(value, 3.0)

    #         # Make sure input image didn't change
    #         for value in np.nditer(img):
    #             self.assertEqual(value, 2.5)

    # def test_bandmath_add_band_image(self):
    #     ''' Test band-math involving adding a band and an image. '''
    #     img = make_image(2, 3, 4)
    #     img[0].fill(1.0)
    #     img[1].fill(2.0)

    #     band = make_band(3, 4)
    #     for y in range(band.shape[1]):
    #         for x in range(band.shape[0]):
    #             band[x, y] = 0.5 * x + 10 * y

    #     expr_info = get_bandmath_expr_info('band + image',
    #         {'image':(VariableType.IMAGE_CUBE, img),
    #          'band':(VariableType.IMAGE_BAND, band)}, {})
    #     result_name = 'test_result'

    #     cache = DataCache()

    #     process_manager = bandmath.eval_bandmath_expr(succeeded_callback=lambda _: None,
    #         status_callback=lambda _: None, error_callback=lambda _: None, 
    #         bandmath_expr='band + image', expr_info=expr_info, result_name=result_name, cache=cache,
    #         variables={'image':(VariableType.IMAGE_CUBE, img),
    #          'band':(VariableType.IMAGE_BAND, band)}, functions={})

    #     process_manager.get_task().wait()
    #     results = process_manager.get_task().get_result()

    #     for result_type, result, result_name, expr_info in results:
    #         assert result_type == RasterDataSet or result_type == VariableType.IMAGE_CUBE

    #         # Check if result_type is RasterDataSet or IMAGE_CUBE and handle accordingly
    #         if result_type == RasterDataSet:
    #             result_dataset = load_image_from_bandmath_result(result_type, result, result_name, None, expr_info, loader, None)
    #             result_arr = result_dataset.get_image_data()
    #             result_shape = result_dataset.get_shape()
    #         elif result_type == VariableType.IMAGE_CUBE:
    #             result_arr = result
    #             result_shape = result.shape
    #         else:
    #             self.fail(f"Unexpected result type: {result_type}")

    #         # Make sure output image has correct results
    #         self.assertEqual(result_shape, (2, 3, 4))
    #         for b in range(result_shape[0]):
    #             for y in range(result_shape[2]):
    #                 for x in range(result_shape[1]):
    #                     self.assertEqual (result_arr[b, x, y], (b + 1.0) + 0.5 * x + 10 * y)

    #         # Make sure input image didn't change

    #         for value in np.nditer(img[0]):
    #             self.assertEqual(value, 1.0)

    #         for value in np.nditer(img[1]):
    #             self.assertEqual(value, 2.0)

    # def test_bandmath_add_band_number(self):
    #     ''' Test band-math involving adding a band and a number. '''
    #     band = make_band(4, 5)
    #     band.fill(2.5)

    #     expr_info = get_bandmath_expr_info('b + 0.5',
    #         {'b':(VariableType.IMAGE_BAND, band)}, {})
    #     result_name = 'test_result'

    #     cache = DataCache()

    #     process_manager = bandmath.eval_bandmath_expr(succeeded_callback=lambda _: None,
    #         status_callback=lambda _: None, error_callback=lambda _: None, 
    #         bandmath_expr='b + 0.5', expr_info=expr_info, result_name=result_name, cache=cache,
    #         variables={'b':(VariableType.IMAGE_BAND, band)}, functions={})

    #     process_manager.get_task().wait()
    #     results = process_manager.get_task().get_result()

    #     for result_type, result_band, result_name, expr_info in results:
    #         assert result_type == RasterDataSet or result_type == VariableType.IMAGE_BAND

    #         # Check if result_type is RasterDataSet or IMAGE_BAND and handle accordingly
    #         if result_type == RasterDataSet:
    #             result_dataset = load_image_from_bandmath_result(result_type, result_band, result_name, None, expr_info, loader, None)
    #             result_band = result_dataset.get_image_data()
    #         elif result_type == VariableType.IMAGE_BAND:
    #             pass
    #         else:
    #             self.fail(f"Unexpected result type: {result_type}")

    #         # Make sure output band has correct results
    #         for value in np.nditer(result_band):
    #             self.assertEqual(value, 3.0)

    #         # Make sure input band didn't change
    #         for value in np.nditer(band):
    #             self.assertEqual(value, 2.5)

    # def test_bandmath_add_spectrum_number(self):
    #     ''' Test band-math involving adding a spectrum and a number. '''
    #     spectrum = make_spectrum(15)
    #     spectrum.fill(2.5)

    #     expr_info = get_bandmath_expr_info('S1 + 0.5',
    #         {'s1':(VariableType.SPECTRUM, spectrum)}, {})
    #     result_name = 'test_result'
        
    #     cache = DataCache()

    #     process_manager = bandmath.eval_bandmath_expr(succeeded_callback=lambda _: None,
    #         status_callback=lambda _: None, error_callback=lambda _: None, 
    #         bandmath_expr='S1 + 0.5', expr_info=expr_info, result_name=result_name, cache=cache,
    #         variables={'s1':(VariableType.SPECTRUM, spectrum)}, functions={})

    #     process_manager.get_task().wait()
    #     results = process_manager.get_task().get_result()

    #     for result_type, result_spectrum, result_name, expr_info in results:
    #         self.assertEqual(result_type, VariableType.SPECTRUM)

    #         # Make sure output spectrum has correct results
    #         for value in np.nditer(result_spectrum):
    #             self.assertEqual(value, 3.0)

    #         # Make sure input spectrum didn't change
    #         for value in np.nditer(spectrum):
    #             self.assertEqual(value, 2.5)

    # def test_bandmath_add_number_image(self):
    #     ''' Test band-math involving adding a number and an image. '''
    #     img = make_image(2, 4, 5)
    #     img.fill(2.5)

    #     expr_info = get_bandmath_expr_info('0.5 + image',
    #         {'image':(VariableType.IMAGE_CUBE, img)}, {})
    #     result_name = 'test_result'
    
    #     cache = DataCache()

    #     process_manager = bandmath.eval_bandmath_expr(succeeded_callback=lambda _: None,
    #         status_callback=lambda _: None, error_callback=lambda _: None, 
    #         bandmath_expr='0.5 + image', expr_info=expr_info, result_name=result_name, cache=cache,
    #         variables={'image':(VariableType.IMAGE_CUBE, img)}, functions={})

    #     process_manager.get_task().wait()
    #     results = process_manager.get_task().get_result()

    #     for result_type, result, result_name, expr_info in results:
    #         assert result_type == RasterDataSet or result_type == VariableType.IMAGE_CUBE

    #         # Check if result_type is RasterDataSet or IMAGE_CUBE and handle accordingly
    #         if result_type == RasterDataSet:
    #             result_dataset = load_image_from_bandmath_result(result_type, result, result_name, None, expr_info, loader, None)
    #             result_arr = result_dataset.get_image_data()
    #         elif result_type == VariableType.IMAGE_CUBE:
    #             result_arr = result
    #         else:
    #             self.fail(f"Unexpected result type: {result_type}")

    #         # Make sure output image has correct results
    #         for value in np.nditer(result_arr):
    #             self.assertEqual(value, 3.0)

    #         # Make sure input image didn't change
    #         for value in np.nditer(img):
    #             self.assertEqual(value, 2.5)

    # def test_bandmath_add_number_band(self):
    #     ''' Test band-math involving adding a number and a band. '''
    #     band = make_band(4, 5)
    #     band.fill(2.5)

    #     expr_info = get_bandmath_expr_info('0.5 + b',
    #         {'b':(VariableType.IMAGE_BAND, band)}, {})
    
    #     cache = DataCache()

    #     process_manager = bandmath.eval_bandmath_expr(succeeded_callback=lambda _: None,
    #         status_callback=lambda _: None, error_callback=lambda _: None, 
    #         bandmath_expr='0.5 + b', expr_info=expr_info, result_name=None, cache=cache,
    #         variables={'b':(VariableType.IMAGE_BAND, band)}, functions={})

    #     process_manager.get_task().wait()
    #     results = process_manager.get_task().get_result()

    #     for result_type, result_band, result_name, expr_info in results:
    #         assert result_type == RasterDataSet or result_type == VariableType.IMAGE_BAND
    #         # Check if result_type is RasterDataSet or IMAGE_BAND and handle accordingly
    #         if result_type == RasterDataSet:
    #             result_dataset = load_image_from_bandmath_result(result_type, result_band, result_name, None, expr_info, loader, None)
    #             result_band = result_dataset.get_image_data()
    #         elif result_type == VariableType.IMAGE_BAND:
    #             pass
    #         else:
    #             self.fail(f"Unexpected result type: {result_type}")

    #         # Make sure output band has correct results
    #         for value in np.nditer(result_band):
    #             self.assertEqual(value, 3.0)

    #         # Make sure input band didn't change
    #         for value in np.nditer(band):
    #             self.assertEqual(value, 2.5)

    # def test_bandmath_add_number_spectrum(self):
    #     ''' Test band-math involving adding a number and a spectrum. '''
    #     spectrum = make_spectrum(15)
    #     spectrum.fill(2.5)

    #     expr_info = get_bandmath_expr_info('0.5 + S1',
    #         {'s1':(VariableType.SPECTRUM, spectrum)}, {})
    #     result_name = 'test_result'

    #     cache = DataCache()

    #     process_manager = bandmath.eval_bandmath_expr(succeeded_callback=lambda _: None,
    #         status_callback=lambda _: None, error_callback=lambda _: None, 
    #         bandmath_expr='0.5 + S1', expr_info=expr_info, result_name=result_name, cache=cache,
    #         variables={'s1':(VariableType.SPECTRUM, spectrum)}, functions={})

    #     process_manager.get_task().wait()
    #     results = process_manager.get_task().get_result()

    #     for result_type, result_spectrum, result_name, expr_info in results:
    #         self.assertEqual(result_type, VariableType.SPECTRUM)

    #         # Make sure output spectrum has correct results
    #         for value in np.nditer(result_spectrum):
    #             self.assertEqual(value, 3.0)

    #         # Make sure input spectrum didn't change
    #         for value in np.nditer(spectrum):
    #             self.assertEqual(value, 2.5)

    # #===========================================================================
    # # SIMPLE SUBTRACTION TESTS

    # def test_bandmath_sub_image_band(self):
    #     ''' Test band-math involving subtracting a band from an image. '''
    #     img = make_image(2, 3, 4)
    #     img[0].fill(1.0)
    #     img[1].fill(2.0)

    #     band = make_band(3, 4)
    #     for y in range(band.shape[1]):
    #         for x in range(band.shape[0]):
    #             band[x, y] = 0.5 * x + 10 * y

    #     expr_info = get_bandmath_expr_info('image - band',
    #         {'image':(VariableType.IMAGE_CUBE, img),
    #          'band':(VariableType.IMAGE_BAND, band)}, {})
    #     result_name = 'test_result'
    
    #     cache = DataCache()

    #     process_manager = bandmath.eval_bandmath_expr(succeeded_callback=lambda _: None,
    #         status_callback=lambda _: None, error_callback=lambda _: None, 
    #         bandmath_expr='image - band', expr_info=expr_info, result_name=result_name, cache=cache,
    #         variables={'image':(VariableType.IMAGE_CUBE, img),
    #          'band':(VariableType.IMAGE_BAND, band)}, functions={})

    #     process_manager.get_task().wait()
    #     results = process_manager.get_task().get_result()

    #     for result_type, result, result_name, expr_info in results:
    #         assert result_type == RasterDataSet or result_type == VariableType.IMAGE_CUBE

    #         # Check if result_type is RasterDataSet or IMAGE_CUBE and handle accordingly
    #         if result_type == RasterDataSet:
    #             result_dataset = load_image_from_bandmath_result(result_type, result, result_name, None, expr_info, loader, None)
    #             result_arr = result_dataset.get_image_data()
    #             result_shape = result_dataset.get_shape()
    #         elif result_type == VariableType.IMAGE_CUBE:
    #             result_arr = result
    #             result_shape = result.shape
    #         else:
    #             self.fail(f"Unexpected result type: {result_type}")

    #         # Make sure output image has correct results
    #         self.assertEqual(result_shape, (2, 3, 4))
    #         for b in range(result_shape[0]):
    #             for y in range(result_shape[2]):
    #                 for x in range(result_shape[1]):
    #                     self.assertEqual(result_arr[b, x, y],
    #                                      (b + 1.0) - (0.5 * x + 10 * y))

    #         # Make sure input image didn't change

    #         for value in np.nditer(img[0]):
    #             self.assertEqual(value, 1.0)

    #         for value in np.nditer(img[1]):
    #             self.assertEqual(value, 2.0)

    # def test_bandmath_sub_image_number(self):
    #     ''' Test band-math involving subtracting a number from an image. '''
    #     img = make_image(2, 4, 5)
    #     img.fill(2.5)

    #     expr_info = get_bandmath_expr_info('image - 0.5',
    #         {'image':(VariableType.IMAGE_CUBE, img)}, {})
    #     result_name = 'test_result'
    
    #     cache = DataCache()

    #     process_manager = bandmath.eval_bandmath_expr(succeeded_callback=lambda _: None,
    #         status_callback=lambda _: None, error_callback=lambda _: None, 
    #         bandmath_expr='image - 0.5', expr_info=expr_info, result_name=result_name, cache=cache,
    #         variables={'image':(VariableType.IMAGE_CUBE, img)}, functions={})

    #     process_manager.get_task().wait()
    #     results = process_manager.get_task().get_result()

    #     for result_type, result, result_name, expr_info in results:
    #         assert result_type == RasterDataSet or result_type == VariableType.IMAGE_CUBE

    #         # Check if result_type is RasterDataSet or IMAGE_CUBE and handle accordingly
    #         if result_type == RasterDataSet:
    #             result_dataset = load_image_from_bandmath_result(result_type, result, result_name, None, expr_info, loader, None)
    #             result_arr = result_dataset.get_image_data()
    #         elif result_type == VariableType.IMAGE_CUBE:
    #             result_arr = result
    #         else:
    #             self.fail(f"Unexpected result type: {result_type}")

    #         # Make sure output image has correct results
    #         for value in np.nditer(result_arr):
    #             self.assertEqual(value, 2.0)

    #         # Make sure input image didn't change
    #         for value in np.nditer(img):
    #             self.assertEqual(value, 2.5)
                

    # def test_bandmath_sub_band_number(self):
    #     ''' Test band-math involving subtracting a number from a band. '''
    #     band = make_band(4, 5)
    #     band.fill(2.5)

    #     expr_info = get_bandmath_expr_info('B1 - 0.5',
    #         {'B1':(VariableType.IMAGE_BAND, band)}, {})
    #     result_name = 'test_result'
    
    #     cache = DataCache()

    #     process_manager = bandmath.eval_bandmath_expr(succeeded_callback=lambda _: None,
    #         status_callback=lambda _: None, error_callback=lambda _: None, 
    #         bandmath_expr='B1 - 0.5', expr_info=expr_info, result_name=result_name, cache=cache,
    #         variables={'B1':(VariableType.IMAGE_BAND, band)}, functions={})

    #     process_manager.get_task().wait()
    #     results = process_manager.get_task().get_result()

    #     for result_type, result_band, result_name, expr_info in results:
    #         self.assertEqual(result_type, VariableType.IMAGE_BAND)

    #         # Make sure output band has correct results
    #         for value in np.nditer(result_band):
    #             self.assertEqual(value, 2.0)

    #         # Make sure input band didn't change
    #         for value in np.nditer(band):
    #             self.assertEqual(value, 2.5)

    # def test_bandmath_sub_spectrum_number(self):
    #     ''' Test band-math involving subtracting a number from a spectrum. '''
    #     spectrum = make_spectrum(15)
    #     spectrum.fill(2.5)

    #     expr_info = get_bandmath_expr_info('s - 0.5',
    #         {'s':(VariableType.SPECTRUM, spectrum)}, {})
    #     result_name = 'test_result'

    #     cache = DataCache()

    #     process_manager = bandmath.eval_bandmath_expr(succeeded_callback=lambda _: None,
    #         status_callback=lambda _: None, error_callback=lambda _: None, 
    #         bandmath_expr='s - 0.5', expr_info=expr_info, result_name=result_name, cache=cache,
    #         variables={'s':(VariableType.SPECTRUM, spectrum)}, functions={})

    #     process_manager.get_task().wait()
    #     results = process_manager.get_task().get_result()

    #     for result_type, result_spectrum, result_name, expr_info in results:
    #         self.assertEqual(result_type, VariableType.SPECTRUM)

    #         # Make sure output image has correct results
    #         for value in np.nditer(result_spectrum):
    #             self.assertEqual(value, 2.0)

    #         # Make sure input image didn't change
    #         for value in np.nditer(spectrum):
    #             self.assertEqual(value, 2.5)

    # #===========================================================================
    # # SIMPLE MULTIPLICATION TESTS

    # def test_bandmath_mul_image_number(self):
    #     ''' Test band-math involving multiplying an image and a number. '''
    #     img = make_image(2, 4, 5)
    #     img.fill(2.5)

    #     expr_info = get_bandmath_expr_info('image * 2',
    #         {'image':(VariableType.IMAGE_CUBE, img)}, {})
    #     result_name = 'test_result'
    
    #     cache = DataCache()

    #     process_manager = bandmath.eval_bandmath_expr(succeeded_callback=lambda _: None,
    #         status_callback=lambda _: None, error_callback=lambda _: None, 
    #         bandmath_expr='image * 2', expr_info=expr_info, result_name=result_name, cache=cache,
    #         variables={'image':(VariableType.IMAGE_CUBE, img)}, functions={})

    #     process_manager.get_task().wait()
    #     results = process_manager.get_task().get_result()

    #     for result_type, result, result_name, expr_info in results:
    #         assert result_type == RasterDataSet or result_type == VariableType.IMAGE_CUBE

    #         # Check if result_type is RasterDataSet or IMAGE_CUBE and handle accordingly
    #         if result_type == RasterDataSet:
    #             result_dataset = load_image_from_bandmath_result(result_type, result, result_name, None, expr_info, loader, None)
    #             result_arr = result_dataset.get_image_data()
    #         elif result_type == VariableType.IMAGE_CUBE:
    #             result_arr = result
    #         else:
    #             self.fail(f"Unexpected result type: {result_type}")

    #         # Make sure output image has correct results
    #         for value in np.nditer(result_arr):
    #             self.assertEqual(value, 5.0)

    #         # Make sure input image didn't change
    #         for value in np.nditer(img):
    #             self.assertEqual(value, 2.5)

    # def test_bandmath_mul_number_image(self):
    #     ''' Test band-math involving multiplying a number and an image. '''
    #     img = make_image(2, 4, 5)
    #     img.fill(2.5)

    #     expr_info = get_bandmath_expr_info('2 * image',
    #         {'image':(VariableType.IMAGE_CUBE, img)}, {})
    #     result_name = 'test_result'

    #     cache = DataCache()

    #     process_manager = bandmath.eval_bandmath_expr(succeeded_callback=lambda _: None,
    #         status_callback=lambda _: None, error_callback=lambda _: None, 
    #         bandmath_expr='2 * image', expr_info=expr_info, result_name=result_name, cache=cache,
    #         variables={'image':(VariableType.IMAGE_CUBE, img)}, functions={})

    #     process_manager.get_task().wait()
    #     results = process_manager.get_task().get_result()

    #     for result_type, result, result_name, expr_info in results:
    #         assert result_type == RasterDataSet or result_type == VariableType.IMAGE_CUBE

    #         # Check if result_type is RasterDataSet or IMAGE_CUBE and handle accordingly
    #         if result_type == RasterDataSet:
    #             result_dataset = load_image_from_bandmath_result(result_type, result, result_name, None, expr_info, loader, None)
    #             result_arr = result_dataset.get_image_data()
    #         elif result_type == VariableType.IMAGE_CUBE:
    #             result_arr = result
    #         else:
    #             self.fail(f"Unexpected result type: {result_type}")

    #         # Make sure output image has correct results
    #         for value in np.nditer(result_arr):
    #             self.assertEqual(value, 5.0)

    #         # Make sure input image didn't change
    #         for value in np.nditer(img):
    #             self.assertEqual(value, 2.5)

    # #===========================================================================
    # # SIMPLE DIVISION TESTS

    # def test_bandmath_div_image_number(self):
    #     ''' Test band-math involving dividing an image by a number. '''
    #     img = make_image(2, 4, 5)
    #     img.fill(2.5)

    #     expr_info = get_bandmath_expr_info('image / 0.5',
    #         {'image':(VariableType.IMAGE_CUBE, img)}, {})
    #     result_name = 'test_result'

    #     cache = DataCache()

    #     process_manager = bandmath.eval_bandmath_expr(succeeded_callback=lambda _: None,
    #         status_callback=lambda _: None, error_callback=lambda _: None, 
    #         bandmath_expr='image / 0.5', expr_info=expr_info, result_name=result_name, cache=cache,
    #         variables={'image':(VariableType.IMAGE_CUBE, img)}, functions={})

    #     process_manager.get_task().wait()
    #     results = process_manager.get_task().get_result()

    #     for result_type, result, result_name, expr_info in results:
    #         assert result_type == RasterDataSet or result_type == VariableType.IMAGE_CUBE

    #         # Check if result_type is RasterDataSet or IMAGE_CUBE and handle accordingly
    #         if result_type == RasterDataSet:
    #             result_dataset = load_image_from_bandmath_result(result_type, result, result_name, None, expr_info, loader, None)
    #             result_arr = result_dataset.get_image_data()
    #         elif result_type == VariableType.IMAGE_CUBE:
    #             result_arr = result
    #         else:
    #             self.fail(f"Unexpected result type: {result_type}")

    #         # Make sure output image has correct results
    #         for value in np.nditer(result_arr):
    #             self.assertEqual(value, 5.0)

    #         # Make sure input image didn't change
    #         for value in np.nditer(img):
    #             self.assertEqual(value, 2.5)

    
    
    # #===========================================================================
    # # COMPLEX OPERATION TESTS
    # def test_bandmath_add_complex_expression(self):
    #     ''' Test band-math with a complex addition expression: (a + b) + (c + d) '''
    #     a = make_image(2, 3, 4)
    #     b = make_band(3, 4)
    #     c = make_image(2, 3, 4)
    #     d = make_spectrum(2)

    #     a.fill(1.0)
    #     b.fill(2.0)
    #     c.fill(3.0)
    #     d.fill(4.0)

    #     expr_info = get_bandmath_expr_info('(a + b) + (c + d)',
    #         {'a':(VariableType.IMAGE_CUBE, a),
    #          'b':(VariableType.IMAGE_BAND, b),
    #          'c':(VariableType.IMAGE_CUBE, c),
    #          'd':(VariableType.SPECTRUM, d)}, {})
    #     result_name = 'test_result'

    #     cache = DataCache()

    #     process_manager = bandmath.eval_bandmath_expr(succeeded_callback=lambda _: None,
    #         status_callback=lambda _: None, error_callback=lambda _: None, 
    #         bandmath_expr='(a + b) + (c + d)', expr_info=expr_info, result_name=result_name, cache=cache,
    #         variables={'a':(VariableType.IMAGE_CUBE, a),
    #          'b':(VariableType.IMAGE_BAND, b),
    #          'c':(VariableType.IMAGE_CUBE, c),
    #          'd':(VariableType.SPECTRUM, d)}, functions={})

    #     process_manager.get_task().wait()
    #     results = process_manager.get_task().get_result()

    #     for result_type, result, result_name, expr_info in results:
    #         assert result_type == RasterDataSet or result_type == VariableType.IMAGE_CUBE

    #         # Check if result_type is RasterDataSet or IMAGE_CUBE and handle accordingly
    #         if result_type == RasterDataSet:
    #             result_dataset = load_image_from_bandmath_result(result_type, result, result_name, None, expr_info, loader, None)
    #             result_arr = result_dataset.get_image_data()
    #         elif result_type == VariableType.IMAGE_CUBE:
    #             result_arr = result
    #         else:
    #             self.fail(f"Unexpected result type: {result_type}")

    #         # Verify the result
    #         for value in np.nditer(result_arr):
    #             self.assertEqual(value, 10.0)
        
    #         # Make sure input values didn't change
    #         for value in np.nditer(a):
    #             self.assertEqual(value, 1.0)
            
    #         for value in np.nditer(b):
    #             self.assertEqual(value, 2.0)
            
    #         for value in np.nditer(c):
    #             self.assertEqual(value, 3.0)
            
    #         for value in np.nditer(d):
    #             self.assertEqual(value, 4.0)


    # def test_bandmath_mul_complex_expression(self):
    #     ''' Test band-math with a complex multiplication expression: (a * b) * (c * d) '''
    #     a = make_image(2, 3, 4)
    #     b = make_band(3, 4)
    #     c = make_image(2, 3, 4)
    #     d = make_spectrum(2)

    #     a.fill(1.0)
    #     b.fill(2.0)
    #     c.fill(3.0)
    #     d.fill(4.0)

    #     expr_info = get_bandmath_expr_info('(a * b) * (c * d)',
    #         {'a':(VariableType.IMAGE_CUBE, a),
    #          'b':(VariableType.IMAGE_BAND, b),
    #          'c':(VariableType.IMAGE_CUBE, c),
    #          'd':(VariableType.SPECTRUM, d)}, {})
    #     result_name = 'test_result'

    #     cache = DataCache()

    #     process_manager = bandmath.eval_bandmath_expr(succeeded_callback=lambda _: None,
    #         status_callback=lambda _: None, error_callback=lambda _: None, 
    #         bandmath_expr='(a * b) * (c * d)', expr_info=expr_info, result_name=result_name, cache=cache,
    #         variables={'a':(VariableType.IMAGE_CUBE, a),
    #          'b':(VariableType.IMAGE_BAND, b),
    #          'c':(VariableType.IMAGE_CUBE, c),
    #          'd':(VariableType.SPECTRUM, d)}, functions={})

    #     process_manager.get_task().wait()
    #     results = process_manager.get_task().get_result()

    #     for result_type, result, result_name, expr_info in results:
    #         assert result_type == RasterDataSet or result_type == VariableType.IMAGE_CUBE

    #         # Check if result_type is RasterDataSet or IMAGE_CUBE and handle accordingly
    #         if result_type == RasterDataSet:
    #             result_dataset = load_image_from_bandmath_result(result_type, result, result_name, None, expr_info, loader, None)
    #             result_arr = result_dataset.get_image_data()
    #         elif result_type == VariableType.IMAGE_CUBE:
    #             result_arr = result
    #         else:
    #             self.fail(f"Unexpected result type: {result_type}")

    #         # Verify the result
    #         for value in np.nditer(result_arr):
    #             self.assertEqual(value, 24.0)  # 1 * 2 * 3 * 4
        
    #         # Make sure input values didn't change
    #         for value in np.nditer(a):
    #             self.assertEqual(value, 1.0)
            
    #         for value in np.nditer(b):
    #             self.assertEqual(value, 2.0)
            
    #         for value in np.nditer(c):
    #             self.assertEqual(value, 3.0)
            
    #         for value in np.nditer(d):
    #             self.assertEqual(value, 4.0)


    # def test_bandmath_div_complex_expression(self):
    #     ''' Test band-math with a complex division expression: (a / b) / (c / d) '''
    #     a = make_image(2, 3, 4)
    #     b = make_band(3, 4)
    #     c = make_image(2, 3, 4)
    #     d = make_spectrum(2)

    #     a.fill(16.0)
    #     b.fill(2.0)
    #     c.fill(4.0)
    #     d.fill(2.0)

    #     expr_info = get_bandmath_expr_info('(a / b) / (c / d)',
    #         {'a':(VariableType.IMAGE_CUBE, a),
    #          'b':(VariableType.IMAGE_BAND, b),
    #          'c':(VariableType.IMAGE_CUBE, c),
    #          'd':(VariableType.SPECTRUM, d)}, {})
    #     result_name = 'test_result'

    #     cache = DataCache()

    #     process_manager = bandmath.eval_bandmath_expr(succeeded_callback=lambda _: None,
    #         status_callback=lambda _: None, error_callback=lambda _: None, 
    #         bandmath_expr='(a / b) / (c / d)', expr_info=expr_info, result_name=result_name, cache=cache,
    #         variables={'a':(VariableType.IMAGE_CUBE, a),
    #          'b':(VariableType.IMAGE_BAND, b),
    #          'c':(VariableType.IMAGE_CUBE, c),
    #          'd':(VariableType.SPECTRUM, d)}, functions={})

    #     process_manager.get_task().wait()
    #     results = process_manager.get_task().get_result()

    #     for result_type, result, result_name, expr_info in results:
    #         assert result_type == RasterDataSet or result_type == VariableType.IMAGE_CUBE

    #         # Check if result_type is RasterDataSet or IMAGE_CUBE and handle accordingly
    #         if result_type == RasterDataSet:
    #             result_dataset = load_image_from_bandmath_result(result_type, result, result_name, None, expr_info, loader, None)
    #             result_arr = result_dataset.get_image_data()
    #         elif result_type == VariableType.IMAGE_CUBE:
    #             result_arr = result
    #         else:
    #             self.fail(f"Unexpected result type: {result_type}")

    #         # Verify the result
    #         for value in np.nditer(result_arr):
    #             self.assertEqual(value, 4.0)  # (16 / 2) / (4 / 2)
        
    #         # Make sure input values didn't change
    #         for value in np.nditer(a):
    #             self.assertEqual(value, 16.0)
            
    #         for value in np.nditer(b):
    #             self.assertEqual(value, 2.0)
            
    #         for value in np.nditer(c):
    #             self.assertEqual(value, 4.0)
            
    #         for value in np.nditer(d):
    #             self.assertEqual(value, 2.0)
        

    # def test_bandmath_sub_complex_expression(self):
    #     ''' Test band-math with a complex subtraction expression: (a - b) - (c - d) '''
    #     a = make_image(2, 3, 4)
    #     b = make_band(3, 4)
    #     c = make_image(2, 3, 4)
    #     d = make_spectrum(2)

    #     a.fill(10.0)
    #     b.fill(2.0)
    #     c.fill(5.0)
    #     d.fill(1.0)

    #     expr_info = get_bandmath_expr_info('(a - b) - (c - d)',
    #         {'a':(VariableType.IMAGE_CUBE, a),
    #          'b':(VariableType.IMAGE_BAND, b),
    #          'c':(VariableType.IMAGE_CUBE, c),
    #          'd':(VariableType.SPECTRUM, d)}, {})
    #     result_name = 'test_result'

    #     cache = DataCache()

    #     process_manager = bandmath.eval_bandmath_expr(succeeded_callback=lambda _: None,
    #         status_callback=lambda _: None, error_callback=lambda _: None, 
    #         bandmath_expr='(a - b) - (c - d)', expr_info=expr_info, result_name=result_name, cache=cache,
    #         variables={'a':(VariableType.IMAGE_CUBE, a),
    #          'b':(VariableType.IMAGE_BAND, b),
    #          'c':(VariableType.IMAGE_CUBE, c),
    #          'd':(VariableType.SPECTRUM, d)}, functions={})

    #     process_manager.get_task().wait()
    #     results = process_manager.get_task().get_result()

    #     for result_type, result, result_name, expr_info in results:
    #         assert result_type == RasterDataSet or result_type == VariableType.IMAGE_CUBE

    #         # Check if result_type is RasterDataSet or IMAGE_CUBE and handle accordingly
    #         if result_type == RasterDataSet:
    #             result_dataset = load_image_from_bandmath_result(result_type, result, result_name, None, expr_info, loader, None)
    #             result_arr = result_dataset.get_image_data()
    #         elif result_type == VariableType.IMAGE_CUBE:
    #             result_arr = result
    #         else:
    #             self.fail(f"Unexpected result type: {result_type}")

    #         # Verify the result
    #         for value in np.nditer(result_arr):
    #             self.assertEqual(value, 4.0)  # (10 - 2) - (5 - 1)
        
    #         # Make sure input values didn't change
    #         for value in np.nditer(a):
    #             self.assertEqual(value, 10.0)
            
    #         for value in np.nditer(b):
    #             self.assertEqual(value, 2.0)
            
    #         for value in np.nditer(c):
    #             self.assertEqual(value, 5.0)
            
    #         for value in np.nditer(d):
    #             self.assertEqual(value, 1.0)
        

    # def test_bandmath_neg_a_plus_1(self):
    #     ''' Test band-math for expression -a + 1 '''
    #     a = make_image(2, 3, 4)
    #     a.fill(10.0)

    #     expr_info = get_bandmath_expr_info('-a + 1',
    #         {'a':(VariableType.IMAGE_CUBE, a)}, {})
    #     result_name = 'test_result'

    #     cache = DataCache()

    #     process_manager = bandmath.eval_bandmath_expr(succeeded_callback=lambda _: None,
    #         status_callback=lambda _: None, error_callback=lambda _: None, 
    #         bandmath_expr='-a + 1', expr_info=expr_info, result_name=result_name, cache=cache,
    #         variables={'a':(VariableType.IMAGE_CUBE, a)}, functions={})

    #     process_manager.get_task().wait()
    #     results = process_manager.get_task().get_result()

    #     for result_type, result, result_name, expr_info in results:
    #         assert result_type == RasterDataSet or result_type == VariableType.IMAGE_CUBE

    #         # Check if result_type is RasterDataSet or IMAGE_CUBE and handle accordingly
    #         if result_type == RasterDataSet:
    #             result_dataset = load_image_from_bandmath_result(result_type, result, result_name, None, expr_info, loader, None)
    #             result_arr = result_dataset.get_image_data()
    #         elif result_type == VariableType.IMAGE_CUBE:
    #             result_arr = result
    #         else:
    #             self.fail(f"Unexpected result type: {result_type}")

    #         # Verify the result
    #         for value in np.nditer(result_arr):
    #             self.assertEqual(value, -9.0)  # -10 + 1
        
    #         # Make sure input values didn't change
    #         for value in np.nditer(a):
    #             self.assertEqual(value, 10.0)
        

    # def test_bandmath_a_pow_b_minus_sqrt_a(self):
    #     ''' Test band-math for expression a**b - (a**0.5) '''
    #     a = make_image(2, 3, 4)
    #     b = make_band(3, 4)

    #     a.fill(4.0)
    #     b.fill(2.0)

    #     expr_info = get_bandmath_expr_info('a**b - (a**0.5)',
    #         {'a':(VariableType.IMAGE_CUBE, a),
    #          'b':(VariableType.IMAGE_BAND, b)}, {})
    #     result_name = 'test_result'

    #     cache = DataCache()

    #     process_manager = bandmath.eval_bandmath_expr(succeeded_callback=lambda _: None,
    #         status_callback=lambda _: None, error_callback=lambda _: None, 
    #         bandmath_expr='a**b - (a**0.5)', expr_info=expr_info, result_name=result_name, cache=cache,
    #         variables={'a':(VariableType.IMAGE_CUBE, a),
    #          'b':(VariableType.IMAGE_BAND, b)}, functions={})

    #     process_manager.get_task().wait()
    #     results = process_manager.get_task().get_result()

    #     for result_type, result, result_name, expr_info in results:
    #         assert result_type == RasterDataSet or result_type == VariableType.IMAGE_CUBE

    #         # Check if result_type is RasterDataSet or IMAGE_CUBE and handle accordingly
    #         if result_type == RasterDataSet:
    #             result_dataset = load_image_from_bandmath_result(result_type, result, result_name, None, expr_info, loader, None)
    #             result_arr = result_dataset.get_image_data()
    #         elif result_type == VariableType.IMAGE_CUBE:
    #             result_arr = result
    #         else:
    #             self.fail(f"Unexpected result type: {result_type}")

    #         # Verify the result
    #         for value in np.nditer(result_arr):
    #             self.assertEqual(value, 14.0)  # 4**2 - (4**0.5) => 16 - 2
        
    #         for value in np.nditer(a):
    #             self.assertEqual(value, 4.0)
            
    #         for value in np.nditer(b):
    #             self.assertEqual(value, 2.0)
        
    
    # def test_bandmath_all_operations(self):
    #     ''' Test band-math with a complex expression: (a / b) - (c * d) + a**0.5 '''
    #     bands = 2
    #     samples = 3
    #     lines = 4
    #     a = make_image(bands, samples, lines)
    #     b = make_band(samples, lines)
    #     c = make_image(bands, samples, lines)
    #     d = make_spectrum(bands)

    #     a.fill(16.0)
    #     b.fill(2.0)
    #     c.fill(4.0)
    #     d.fill(2.0)

    #     expr_info = get_bandmath_expr_info('(a / b) - (c * d) + a**0.5',
    #         {'a': (VariableType.IMAGE_CUBE, a),
    #          'b': (VariableType.IMAGE_BAND, b),
    #          'c': (VariableType.IMAGE_CUBE, c),
    #          'd': (VariableType.SPECTRUM, d)}, {})
    #     result_name = 'test_result'

    #     cache = DataCache()

    #     process_manager = bandmath.eval_bandmath_expr(succeeded_callback=lambda _: None,
    #         status_callback=lambda _: None, error_callback=lambda _: None, 
    #         bandmath_expr='(a / b) - (c * d) + a**0.5', expr_info=expr_info, result_name=result_name, cache=cache,
    #         variables={'a': (VariableType.IMAGE_CUBE, a),
    #          'b': (VariableType.IMAGE_BAND, b),
    #          'c': (VariableType.IMAGE_CUBE, c),
    #          'd': (VariableType.SPECTRUM, d)}, functions={})

    #     process_manager.get_task().wait()
    #     results = process_manager.get_task().get_result()

    #     for result_type, result, result_name, expr_info in results:
    #         assert result_type == RasterDataSet or result_type == VariableType.IMAGE_CUBE

    #         # Check if result_type is RasterDataSet or IMAGE_CUBE and handle accordingly
    #         if result_type == RasterDataSet:
    #             result_dataset = load_image_from_bandmath_result(result_type, result, result_name, None, expr_info, loader, None)
    #             result_arr = result_dataset.get_image_data()
    #         elif result_type == VariableType.IMAGE_CUBE:
    #             result_arr = result
    #         else:
    #             self.fail(f"Unexpected result type: {result_type}")
        
    #         # Verify the result
    #         for value in np.nditer(result_arr):
    #             expected_value = (16.0 / 2.0) - (4.0 * 2.0) + (16.0 ** 0.5)
    #             self.assertEqual(value, expected_value)  # (16 / 2) - (4 * 2) + sqrt(16)

    #         # Make sure input values didn't change
    #         for value in np.nditer(a):
    #             self.assertEqual(value, 16.0)

    #         for value in np.nditer(b):
    #             self.assertEqual(value, 2.0)

    #         for value in np.nditer(c):
    #             self.assertEqual(value, 4.0)

    #         for value in np.nditer(d):
    #             self.assertEqual(value, 2.0)


    # def test_bandmath_image_metadata_copying(self):
    #     """Tests that a GeoTIFF `.tiff` file can be successfully opened and loaded into WISER."""
    #     current_dir = os.path.dirname(os.path.abspath(__file__))

    #     target_path = os.path.normpath(os.path.join(current_dir, "..", "test_utils", "test_datasets", "caltech_4_100_150_nm_epsg4087.tif"))

    #     caltech_ds = self.test_model.load_dataset(target_path)

    #     expr_info = get_bandmath_expr_info('a + 0', {'a': (VariableType.IMAGE_CUBE, caltech_ds)}, {})
    #     result_name = 'test_result'
    #     cache = DataCache()
    #     process_manager = bandmath.eval_bandmath_expr(succeeded_callback=lambda _: None,
    #         status_callback=lambda _: None, error_callback=lambda _: None, 
    #         bandmath_expr='a + 0', expr_info=expr_info, result_name=result_name, cache=cache,
    #         variables={'a': (VariableType.IMAGE_CUBE, caltech_ds)}, functions={})

    #     process_manager.get_task().wait()
    #     results = process_manager.get_task().get_result()

    #     for result_type, result, result_name, expr_info in results:
    #         assert result_type == RasterDataSet or result_type == VariableType.IMAGE_CUBE
    #         if result_type == RasterDataSet:
    #             result_dataset = load_image_from_bandmath_result(result_type, result, result_name, None, expr_info, loader, None)
    #             result_arr = result_dataset.get_image_data()
    #         elif result_type == VariableType.IMAGE_CUBE:
    #             result_arr = result
    #         else:
    #             self.fail(f"Unexpected result type: {result_type}")
            
    #         assert np.allclose(result_arr, caltech_ds.get_image_data())

    #         assert expr_info.spatial_metadata_source is not None
    #         assert expr_info.spectral_metadata_source is not None

    #         assert expr_info.spatial_metadata_source == caltech_ds.get_spatial_metadata()
    #         assert expr_info.spectral_metadata_source == caltech_ds.get_spectral_metadata()

    # def bandmath_preloaded_data_with_batch_helper(self, run_sync: bool):
    #     # Load in caltech_4_100_150nm
    #     current_dir = os.path.dirname(os.path.abspath(__file__))

    #     target_path = os.path.normpath(os.path.join(current_dir, "..", "test_utils", "test_datasets", "caltech_4_100_150_nm"))

    #     batch_test_folder = os.path.normpath(
    #         os.path.join(current_dir, "..", "test_utils", "test_datasets", "bandmath_batch_test_input_folder"))

    #     caltech_ds = self.test_model.load_dataset(target_path)

    #     band = RasterDataBand(caltech_ds, 0)

    #     spectrum = SpectrumAtPoint(caltech_ds, (1, 1))

    #     vars = [(VariableType.IMAGE_CUBE, caltech_ds), \
    #             (VariableType.IMAGE_BAND, band), \
    #             (VariableType.SPECTRUM, spectrum)]

    #     for var in vars:
    #         expr = 'a + b'
    #         variables = {'a': var, 'b': (VariableType.IMAGE_CUBE_BATCH, batch_test_folder)}
    #         expr_info = get_bandmath_expr_info(expr, variables, {})
    #         suffix = 'test_result'
    #         cache = DataCache()
    #         process_manager = bandmath.eval_bandmath_expr(succeeded_callback=lambda _: None,
    #             status_callback=lambda _: None, error_callback=lambda _: None, 
    #             bandmath_expr=expr, expr_info=expr_info, result_name=suffix, cache=cache,
    #             variables=variables, functions={}, use_synchronous_method=run_sync)
    #         process_manager.get_task().wait()
    #         results = process_manager.get_task().get_result()
    #         for result_type, result, result_name, expr_info in results:
    #             assert result_type == RasterDataSet or result_type == VariableType.IMAGE_CUBE
    #             if result_type == RasterDataSet or result_type == VariableType.IMAGE_CUBE:
    #                 result_ds = load_image_from_bandmath_result(result_type, result, result_name, expr, expr_info, loader, None)
    #                 result_arr = result_ds.get_image_data()
    #                 original_file_name = result_name[:-(len(suffix))] if result_name.endswith(suffix) else result_name
    #                 original_ds = loader.load_from_file(path=os.path.normpath(os.path.join(batch_test_folder, original_file_name)))[0]
    #                 if var[0] == VariableType.IMAGE_CUBE:
    #                     var_arr = var[1].get_image_data()
    #                 elif var[0] == VariableType.IMAGE_BAND:
    #                     var_arr = var[1].get_data()
    #                 elif var[0] == VariableType.SPECTRUM:
    #                     var_arr = var[1].get_spectrum()
    #                     if var_arr.ndim == 1:
    #                         var_arr = var_arr[:, np.newaxis, np.newaxis]
    #                 else:
    #                     self.fail(f"Unexpected variable type: {var[0]}")

    #                 assert np.allclose(result_arr, original_ds.get_image_data() + var_arr)
                    
                    
    #                 assert expr_info.spatial_metadata_source is not None
    #                 assert expr_info.spectral_metadata_source is not None
    #                 assert expr_info.spatial_metadata_source == original_ds.get_spatial_metadata()
    #                 assert expr_info.spectral_metadata_source == original_ds.get_spectral_metadata()
                    
    #                 del result
    #                 del original_ds

    # def test_bandmath_preloaded_data_with_image_batch_sync(self):
    #     self.bandmath_preloaded_data_with_batch_helper(run_sync=True)
    
    # def test_bandmath_preloaded_data_with_image_batch_async(self):
    #     self.bandmath_preloaded_data_with_batch_helper(run_sync=False)

    def bandmath_preloaded_data_with_band_batch_helper(self,
                                                       raster_batch_band: RasterDataBatchBand,
                                                       run_sync: bool,
                                                       success_callback: Callable = lambda _: None):
        current_dir = os.path.dirname(os.path.abspath(__file__))

        target_path = os.path.normpath(os.path.join(current_dir, "..", "test_utils", "test_datasets", "caltech_4_100_150_nm"))

        caltech_ds = self.test_model.load_dataset(target_path)

        band = RasterDataBand(caltech_ds, 0)

        vars = [(VariableType.IMAGE_CUBE, caltech_ds), \
                (VariableType.IMAGE_BAND, band)]

        status_callback = lambda msg: print(f"Msg from process: {msg}")

        print(f"going into vars")
        for var in vars:
            print(f"var to run on: {var}")
            expr = 'a + b'
            variables = {'a': var, 'b': (VariableType.IMAGE_BAND_BATCH, raster_batch_band)}
            expr_info = get_bandmath_expr_info(expr, variables, {})
            suffix = 'test_result'
            cache = DataCache()
            process_manager = bandmath.eval_bandmath_expr(succeeded_callback=success_callback,
                status_callback=status_callback, error_callback=lambda _: None, 
                bandmath_expr=expr, expr_info=expr_info, result_name=suffix, cache=cache,
                variables=variables, functions={}, use_synchronous_method=run_sync)
            process_manager.get_task().wait()
            results = process_manager.get_task().get_result()
            for result_type, result, result_name, expr_info in results:
                print(f"result_type: {result_type}")
                original_file_name = result_name[:-(len(suffix))] if result_name.endswith(suffix) else result_name
                original_ds = loader.load_from_file(path=os.path.normpath(os.path.join(raster_batch_band.get_folderpath(),
                                                                                    original_file_name)), interactive=False)[0]
                original_band = RasterDataDynamicBand(original_ds, raster_batch_band.get_band_index(),
                                                      raster_batch_band.get_wavelength_value(),
                                                      raster_batch_band.get_wavelength_units(),
                                                      raster_batch_band.get_epsilon())
                original_band_arr = original_band.get_data()
                print(f"$$#$ result instance: {type(result)}")
                if result_type == VariableType.IMAGE_CUBE or result_type == RasterDataSet:
                    assert isinstance(result, (np.ndarray, SerializedForm))
                    result_ds = load_image_from_bandmath_result(result_type, result, result_name, expr, expr_info, loader, None)
                    result_arr = result_ds.get_image_data()
                elif result_type == VariableType.IMAGE_BAND:
                    result_ds = load_band_from_bandmath_result(result=result, result_name=result_name,
                                                               expression=expr, expr_info=expr_info, parent=None,
                                                               loader=loader, app_state=None)
                    result_arr = result_ds.get_image_data()
                else:
                    self.fail(f"Unexpected result type: {result_type}")
                if var[0] == VariableType.IMAGE_CUBE:
                    var_arr = var[1].get_image_data()
                elif var[0] == VariableType.IMAGE_BAND:
                    var_arr = var[1].get_data()
                else:
                    self.fail(f"Unexpected variable type: {var[0]}")

                assert np.allclose(result_arr, original_band_arr + var_arr)
                
                
                assert expr_info.spatial_metadata_source is not None
                assert expr_info.spatial_metadata_source == original_ds.get_spatial_metadata()
                
                del result
                del original_ds


    # def test_bandmath_preloaded_data_with_band_index_batch_sync(self):
    #     current_dir = os.path.dirname(os.path.abspath(__file__))
    #     batch_test_folder = os.path.normpath(
    #         os.path.join(current_dir, "..", "test_utils", "test_datasets", "bandmath_batch_test_input_folder"))
    #     raster_batch_band = RasterDataBatchBand(batch_test_folder, 0)
    #     self.bandmath_preloaded_data_with_band_batch_helper(raster_batch_band, run_sync=True)

    def test_bandmath_preloaded_data_with_band_index_batch_async(self):
        print(f"Test is running")
        current_dir = os.path.dirname(os.path.abspath(__file__))
        batch_test_folder = os.path.normpath(
            os.path.join(current_dir, "..", "test_utils", "test_datasets", "bandmath_batch_test_input_folder"))
        raster_batch_band = RasterDataBatchBand(batch_test_folder, 0)
        print(f"calling helper")
        self.bandmath_preloaded_data_with_band_batch_helper(raster_batch_band, run_sync=False)

    # def test_bandmath_preloaded_data_with_band_wvl_batch_sync(self):
    #     current_dir = os.path.dirname(os.path.abspath(__file__))
    #     batch_test_folder = os.path.normpath(
    #         os.path.join(current_dir, "..", "test_utils", "test_datasets", "bandmath_batch_test_input_folder"))
    #     raster_batch_band = RasterDataBatchBand(
    #         batch_test_folder, band_index=None, wavelength_value=700, wavelength_units=u.nm, epsilon=20)
    #     self.bandmath_preloaded_data_with_band_batch_helper(raster_batch_band, run_sync=True)

    # def test_bandmath_preloaded_data_with_band_wvl_batch_async(self):
    #     current_dir = os.path.dirname(os.path.abspath(__file__))
    #     batch_test_folder = os.path.normpath(
    #         os.path.join(current_dir, "..", "test_utils", "test_datasets", "bandmath_batch_test_input_folder"))
    #     raster_batch_band = RasterDataBatchBand(
    #         batch_test_folder, band_index=None, wavelength_value=700, wavelength_units=u.nm, epsilon=20)
    #     self.bandmath_preloaded_data_with_band_batch_helper(raster_batch_band, run_sync=False)

    # # TODO (Joshua G-K): Write these tests for bandmath batch buttons.
    # def test_bandmath_batch_cancel(self):
    #     pass

    # def test_bandmath_batch_remove(self):
    #     pass

    # def test_bandmath_batch_run_multiple(self):
    #     pass

    # def test_bandmath_batch_progress_bar(self):
    #     pass

    # def test_bandmath_batch_view_errors(self):
    #     pass

if __name__ == '__main__':
    test_class = TestBandmathEvaluator()
    test_class.setUp()
    test_class.test_bandmath_preloaded_data_with_band_index_batch_sync()
    test_class.tearDown()
