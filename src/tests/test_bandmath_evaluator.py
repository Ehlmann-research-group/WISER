import sys
import os

import unittest
import numpy as np

import tests.context

from wiser import bandmath
from wiser.bandmath import VariableType
from wiser.bandmath.analyzer import get_bandmath_expr_info
from wiser.raster.dataset import RasterDataSet
from wiser.raster.loader import RasterDataLoader

from wiser.raster.data_cache import DataCache


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

    #===========================================================================
    # NUMBERS-ONLY TEST

    def test_bandmath_numbers_only(self):
        ''' Test band-math involving only numbers. '''
        expr_info = get_bandmath_expr_info('(2 * 3 + 4) / 2',
            {}, {})
        result_name = 'test_result'

        cache = DataCache()

        (result_type, result_value) = \
            bandmath.eval_bandmath_expr('(2 * 3 + 4) / 2', expr_info, result_name, cache, {}, {})

        self.assertEqual(result_type, VariableType.NUMBER)
        self.assertEqual(result_value, 5)

    #===========================================================================
    # SIMPLE ADDITION TESTS

    def test_bandmath_add_image_band(self):
        ''' Test band-math involving adding an image and a band. '''
        img = make_image(2, 3, 4)
        img[0].fill(1.0)
        img[1].fill(2.0)

        band = make_band(3, 4)
        for y in range(band.shape[1]):
            for x in range(band.shape[0]):
                band[x, y] = 0.5 * x + 10 * y

        expr_info = get_bandmath_expr_info('image + band',
            {'image':(VariableType.IMAGE_CUBE, img),
             'band':(VariableType.IMAGE_BAND, band)}, {})
        result_name = 'test_result'
    
        cache = DataCache()

        (result_type, result_dataset) = bandmath.eval_bandmath_expr('image + band', expr_info, result_name, cache,
            {'image':(VariableType.IMAGE_CUBE, img),
             'band':(VariableType.IMAGE_BAND, band)}, {})

        assert result_type == RasterDataSet or result_type == VariableType.IMAGE_CUBE

        # Check if result_type is RasterDataSet or IMAGE_CUBE and handle accordingly
        if result_type == RasterDataSet:
            result_arr = result_dataset.get_image_data()
            result_shape = result_dataset.get_shape()
        elif result_type == VariableType.IMAGE_CUBE:
            result_arr = result_dataset
            result_shape = result_dataset.shape
        else:
            self.fail(f"Unexpected result type: {result_type}")

        # Make sure output image has correct results
        self.assertEqual(result_shape, (2, 3, 4))
        for b in range(result_shape[0]):
            for y in range(result_shape[2]):
                for x in range(result_shape[1]):
                    self.assertEqual(result_arr[b, x, y], (b + 1.0) + 0.5 * x + 10 * y)
        # Make sure input image didn't change

        for value in np.nditer(img[0]):
            self.assertEqual(value, 1.0)

        for value in np.nditer(img[1]):
            self.assertEqual(value, 2.0)

    def test_bandmath_add_image_number(self):
        ''' Test band-math involving adding an image and a number. '''
        img = make_image(2, 4, 5)
        img.fill(2.5)

        expr_info = get_bandmath_expr_info('image + 0.5',
            {'image':(VariableType.IMAGE_CUBE, img)}, {})
        result_name = 'test_result'

        cache = DataCache()

        (result_type, result_dataset) = bandmath.eval_bandmath_expr('image + 0.5', expr_info, result_name, cache,
            {'image':(VariableType.IMAGE_CUBE, img)}, {})

        assert result_type == RasterDataSet or result_type == VariableType.IMAGE_CUBE
        
        # Check if result_type is RasterDataSet or IMAGE_CUBE and handle accordingly
        if result_type == RasterDataSet:
            result_arr = result_dataset.get_image_data()
        elif result_type == VariableType.IMAGE_CUBE:
            result_arr = result_dataset
        else:
            self.fail(f"Unexpected result type: {result_type}")

        # Make sure output image has correct results
        for value in np.nditer(result_arr):
            self.assertEqual(value, 3.0)

        # Make sure input image didn't change
        for value in np.nditer(img):
            self.assertEqual(value, 2.5)

    def test_bandmath_add_band_image(self):
        ''' Test band-math involving adding a band and an image. '''
        img = make_image(2, 3, 4)
        img[0].fill(1.0)
        img[1].fill(2.0)

        band = make_band(3, 4)
        for y in range(band.shape[1]):
            for x in range(band.shape[0]):
                band[x, y] = 0.5 * x + 10 * y

        expr_info = get_bandmath_expr_info('band + image',
            {'image':(VariableType.IMAGE_CUBE, img),
             'band':(VariableType.IMAGE_BAND, band)}, {})
        result_name = 'test_result'

        cache = DataCache()

        (result_type, result_dataset) = bandmath.eval_bandmath_expr('band + image', expr_info, result_name, cache,
            {'image':(VariableType.IMAGE_CUBE, img),
             'band':(VariableType.IMAGE_BAND, band)}, {})

        assert result_type == RasterDataSet or result_type == VariableType.IMAGE_CUBE

        
        # Check if result_type is RasterDataSet or IMAGE_CUBE and handle accordingly
        if result_type == RasterDataSet:
            result_arr = result_dataset.get_image_data()
            result_shape = result_dataset.get_shape()
        elif result_type == VariableType.IMAGE_CUBE:
            result_arr = result_dataset
            result_shape = result_dataset.shape
        else:
            self.fail(f"Unexpected result type: {result_type}")

        # Make sure output image has correct results
        self.assertEqual(result_shape, (2, 3, 4))
        for b in range(result_shape[0]):
            for y in range(result_shape[2]):
                for x in range(result_shape[1]):
                    self.assertEqual (result_arr[b, x, y], (b + 1.0) + 0.5 * x + 10 * y)

        # Make sure input image didn't change

        for value in np.nditer(img[0]):
            self.assertEqual(value, 1.0)

        for value in np.nditer(img[1]):
            self.assertEqual(value, 2.0)

    def test_bandmath_add_band_number(self):
        ''' Test band-math involving adding a band and a number. '''
        band = make_band(4, 5)
        band.fill(2.5)

        expr_info = get_bandmath_expr_info('b + 0.5',
            {'b':(VariableType.IMAGE_BAND, band)}, {})
        result_name = 'test_result'

        cache = DataCache()

        (result_type, result_band) = bandmath.eval_bandmath_expr('b + 0.5', expr_info, result_name, cache,
            {'b':(VariableType.IMAGE_BAND, band)}, {})

        self.assertEqual(result_type, VariableType.IMAGE_BAND)

        # Make sure output band has correct results
        for value in np.nditer(result_band):
            self.assertEqual(value, 3.0)

        # Make sure input band didn't change
        for value in np.nditer(band):
            self.assertEqual(value, 2.5)

    def test_bandmath_add_spectrum_number(self):
        ''' Test band-math involving adding a spectrum and a number. '''
        spectrum = make_spectrum(15)
        spectrum.fill(2.5)

        expr_info = get_bandmath_expr_info('S1 + 0.5',
            {'s1':(VariableType.SPECTRUM, spectrum)}, {})
        result_name = 'test_result'
        
        cache = DataCache()

        (result_type, result_spectrum) = bandmath.eval_bandmath_expr('S1 + 0.5', expr_info, result_name, cache,
            {'s1':(VariableType.SPECTRUM, spectrum)}, {})

        self.assertEqual(result_type, VariableType.SPECTRUM)

        # Make sure output spectrum has correct results
        for value in np.nditer(result_spectrum):
            self.assertEqual(value, 3.0)

        # Make sure input spectrum didn't change
        for value in np.nditer(spectrum):
            self.assertEqual(value, 2.5)

    def test_bandmath_add_number_image(self):
        ''' Test band-math involving adding a number and an image. '''
        img = make_image(2, 4, 5)
        img.fill(2.5)

        expr_info = get_bandmath_expr_info('0.5 + image',
            {'image':(VariableType.IMAGE_CUBE, img)}, {})
        result_name = 'test_result'
    
        cache = DataCache()

        (result_type, result_dataset) = bandmath.eval_bandmath_expr('0.5 + image', expr_info, result_name, cache,
            {'image':(VariableType.IMAGE_CUBE, img)}, {})

        assert result_type == RasterDataSet or result_type == VariableType.IMAGE_CUBE

        # Check if result_type is RasterDataSet or IMAGE_CUBE and handle accordingly
        if result_type == RasterDataSet:
            result_arr = result_dataset.get_image_data()
        elif result_type == VariableType.IMAGE_CUBE:
            result_arr = result_dataset
        else:
            self.fail(f"Unexpected result type: {result_type}")

        # Make sure output image has correct results
        for value in np.nditer(result_arr):
            self.assertEqual(value, 3.0)

        # Make sure input image didn't change
        for value in np.nditer(img):
            self.assertEqual(value, 2.5)
            
        del result_dataset

    def test_bandmath_add_number_band(self):
        ''' Test band-math involving adding a number and a band. '''
        band = make_band(4, 5)
        band.fill(2.5)

        expr_info = get_bandmath_expr_info('0.5 + b',
            {'b':(VariableType.IMAGE_BAND, band)}, {})
    
        cache = DataCache()

        (result_type, result_band) = bandmath.eval_bandmath_expr('0.5 + b', expr_info, None, cache,
            {'b':(VariableType.IMAGE_BAND, band)}, {})

        self.assertEqual(result_type, VariableType.IMAGE_BAND)

        # Make sure output band has correct results
        for value in np.nditer(result_band):
            self.assertEqual(value, 3.0)

        # Make sure input band didn't change
        for value in np.nditer(band):
            self.assertEqual(value, 2.5)

    def test_bandmath_add_number_spectrum(self):
        ''' Test band-math involving adding a number and a spectrum. '''
        spectrum = make_spectrum(15)
        spectrum.fill(2.5)

        expr_info = get_bandmath_expr_info('0.5 + S1',
            {'s1':(VariableType.SPECTRUM, spectrum)}, {})
        result_name = 'test_result'

        cache = DataCache()

        (result_type, result_spectrum) = bandmath.eval_bandmath_expr('0.5 + S1', expr_info, result_name, cache,
            {'s1':(VariableType.SPECTRUM, spectrum)}, {})

        self.assertEqual(result_type, VariableType.SPECTRUM)

        # Make sure output spectrum has correct results
        for value in np.nditer(result_spectrum):
            self.assertEqual(value, 3.0)

        # Make sure input spectrum didn't change
        for value in np.nditer(spectrum):
            self.assertEqual(value, 2.5)

    #===========================================================================
    # SIMPLE SUBTRACTION TESTS

    def test_bandmath_sub_image_band(self):
        ''' Test band-math involving subtracting a band from an image. '''
        img = make_image(2, 3, 4)
        img[0].fill(1.0)
        img[1].fill(2.0)

        band = make_band(3, 4)
        for y in range(band.shape[1]):
            for x in range(band.shape[0]):
                band[x, y] = 0.5 * x + 10 * y

        expr_info = get_bandmath_expr_info('image - band',
            {'image':(VariableType.IMAGE_CUBE, img),
             'band':(VariableType.IMAGE_BAND, band)}, {})
        result_name = 'test_result'
    
        cache = DataCache()

        (result_type, result_dataset) = bandmath.eval_bandmath_expr('image - band', expr_info, result_name, cache,
            {'image':(VariableType.IMAGE_CUBE, img),
             'band':(VariableType.IMAGE_BAND, band)}, {})

        assert result_type == RasterDataSet or result_type == VariableType.IMAGE_CUBE

        # Check if result_type is RasterDataSet or IMAGE_CUBE and handle accordingly
        if result_type == RasterDataSet:
            result_arr = result_dataset.get_image_data()
            result_shape = result_dataset.get_shape()
        elif result_type == VariableType.IMAGE_CUBE:
            result_arr = result_dataset
            result_shape = result_dataset.shape
        else:
            self.fail(f"Unexpected result type: {result_type}")

        # Make sure output image has correct results
        self.assertEqual(result_shape, (2, 3, 4))
        for b in range(result_shape[0]):
            for y in range(result_shape[2]):
                for x in range(result_shape[1]):
                    self.assertEqual(result_arr[b, x, y],
                                     (b + 1.0) - (0.5 * x + 10 * y))

        # Make sure input image didn't change

        for value in np.nditer(img[0]):
            self.assertEqual(value, 1.0)

        for value in np.nditer(img[1]):
            self.assertEqual(value, 2.0)
            
        del result_dataset

    def test_bandmath_sub_image_number(self):
        ''' Test band-math involving subtracting a number from an image. '''
        img = make_image(2, 4, 5)
        img.fill(2.5)

        expr_info = get_bandmath_expr_info('image - 0.5',
            {'image':(VariableType.IMAGE_CUBE, img)}, {})
        result_name = 'test_result'
    
        cache = DataCache()

        (result_type, result_dataset) = bandmath.eval_bandmath_expr('image - 0.5', expr_info, result_name, cache,
            {'image':(VariableType.IMAGE_CUBE, img)}, {})

        assert result_type == RasterDataSet or result_type == VariableType.IMAGE_CUBE

        # Check if result_type is RasterDataSet or IMAGE_CUBE and handle accordingly
        if result_type == RasterDataSet:
            result_arr = result_dataset.get_image_data()
        elif result_type == VariableType.IMAGE_CUBE:
            result_arr = result_dataset
        else:
            self.fail(f"Unexpected result type: {result_type}")

        # Make sure output image has correct results
        for value in np.nditer(result_arr):
            self.assertEqual(value, 2.0)

        # Make sure input image didn't change
        for value in np.nditer(img):
            self.assertEqual(value, 2.5)
            
        del result_dataset

    def test_bandmath_sub_band_number(self):
        ''' Test band-math involving subtracting a number from a band. '''
        band = make_band(4, 5)
        band.fill(2.5)

        expr_info = get_bandmath_expr_info('B1 - 0.5',
            {'B1':(VariableType.IMAGE_BAND, band)}, {})
        result_name = 'test_result'
    
        cache = DataCache()

        (result_type, result_band) = bandmath.eval_bandmath_expr('B1 - 0.5', expr_info, result_name, cache,
            {'B1':(VariableType.IMAGE_BAND, band)}, {})

        self.assertEqual(result_type, VariableType.IMAGE_BAND)

        # Make sure output band has correct results
        for value in np.nditer(result_band):
            self.assertEqual(value, 2.0)

        # Make sure input band didn't change
        for value in np.nditer(band):
            self.assertEqual(value, 2.5)

    def test_bandmath_sub_spectrum_number(self):
        ''' Test band-math involving subtracting a number from a spectrum. '''
        spectrum = make_spectrum(15)
        spectrum.fill(2.5)

        expr_info = get_bandmath_expr_info('s - 0.5',
            {'s':(VariableType.SPECTRUM, spectrum)}, {})
        result_name = 'test_result'

        cache = DataCache()

        (result_type, result_spectrum) = bandmath.eval_bandmath_expr('s - 0.5', expr_info, result_name, cache,
            {'s':(VariableType.SPECTRUM, spectrum)}, {})

        self.assertEqual(result_type, VariableType.SPECTRUM)

        # Make sure output image has correct results
        for value in np.nditer(result_spectrum):
            self.assertEqual(value, 2.0)

        # Make sure input image didn't change
        for value in np.nditer(spectrum):
            self.assertEqual(value, 2.5)

    #===========================================================================
    # SIMPLE MULTIPLICATION TESTS

    def test_bandmath_mul_image_number(self):
        ''' Test band-math involving multiplying an image and a number. '''
        img = make_image(2, 4, 5)
        img.fill(2.5)

        expr_info = get_bandmath_expr_info('image * 2',
            {'image':(VariableType.IMAGE_CUBE, img)}, {})
        result_name = 'test_result'
    
        cache = DataCache()

        (result_type, result_dataset) = bandmath.eval_bandmath_expr('image * 2', expr_info, result_name, cache,
            {'image':(VariableType.IMAGE_CUBE, img)}, {})

        assert result_type == RasterDataSet or result_type == VariableType.IMAGE_CUBE

        # Check if result_type is RasterDataSet or IMAGE_CUBE and handle accordingly
        if result_type == RasterDataSet:
            result_arr = result_dataset.get_image_data()
        elif result_type == VariableType.IMAGE_CUBE:
            result_arr = result_dataset
        else:
            self.fail(f"Unexpected result type: {result_type}")

        # Make sure output image has correct results
        for value in np.nditer(result_arr):
            self.assertEqual(value, 5.0)

        # Make sure input image didn't change
        for value in np.nditer(img):
            self.assertEqual(value, 2.5)
            
        del result_dataset

    def test_bandmath_mul_number_image(self):
        ''' Test band-math involving multiplying a number and an image. '''
        img = make_image(2, 4, 5)
        img.fill(2.5)

        expr_info = get_bandmath_expr_info('2 * image',
            {'image':(VariableType.IMAGE_CUBE, img)}, {})
        result_name = 'test_result'

        cache = DataCache()

        (result_type, result_dataset) = bandmath.eval_bandmath_expr('2 * image', expr_info, result_name, cache,
            {'image':(VariableType.IMAGE_CUBE, img)}, {})

        assert result_type == RasterDataSet or result_type == VariableType.IMAGE_CUBE

        # Check if result_type is RasterDataSet or IMAGE_CUBE and handle accordingly
        if result_type == RasterDataSet:
            result_arr = result_dataset.get_image_data()
        elif result_type == VariableType.IMAGE_CUBE:
            result_arr = result_dataset
        else:
            self.fail(f"Unexpected result type: {result_type}")

        # Make sure output image has correct results
        for value in np.nditer(result_arr):
            self.assertEqual(value, 5.0)

        # Make sure input image didn't change
        for value in np.nditer(img):
            self.assertEqual(value, 2.5)
            
        del result_dataset

    #===========================================================================
    # SIMPLE DIVISION TESTS

    def test_bandmath_div_image_number(self):
        ''' Test band-math involving dividing an image by a number. '''
        img = make_image(2, 4, 5)
        img.fill(2.5)

        expr_info = get_bandmath_expr_info('image / 0.5',
            {'image':(VariableType.IMAGE_CUBE, img)}, {})
        result_name = 'test_result'

        cache = DataCache()

        (result_type, result_dataset) = bandmath.eval_bandmath_expr('image / 0.5', expr_info, result_name, cache,
            {'image':(VariableType.IMAGE_CUBE, img)}, {})

        assert result_type == RasterDataSet or result_type == VariableType.IMAGE_CUBE

        # Check if result_type is RasterDataSet or IMAGE_CUBE and handle accordingly
        if result_type == RasterDataSet:
            result_arr = result_dataset.get_image_data()
        elif result_type == VariableType.IMAGE_CUBE:
            result_arr = result_dataset
        else:
            self.fail(f"Unexpected result type: {result_type}")

        # Make sure output image has correct results
        for value in np.nditer(result_arr):
            self.assertEqual(value, 5.0)

        # Make sure input image didn't change
        for value in np.nditer(img):
            self.assertEqual(value, 2.5)
            
        del result_dataset
    
    
    #===========================================================================
    # COMPLEX OPERATION TESTS
    def test_bandmath_add_complex_expression(self):
        ''' Test band-math with a complex addition expression: (a + b) + (c + d) '''
        a = make_image(2, 3, 4)
        b = make_band(3, 4)
        c = make_image(2, 3, 4)
        d = make_spectrum(2)

        a.fill(1.0)
        b.fill(2.0)
        c.fill(3.0)
        d.fill(4.0)

        expr_info = get_bandmath_expr_info('(a + b) + (c + d)',
            {'a':(VariableType.IMAGE_CUBE, a),
             'b':(VariableType.IMAGE_BAND, b),
             'c':(VariableType.IMAGE_CUBE, c),
             'd':(VariableType.SPECTRUM, d)}, {})
        result_name = 'test_result'

        cache = DataCache()

        (result_type, result_dataset) = bandmath.eval_bandmath_expr('(a + b) + (c + d)', expr_info, result_name, cache,
            {'a':(VariableType.IMAGE_CUBE, a),
             'b':(VariableType.IMAGE_BAND, b),
             'c':(VariableType.IMAGE_CUBE, c),
             'd':(VariableType.SPECTRUM, d)}, {})

        assert result_type == RasterDataSet or result_type == VariableType.IMAGE_CUBE

        # Check if result_type is RasterDataSet or IMAGE_CUBE and handle accordingly
        if result_type == RasterDataSet:
            result_arr = result_dataset.get_image_data()
        elif result_type == VariableType.IMAGE_CUBE:
            result_arr = result_dataset
        else:
            self.fail(f"Unexpected result type: {result_type}")

        # Verify the result
        for value in np.nditer(result_arr):
            self.assertEqual(value, 10.0)
        
        # Make sure input values didn't change
        for value in np.nditer(a):
            self.assertEqual(value, 1.0)
        
        for value in np.nditer(b):
            self.assertEqual(value, 2.0)
        
        for value in np.nditer(c):
            self.assertEqual(value, 3.0)
        
        for value in np.nditer(d):
            self.assertEqual(value, 4.0)

        del result_dataset

    def test_bandmath_mul_complex_expression(self):
        ''' Test band-math with a complex multiplication expression: (a * b) * (c * d) '''
        a = make_image(2, 3, 4)
        b = make_band(3, 4)
        c = make_image(2, 3, 4)
        d = make_spectrum(2)

        a.fill(1.0)
        b.fill(2.0)
        c.fill(3.0)
        d.fill(4.0)

        expr_info = get_bandmath_expr_info('(a * b) * (c * d)',
            {'a':(VariableType.IMAGE_CUBE, a),
             'b':(VariableType.IMAGE_BAND, b),
             'c':(VariableType.IMAGE_CUBE, c),
             'd':(VariableType.SPECTRUM, d)}, {})
        result_name = 'test_result'

        cache = DataCache()

        (result_type, result_dataset) = bandmath.eval_bandmath_expr('(a * b) * (c * d)', expr_info, result_name, cache,
            {'a':(VariableType.IMAGE_CUBE, a),
             'b':(VariableType.IMAGE_BAND, b),
             'c':(VariableType.IMAGE_CUBE, c),
             'd':(VariableType.SPECTRUM, d)}, {})

        assert result_type == RasterDataSet or result_type == VariableType.IMAGE_CUBE

        # Check if result_type is RasterDataSet or IMAGE_CUBE and handle accordingly
        if result_type == RasterDataSet:
            result_arr = result_dataset.get_image_data()
        elif result_type == VariableType.IMAGE_CUBE:
            result_arr = result_dataset
        else:
            self.fail(f"Unexpected result type: {result_type}")

        # Verify the result
        for value in np.nditer(result_arr):
            self.assertEqual(value, 24.0)  # 1 * 2 * 3 * 4
        
        # Make sure input values didn't change
        for value in np.nditer(a):
            self.assertEqual(value, 1.0)
        
        for value in np.nditer(b):
            self.assertEqual(value, 2.0)
        
        for value in np.nditer(c):
            self.assertEqual(value, 3.0)
        
        for value in np.nditer(d):
            self.assertEqual(value, 4.0)
        
        del result_dataset

    def test_bandmath_div_complex_expression(self):
        ''' Test band-math with a complex division expression: (a / b) / (c / d) '''
        a = make_image(2, 3, 4)
        b = make_band(3, 4)
        c = make_image(2, 3, 4)
        d = make_spectrum(2)

        a.fill(16.0)
        b.fill(2.0)
        c.fill(4.0)
        d.fill(2.0)

        expr_info = get_bandmath_expr_info('(a / b) / (c / d)',
            {'a':(VariableType.IMAGE_CUBE, a),
             'b':(VariableType.IMAGE_BAND, b),
             'c':(VariableType.IMAGE_CUBE, c),
             'd':(VariableType.SPECTRUM, d)}, {})
        result_name = 'test_result'

        cache = DataCache()

        (result_type, result_dataset) = bandmath.eval_bandmath_expr('(a / b) / (c / d)', expr_info, result_name, cache,
            {'a':(VariableType.IMAGE_CUBE, a),
             'b':(VariableType.IMAGE_BAND, b),
             'c':(VariableType.IMAGE_CUBE, c),
             'd':(VariableType.SPECTRUM, d)}, {})

        assert result_type == RasterDataSet or result_type == VariableType.IMAGE_CUBE

        # Check if result_type is RasterDataSet or IMAGE_CUBE and handle accordingly
        if result_type == RasterDataSet:
            result_arr = result_dataset.get_image_data()
        elif result_type == VariableType.IMAGE_CUBE:
            result_arr = result_dataset
        else:
            self.fail(f"Unexpected result type: {result_type}")

        # Verify the result
        for value in np.nditer(result_arr):
            self.assertEqual(value, 4.0)  # (16 / 2) / (4 / 2)
        
        # Make sure input values didn't change
        for value in np.nditer(a):
            self.assertEqual(value, 16.0)
        
        for value in np.nditer(b):
            self.assertEqual(value, 2.0)
        
        for value in np.nditer(c):
            self.assertEqual(value, 4.0)
        
        for value in np.nditer(d):
            self.assertEqual(value, 2.0)
        
        del result_dataset

    def test_bandmath_sub_complex_expression(self):
        ''' Test band-math with a complex subtraction expression: (a - b) - (c - d) '''
        a = make_image(2, 3, 4)
        b = make_band(3, 4)
        c = make_image(2, 3, 4)
        d = make_spectrum(2)

        a.fill(10.0)
        b.fill(2.0)
        c.fill(5.0)
        d.fill(1.0)

        expr_info = get_bandmath_expr_info('(a - b) - (c - d)',
            {'a':(VariableType.IMAGE_CUBE, a),
             'b':(VariableType.IMAGE_BAND, b),
             'c':(VariableType.IMAGE_CUBE, c),
             'd':(VariableType.SPECTRUM, d)}, {})
        result_name = 'test_result'

        cache = DataCache()

        (result_type, result_dataset) = bandmath.eval_bandmath_expr('(a - b) - (c - d)', expr_info, result_name, cache,
            {'a':(VariableType.IMAGE_CUBE, a),
             'b':(VariableType.IMAGE_BAND, b),
             'c':(VariableType.IMAGE_CUBE, c),
             'd':(VariableType.SPECTRUM, d)}, {})

        assert result_type == RasterDataSet or result_type == VariableType.IMAGE_CUBE

        # Check if result_type is RasterDataSet or IMAGE_CUBE and handle accordingly
        if result_type == RasterDataSet:
            result_arr = result_dataset.get_image_data()
        elif result_type == VariableType.IMAGE_CUBE:
            result_arr = result_dataset
        else:
            self.fail(f"Unexpected result type: {result_type}")

        # Verify the result
        for value in np.nditer(result_arr):
            self.assertEqual(value, 4.0)  # (10 - 2) - (5 - 1)
        
        # Make sure input values didn't change
        for value in np.nditer(a):
            self.assertEqual(value, 10.0)
        
        for value in np.nditer(b):
            self.assertEqual(value, 2.0)
        
        for value in np.nditer(c):
            self.assertEqual(value, 5.0)
        
        for value in np.nditer(d):
            self.assertEqual(value, 1.0)
        
        del result_dataset

    def test_bandmath_neg_a_plus_1(self):
        ''' Test band-math for expression -a + 1 '''
        a = make_image(2, 3, 4)
        a.fill(10.0)

        expr_info = get_bandmath_expr_info('-a + 1',
            {'a':(VariableType.IMAGE_CUBE, a)}, {})
        result_name = 'test_result'

        cache = DataCache()

        (result_type, result_dataset) = bandmath.eval_bandmath_expr('-a + 1', expr_info, result_name, cache,
            {'a':(VariableType.IMAGE_CUBE, a)}, {})

        assert result_type == RasterDataSet or result_type == VariableType.IMAGE_CUBE

        # Check if result_type is RasterDataSet or IMAGE_CUBE and handle accordingly
        if result_type == RasterDataSet:
            result_arr = result_dataset.get_image_data()
        elif result_type == VariableType.IMAGE_CUBE:
            result_arr = result_dataset
        else:
            self.fail(f"Unexpected result type: {result_type}")

        # Verify the result
        for value in np.nditer(result_arr):
            self.assertEqual(value, -9.0)  # -10 + 1
        
        # Make sure input values didn't change
        for value in np.nditer(a):
            self.assertEqual(value, 10.0)
        
        del result_dataset

    def test_bandmath_a_pow_b_minus_sqrt_a(self):
        ''' Test band-math for expression a**b - (a**0.5) '''
        a = make_image(2, 3, 4)
        b = make_band(3, 4)

        a.fill(4.0)
        b.fill(2.0)

        expr_info = get_bandmath_expr_info('a**b - (a**0.5)',
            {'a':(VariableType.IMAGE_CUBE, a),
             'b':(VariableType.IMAGE_BAND, b)}, {})
        result_name = 'test_result'

        cache = DataCache()

        (result_type, result_dataset) = bandmath.eval_bandmath_expr('a**b - (a**0.5)', expr_info, result_name, cache,
            {'a':(VariableType.IMAGE_CUBE, a),
             'b':(VariableType.IMAGE_BAND, b)}, {})

        assert result_type == RasterDataSet or result_type == VariableType.IMAGE_CUBE

        # Check if result_type is RasterDataSet or IMAGE_CUBE and handle accordingly
        if result_type == RasterDataSet:
            result_arr = result_dataset.get_image_data()
        elif result_type == VariableType.IMAGE_CUBE:
            result_arr = result_dataset
        else:
            self.fail(f"Unexpected result type: {result_type}")

        # Verify the result
        for value in np.nditer(result_arr):
            self.assertEqual(value, 14.0)  # 4**2 - (4**0.5) => 16 - 2
        
        for value in np.nditer(a):
            self.assertEqual(value, 4.0)
        
        for value in np.nditer(b):
            self.assertEqual(value, 2.0)
        
        del result_dataset
    
    def test_bandmath_all_operations(self):
        ''' Test band-math with a complex expression: (a / b) - (c * d) + a**0.5 '''
        bands = 2
        samples = 3
        lines = 4
        a = make_image(bands, samples, lines)
        b = make_band(samples, lines)
        c = make_image(bands, samples, lines)
        d = make_spectrum(bands)

        a.fill(16.0)
        b.fill(2.0)
        c.fill(4.0)
        d.fill(2.0)

        expr_info = get_bandmath_expr_info('(a / b) - (c * d) + a**0.5',
            {'a': (VariableType.IMAGE_CUBE, a),
             'b': (VariableType.IMAGE_BAND, b),
             'c': (VariableType.IMAGE_CUBE, c),
             'd': (VariableType.SPECTRUM, d)}, {})
        result_name = 'test_result'

        cache = DataCache()

        (result_type, result_dataset) = bandmath.eval_bandmath_expr('(a / b) - (c * d) + a**0.5', expr_info, result_name, cache,
            {'a': (VariableType.IMAGE_CUBE, a),
             'b': (VariableType.IMAGE_BAND, b),
             'c': (VariableType.IMAGE_CUBE, c),
             'd': (VariableType.SPECTRUM, d)}, {})
    
        assert result_type == RasterDataSet or result_type == VariableType.IMAGE_CUBE

        # Check if result_type is RasterDataSet or IMAGE_CUBE and handle accordingly
        if result_type == RasterDataSet:
            result_arr = result_dataset.get_image_data()
        elif result_type == VariableType.IMAGE_CUBE:
            result_arr = result_dataset
        else:
            self.fail(f"Unexpected result type: {result_type}")
    
        # Verify the result
        for value in np.nditer(result_arr):
            expected_value = (16.0 / 2.0) - (4.0 * 2.0) + (16.0 ** 0.5)
            self.assertEqual(value, expected_value)  # (16 / 2) - (4 * 2) + sqrt(16)

        # Make sure input values didn't change
        for value in np.nditer(a):
            self.assertEqual(value, 16.0)

        for value in np.nditer(b):
            self.assertEqual(value, 2.0)

        for value in np.nditer(c):
            self.assertEqual(value, 4.0)

        for value in np.nditer(d):
            self.assertEqual(value, 2.0)

        del result_dataset

if __name__ == '__main__':
    test_class = TestBandmathEvaluator()
    test_class.test_bandmath_all_operations()

    
