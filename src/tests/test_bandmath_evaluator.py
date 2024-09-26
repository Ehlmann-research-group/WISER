import sys
sys.path.append("C:\\Users\\jgarc\\OneDrive\\Documents\\Schmidt-Code\\WISER\\src\\wiser")
sys.path.append("C:\\Users\\jgarc\\OneDrive\\Documents\\Schmidt-Code\\WISER\\src")

import unittest
import numpy as np

# import tests.context

from wiser import bandmath
from wiser.bandmath import VariableType
from wiser.bandmath.analyzer import get_bandmath_expr_info
from wiser.raster.dataset import RasterDataSet
from wiser.raster.loader import RasterDataLoader


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
        (result_type, result_value) = \
            bandmath.eval_bandmath_expr('(2 * 3 + 4) / 2', expr_info, result_name, {}, {})

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
    
        (result_type, result_dataset) = bandmath.eval_bandmath_expr('image + band', expr_info, result_name,
            {'image':(VariableType.IMAGE_CUBE, img),
             'band':(VariableType.IMAGE_BAND, band)}, {})

        self.assertEqual(result_type, RasterDataSet)

        # Make sure output image has correct results
        self.assertEqual(result_dataset.get_shape(), (2, 3, 4))
        for b in range(result_dataset.get_shape()[0]):
            for y in range(result_dataset.get_shape()[2]):
                for x in range(result_dataset.get_shape()[1]):
                    self.assertEqual(result_dataset.get_image_data()[b, x, y],
                                     (b + 1.0) + 0.5 * x + 10 * y)

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

        (result_type, result_dataset) = bandmath.eval_bandmath_expr('image + 0.5', expr_info, result_name,
            {'image':(VariableType.IMAGE_CUBE, img)}, {})

        self.assertEqual(result_type, RasterDataSet)
        
        # Make sure output image has correct results
        for value in np.nditer(result_dataset.get_image_data()):
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

        (result_type, result_dataset) = bandmath.eval_bandmath_expr('band + image', expr_info, result_name,
            {'image':(VariableType.IMAGE_CUBE, img),
             'band':(VariableType.IMAGE_BAND, band)}, {})

        self.assertEqual(result_type, RasterDataSet)

        # Make sure output image has correct results
        self.assertEqual(result_dataset.get_shape(), (2, 3, 4))
        for b in range(result_dataset.get_shape()[0]):
            for y in range(result_dataset.get_shape()[2]):
                for x in range(result_dataset.get_shape()[1]):
                    self.assertEqual (result_dataset.get_image_data()[b, x, y], (b + 1.0) + 0.5 * x + 10 * y)

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

        (result_type, result_band) = bandmath.eval_bandmath_expr('b + 0.5', expr_info, result_name,
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
        (result_type, result_spectrum) = bandmath.eval_bandmath_expr('S1 + 0.5', expr_info, result_name,
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
    
        (result_type, result_dataset) = bandmath.eval_bandmath_expr('0.5 + image', expr_info, result_name,
            {'image':(VariableType.IMAGE_CUBE, img)}, {})

        self.assertEqual(result_type, RasterDataSet)

        # Make sure output image has correct results
        for value in np.nditer(result_dataset.get_image_data()):
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
    
        (result_type, result_band) = bandmath.eval_bandmath_expr('0.5 + b', expr_info, None,
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

        (result_type, result_spectrum) = bandmath.eval_bandmath_expr('0.5 + S1', expr_info, result_name,
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
    
        (result_type, result_dataset) = bandmath.eval_bandmath_expr('image - band', expr_info, result_name,
            {'image':(VariableType.IMAGE_CUBE, img),
             'band':(VariableType.IMAGE_BAND, band)}, {})

        self.assertEqual(result_type, RasterDataSet)

        # Make sure output image has correct results
        self.assertEqual(result_dataset.get_shape(), (2, 3, 4))
        for b in range(result_dataset.get_shape()[0]):
            for y in range(result_dataset.get_shape()[2]):
                for x in range(result_dataset.get_shape()[1]):
                    self.assertEqual(result_dataset.get_image_data()[b, x, y],
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
    
        (result_type, result_dataset) = bandmath.eval_bandmath_expr('image - 0.5', expr_info, result_name,
            {'image':(VariableType.IMAGE_CUBE, img)}, {})

        self.assertEqual(result_type, RasterDataSet)

        # Make sure output image has correct results
        for value in np.nditer(result_dataset.get_image_data()):
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
    
        (result_type, result_band) = bandmath.eval_bandmath_expr('B1 - 0.5', expr_info, result_name,
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

        (result_type, result_spectrum) = bandmath.eval_bandmath_expr('s - 0.5', expr_info, result_name,
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
    
        (result_type, result_dataset) = bandmath.eval_bandmath_expr('image * 2', expr_info, result_name,
            {'image':(VariableType.IMAGE_CUBE, img)}, {})

        self.assertEqual(result_type, RasterDataSet)

        # Make sure output image has correct results
        for value in np.nditer(result_dataset.get_image_data()):
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

        (result_type, result_dataset) = bandmath.eval_bandmath_expr('2 * image', expr_info, result_name,
            {'image':(VariableType.IMAGE_CUBE, img)}, {})

        self.assertEqual(result_type, RasterDataSet)

        # Make sure output image has correct results
        for value in np.nditer(result_dataset.get_image_data()):
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

        (result_type, result_dataset) = bandmath.eval_bandmath_expr('image / 0.5', expr_info, result_name,
            {'image':(VariableType.IMAGE_CUBE, img)}, {})

        self.assertEqual(result_type, RasterDataSet)

        # Make sure output image has correct results
        for value in np.nditer(result_dataset.get_image_data()):
            self.assertEqual(value, 5.0)

        # Make sure input image didn't change
        for value in np.nditer(img):
            self.assertEqual(value, 2.5)
            
        del result_dataset
