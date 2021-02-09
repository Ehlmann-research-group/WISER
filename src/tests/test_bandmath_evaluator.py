import unittest
import numpy as np

import tests.context

import bandmath
from bandmath import VariableType


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


class TestBandmathEvaluator(unittest.TestCase):
    '''
    Exercise code in the bandmath.evaluator module.
    '''

    def test_bandmath_add_image_number(self):
        ''' Test band-math involving adding an image and a number. '''
        img = make_image(2, 4, 5)
        img.fill(2.5)

        (result_type, result_img) = bandmath.eval_bandmath_expr('image + 0.5',
            {'image':(VariableType.IMAGE_CUBE, img)}, {})

        self.assertEqual(result_type, VariableType.IMAGE_CUBE)

        # Make sure output image has correct results
        for value in np.nditer(result_img):
            self.assertEqual(value, 3.0)

        # Make sure input image didn't change
        for value in np.nditer(img):
            self.assertEqual(value, 2.5)

    def test_bandmath_add_number_image(self):
        ''' Test band-math involving adding a number and an image. '''
        img = make_image(2, 4, 5)
        img.fill(2.5)

        (result_type, result_img) = bandmath.eval_bandmath_expr('0.5 + image',
            {'image':(VariableType.IMAGE_CUBE, img)}, {})

        self.assertEqual(result_type, VariableType.IMAGE_CUBE)

        # Make sure output image has correct results
        for value in np.nditer(result_img):
            self.assertEqual(value, 3.0)

        # Make sure input image didn't change
        for value in np.nditer(img):
            self.assertEqual(value, 2.5)

    def test_bandmath_sub_image_number(self):
        ''' Test band-math involving subtracting a number from an image. '''
        img = make_image(2, 4, 5)
        img.fill(2.5)

        (result_type, result_img) = bandmath.eval_bandmath_expr('image - 0.5',
            {'image':(VariableType.IMAGE_CUBE, img)}, {})

        self.assertEqual(result_type, VariableType.IMAGE_CUBE)

        # Make sure output image has correct results
        for value in np.nditer(result_img):
            self.assertEqual(value, 2.0)

        # Make sure input image didn't change
        for value in np.nditer(img):
            self.assertEqual(value, 2.5)

    def test_bandmath_mul_image_number(self):
        ''' Test band-math involving multiplying an image and a number. '''
        img = make_image(2, 4, 5)
        img.fill(2.5)

        (result_type, result_img) = bandmath.eval_bandmath_expr('image * 2',
            {'image':(VariableType.IMAGE_CUBE, img)}, {})

        self.assertEqual(result_type, VariableType.IMAGE_CUBE)

        # Make sure output image has correct results
        for value in np.nditer(result_img):
            self.assertEqual(value, 5.0)

        # Make sure input image didn't change
        for value in np.nditer(img):
            self.assertEqual(value, 2.5)

    def test_bandmath_mul_number_image(self):
        ''' Test band-math involving multiplying a number and an image. '''
        img = make_image(2, 4, 5)
        img.fill(2.5)

        (result_type, result_img) = bandmath.eval_bandmath_expr('2 * image',
            {'image':(VariableType.IMAGE_CUBE, img)}, {})

        self.assertEqual(result_type, VariableType.IMAGE_CUBE)

        # Make sure output image has correct results
        for value in np.nditer(result_img):
            self.assertEqual(value, 5.0)

        # Make sure input image didn't change
        for value in np.nditer(img):
            self.assertEqual(value, 2.5)

    def test_bandmath_div_image_number(self):
        ''' Test band-math involving dividing an image by a number. '''
        img = make_image(2, 4, 5)
        img.fill(2.5)

        (result_type, result_img) = bandmath.eval_bandmath_expr('image / 0.5',
            {'image':(VariableType.IMAGE_CUBE, img)}, {})

        self.assertEqual(result_type, VariableType.IMAGE_CUBE)

        # Make sure output image has correct results
        for value in np.nditer(result_img):
            self.assertEqual(value, 5.0)

        # Make sure input image didn't change
        for value in np.nditer(img):
            self.assertEqual(value, 2.5)
