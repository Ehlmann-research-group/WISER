import enum

from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import lark
import numpy as np

from .common import VariableType, BandMathValue, BandMathEvalError
from .functions import get_builtin_functions


class BandMathOperation(enum.Enum):
    ADD = 1

    SUBTRACT = 2

    MULTIPLY = 3

    DIVIDE = 4

    POWER = 5

    UNARY_NEGATE = 6


class BandMathTypeError(TypeError):
    '''
    This subtype of the TypeError exception is raised when a band-math operation
    won't work for the given types.
    '''
    def __init__(self, msg: str,
                 operation: Optional[BandMathOperation] = None,
                 lhs_type: Optional[VariableType] = None,
                 rhs_type: Optional[VariableType] = None):
        super().__init__(msg)
        self.lhs_type = lhs_type
        self.rhs_type = rhs_type


def add_values(lhs, rhs):
    '''
    Add the LHS and RHS and return the result.
    '''

    # Take care of the simple case first, where it's just two numbers.
    if lhs.type == VariableType.NUMBER and rhs.type == VariableType.NUMBER:
        return BandMathValue(VariableType.NUMBER, lhs.value + rhs.value)

    # Since addition is commutative, arrange the arguments to make the
    # calculation logic easier.
    if lhs.type == VariableType.IMAGE_CUBE or rhs.type == VariableType.IMAGE_CUBE:
        # If there is only one image cube, make sure it is on the LHS.
        if lhs.type != VariableType.IMAGE_CUBE:
            (rhs, lhs) = (lhs, rhs)
    elif lhs.type == VariableType.IMAGE_BAND or rhs.type == VariableType.IMAGE_BAND:
        # No image cubes.
        # If there is only one image band, make sure it is on the LHS.
        if lhs.type != VariableType.IMAGE_BAND:
            (rhs, lhs) = (lhs, rhs)

    elif lhs.type == VariableType.SPECTRUM or rhs.type == VariableType.SPECTRUM:
        # No image bands.
        # If there is only one spectrum, make sure it is on the LHS.
        if lhs.type != VariableType.SPECTRUM:
            (rhs, lhs) = (lhs, rhs)

    if lhs.type == VariableType.IMAGE_CUBE:
        # Dimensions:  [band][x][y]
        lhs_arr = lhs.as_numpy_array()
        assert lhs_arr.ndim == 3

        if rhs.type == VariableType.IMAGE_CUBE:
            # Dimensions:  [band][x][y]
            rhs_arr = rhs.as_numpy_array()
            result_arr = lhs_arr + rhs_arr

        elif rhs.type == VariableType.IMAGE_BAND:
            # Dimensions:  [x][y]
            rhs_arr = rhs.as_numpy_array()
            assert rhs_arr.ndim == 2

            # NumPy will broadcast the band across the entire image.
            result_arr = lhs_arr + rhs_arr

        elif rhs.type == VariableType.SPECTRUM:
            # Dimensions:  [band]
            rhs_arr = rhs.as_numpy_array()
            assert rhs_arr.ndim == 1

            # To ensure the spectrum is added to the image's pixels, reshape to
            # effectively create a 1x1 image.
            # New dimensions:  [band][x=1][y=1]
            rhs_arr = rhs_arr[:, np.newaxis, np.newaxis]

            result_arr = lhs_arr + rhs_arr

        elif rhs.type == VariableType.NUMBER:
            result_arr = lhs_arr + rhs.value

        else:
            raise BandMathTypeError(
                f'Don\'t know how to add {lhs.type} and {rhs.type}.',
                lhs.type, rhs.type)

        # The result array should have the same dimensions as the LHS input
        # array.
        assert result_arr.ndim == 3
        assert result_arr.shape == lhs_arr.shape
        return BandMathValue(VariableType.IMAGE_CUBE, result_arr)

    elif lhs.type == VariableType.IMAGE_BAND:
        # Dimensions:  [x][y]
        lhs_arr = lhs.as_numpy_array()
        assert lhs_arr.ndim == 2

        if rhs.type == VariableType.IMAGE_BAND:
            # Dimensions:  [x][y]
            rhs_arr = rhs.as_numpy_array()
            assert rhs_arr.ndim == 2

            result_arr = lhs_arr + rhs_arr

        elif rhs.type == VariableType.NUMBER:
            result_arr = lhs_arr + rhs.value

        else:
            raise BandMathTypeError(
                f'Don\'t know how to add {lhs.type} and {rhs.type}.',
                lhs.type, rhs.type)

        # The result array should have the same dimensions as the LHS input
        # array.
        assert result_arr.ndim == 2
        assert result_arr.shape == lhs_arr.shape
        return BandMathValue(VariableType.IMAGE_BAND, result_arr)

    elif lhs.type == VariableType.SPECTRUM:
        # Dimensions:  [band]
        lhs_arr = lhs.as_numpy_array()
        assert lhs_arr.ndim == 1

        if rhs.type == VariableType.SPECTRUM:
            # Dimensions:  [band]
            rhs_arr = rhs.as_numpy_array()
            assert rhs_arr.ndim == 1

            result_arr = lhs_arr + rhs_arr

        elif rhs.type == VariableType.NUMBER:
            result_arr = lhs_arr + rhs.value

        else:
            raise BandMathTypeError(
                f'Don\'t know how to add {lhs.type} and {rhs.type}.',
                lhs.type, rhs.type)

        # The result array should have the same dimensions as the LHS input
        # array.
        assert result_arr.ndim == 1
        assert result_arr.shape == lhs_arr.shape
        return BandMathValue(VariableType.SPECTRUM, result_arr)

    # If we get here, we don't know how to add the two types.
    raise BandMathTypeError(f'Don\'t know how to add {lhs.type} and {rhs.type}.',
        lhs.type, rhs.type)


def subtract_values(lhs, rhs):
    '''
    Subtract the RHS from the LHS and return the result.
    '''
    # Take care of the simple case first, where it's just two numbers.
    if lhs.type == VariableType.NUMBER and rhs.type == VariableType.NUMBER:
        return BandMathValue(VariableType.NUMBER, lhs.value - rhs.value)

    if lhs.type == VariableType.IMAGE_CUBE:
        # Dimensions:  [band][x][y]
        lhs_arr = lhs.as_numpy_array()
        assert lhs_arr.ndim == 3

        if rhs.type == VariableType.IMAGE_CUBE:
            # Dimensions:  [band][x][y]
            rhs_arr = rhs.as_numpy_array()
            result_arr = lhs_arr - rhs_arr

        elif rhs.type == VariableType.IMAGE_BAND:
            # Dimensions:  [x][y]
            rhs_arr = rhs.as_numpy_array()
            assert rhs_arr.ndim == 2

            # NumPy will broadcast the band across the entire image.
            result_arr = lhs_arr - rhs_arr

        elif rhs.type == VariableType.SPECTRUM:
            # Dimensions:  [band]
            rhs_arr = rhs.as_numpy_array()
            assert rhs_arr.ndim == 1

            # To ensure the spectrum is added to the image's pixels, reshape to
            # effectively create a 1x1 image.
            # New dimensions:  [band][x=1][y=1]
            rhs_arr = rhs_arr[:, np.newaxis, np.newaxis]

            result_arr = lhs_arr - rhs_arr

        elif rhs.type == VariableType.NUMBER:
            result_arr = lhs_arr - rhs.value

        else:
            raise TypeError(f'Don\'t know how to subtract {lhs.type} and {rhs.type}.')

        # The result array should have the same dimensions as the LHS input
        # array.
        assert result_arr.ndim == 3
        assert result_arr.shape == lhs_arr.shape
        return BandMathValue(VariableType.IMAGE_CUBE, result_arr)

    elif lhs.type == VariableType.IMAGE_BAND:
        # Dimensions:  [x][y]
        lhs_arr = lhs.as_numpy_array()
        assert lhs_arr.ndim == 2

        if rhs.type == VariableType.IMAGE_BAND:
            # Dimensions:  [x][y]
            rhs_arr = rhs.as_numpy_array()
            assert rhs_arr.ndim == 2

            result_arr = lhs_arr - rhs_arr

        elif rhs.type == VariableType.NUMBER:
            result_arr = lhs_arr - rhs.value

        else:
            raise TypeError(f'Don\'t know how to subtract {lhs.type} and {rhs.type}.')

        # The result array should have the same dimensions as the LHS input
        # array.
        assert result_arr.ndim == 2
        assert result_arr.shape == lhs_arr.shape
        return BandMathValue(VariableType.IMAGE_BAND, result_arr)

    elif lhs.type == VariableType.SPECTRUM:
        # Dimensions:  [band]
        lhs_arr = lhs.as_numpy_array()
        assert lhs_arr.ndim == 1

        if rhs.type == VariableType.SPECTRUM:
            # Dimensions:  [band]
            rhs_arr = rhs.as_numpy_array()
            assert rhs_arr.ndim == 1

            result_arr = lhs_arr - rhs_arr

        elif rhs.type == VariableType.NUMBER:
            result_arr = lhs_arr - rhs.value

        else:
            raise TypeError(f'Don\'t know how to subtract {lhs.type} and {rhs.type}.')

        # The result array should have the same dimensions as the LHS input
        # array.
        assert result_arr.ndim == 1
        assert result_arr.shape == lhs_arr.shape
        return BandMathValue(VariableType.SPECTRUM, result_arr)

    raise TypeError(f'Don\'t know how to subtract {lhs.type} and {rhs.type}.')


def multiply_values(lhs, rhs):
    '''
    Multiply the LHS and RHS and return the result.
    '''

    # Take care of the simple case first, where it's just two numbers.
    if lhs.type == VariableType.NUMBER and rhs.type == VariableType.NUMBER:
        return BandMathValue(VariableType.NUMBER, lhs.value * rhs.value)

    # Since multiplication is commutative, arrange the arguments to make the
    # calculation logic easier.
    if lhs.type == VariableType.IMAGE_CUBE or rhs.type == VariableType.IMAGE_CUBE:
        # If there is only one image cube, make sure it is on the LHS.
        if lhs.type != VariableType.IMAGE_CUBE:
            (rhs, lhs) = (lhs, rhs)
    elif lhs.type == VariableType.IMAGE_BAND or rhs.type == VariableType.IMAGE_BAND:
        # No image cubes.
        # If there is only one image band, make sure it is on the LHS.
        if lhs.type != VariableType.IMAGE_BAND:
            (rhs, lhs) = (lhs, rhs)

    elif lhs.type == VariableType.SPECTRUM or rhs.type == VariableType.SPECTRUM:
        # No image bands.
        # If there is only one spectrum, make sure it is on the LHS.
        if lhs.type != VariableType.SPECTRUM:
            (rhs, lhs) = (lhs, rhs)

    if lhs.type == VariableType.IMAGE_CUBE:
        # Dimensions:  [band][x][y]
        lhs_arr = lhs.as_numpy_array()
        assert lhs_arr.ndim == 3

        # Image * Spectrum is excluded because it has multiple useful
        # definitions, and we want to support them all.

        if rhs.type == VariableType.IMAGE_CUBE:
            # Dimensions:  [band][x][y]
            rhs_arr = rhs.as_numpy_array()
            result_arr = lhs_arr * rhs_arr

        elif rhs.type == VariableType.IMAGE_BAND:
            # Dimensions:  [x][y]
            rhs_arr = rhs.as_numpy_array()
            assert rhs_arr.ndim == 2

            # NumPy will broadcast the band across the entire image.
            result_arr = lhs_arr * rhs_arr

        elif rhs.type == VariableType.NUMBER:
            result_arr = lhs_arr * rhs.value

        else:
            raise BandMathTypeError(BandMathOperation.MULTIPLY, lhs.type, rhs.type,
                f'Don\'t know how to multiply {lhs.type} and {rhs.type}.')

        # The result array should have the same dimensions as the LHS input
        # array.
        assert result_arr.ndim == 3
        assert result_arr.shape == lhs_arr.shape
        return BandMathValue(VariableType.IMAGE_CUBE, result_arr)

    elif lhs.type == VariableType.IMAGE_BAND:
        # Dimensions:  [x][y]
        lhs_arr = lhs.as_numpy_array()
        assert lhs_arr.ndim == 2

        if rhs.type == VariableType.IMAGE_BAND:
            # Dimensions:  [x][y]
            rhs_arr = rhs.as_numpy_array()
            assert rhs_arr.ndim == 2

            result_arr = lhs_arr * rhs_arr

        elif rhs.type == VariableType.NUMBER:
            result_arr = lhs_arr * rhs.value

        else:
            raise BandMathTypeError(BandMathOperation.MULTIPLY, lhs.type, rhs.type,
                f'Don\'t know how to multiply {lhs.type} and {rhs.type}.')

        # The result array should have the same dimensions as the LHS input
        # array.
        assert result_arr.ndim == 2
        assert result_arr.shape == lhs_arr.shape
        return BandMathValue(VariableType.IMAGE_BAND, result_arr)

    elif lhs.type == VariableType.SPECTRUM:
        # Dimensions:  [band]
        lhs_arr = lhs.as_numpy_array()
        assert lhs_arr.ndim == 1

        if rhs.type == VariableType.SPECTRUM:
            # Dimensions:  [band]
            rhs_arr = rhs.as_numpy_array()
            assert rhs_arr.ndim == 1

            result_arr = lhs_arr * rhs_arr

        elif rhs.type == VariableType.NUMBER:
            result_arr = lhs_arr * rhs.value

        else:
            raise BandMathTypeError(BandMathOperation.MULTIPLY, lhs.type, rhs.type,
                f'Don\'t know how to multiply {lhs.type} and {rhs.type}.')

        # The result array should have the same dimensions as the LHS input
        # array.
        assert result_arr.ndim == 1
        assert result_arr.shape == lhs_arr.shape
        return BandMathValue(VariableType.SPECTRUM, result_arr)

    # If we get here, we don't know how to multiply the two types.
    raise BandMathTypeError(BandMathOperation.MULTIPLY, lhs.type, rhs.type,
        f'Don\'t know how to multiply {lhs.type} and {rhs.type}.')


def divide_values(lhs, rhs):
    '''
    Divide the LHS by the RHS and return the result.
    '''

    # Take care of the simple case first, where it's just two numbers.
    if lhs.type == VariableType.NUMBER and rhs.type == VariableType.NUMBER:
        return BandMathValue(VariableType.NUMBER, lhs.value / rhs.value)

    if lhs.type == VariableType.IMAGE_CUBE:
        # Dimensions:  [band][x][y]
        lhs_arr = lhs.as_numpy_array()
        assert lhs_arr.ndim == 3

        if rhs.type == VariableType.IMAGE_CUBE:
            # Dimensions:  [band][x][y]
            rhs_arr = rhs.as_numpy_array()
            result_arr = lhs_arr / rhs_arr

        elif rhs.type == VariableType.IMAGE_BAND:
            # Dimensions:  [x][y]
            rhs_arr = rhs.as_numpy_array()
            assert rhs_arr.ndim == 2

            # NumPy will broadcast the band across the entire image.
            result_arr = lhs_arr / rhs_arr

        elif rhs.type == VariableType.SPECTRUM:
            # Dimensions:  [band]
            rhs_arr = rhs.as_numpy_array()
            assert rhs_arr.ndim == 1

            # To ensure the spectrum is added to the image's pixels, reshape to
            # effectively create a 1x1 image.
            # New dimensions:  [band][x=1][y=1]
            rhs_arr = rhs_arr[:, np.newaxis, np.newaxis]

            result_arr = lhs_arr / rhs_arr

        elif rhs.type == VariableType.NUMBER:
            result_arr = lhs_arr / rhs.value

        else:
            raise BandMathTypeError(BandMathOperation.DIVIDE, lhs.type, rhs.type,
                f'Don\'t know how to divide {lhs.type} and {rhs.type}.')

        # The result array should have the same dimensions as the LHS input
        # array.
        assert result_arr.ndim == 3
        assert result_arr.shape == lhs_arr.shape
        return BandMathValue(VariableType.IMAGE_CUBE, result_arr)

    elif lhs.type == VariableType.IMAGE_BAND:
        # Dimensions:  [x][y]
        lhs_arr = lhs.as_numpy_array()
        assert lhs_arr.ndim == 2

        if rhs.type == VariableType.IMAGE_BAND:
            # Dimensions:  [x][y]
            rhs_arr = rhs.as_numpy_array()
            assert rhs_arr.ndim == 2

            result_arr = lhs_arr / rhs_arr

        elif rhs.type == VariableType.NUMBER:
            result_arr = lhs_arr / rhs.value

        else:
            raise BandMathTypeError(BandMathOperation.DIVIDE, lhs.type, rhs.type,
                f'Don\'t know how to divide {lhs.type} and {rhs.type}.')

        # The result array should have the same dimensions as the LHS input
        # array.
        assert result_arr.ndim == 2
        assert result_arr.shape == lhs_arr.shape
        return BandMathValue(VariableType.IMAGE_BAND, result_arr)

    elif lhs.type == VariableType.SPECTRUM:
        # Dimensions:  [band]
        lhs_arr = lhs.as_numpy_array()
        assert lhs_arr.ndim == 1

        if rhs.type == VariableType.SPECTRUM:
            # Dimensions:  [band]
            rhs_arr = rhs.as_numpy_array()
            assert rhs_arr.ndim == 1

            result_arr = lhs_arr / rhs_arr

        elif rhs.type == VariableType.NUMBER:
            result_arr = lhs_arr / rhs.value

        else:
            raise BandMathTypeError(BandMathOperation.DIVIDE, lhs.type, rhs.type,
                f'Don\'t know how to divide {lhs.type} and {rhs.type}.')

        # The result array should have the same dimensions as the LHS input
        # array.
        assert result_arr.ndim == 1
        assert result_arr.shape == lhs_arr.shape
        return BandMathValue(VariableType.SPECTRUM, result_arr)

    # If we get here, we don't know how to multiply the two types.
    raise BandMathTypeError(BandMathOperation.DIVIDE, lhs.type, rhs.type,
        f'Don\'t know how to divide {lhs.type} and {rhs.type}.')


def negate_value(arg):
    '''
    Perform unary negation on the argument and return the result.
    '''
    if arg.type == VariableType.NUMBER:
        return BandMathValue(VariableType.NUMBER, -arg.value)

    elif arg.type in [VariableType.IMAGE_CUBE, VariableType.IMAGE_BAND,
                      VariableType.SPECTRUM]:
        arr = arg.as_numpy_array()
        result_arr = -arr
        return BandMathValue(arg.type, result_arr)

    raise BandMathTypeError(BandMathOperation.UNARY_NEGATE, None, rhs.type,
        f'Don\'t know how to unary-negate {rhs.type}.')


class BandMathEvaluator(lark.visitors.Transformer):
    '''
    A Lark Transformer for evaluating band-math expressions.
    '''
    def __init__(self, variables: Dict[str, Tuple[VariableType, Any]],
                       functions: Dict[str, Callable]):
        self._variables = variables
        self._functions = functions

    def comparison(self, args):
        raise NotImplementedError(f'TODO:  comparison({args})')


    def add_expr(self, values):
        '''
        Implementation of addition and subtraction operations in the
        transformer.
        '''
        lhs = values[0]
        oper = values[1]
        rhs = values[2]

        if oper == '+':
            return add_values(lhs, rhs)

        elif oper == '-':
            return subtract_values(lhs, rhs)

        raise RuntimeError(f'Unexpected operator {oper}')


    def mul_expr(self, args):
        '''
        Implementation of multiplication and division operations in the
        transformer.
        '''
        lhs = args[0]
        oper = args[1]
        rhs = args[2]

        if oper == '*':
            return multiply_values(lhs, rhs)

        elif oper == '/':
            return divide_values(lhs, rhs)

        raise RuntimeError(f'Unexpected operator {oper}')


    def unary_op(self, args):
        '''
        Implementation of unary operations in the transformer.
        '''
        if args[0] == '-':
            return negate_value(args[1])

        # Sanity check - shouldn't be possible
        if args[0] != '+':
            raise RuntimeError(f'Unexpected operator {args[0]}')


    def true(self, args):
        return True

    def false(self, args):
        return False

    def number(self, args):
        return args[0]

    def variable(self, args):
        name = args[0]
        (type, value) = self._variables[name]
        return BandMathValue(type, value, computed=False)

    def function(self, args):
        func_name = args[0]
        func_args = args[1:]

        if func_name not in self._functions:
            raise BandMathEvalError(f'Unrecognized function "{func_name}"')

        func_impl = self._functions[func_name]
        return func_impl(func_args)


    def NAME(self, token):
        ''' Parse a token as a string variable name. '''
        return str(token).lower()

    def NUMBER(self, token):
        ''' Parse a token as a number. '''
        return BandMathValue(VariableType.NUMBER, float(token), computed=False)


def eval_bandmath_expr(bandmath_expr: str,
        variables: Dict[str, Tuple[VariableType, Any]],
        functions: Dict[str, Callable] = None):
    '''
    Evaluate a band-math expression using the specified variable and function
    definitions.

    Variables are passed in a dictionary of string names that map to 2-tuples:
    (VariableType, value).  The VariableType enum-value specifies the high-level
    type of the value, since multiple specific types are supported.

    *   VariableType.IMAGE_CUBE:  RasterDataSet, 3D np.ndarray [band][x][y]
    *   VariableType.IMAGE_BAND:  RasterDataBand, 2D np.ndarray [x][y]
    *   VariableType.SPECTRUM:  SpectrumInfo, 1D np.ndarray [band]

    Functions are passed in a dictionary of string names that map to a callable.
    TODO:  MORE DETAIL HERE, ONCE WE IMPLEMENT THIS.

    If successful, the result of the calculation is returned as a 2-tuple of the
    same form as the variables, although the value is always either a number or
    a NumPy array:

    *   VariableType.IMAGE_CUBE:  3D np.ndarray [band][x][y]
    *   VariableType.IMAGE_BAND:  2D np.ndarray [x][y]
    *   VariableType.SPECTRUM:  1D np.ndarray [band]
    *   VariableType.NUMBER:  float
    '''

    # Just to be defensive against potentially bad inputs, make sure all names
    # of variables and functions are lowercase.
    # TODO(donnie):  Can also make sure they are valid, trimmed of whitespace,
    #     etc.

    lower_variables = {}
    for name, value in variables.items():
        lower_variables[name.lower()] = value

    lower_functions = get_builtin_functions()
    if functions:
        for name, function in functions.items():
            lower_functions[name.lower()] = function

    parser = lark.Lark.open('bandmath.lark', rel_to=__file__, start='expression')
    tree = parser.parse(bandmath_expr)
    eval = BandMathEvaluator(lower_variables, lower_functions)
    result_value = eval.transform(tree)

    return (result_value.type, result_value.value)
