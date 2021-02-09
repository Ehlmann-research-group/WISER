import enum

from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import lark

from .common import VariableType


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


class BandMathValue:
    '''
    This is a value created or consumed by a band-math expression during
    evaluation.  The high-level type of the variable is stored, along with the
    actual value.  The value may be one of several possible types, since most
    band-math operations work directly on NumPy arrays rather than other WISER
    types.

    Whether the band-math value is a computed result or not is also recorded in
    this type, so that math operations can reuse an argument's memory where that
    would be more efficient.
    '''
    def __init__(self, type: VariableType, value: Any, computed: bool = True):
        if type not in VariableType:
            raise ValueError(f'Unrecognized variable-type {type}')

        self.type = type
        self.value = value
        self.computed = computed

    def as_numpy_array(self):
        # If the value is already a NumPy array, we are done!
        if isinstance(self.value, np.ndarray):
            return self.value

        if self.type == VariableType.IMAGE_CUBE:
            if isinstance(self.value, RasterDataSet):
                return self.value.get_image_data()

        elif self.type == VariableType.IMAGE_BAND:
            if isinstance(self.value, RasterDataBand):
                return self.value.get_data()

        elif self.type == VariableType.SPECTRUM:
            if isinstance(self.value, SpectrumInfo):
                return self.value.get_spectrum()

        # If we got here, we don't know how to convert the value into a NumPy
        # array.
        raise TypeError(f'Don\'t know how to convert {self.type} ' +
                        f'value {self.value} into a NumPy array')


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


    def add_oper(self, values):
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


    def mul_oper(self, args):
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
        print(f'Lookup on variable({args[0]})')
        return self._variables[args[0]]

    def function(self, args):
        print(f'TODO:  function({args})')


    def NAME(self, token):
        ''' Parse a token as a string variable name. '''
        return str(token)

    def NUMBER(self, token):
        ''' Parse a token as a number. '''
        return float(token)


def eval_bandmath_expr(bandmath_expr: str,
        variables: Dict[str, Tuple[VariableType, Any]],
        functions: Dict[str, Callable]):

    # Just to be defensive against potentially bad inputs, make sure all names
    # of variables and functions are lowercase.
    # TODO(donnie):  Can also make sure they are valid, trimmed of whitespace,
    #     etc.

    lower_variables = {}
    for name, value in variables.items():
        lower_variables[name.lower()] = value

    lower_functions = {}
    for name, function in functions.items():
        lower_functions[name.lower()] = function

    parser = lark.Lark.open('bandmath.lark', rel_to=__file__, start='expression')
    tree = parser.parse(bandmath_expr)
    eval = BandMathEvaluator(lower_variables, lower_functions)
    result_value = eval.transform(tree)

    return (result_value.type, result_value.value)
