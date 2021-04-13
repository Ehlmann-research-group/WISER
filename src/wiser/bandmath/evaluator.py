import enum

from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import lark
import numpy as np

from .common import VariableType, BandMathValue, BandMathEvalError
from .functions import BandMathFunction, get_builtin_functions

from .builtins import OperatorAdd, OperatorSub, OperatorMul, OperatorDiv
from .builtins import OperatorUnaryNegate, OperatorPower


class BandMathOperation(enum.Enum):
    ADD = 1

    SUBTRACT = 2

    MULTIPLY = 3

    DIVIDE = 4

    POWER = 5

    UNARY_NEGATE = 6


class BandMathEvaluator(lark.visitors.Transformer):
    '''
    A Lark Transformer for evaluating band-math expressions.
    '''
    def __init__(self, variables: Dict[str, Tuple[VariableType, Any]],
                       functions: Dict[str, Callable]):
        self._variables = variables
        self._functions = functions

    def and_expr(self, values):
        raise NotImplementedError(f'TODO:  and_expr({values})')

    def or_expr(self, values):
        raise NotImplementedError(f'TODO:  or_expr({values})')

    def not_expr(self, values):
        raise NotImplementedError(f'TODO:  not_expr({values})')

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
            return OperatorAdd().apply([lhs, rhs])

        elif oper == '-':
            return OperatorSub().apply([lhs, rhs])

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
            return OperatorMul().apply([lhs, rhs])

        elif oper == '/':
            return OperatorDiv().apply([lhs, rhs])

        raise RuntimeError(f'Unexpected operator {oper}')


    def power_expr(self, args):
        '''
        Implementation of power operation in the transformer.
        '''
        return OperatorPower().apply([args[0], args[2]])


    def unary_op_expr(self, args):
        '''
        Implementation of unary operations in the transformer.
        '''
        if args[0] == '-':
            return OperatorUnaryNegate().apply([args[1]])

        # Sanity check - shouldn't be possible
        if args[0] != '+':
            raise RuntimeError(f'Unexpected operator {args[0]}')


    def true(self, args):
        return BandMathValue(VariableType.BOOLEAN, True, computed=False)

    def false(self, args):
        return BandMathValue(VariableType.BOOLEAN, False, computed=False)

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
        return func_impl.apply(func_args)

    def NAME(self, token):
        ''' Parse a token as a string variable name. '''
        return str(token).lower()

    def NUMBER(self, token):
        ''' Parse a token as a number. '''
        return BandMathValue(VariableType.NUMBER, float(token), computed=False)


def eval_bandmath_expr(bandmath_expr: str,
        variables: Dict[str, Tuple[VariableType, Any]],
        functions: Dict[str, BandMathFunction] = None) -> BandMathValue:
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
