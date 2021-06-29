import enum

from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import lark
import numpy as np

from .types import VariableType, BandMathValue, BandMathEvalError
from .functions import BandMathFunction, get_builtin_functions

from .builtins import (
    OperatorCompare,
    OperatorAdd, OperatorSubtract, OperatorMultiply, OperatorDivide,
    OperatorUnaryNegate, OperatorPower,
    )


class BandMathEvaluator(lark.visitors.Transformer):
    '''
    A Lark Transformer for evaluating band-math expressions.
    '''
    def __init__(self, variables: Dict[str, Tuple[VariableType, Any]],
                       functions: Dict[str, Callable]):
        self._variables = variables
        self._functions = functions

    def comparison(self, args):
        lhs = args[0]
        oper = args[1]
        rhs = args[2]
        return OperatorCompare(oper).apply([lhs, rhs])


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
            return OperatorSubtract().apply([lhs, rhs])

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
            return OperatorMultiply().apply([lhs, rhs])

        elif oper == '/':
            return OperatorDivide().apply([lhs, rhs])

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
        ''' Returns a BandMathValue of True. '''
        return BandMathValue(VariableType.BOOLEAN, True, computed=False)

    def false(self, args):
        ''' Returns a BandMathValue of False. '''
        return BandMathValue(VariableType.BOOLEAN, False, computed=False)

    def number(self, args):
        ''' Returns a BandMathValue containing a specific number. '''
        return args[0]

    def variable(self, args) -> BandMathValue:
        '''
        Returns a BandMathValue containing the value of the specified variable.
        '''
        name = args[0]
        if name not in self._variables or self._variables[name][1] is None:
            raise BandMathEvalError(f'Variable "{name}" is unspecified')

        (type, value) = self._variables[name]
        return BandMathValue(type, value, computed=False)

    def function(self, args) -> BandMathValue:
        '''
        Calls the function named in args[0], passing it args[1:], and returns
        the result as a BandMathValue.
        '''
        func_name = args[0]
        func_args = args[1:]

        if func_name not in self._functions:
            raise BandMathEvalError(f'Unrecognized function "{func_name}"')

        func_impl = self._functions[func_name]
        return func_impl.apply(func_args)

    def NAME(self, token) -> str:
        '''
        Parse a token as a string variable name.  The variable name is converted
        to lowercase.
        '''
        return str(token).lower()

    def NUMBER(self, token) -> BandMathValue:
        '''
        Parse a token as a number.  The number is represented as a Python float,
        and is wrapped in a BandMathValue object.
        '''
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

    *   VariableType.IMAGE_CUBE:  RasterDataSet, 3D np.ndarray [band][y][x]
    *   VariableType.IMAGE_BAND:  RasterDataBand, 2D np.ndarray [y][x]
    *   VariableType.SPECTRUM:  Spectrum, 1D np.ndarray [band]

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
