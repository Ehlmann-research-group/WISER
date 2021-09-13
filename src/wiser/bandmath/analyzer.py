import enum
import logging

from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import lark
import numpy as np

from .types import VariableType, BandMathValue, BandMathEvalError, BandMathExprInfo
from .functions import BandMathFunction, get_builtin_functions

from .builtins import (
    OperatorCompare,
    OperatorAdd, OperatorSubtract, OperatorMultiply, OperatorDivide,
    OperatorUnaryNegate, OperatorPower,
)


logger = logging.getLogger(__name__)


class BandMathAnalyzer(lark.visitors.Transformer):
    '''
    A Lark Transformer for analyzing band-math expressions.  Analysis involves
    identifying any errors in the band-math expression, and if no errors are
    found, reporting the overall result-type of the expression.
    '''
    def __init__(self, variables: Dict[str, Tuple[VariableType, Any]],
                       functions: Dict[str, BandMathFunction]):
        self._variables = variables
        self._functions = functions

        self._errors = []
        self._result_type = None


    def comparison(self, args) -> BandMathExprInfo:
        lhs: BandMathExprInfo = args[0]
        oper = args[1]
        rhs: BandMathExprInfo = args[2]

        # All comparisons produce the same results regardless of the specific
        # operator.
        return OperatorCompare(oper).analyze([lhs, rhs])


    def add_expr(self, values):
        '''
        Implementation of addition and subtraction operations in the
        transformer.
        '''
        lhs = values[0]
        oper = values[1]
        rhs = values[2]

        if oper == '+':
            return OperatorAdd().analyze([lhs, rhs])

        elif oper == '-':
            return OperatorSubtract().analyze([lhs, rhs])

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
            return OperatorMultiply().analyze([lhs, rhs])

        elif oper == '/':
            return OperatorDivide().analyze([lhs, rhs])

        raise RuntimeError(f'Unexpected operator {oper}')


    def power_expr(self, args):
        '''
        Implementation of power operation in the transformer.
        '''
        return OperatorPower().analyze([args[0], args[1]])


    def unary_negate_expr(self, args):
        '''
        Implementation of unary operations in the transformer.
        '''
        # args[0] is the '-' character
        return OperatorUnaryNegate().analyze([args[1]])


    def true(self, args) -> BandMathExprInfo:
        return BandMathExprInfo(VariableType.BOOLEAN)

    def false(self, args) -> BandMathExprInfo:
        return BandMathExprInfo(VariableType.BOOLEAN)

    def number(self, args) -> BandMathExprInfo:
        return BandMathExprInfo(VariableType.NUMBER)

    def string(self, args) -> BandMathExprInfo:
        return BandMathExprInfo(VariableType.STRING)

    def variable(self, args) -> BandMathExprInfo:
        # Look up the variable's type and value.
        name = args[0]
        (type, value) = self._variables[name]

        info = BandMathExprInfo(type)
        if type in [VariableType.IMAGE_CUBE,
                    VariableType.IMAGE_BAND,
                    VariableType.SPECTRUM]:
            # These types also have a shape and an element-type.
            bmv = BandMathValue(type, value)
            info.elem_type = bmv.get_elem_type()
            info.shape = bmv.get_shape()

        logger.debug(f'Variable "{name}":  {info}')

        return info

    def named_expression(self, args) -> BandMathExprInfo:
        # args[0] is the name, args[1] is the expression's info
        return args[1]

    def function(self, args):
        func_name = args[0]
        func_args = args[1:]

        if func_name not in self._functions:
            raise ValueError(f'Unrecognized function "{func_name}"')

        func_impl = self._functions[func_name]
        try:
            return func_impl.analyze(func_args)
        except Exception as e:
            raise RuntimeError(f'Function "{func_name}" analysis error', e)

    def NAME(self, token):
        ''' Parse a token as a string variable name. '''
        return str(token).lower()


def get_bandmath_expr_info(bandmath_expr: str,
        variables: Dict[str, Tuple[VariableType, Any]],
        functions: Dict[str, BandMathFunction] = None) -> BandMathExprInfo:
    '''
    Determine the return-type of a band-math expression using the specified
    variable and function definitions.

    Variables are passed in a dictionary of string names that map to 2-tuples:
    (VariableType, value).  The VariableType enum-value specifies the high-level
    type of the value, since multiple specific types are supported.

    *   VariableType.IMAGE_CUBE:  RasterDataSet, 3D np.ndarray [band][x][y]
    *   VariableType.IMAGE_BAND:  RasterDataBand, 2D np.ndarray [x][y]
    *   VariableType.SPECTRUM:  Spectrum, 1D np.ndarray [band]

    Functions are passed in a dictionary of string names that map to a
    BandMathFunction object.

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
    analyzer = BandMathAnalyzer(lower_variables, lower_functions)
    return analyzer.transform(tree)
