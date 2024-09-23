import enum
import logging

from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import lark
import numpy as np

from .types import VariableType, BandMathValue, BandMathEvalError, BandMathExprInfo
from .functions import BandMathFunction, get_builtin_functions

from wiser.raster.dataset_impl import InterleaveType

from .builtins import (
    OperatorCompare,
    OperatorAdd, OperatorSubtract, OperatorMultiply, OperatorDivide,
    OperatorUnaryNegate, OperatorPower,
    )


logger = logging.getLogger(__name__)


class BandMathEvaluator(lark.visitors.Transformer):
    '''
    A Lark Transformer for evaluating band-math expressions.
    '''
    def __init__(self, variables: Dict[str, Tuple[VariableType, Any]],
                       functions: Dict[str, Callable],
                       interleave_type: InterleaveType,
                       shape: Tuple[int, int, int]):
        self._variables = variables
        self._functions = functions
        self.index = 0
        self._interleave = interleave_type
        self._shape = shape


    def comparison(self, args):
        logger.debug(' * comparison')
        lhs = args[0]
        oper = args[1]
        rhs = args[2]
        return OperatorCompare(oper).apply([lhs, rhs])


    def add_expr(self, values):
        '''
        Implementation of addition and subtraction operations in the
        transformer.
        '''
        logger.debug(' * add_expr')
        lhs = values[0]
        oper = values[1]
        rhs = values[2]

        if oper == '+':
            return OperatorAdd().apply([lhs, rhs], self.index, self._interleave)

        elif oper == '-':
            return OperatorSubtract().apply([lhs, rhs])

        raise RuntimeError(f'Unexpected operator {oper}')


    def mul_expr(self, args):
        '''
        Implementation of multiplication and division operations in the
        transformer.
        '''
        logger.debug(' * mul_expr')
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
        logger.debug(' * power_expr')
        return OperatorPower().apply([args[0], args[1]])


    def unary_negate_expr(self, args):
        '''
        Implementation of unary negation in the transformer.
        '''
        logger.debug(' * unary_negate_expr')
        # args[0] is the '-' character
        return OperatorUnaryNegate().apply([args[1]])


    def true(self, args):
        ''' Returns a BandMathValue of True. '''
        logger.debug(' * true')
        return BandMathValue(VariableType.BOOLEAN, True, computed=False)

    def false(self, args):
        ''' Returns a BandMathValue of False. '''
        logger.debug(' * false')
        return BandMathValue(VariableType.BOOLEAN, False, computed=False)

    def number(self, args):
        ''' Returns a BandMathValue containing a specific number. '''
        logger.debug(f' * number {args[0]}')
        return args[0]

    def string(self, args):
        ''' Returns a BandMathValue containing a specific string. '''
        logger.debug(f' * string "{args[0]}"')
        return args[0]

    def variable(self, args) -> BandMathValue:
        '''
        Returns a BandMathValue containing the value of the specified variable.
        '''
        logger.debug(' * variable')
        name = args[0]
        if name not in self._variables or self._variables[name][1] is None:
            raise BandMathEvalError(f'Variable "{name}" is unspecified')

        (type, value) = self._variables[name]
        return BandMathValue(type, value, computed=False)

    def named_expression(self, args) -> BandMathValue:
        '''
        Named expressions can appear in function arguments.
        '''
        logger.debug(' * named_expression')
        # The first argument is the name, and the second argument is a
        # BandMathValue object holding the result of the expression evaluation.
        # Set the name and return the object.
        value = args[1]
        value.set_name(args[0])
        return value

    def function(self, args) -> BandMathValue:
        '''
        Calls the function named in args[0], passing it args[1:], and returns
        the result as a BandMathValue.
        '''
        print("IN THAT WEIRD FUNCTION CLASS")
        logger.debug(' * function')
        func_name = args[0]
        func_args = args[1:]

        has_named_args = False
        for fa in func_args:
            if fa.name is None:
                if has_named_args:
                    raise BandMathEvalError('Named arguments must be '
                        'specified after all positional arguments')
            else:
                has_named_args = True

        if func_name not in self._functions:
            raise BandMathEvalError(f'Unrecognized function "{func_name}"')

        func_impl = self._functions[func_name]
        return func_impl.apply(func_args)

    def NAME(self, token) -> str:
        '''
        Parse a token as a string variable name.  The variable name is converted
        to lowercase.
        '''
        logger.debug(' * NAME')
        return str(token).lower()

    def NUMBER(self, token) -> BandMathValue:
        '''
        Parse a token as a number.  The number is represented as a Python float,
        and is wrapped in a BandMathValue object.
        '''
        logger.debug(' * NUMBER')
        return BandMathValue(VariableType.NUMBER, float(token), computed=False)

    def STRING(self, token) -> str:
        '''
        Parse a token as a string literal.  The variable name is converted
        to lowercase.
        '''
        logger.debug(' * STRING')
        # Chop the quotes off of the string value
        return str(token)[1:-1]



def eval_bandmath_expr(bandmath_expr: str, expr_info: BandMathExprInfo, result_name: str,
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
    print("!!!!!!!!!!!!!!!INFO!!!!!!!!!!!!!")
    print(f"expr_info.interleave_type: {expr_info.interleave_type}")
    print(f"expr_info.shape: {expr_info.shape}")

    lower_variables = {}
    for name, value in variables.items():
        lower_variables[name.lower()] = value

    lower_functions = get_builtin_functions()
    if functions:
        for name, function in functions.items():
            lower_functions[name.lower()] = function

    parser = lark.Lark.open('bandmath.lark', rel_to=__file__, start='expression')
    tree = parser.parse(bandmath_expr)
    logger.info(f'Band-math parse tree:\n{tree.pretty()}')

    print("===============TREE VALUE===============")
    print(tree)
    
    logger.debug('Beginning band-math evaluation')
    expr_interleave = expr_info.interleave_type
    eval = BandMathEvaluator(lower_variables, lower_functions, expr_info.interleave_type, expr_info.shape)
    # Now I need logic to call transform multiple times on the tree and update index each time its called
    # then combine everything into the right form
    index_max = -1
    result_shape = None
    bands, lines, samples = expr_info.shape
    if expr_interleave == InterleaveType.BSQ:
        result_shape = (bands, samples, lines)
    elif expr_interleave == InterleaveType.BIL:
        result_shape = (lines, bands, samples)
    elif expr_interleave == InterleaveType.BIP:
        result_shape = (samples, lines, bands)
    else:
        result_shape = (bands, samples, lines)
    index_max = result_shape[0]
    # result_memmap = np.memmap(f"result_oper_add_{result_name}.bat", mode='w+', dtype=expr_info.elem_type, shape=result_shape)
    result_memmap = np.zeros(result_shape, dtype=expr_info.elem_type)
    result_type = None
    for index in range(index_max):
        eval.index = index
        result_value = eval.transform(tree)
        result_type = result_value.type
        if index % 50 == 0:
            print(f"Evaluator index: {index}")
        result_memmap[index:index+1,:,:] = result_value.value
        del result_value
    
    # print("===============RESULT VALUE===============")
    # print(f"type(result_value): {type(result_value)}")
    # print(f"type(result_value.value): {type(result_value.value)}")
    # print(f"result_value.value.shape: {result_value.value.shape}")

    print("===============RESULT VALUE===============")
    print(f"type(result_memmap): {type(np.array(result_memmap))}")
    print(f"result_memmap.shape: {result_memmap.shape}")
    lhs = eval.variable(['a'])
    filepath = lhs.value._impl.get_filepaths()[0]
    print(f"filepath: {filepath}")
    lhs_value_memmap = np.memmap(filepath, np.float32, 'r', offset=0, shape=result_shape)
    print(f"all close: {np.allclose(result_memmap, lhs_value_memmap)}")
    result_arr = np.array(result_memmap)
    result_arr_1 = np.zeros((bands, lines, samples), dtype=expr_info.elem_type)
    if expr_interleave == InterleaveType.BSQ:
        pass
    elif expr_interleave == InterleaveType.BIL:
        print("DOING FOR LOOPAGE")
        for line_index in range(result_arr.shape[0]):
            arr_to_add = result_arr[line_index:line_index+1,:,:].reshape((425, 1, 680))
            result_arr_1[:,line_index:line_index+1,:] = arr_to_add
        result_arr_1 = result_arr.reshape((bands, lines, samples), order='F')
    elif expr_interleave == InterleaveType.BIP:
        result_arr_1 = result_arr.reshape((bands, lines, samples), order='F')
    else:
        result_arr_1 = result_arr.reshape((bands, lines, samples), order='F')

    print("==================IS VIEW?==================")
    print(f"{result_arr_1.base is result_arr}")
    print(f"Final shape: {result_arr_1.shape}")
    return (result_type, result_arr_1)
