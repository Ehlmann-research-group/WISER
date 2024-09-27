import os
import logging

from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import lark
import numpy as np

from .types import VariableType, BandMathValue, BandMathEvalError, BandMathExprInfo
from .functions import BandMathFunction, get_builtin_functions
from .utils import TEMP_FOLDER_PATH

from wiser.raster.dataset import RasterDataSet

from osgeo import gdal

from .builtins import (
    OperatorCompare,
    OperatorAdd, OperatorSubtract, OperatorMultiply, OperatorDivide,
    OperatorUnaryNegate, OperatorPower,
    )

from wiser.raster.loader import RasterDataLoader
from wiser.raster.dataset_impl import SaveState

from .builtins.constants import MAX_RAM_BYTES, SCALAR_BYTES


logger = logging.getLogger(__name__)


class BandMathEvaluator(lark.visitors.Transformer):
    '''
    A Lark Transformer for evaluating band-math expressions.
    '''
    def __init__(self, variables: Dict[str, Tuple[VariableType, Any]],
                       functions: Dict[str, Callable],
                       shape: Tuple[int, int, int] = None):
        self._variables = variables
        self._functions = functions
        self.index_list = None
        self._shape = shape


    def comparison(self, args):
        logger.debug(' * comparison')
        lhs = args[0]
        oper = args[1]
        rhs = args[2]
        # It is okay if we don't want to use index with bands or spectrum 
        # because in each operator, index is only used with image cubes
        return OperatorCompare(oper).apply([lhs, rhs], self.index_list)


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
            return OperatorAdd().apply([lhs, rhs], self.index_list)

        elif oper == '-':
            return OperatorSubtract().apply([lhs, rhs], self.index_list)

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
            return OperatorMultiply().apply([lhs, rhs], self.index_list)

        elif oper == '/':
            return OperatorDivide().apply([lhs, rhs], self.index_list)

        raise RuntimeError(f'Unexpected operator {oper}')


    def power_expr(self, args):
        '''
        Implementation of power operation in the transformer.
        '''
        logger.debug(' * power_expr')
        return OperatorPower().apply([args[0], args[1]], self.index_list)


    def unary_negate_expr(self, args):
        '''
        Implementation of unary negation in the transformer.
        '''
        logger.debug(' * unary_negate_expr')
        # args[0] is the '-' character
        return OperatorUnaryNegate().apply([args[1]], self.index_list)


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

import re
def remove_trailing_number(filepath):
    # Regular expression pattern to match " space followed by digits" at the end of the path
    pattern = r"(.*)\s\d+$"
    
    # Use re.match to see if the pattern matches the filepath
    match = re.match(pattern, filepath)
    
    # If a match is found, return the group without the trailing space and number
    if match:
        return match.group(1)
    
    # Otherwise, return the original filepath
    return filepath

def eval_bandmath_expr(bandmath_expr: str, expr_info: BandMathExprInfo, result_name: str,
        variables: Dict[str, Tuple[VariableType, Any]],
        functions: Dict[str, BandMathFunction] = None,
        use_old_method = False) -> BandMathValue:
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
    logger.info(f'Band-math parse tree:\n{tree.pretty()}')
    '''
    Okay so I think to make this parallel, in the tree when a data piece is retrieve we actually retrieve that data piece from the queue. But where is the queue? The queue could be stored in the evaluator and passed
    in to any of the Operators. Also we could have a processor class in BandMathEvaluator that spawns a ProcessPoolExecutor where we can run the operations on. Everytime a node of the tree is processed, we will add that Operator
    to the process, maybe wrapped in a function. Then writing to disk we can simply do as is done in the chatpgt code.
    '''
    logger.debug('Beginning band-math evaluation')
    if expr_info.result_type == VariableType.IMAGE_CUBE and not use_old_method:
        eval = BandMathEvaluator(lower_variables, lower_functions, expr_info.shape)

        bands, lines, samples = expr_info.shape
        result_path = os.path.join(TEMP_FOLDER_PATH, result_name)
        count = 2
        while (os.path.exists(result_path)):
            result_path = remove_trailing_number(result_path)
            result_path+=f" {count}"
            count+=1
        folder_path = os.path.dirname(result_path)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        out_dataset_gdal = gdal.GetDriverByName('ENVI').Create(result_path, samples, lines, bands, gdal.GDT_CFloat32)

        bytes_per_scalar = SCALAR_BYTES
        # if expr_info.elem_type is not None:
        #     print("itemsize setting")
        #     bytes_per_scalar = expr_info.elem_type.itemsize
        # else:
        #     bytes_per_scalar = SCALAR_BYTES
        max_bytes = MAX_RAM_BYTES/bytes_per_scalar
        num_bands = int(np.floor(max_bytes / (lines*samples)))
        for band_index in range(0, bands, num_bands):
            print(f"bands: {bands}")
            band_index_list = [band for band in range(band_index, band_index+num_bands) if band < bands]
            print(f"band_index_list : {band_index_list}")
            eval.index_list = band_index_list
            result_value = eval.transform(tree)
            res = result_value.value
            if isinstance(result_value.value, np.ma.MaskedArray):
                if not np.issubdtype(res.dtype, np.floating):
                    res = res.astype(np.float32)
                res[result_value.value.mask] = np.nan

            for gdal_band_index in band_index_list:
                print(f"gdal_band_index-band_index: {gdal_band_index-band_index}")
                print(f"res.shape: {res.shape}")
                band_to_write = None
                if len(band_index_list) == 1:
                    band_to_write = res
                else:
                    band_to_write = res[gdal_band_index-band_index]
                band = out_dataset_gdal.GetRasterBand(gdal_band_index+1)
                band.WriteArray(band_to_write)
                band.FlushCache()

        out_dataset = RasterDataLoader().dataset_from_gdal_dataset(out_dataset_gdal)
        out_dataset.set_save_state(SaveState.IN_DISK_NOT_SAVED)
        out_dataset.set_dirty()

        return (RasterDataSet, out_dataset)
    else:
        print("OLD METHOD")
        eval = BandMathEvaluator(lower_variables, lower_functions)
        result_value = eval.transform(tree)
        return (result_value.type, result_value.value)
