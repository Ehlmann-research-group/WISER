import os
import logging

from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Coroutine

import lark
from lark import Visitor, Tree, Token, v_args, Discard
from lark.exceptions import VisitError, GrammarError
import inspect
import numpy as np

import queue
import asyncio
import threading
import concurrent.futures

from .types import VariableType, BandMathValue, BandMathEvalError, BandMathExprInfo
from .functions import BandMathFunction, get_builtin_functions
from .utils import TEMP_FOLDER_PATH, print_tree_with_meta, get_unused_file_path_in_folder

from wiser.raster.dataset import RasterDataSet

from osgeo import gdal

from .builtins import (
    OperatorCompare,
    OperatorAdd, OperatorSubtract, OperatorMultiply, OperatorDivide,
    OperatorUnaryNegate, OperatorPower, OperatorCompareOrig, OperatorAddOrig, OperatorSubtractOrig,
    OperatorMultiplyOrig, OperatorDivideOrig, OperatorUnaryNegateOrig, OperatorPowerOrig, 
    )

from wiser.raster.loader import RasterDataLoader
from wiser.raster.dataset_impl import SaveState, GDALRasterDataImpl

from .builtins.constants import (
    MAX_RAM_BYTES, SCALAR_BYTES, NUM_READERS, \
    NUM_WRITERS, LHS_KEY, RHS_KEY
    )

logger = logging.getLogger(__name__)

class UniqueIDAssigner(Visitor):
    def __init__(self):
        self.current_id = 0

    def _assign_id(self, tree):
        self.current_id += 1
        tree.meta.unique_id = self.current_id
        print(f"Giving unique ID: {self.current_id}")

    def comparison(self, tree):
        self._assign_id(tree)

    def add_expr(self, tree):
        self._assign_id(tree)

    def mul_expr(self, tree):
        self._assign_id(tree)

    def unary_negate_expr(self, tree):
        self._assign_id(tree)

    def power_expr(self, tree):
        self._assign_id(tree)

class NumberOfIntermediatesFinder(lark.visitors.Transformer):
    '''
    A Lark Transformer for evaluating band-math expressions.
    '''
    def __init__(self, variables: Dict[str, Tuple[VariableType, Any]],
                       functions: Dict[str, Callable],
                       shape: Tuple[int, int, int] = None):
        self._variables = variables
        self._functions = functions
        self._shape = shape
        self._intermediate_running_total = 0
        self._max_intermediates = 0
    
    def increment_interm_running_total(self):
        self._intermediate_running_total += 1
        if self._intermediate_running_total > self._max_intermediates:
            self._max_intermediates = self._intermediate_running_total
    
    def decrement_interm_running_total(self):
        if self._intermediate_running_total > 0:
            self._intermediate_running_total -= 1
    
    def update_interm_running_total(self, update: int):
        self._intermediate_running_total += update
        if self._intermediate_running_total > self._max_intermediates:
            self._max_intermediates = self._intermediate_running_total

    def get_max_intermediates(self):
        return self._max_intermediates

    def find_current_interm_and_update_max(self, lhs, rhs):
        has_intermediate = 0
        if isinstance(lhs, BandMathValue) and isinstance(rhs, BandMathValue):
            # If both lhs and rhs are bandmath value image cubes then we are at a leaf node so we want
            # to incrememnt the running total and we will currently have two intermediates
            if lhs.type == VariableType.IMAGE_CUBE and rhs.type == VariableType.IMAGE_CUBE:
                self.increment_interm_running_total()
                self.increment_interm_running_total()
                self.decrement_interm_running_total()
                has_intermediate = 1
            # If either lhs and rhs are image cubes, then we will incrememnt the counter by one
            # and make current intermediates = 1
            elif lhs.type == VariableType.IMAGE_CUBE or rhs.type == VariableType.IMAGE_CUBE:
                self.increment_interm_running_total()
                has_intermediate = 1
        # The case when we just got up the tree from an expression node and we have a 
        # band math value. If lhs is an image cube we want to increment curr intermediates.
        # If rhs is an int that is not zero, then we 
        elif isinstance(lhs, BandMathValue) and isinstance(rhs, int):
            # In this case, both things are counted as intermediates
            if lhs.type == VariableType.IMAGE_CUBE and rhs > 0:
                self.increment_interm_running_total()  # Because lhs is new since it is an image cube bandmath value
                self.decrement_interm_running_total()
                has_intermediate = 1
            # elif rhs > 0: # This intermediate has already been added to the running total so we do nothing in this case
            
            elif lhs.type == VariableType.IMAGE_CUBE:
                self.increment_interm_running_total() # We don't decrement because we aren't combining two values
                has_intermediate = 1
        elif isinstance(lhs, int) and isinstance(rhs, BandMathValue):
            if rhs.type == VariableType.IMAGE_CUBE and lhs > 0:
                self.increment_interm_running_total()
                self.decrement_interm_running_total()
                has_intermediate = 1
            elif rhs.type == VariableType.IMAGE_CUBE:
                self.increment_interm_running_total() # We don't decrement because we aren't combining two values
                has_intermediate = 1
        elif isinstance(lhs, int) and isinstance(rhs, int):
            if lhs > 0 and rhs > 0:
                self.decrement_interm_running_total()
                has_intermediate = 1
        else:
            raise TypeError(f' Got wrong type in either argument. Arg1 {lhs}, arg2: {rhs}')

        return has_intermediate

    def comparison(self, args):
        logger.debug(' * comparison')
        lhs = args[0]
        oper = args[1]
        rhs = args[2]
        return self.find_current_interm_and_update_max(lhs, rhs)

    def add_expr(self, values):
        '''
        Implementation of addition and subtraction operations in the
        transformer.
        '''
        logger.debug(' * add_expr')
        lhs = values[0]
        oper = values[1]
        rhs = values[2]

        if oper != '+' and oper != '-':
            raise RuntimeError(f'Unexpected operator {oper}')

        return self.find_current_interm_and_update_max(lhs, rhs)

    def mul_expr(self, args):
        '''
        Implementation of multiplication and division operations in the
        transformer.
        '''
        logger.debug(' * mul_expr')
        lhs = args[0]
        oper = args[1]
        rhs = args[2]

        if oper != '*' and oper != '/':
            raise RuntimeError(f'Unexpected operator {oper}')
        
        return self.find_current_interm_and_update_max(lhs, rhs)

    def power_expr(self, args):
        '''
        Implementation of power operation in the transformer.
        '''
        logger.debug(' * power_expr')
        return self.find_current_interm_and_update_max(args[0], args[1])

    def unary_negate_expr(self, args):
        '''
        Implementation of unary negation in the transformer.
        '''
        logger.debug(' * unary_negate_expr')
        return self.find_current_interm_and_update_max(args[1], 0)


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

class BandMathEvaluator(lark.visitors.Transformer):
    '''
    A Lark Transformer for evaluating band-math expressions.
    '''
    def __init__(self, variables: Dict[str, Tuple[VariableType, Any]],
                       functions: Dict[str, Callable]):
        self._variables = variables
        self._functions = functions

    def comparison(self, args):
        logger.debug(' * comparison')
        lhs = args[0]
        oper = args[1]
        rhs = args[2]
        return OperatorCompareOrig(oper).apply([lhs, rhs])


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
            return OperatorAddOrig().apply([lhs, rhs])

        elif oper == '-':
            return OperatorSubtractOrig().apply([lhs, rhs])

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
            return OperatorMultiplyOrig().apply([lhs, rhs])

        elif oper == '/':
            return OperatorDivideOrig().apply([lhs, rhs])

        raise RuntimeError(f'Unexpected operator {oper}')


    def power_expr(self, args):
        '''
        Implementation of power operation in the transformer.
        '''
        logger.debug(' * power_expr')
        return OperatorPowerOrig().apply([args[0], args[1]])


    def unary_negate_expr(self, args):
        '''
        Implementation of unary negation in the transformer.
        '''
        logger.debug(' * unary_negate_expr')
        # args[0] is the '-' character
        return OperatorUnaryNegateOrig().apply([args[1]])


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

    # print(f"GDAL Python Version: {gdal.__version__}")
    lower_variables = {}
    for name, value in variables.items():
        lower_variables[name.lower()] = value

    lower_functions = get_builtin_functions()
    if functions:
        for name, function in functions.items():
            lower_functions[name.lower()] = function

    parser = lark.Lark.open('bandmath.lark', rel_to=__file__, start='expression', 
                            propagate_positions=True)
    tree = parser.parse(bandmath_expr)

    logger.info(f'Band-math parse tree:\n{tree.pretty()}')
    logger.debug('Beginning band-math evaluation')

    id_assigner = UniqueIDAssigner()
    id_assigner.visit(tree)
    # print(f"TREE: \n")
    # print_tree_with_meta(tree)

    numInterFinder = NumberOfIntermediatesFinder(lower_variables, lower_functions, expr_info.shape)
    numInterFinder.transform(tree)
    number_of_intermediates = numInterFinder.get_max_intermediates()
    # print(f"Number of intermediates: {number_of_intermediates}")

    def write_raster(out_dataset_gdal, band_index_list_current: List[int], result: np.ndarray):
        # print("ABOUT TO WRITE DATA")
        gdal_band_list_current = [band+1 for band in band_index_list_current]
        # We could check the type of out_dataset_gdal here to make sure 
        # that it's a gdal even though we are only using this function here
        out_dataset_gdal.WriteRaster(
            0, 0, out_dataset_gdal.RasterXSize, out_dataset_gdal.RasterYSize,
            result.tobytes(),
            buf_xsize = out_dataset_gdal.RasterXSize, buf_ysize=out_dataset_gdal.RasterYSize,
            buf_type=gdal.GDT_Float32,
            band_list=gdal_band_list_current
        )
        out_dataset_gdal.FlushCache()
        # print("FINISHED FLUSHING DATA")

    if expr_info.result_type == VariableType.IMAGE_CUBE and not use_old_method:
        eval = None
        try:
            # eval = BandMathEvaluatorSync(lower_variables, lower_functions, expr_info.shape)
            eval = BandMathEvaluatorAsync(lower_variables, lower_functions, expr_info.shape)

            bands, lines, samples = expr_info.shape
            
            # Gets the correct file path to make our temporary file
            result_path = get_unused_file_path_in_folder(TEMP_FOLDER_PATH, result_name)
            folder_path = os.path.dirname(result_path)
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            
            out_dataset_gdal = gdal.GetDriverByName('ENVI').Create(result_path, samples, lines, bands, gdal.GDT_Float32)
            # We declare the dataset write after so if any errors occur below,
            # the file gets destroyed (which happens in del of RasterDataSet)
            out_dataset = RasterDataLoader().dataset_from_gdal_dataset(out_dataset_gdal)
            out_dataset.set_save_state(SaveState.IN_DISK_NOT_SAVED)
            out_dataset.set_dirty()
            
            # Based on memory limits (currently set in constants,, but we could make it more adjustable)
            # find the number of bands that we can access without exceeding it
            bytes_per_element = np.dtype(expr_info.elem_type).itemsize if expr_info.elem_type is not None else SCALAR_BYTES
            bytes_per_scalar = bytes_per_element
            max_bytes = MAX_RAM_BYTES/bytes_per_scalar
            max_bytes_per_intermediate = max_bytes / number_of_intermediates
            num_bands = int(np.floor(max_bytes_per_intermediate / (lines*samples)))

            writing_futures = []
            for band_index in range(0, bands, num_bands):
                band_index_list_current = [band for band in range(band_index, band_index+num_bands) if band < bands]
                band_index_list_next = [band for band in range(band_index+num_bands, band_index+2*num_bands) if band < bands]

                eval.index_list_current = band_index_list_current
                eval.index_list_next = band_index_list_next
                
                result_value = eval.transform(tree)
                if isinstance(result_value, (asyncio.Future, Coroutine)):
                    result_value = asyncio.run_coroutine_threadsafe(eval.transform(tree), eval._event_loop).result()
                res = result_value.value
                
                if isinstance(res, np.ma.MaskedArray):
                    if not np.issubdtype(res.dtype, np.floating):
                        res = res.astype(np.float32)
                    res[res.mask] = np.nan
                    
                assert (res.shape[0] == out_dataset_gdal.RasterXSize, \
                        res.shape[1] == out_dataset_gdal.RasterYSize)
                
                future = eval._write_thread_pool.submit(write_raster, \
                                                    out_dataset_gdal, band_index_list_current, \
                                                    res)
                writing_futures.append(future)
            concurrent.futures.wait(writing_futures)
            # print(f"DONE WRITING ARRAY")
        except BaseException as e:
            if eval is not None:
                eval.stop()
                raise e
        finally:
            eval.stop()

        return (RasterDataSet, out_dataset)
    else:
        # print("OLD METHOD")
        try:
            eval = BandMathEvaluator(lower_variables, lower_functions)
            result_value = eval.transform(tree)
            if isinstance(result_value, Coroutine): 
                result_value = \
                    asyncio.run_coroutine_threadsafe(result_value, eval._event_loop).result()
        except BaseException as e:
            raise e
        return (result_value.type, result_value.value)
