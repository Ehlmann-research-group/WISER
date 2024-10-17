import os
import logging

from typing import Any, Callable, Dict, List, Set, Tuple

import lark
import numpy as np

import concurrent.futures

from .types import VariableType, BandMathValue, BandMathEvalError, BandMathExprInfo
from .functions import BandMathFunction, get_builtin_functions
from .utils import (
    TEMP_FOLDER_PATH, get_unused_file_path_in_folder, \
    np_dtype_to_gdal, write_raster_to_dataset, max_bytes_to_chunk,
)

from wiser.raster.dataset import RasterDataSet

from osgeo import gdal

from .builtins import (
    OperatorCompare,
    OperatorAdd, OperatorSubtract, OperatorMultiply, OperatorDivide,
    OperatorUnaryNegate, OperatorPower,
    )

from wiser.raster.loader import RasterDataLoader
from wiser.raster.dataset_impl import SaveState

from .builtins.constants import SCALAR_BYTES, NUM_WRITERS

logger = logging.getLogger(__name__)

class BandMathEvaluator(lark.visitors.Transformer):
    '''
    A Lark Transformer for evaluating band-math expressions.
    '''
    def __init__(self, variables: Dict[str, Tuple[VariableType, Any]],
                       functions: Dict[str, Callable]):
        self._variables = variables
        self._functions = functions
        self.index_list = None

    def comparison(self, args):
        logger.debug(' * comparison')
        lhs = args[0]
        oper = args[1]
        rhs = args[2]
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

class BandMathEvaluatorChunking(BandMathEvaluator):
    '''
    A Lark Transformer for evaluating band-math expressions.
    '''
    def __init__(self, variables: Dict[str, Tuple[VariableType, Any]],
                       functions: Dict[str, Callable]):
        super().__init__(variables, functions)
        self.write_thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=NUM_WRITERS)
    
    def stop(self):
        """Gracefully stop the event loop and wait for the thread to finish."""
        self.write_thread_pool.shutdown(wait=False, cancel_futures=True)

    def __del__(self):
        self.stop()  # Ensure the loop and thread are stopped
        print("Eval operator event loop and thread cleaned up")

class NumberOfIntermediatesFinder(BandMathEvaluator):
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

    numInterFinder = NumberOfIntermediatesFinder(lower_variables, lower_functions, expr_info.shape)
    numInterFinder.transform(tree)
    number_of_intermediates = numInterFinder.get_max_intermediates()
    logger.debug(f'Number of intermediates: {number_of_intermediates}')

    gdal_type = np_dtype_to_gdal(np.dtype(expr_info.elem_type))
    
    max_chunking_bytes = max_bytes_to_chunk(expr_info.result_size()*number_of_intermediates)
    logger.debug(f"Max chunking bytes: {max_chunking_bytes}")
    # max_chunking_bytes = 4000000000
    if expr_info.result_type == VariableType.IMAGE_CUBE and max_chunking_bytes is not None and not use_old_method:
        try:
            eval = BandMathEvaluatorChunking(lower_variables, lower_functions)

            bands, lines, samples = expr_info.shape
            # Gets the correct file path to make our temporary file
            result_path = get_unused_file_path_in_folder(TEMP_FOLDER_PATH, result_name)
            folder_path = os.path.dirname(result_path)
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            
            out_dataset_gdal = gdal.GetDriverByName('ENVI').Create(result_path, samples, lines, bands, gdal_type)
            # We declare the dataset write after so if any errors occur below,
            # the file gets destroyed (which happens in del of RasterDataSet)
            out_dataset = RasterDataLoader().dataset_from_gdal_dataset(out_dataset_gdal)
            out_dataset.set_save_state(SaveState.IN_DISK_NOT_SAVED)
            out_dataset.set_dirty()
            
            # Based on memory limits (currently set in constants,, but we could make it more adjustable)
            # find the number of bands that we can access without exceeding it
            bytes_per_element = np.dtype(expr_info.elem_type).itemsize if expr_info.elem_type is not None else SCALAR_BYTES
            bytes_per_scalar = bytes_per_element
            max_bytes = max_chunking_bytes/bytes_per_scalar
            max_bytes_per_intermediate = max_bytes / number_of_intermediates
            num_bands = int(np.floor(max_bytes_per_intermediate / (lines*samples)))
            num_bands = 1 if num_bands < 1 else num_bands

            for band_index in range(0, bands, num_bands):
                band_index_list = [band for band in range(band_index, band_index+num_bands) if band < bands]
                eval.index_list = band_index_list
                
                result_value = eval.transform(tree)
                res = result_value.value
                print(f"Type new: {type(res)}")
    
                assert (res.shape[0] == out_dataset_gdal.RasterXSize, \
                        res.shape[1] == out_dataset_gdal.RasterYSize)
                
                write_raster_to_dataset(out_dataset_gdal, band_index_list, res, gdal_type)
        except BaseException as e:
            if eval is not None:
                eval.stop()
                raise e
        finally:
            eval.stop()

        return (RasterDataSet, out_dataset)
    else:
        try:
            eval = BandMathEvaluator(lower_variables, lower_functions)
            result_value = eval.transform(tree)
            res = result_value.value
            print(f"Type old: {type(res)}")
        except BaseException as e:
            raise e
        return (result_value.type, result_value.value)
