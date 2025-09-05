import os
import logging
import inspect

from typing import Any, Callable, Dict, Tuple, Coroutine, Union, List, TYPE_CHECKING, Optional

import lark
from lark import Visitor, Tree, Token, v_args
from lark.exceptions import VisitError, GrammarError
import numpy as np

import queue
import asyncio
import threading
import concurrent.futures

from .types import VariableType, BandMathValue, BandMathEvalError, BandMathExprInfo
from .functions import BandMathFunction, get_builtin_functions
from .utils import (
    TEMP_FOLDER_PATH, get_unused_file_path_in_folder, \
    np_dtype_to_gdal, write_raster_to_dataset, max_bytes_to_chunk,
    get_valid_ignore_value,
)

from wiser import bandmath
from wiser.bandmath.types import BANDMATH_VALUE_TYPE
from wiser.raster.serializable import Serializable, SerializedForm

from wiser.raster.data_cache import DataCache
from wiser.raster.loader import RasterDataLoader

from wiser.raster.dataset import RasterDataSet, RasterDataBand, SpectralMetadata, RasterDataBatchBand, RasterDataDynamicBand
from wiser.raster.spectrum import Spectrum
from wiser.raster.loader import RasterDataLoader
from wiser.raster.dataset_impl import SaveState

if TYPE_CHECKING:
    from wiser.gui.app_state import ApplicationState

from wiser.gui.subprocessing_manager import ProcessManager

from osgeo import gdal
import multiprocessing as mp
import multiprocessing.connection as mp_conn

from .builtins import (
    OperatorCompare,
    OperatorAdd, OperatorSubtract, OperatorMultiply, OperatorDivide,
    OperatorUnaryNegate, OperatorPower,
    )


from .builtins.constants import SCALAR_BYTES, NUM_WRITERS, DEFAULT_IGNORE_VALUE, NUM_READERS, LHS_KEY, RHS_KEY

import traceback

logger = logging.getLogger(__name__)


class UniqueIDAssigner(Visitor):
    def __init__(self):
        self.current_id = 0

    def _assign_id(self, tree):
        self.current_id += 1
        tree.meta.unique_id = self.current_id

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

class AsyncTransformer(lark.visitors.Transformer):
    """
    Custom Transformer class that supports asynchronous methods.
    This class mirrors the functionality of Lark's `Transformer` class,
    but allows the use of `async` methods for transforming tree nodes.
    """

    async def _call_userfunc(self, tree, new_children=None):
        """
        Call the appropriate transformation method for a given tree node.
        Handles both asynchronous and synchronous transformation methods.
        """
        children = new_children if new_children is not None else tree.children
        try:
            f = getattr(self, tree.data)
        except AttributeError:
            return await self.__default__(tree.data, children, tree.meta)  # Ensure we await the default method if overridden
        else:
            try:
                wrapper = getattr(f, 'visit_wrapper', None)
                if wrapper is not None:
                    return await f.visit_wrapper(f, tree.data, children, tree.meta)
                else:
                    # Check if the transformation method is async or sync
                    if inspect.isawaitable(f):
                        return await f(children)
                    else:
                        return f(children)
            except GrammarError:
                raise
            except Exception as e:
                raise VisitError(tree.data, tree, e)

    async def _call_userfunc_token(self, token):
        """
        Call the appropriate transformation method for a given token.
        Handles both asynchronous and synchronous methods.
        """
        try:
            f = getattr(self, token.type)
        except AttributeError:
            return await self.__default_token__(token)  # Ensure we await the default token method if overridden
        else:
            try:
                if inspect.isawaitable(f):
                    return await f(token)
                else:
                    return f(token)
            except GrammarError:
                raise
            except Exception as e:
                raise VisitError(token.type, token, e)

    async def _transform_children(self, children):
        """
        Asynchronously transform a list of children, yielding transformed children.
        Handles both Tree nodes and Token nodes.
        """
        child_tasks = []

        for c in children:
            if isinstance(c, Tree):
                # Create a separate task to transform each subtree
                child_tasks.append(asyncio.create_task(self._transform_tree(c)))
            elif self.__visit_tokens__ and isinstance(c, Token):
                # Create a separate task for transforming tokens if `visit_tokens` is set to True
                child_tasks.append(asyncio.create_task(self._call_userfunc_token(c)))
            else:
                # Directly append non-tree, non-token objects without a task
                child_tasks.append(asyncio.create_task(asyncio.sleep(0, c)))  # Wrap raw values as completed tasks

        # Await all child tasks concurrently and gather results into a list
        transformed_children = await asyncio.gather(*child_tasks)
        return transformed_children

    async def _transform_tree(self, tree):
        """
        Asynchronously transform a tree node.
        This function recursively transforms the children first, and then calls the transformation method for the node.
        """
        children_tasks = [asyncio.create_task(self._transform_children(tree.children))]
        children = await asyncio.gather(*children_tasks)
        flattened_children = [item for sublist in children for item in sublist]
        return await self._call_userfunc(tree, flattened_children)

    async def transform(self, tree):
        """
        Asynchronously transform the given tree and return the final result.
        """
        root_task = asyncio.create_task(self._transform_tree(tree))
        
        # Await the top-level task and get the result
        result = await root_task
        return result

    async def __default__(self, data, children, meta):
        """
        Default function called if no attribute matches `data`.
        This function can be overridden in subclasses if needed.
        """
        return Tree(data, children, meta)

    async def __default_token__(self, token):
        """
        Default function called if no attribute matches `token.type`.
        This function can be overridden in subclasses if needed.
        """
        return token

class BandMathEvaluatorAsync(AsyncTransformer):
    '''
    A Lark Transformer for evaluating band-math expressions.
    '''
    def __init__(self, variables: Dict[str, Tuple[VariableType, BANDMATH_VALUE_TYPE]],
                       functions: Dict[str, Callable],
                       shape: Tuple[int, int, int] = None,
                       use_parallelization = True):
        self._variables = variables
        self._functions = functions
        self.index_list_current = None
        self.index_list_next = None
        self._shape = shape
        if use_parallelization:
            self._read_data_queue_dict = {}
            self._write_data_queue = queue.Queue()
            self._read_thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=NUM_READERS)
            self._write_thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=NUM_WRITERS)
            self._event_loop = asyncio.new_event_loop()
            self._loop_thread = threading.Thread(target=self._event_loop.run_forever, daemon=False)
            self._loop_thread.start()
        else:
            self._read_data_queue = None
            self._write_data_queue = None

    @v_args(meta=True)
    async def comparison(self, meta, args):
        logger.debug(' * comparison')
        node_id = getattr(meta, 'unique_id', None)
        if node_id not in self._read_data_queue_dict:
            self._read_data_queue_dict[node_id] = {}
            self._read_data_queue_dict[node_id][LHS_KEY] = queue.Queue()
            self._read_data_queue_dict[node_id][RHS_KEY] = queue.Queue()

        lhs = args[0]
        oper = args[1]
        rhs = args[2]
        
        # Schedule this operation as a background task
        addition_task = asyncio.ensure_future(OperatorCompare(oper).apply(
            [lhs, rhs],
            self.index_list_current,
            self.index_list_next,
            self._read_data_queue_dict[node_id],
            self._read_thread_pool,
            self._event_loop,
            node_id
        ))
        return await addition_task

    @v_args(meta=True)
    async def add_expr(self, meta, values):
        '''
        Implementation of addition and subtraction operations in the
        transformer.
        '''
        logger.debug(' * add_expr')
        node_id = getattr(meta, 'unique_id', None)
        if node_id not in self._read_data_queue_dict:
            self._read_data_queue_dict[node_id] = {}
            self._read_data_queue_dict[node_id][LHS_KEY] = queue.Queue()
            self._read_data_queue_dict[node_id][RHS_KEY] = queue.Queue()

        lhs = values[0]
        oper = values[1]
        rhs = values[2]

        if oper == '+':
            # Schedule this operation as a background task
            addition_task = asyncio.ensure_future(OperatorAdd().apply(
                [lhs, rhs],
                self.index_list_current,
                self.index_list_next,
                self._read_data_queue_dict[node_id],
                self._read_thread_pool,
                self._event_loop,
                node_id
            ))
            return await addition_task

        elif oper == '-':
            # Schedule this operation as a background task
            addition_task = asyncio.ensure_future(OperatorSubtract().apply(
                [lhs, rhs],
                self.index_list_current,
                self.index_list_next,
                self._read_data_queue_dict[node_id],
                self._read_thread_pool,
                self._event_loop,
                node_id
            ))
            return await addition_task

        raise RuntimeError(f'Unexpected operator {oper}')


    @v_args(meta=True)
    async def mul_expr(self, meta, args):
        '''
        Implementation of multiplication and division operations in the
        transformer.
        '''
        logger.debug(' * mul_expr')
        node_id = getattr(meta, 'unique_id', None)
        if node_id not in self._read_data_queue_dict:
            self._read_data_queue_dict[node_id] = {}
            self._read_data_queue_dict[node_id][LHS_KEY] = queue.Queue()
            self._read_data_queue_dict[node_id][RHS_KEY] = queue.Queue()
            
        lhs = args[0]
        oper = args[1]
        rhs = args[2]

        if oper == '*':
            # Schedule this operation as a background task
            addition_task = asyncio.ensure_future(OperatorMultiply().apply(
                [lhs, rhs],
                self.index_list_current,
                self.index_list_next,
                self._read_data_queue_dict[node_id],
                self._read_thread_pool,
                self._event_loop,
                node_id
            ))
            return await addition_task

        elif oper == '/':
            # Schedule this operation as a background task
            addition_task = asyncio.ensure_future(OperatorDivide().apply(
                [lhs, rhs],
                self.index_list_current,
                self.index_list_next,
                self._read_data_queue_dict[node_id],
                self._read_thread_pool,
                self._event_loop,
                node_id
            ))
            return await addition_task

        raise RuntimeError(f'Unexpected operator {oper}')


    @v_args(meta=True)
    async def power_expr(self, meta, args):
        '''
        Implementation of power operation in the transformer.
        '''
        logger.debug(' * power_expr')
        node_id = getattr(meta, 'unique_id', None)
        if node_id not in self._read_data_queue_dict:
            self._read_data_queue_dict[node_id] = {}
            self._read_data_queue_dict[node_id][LHS_KEY] = queue.Queue()
            self._read_data_queue_dict[node_id][RHS_KEY] = queue.Queue()

        addition_task = asyncio.ensure_future(OperatorPower().apply(
                [args[0], args[1]],
                self.index_list_current,
                self.index_list_next,
                self._read_data_queue_dict[node_id],
                self._read_thread_pool,
                self._event_loop,
                node_id
            ))
        return await addition_task


    @v_args(meta=True)
    async def unary_negate_expr(self, meta, args):
        '''
        Implementation of unary negation in the transformer.
        '''
        logger.debug(' * unary_negate_expr')
        node_id = getattr(meta, 'unique_id', None)
        if node_id not in self._read_data_queue_dict:
            self._read_data_queue_dict[node_id] = {}
            self._read_data_queue_dict[node_id][LHS_KEY] = queue.Queue()

        addition_task = asyncio.ensure_future(OperatorUnaryNegate().apply(
            [args[1]],
            self.index_list_current,
            self.index_list_next,
            self._read_data_queue_dict[node_id],
            self._read_thread_pool,
            self._event_loop,
            node_id
        ))
        return await addition_task


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

    def stop(self):
        """Gracefully stop the event loop and wait for the thread to finish."""
        if self._event_loop.is_running():
            self._event_loop.call_soon_threadsafe(self._event_loop.stop)  # Safely stop the loop
        self._loop_thread.join()  # Wait for the thread to finish
        self._read_thread_pool.shutdown(wait=False, cancel_futures=True)
        self._write_thread_pool.shutdown(wait=False, cancel_futures=True)

    def __del__(self):
        self.stop()  # Ensure the loop and thread are stopped

class BandMathEvaluator(lark.visitors.Transformer):
    '''
    A Lark Transformer for evaluating band-math expressions.
    '''
    def __init__(self, variables: Dict[str, Tuple[VariableType, BANDMATH_VALUE_TYPE]],
                       functions: Dict[str, Callable]):
        self._variables = variables
        self._functions = functions
        self.index_list = None
        self._event_loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(target=self._event_loop.run_forever, daemon=False)
        self._loop_thread.start()

    def comparison(self, args):
        logger.debug(' * comparison')
        lhs = args[0]
        oper = args[1]
        rhs = args[2]

        future = asyncio.run_coroutine_threadsafe(
            OperatorCompare(oper).apply([lhs, rhs], self.index_list), \
            self._event_loop)
        return future.result()


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
            future = asyncio.run_coroutine_threadsafe(
                OperatorAdd().apply([lhs, rhs], self.index_list), \
                self._event_loop)
            return future.result()

        elif oper == '-':
            future = asyncio.run_coroutine_threadsafe(
                OperatorSubtract().apply([lhs, rhs], self.index_list), \
                self._event_loop)
            return future.result()

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
            future = asyncio.run_coroutine_threadsafe(
                OperatorMultiply().apply([lhs, rhs], self.index_list), \
                self._event_loop)
            return future.result()

        elif oper == '/':
            future = asyncio.run_coroutine_threadsafe(
                OperatorDivide().apply([lhs, rhs], self.index_list), \
                self._event_loop)
            return future.result()

        raise RuntimeError(f'Unexpected operator {oper}')


    def power_expr(self, args):
        '''
        Implementation of power operation in the transformer.
        '''
        logger.debug(' * power_expr')

        future = asyncio.run_coroutine_threadsafe(
            OperatorPower().apply([args[0], args[1]], self.index_list), \
            self._event_loop)
        return future.result()


    def unary_negate_expr(self, args):
        '''
        Implementation of unary negation in the transformer.
        '''
        logger.debug(' * unary_negate_expr')
        # args[0] is the '-' character
        
        future = asyncio.run_coroutine_threadsafe(
            OperatorUnaryNegate().apply([args[1]], self.index_list), \
            self._event_loop)
        return future.result()


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
    
    def stop(self):
        """Gracefully stop the event loop and wait for the thread to finish."""

        if hasattr(self, '_event_loop') and self._event_loop.is_running():
            self._event_loop.call_soon_threadsafe(self._event_loop.stop)  # Safely stop the loop
        if hasattr(self, '_loop_thread'):
            self._loop_thread.join()
    def __del__(self):
        self.stop()  # Ensure the loop and thread are stopped

class NumberOfIntermediatesFinder(BandMathEvaluator):
    '''
    A Lark Transformer for evaluating band-math expressions.
    '''
    def __init__(self, variables: Dict[str, Tuple[VariableType, BANDMATH_VALUE_TYPE]],
                       functions: Dict[str, Callable]):
        self._variables = variables
        self._functions = functions
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


def eval_bandmath_expr(
        bandmath_expr: str, expr_info: BandMathExprInfo, result_name: str, cache: DataCache,
        variables: Dict[str, Tuple[VariableType, BANDMATH_VALUE_TYPE]],
        functions: Dict[str, BandMathFunction] = None, succeeded_callback: Callable = lambda _: None, \
        status_callback: Callable = lambda _: None, error_callback: Callable = lambda _: None, \
        started_callback: Callable = lambda _: None, cancelled_callback: Callable = lambda _: None, \
        app_state: 'ApplicationState' = None, use_synchronous_method = True
        ) -> ProcessManager:
    '''
    Evaluate a band-math expression using the specified variable and function
    definitions.

    Variables are passed in a dictionary of string names that map to 2-tuples:
    (VariableType, value).  The VariableType enum-value specifies the high-level
    type of the value, since multiple specific types are supported.

    *   VariableType.IMAGE_CUBE:  RasterDataSet, 3D np.ndarray [band][y][x]
    *   VariableType.IMAGE_BAND:  RasterDataBand, 2D np.ndarray [y][x]
    *   VariableType.SPECTRUM:  Spectrum, 1D np.ndarray [band]

    Functions are passed in a dictionary of string names that map to the class
    BandMathFunction

    If successful, the result of the calculation is returned as a 2-tuple of the
    same form as the variables or as a 2-tuple where the first variable is the 
    class RasterDataSet and the second is an instantiation of that class.
    This 2-tuple value is sent to the passed in callable.

    *   VariableType.IMAGE_CUBE:  3D np.ndarray [band][x][y]
    *   VariableType.IMAGE_BAND:  2D np.ndarray [x][y]
    *   VariableType.SPECTRUM:  1D np.ndarray [band]
    *   VariableType.NUMBER:  float
    *   RasterDataSet:  RasterDataSet (instantiation)

    The function returns the ProcessManager object that is managing the underlying
    QThread and subprocess.
    '''

    # Just to be defensive against potentially bad inputs, make sure all names
    # of variables and functions are lowercase.
    # TODO(donnie):  Can also make sure they are valid, trimmed of whitespace,
    #     etc.

    lower_variables: Dict[str, Tuple[VariableType, BANDMATH_VALUE_TYPE]] = {}
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

    numInterFinder = NumberOfIntermediatesFinder(lower_variables, lower_functions)
    numInterFinder.transform(tree)
    number_of_intermediates = numInterFinder.get_max_intermediates()
    number_of_intermediates += 1
    logger.debug(f'Number of intermediates: {number_of_intermediates}')

    # We must serialize RasterDataSet, RasterBand, and Spectrum objects because they
    # could have an underlying gdal or osgeo object that can't be pickled
    serialized_variables = serialize_bandmath_variables(lower_variables)

    kwargs = {
        "bandmath_expr": bandmath_expr,
        "expr_info": expr_info,
        "result_name": result_name,
        "cache": None,
        "serialized_variables": serialized_variables,
        "lower_functions": lower_functions,
        "number_of_intermediates": number_of_intermediates,
        "tree": tree,
        "use_synchronous_method": use_synchronous_method,
    }

    process_manager = ProcessManager(subprocess_bandmath, kwargs)
    if app_state:
        app_state.add_running_process(process_manager)

    task = process_manager.get_task()
    task.cancelled.connect(cancelled_callback)
    # The started slot is passed the task
    task.started.connect(started_callback)
    # The error slot is passed the process_manager's task
    task.error.connect(error_callback)
    # The progress slot is passed the message that subprocess_bandmath
    # sends over the pipe
    task.status.connect(status_callback)
    task.succeeded.connect(lambda task: succeeded_callback(task.get_result()))
    process_manager.start_task()
    return process_manager

def serialize_bandmath_variables(variables: Dict[str, Tuple[VariableType, BANDMATH_VALUE_TYPE]]) -> \
    Dict[str, Tuple[VariableType, Union[SerializedForm, str, bool]]]:
    '''
    This function is meant to serialize the 'variables' and 'functions' dictionaries into a format that can be
    passed to the sub process. In the subprocess, we will deserialize the variables and functions and then pass them
    to the eval_bandmath_expr function.
    '''
    variables_serialized = {}
    for var_name, var_tuple in variables.items():
        var_type = var_tuple[0]
        var_value = var_tuple[1]
        if isinstance(var_value, Serializable):
            variables_serialized[var_name] = (var_type, var_value.get_serialized_form())
        else:
            # else, the variable is either a numpy array or a string which is already serializeable
            variables_serialized[var_name] = var_tuple
    return variables_serialized

def subprocess_bandmath(bandmath_expr: str, expr_info: BandMathExprInfo, result_name: str, cache: DataCache,
                        serialized_variables: Dict[str, Tuple[VariableType, Union[SerializedForm, str, bool]]],
                        lower_functions: Dict[str, BandMathFunction], number_of_intermediates: int, tree: lark.ParseTree,
                        use_synchronous_method: bool, child_conn: mp_conn.Connection, return_queue: mp.Queue):
    # First we will decide if we are doing batching or not. If we are doing batching we get the filepaths, if we are not doing
    # batching we will make the file paths = [None]
    is_batch = is_batch_job(serialized_variables)
    filepaths = []
    if is_batch:
        filepaths = get_batch_filepaths(serialized_variables)

    eval_all_bandmath_expr(filepaths=filepaths, bandmath_expr=bandmath_expr, expr_info=expr_info,
                           result_name=result_name, cache=cache, serialized_variables=serialized_variables,
                           lower_functions=lower_functions, number_of_intermediates=number_of_intermediates,
                           tree=tree, use_synchronous_method=use_synchronous_method, child_conn=child_conn,
                           return_queue=return_queue)

    # # This is where we take everything out of the batch folder and make individual pairs of BandMathExprInfo and
    # # variable dictionaries for each. That way we can just use the same machinery as we do for regular bandmath.
    # prepared_variables_list = prepare_bandmath_variables(serialized_variables, filepaths)
    # # Because the first pass through of expr_info (which happened in bandmath_dialog.py) used
    # # either IMAGE_CUBE_BATCH or IMAGE_BAND_BATCH as a variable type (and these have no metadata)
    # # information, we have to get the expr_info again for each filepath in the folder we are batch processing
    # prepared_expr_info_list = prepare_expr_info(bandmath_expr, prepared_variables_list, lower_functions)
    # prepared_result_names_list = prepare_result_names(result_name, filepaths)
    # # This function actually calls the lark transformer and does the heavy lifting.
    # results = eval_full_bandmath_expr(prepared_expr_info_list, prepared_result_names_list, cache, prepared_variables_list, lower_functions, \
    #                         number_of_intermediates, tree, use_synchronous_method, child_conn)
    # # At this point, everything in the folder has been processed. Now we collect the results to put them
    # # on the return queue.
    # serialized_results: List[Tuple[VariableType, SerializedForm, str, BandMathExprInfo]] = []
    # for result_type, result_value, result_name, result_expr_info in results:
    #     if isinstance(result_value, Serializable):
    #         serialized_results.append((result_type, result_value.get_serialized_form(), result_name, result_expr_info))
    #     else:
    #         serialized_results.append((result_type, result_value, result_name, result_expr_info))
    # return_queue.put(serialized_results)

def get_batch_filepaths(serialized_variables: Dict[str, Tuple[VariableType, Union[SerializedForm, str, bool]]]) -> List[str]:
    filepaths = []
    # We need to check if we are doing batching or not. If we are, then we need to 
    # make a list of the filepaths and then we need to make a list of the variables.
    for var_name, var_tuple in serialized_variables.items():
        var_type = var_tuple[0]
        var_value = var_tuple[1]

        # All batch variables 
        if var_type == VariableType.IMAGE_CUBE_BATCH:
            assert isinstance(var_value, str), "Image Cube Batch variables should be strings"
            folder_path = var_value
            filepaths = get_unique_filepaths(folder_path)
            break
        elif var_type == VariableType.IMAGE_BAND_BATCH:
            assert isinstance(var_value, SerializedForm), "Image Band Batch variables should be SerializedForm"
            folder_path = var_value.get_serialize_value()
            filepaths = get_unique_filepaths(folder_path)
            break
    return filepaths


def is_batch_job(serialized_variables: Dict[str, Tuple[VariableType, Union[SerializedForm, str, bool]]]) -> bool:
    '''
    This function is used to decide if we are doing batching or not.
    '''
    for var_name, var_tuple in serialized_variables.items():
        var_type = var_tuple[0]
        var_value = var_tuple[1]
        if var_type == VariableType.IMAGE_CUBE_BATCH or var_type == VariableType.IMAGE_BAND_BATCH:
            return True
    return False

def serialized_form_to_variable(var_name: str, var_type: VariableType, var_value: Union[SerializedForm, str, bool], \
                                loader: RasterDataLoader, filepath: str = None) -> Dict[str, Tuple[VariableType, BANDMATH_VALUE_TYPE]]:
    '''
    This function is used to convert a serialized form of an object back into the original object.
    '''
    if var_type == VariableType.IMAGE_CUBE:
        if isinstance(var_value, SerializedForm):
            serialize_value = var_value.get_serialize_value()
            serialize_metadata = var_value.get_metadata()
            obj = RasterDataSet.deserialize_into_class(serialize_value, serialize_metadata)
            return{var_name: (var_type, obj)}
        else:
            assert isinstance(var_value, np.ndarray), "Image Cube variables should be either a SerializedForm or a numpy array"
            return {var_name: (var_type, var_value)}
    elif var_type == VariableType.IMAGE_CUBE_BATCH:
        assert isinstance(var_value, str), "Image Cube Batch variables should be strings"
        assert filepath is not None, "Filepath is required for Image Cube Batch variables"
        dataset = loader.load_from_file(filepath, interactive=False)[0]
        return {var_name: (VariableType.IMAGE_CUBE, dataset)}

    elif var_type == VariableType.IMAGE_BAND:
        if isinstance(var_value, SerializedForm):
            serialize_value = var_value.get_serialize_value()
            serialize_metadata = var_value.get_metadata()
            obj = RasterDataBand.deserialize_into_class(serialize_value, serialize_metadata)
            return {var_name: (var_type, obj)}
        else:
            assert isinstance(var_value, np.ndarray), "Image Band variables should be either a SerializedForm or a numpy array"
            return {var_name: (var_type, var_value)}

    elif var_type == VariableType.IMAGE_BAND_BATCH:
        assert isinstance(var_value, SerializedForm), "Image Band Batch variables should be SerializedForm"
        assert filepath is not None, "Filepath is required for Image Band Batch variables"
        assert ('band_index' in var_value.get_metadata() and var_value.get_metadata()['band_index'] is not None) or \
            ('wavelength_value' in var_value.get_metadata() and var_value.get_metadata()['wavelength_value'] is not None), \
            "Band index or wavelength value is required for Image Band Batch variables"
        serializable_class = var_value.get_serializable_class()
        # This should never occur, but if it does we make it a RasterDataDynamicBand
        if issubclass(serializable_class, RasterDataBatchBand):
            band_index = var_value.get_metadata().get('band_index', None)
            wavelength_value = var_value.get_metadata().get('wavelength_value', None)
            wavelength_units = var_value.get_metadata().get('wavelength_units', None)
            epsilon = var_value.get_metadata().get('epsilon', None)
            dataset = loader.load_from_file(filepath)
            band = RasterDataDynamicBand(dataset, band_index=band_index, \
                                        wavelength_value=wavelength_value, \
                                        wavelength_units=wavelength_units, epsilon=epsilon)
        else:
            serialize_metadata = var_value.get_metadata()
            serialize_metadata.update({'filepath': filepath})
            band_index  = serialize_metadata.get('band_index', None)
            if band_index is not None:
                band_index = int(band_index)
            band = serializable_class.deserialize_into_class(band_index, serialize_metadata)
        return {var_name: (VariableType.IMAGE_BAND, band)}

    elif var_type == VariableType.SPECTRUM:
        if isinstance(var_value, SerializedForm):
            serialize_value = var_value.get_serialize_value()
            serialize_metadata = var_value.get_metadata()
            obj = Spectrum.deserialize_into_class(serialize_value, serialize_metadata)
            return {var_name: (var_type, obj)}
        else:
            assert isinstance(var_value, np.ndarray), "Spectrum variables should be either a SerializedForm or a numpy array"
            return {var_name: (var_type, var_value)}

    elif var_type == VariableType.NUMBER:
        return {var_name: (var_type, var_value)}

    elif var_type == VariableType.BOOLEAN:
        return {var_name: (var_type, var_value)}

    else:
        raise ValueError(f"Unsupported variable type: {var_type}")

def get_unique_filepaths(folder: str):
    """
    Get all file paths in a folder, but ignore duplicates with the same
    base name. Prefer files with an extension over those without.

    Args:
        folder (str): Path to the folder to scan.

    Returns:
        list[str]: List of file paths.
    """
    files_seen = {}
    for entry in os.listdir(folder):
        full_path = os.path.join(folder, entry)
        if not os.path.isfile(full_path):
            continue

        base, ext = os.path.splitext(entry)
        has_ext = bool(ext)

        # If we've never seen this base name, store it
        if base not in files_seen:
            files_seen[base] = (full_path, has_ext)
        else:
            # Prefer the version with extension
            existing_path, existing_has_ext = files_seen[base]
            if not existing_has_ext and has_ext:
                files_seen[base] = (full_path, has_ext)
            # If both have extensions or both don't, keep the first one

    return [path for path, _ in files_seen.values()]

def prepare_result_names(result_name: str, filepaths: List[str]) -> List[str]:
    """
    Prepare result names by taking the base name of each file (without extension)
    and appending the given suffix, then re-adding the original extension.
    Example:
        result_name="processed", filepath="data/sample.csv"
        -> "sample_processed.csv"
    """
    if not filepaths:
        return [result_name]
    
    result_name_list = []
    for filepath in filepaths:
        base = os.path.basename(filepath)
        name, ext = os.path.splitext(base)
        new_name = f"{name}{result_name}"
        result_name_list.append(new_name)
    return result_name_list

def prepare_expr_info(bandmath_expr: str, \
                      variables_list: List[Dict[str, Tuple[VariableType, Union[SerializedForm, str, bool]]]], \
                      functions: Dict[str, BandMathFunction]) -> List[BandMathExprInfo]:
    '''
    This function is used to expand expr_info to include expression info for each of the variable dictionaries
    in the list 
    '''
    # Go through each of the variables in the list and get the expr_info for each of them
    expr_info_list = []
    for variables in variables_list:
        expr_info_list.append(bandmath.get_bandmath_expr_info(bandmath_expr, variables, functions))
    return expr_info_list

def prepare_bandmath_variables(serialized_variables: Dict[str, Tuple[VariableType, Union[SerializedForm, str, bool]]], \
                               filepaths: List[str]) ->  List[Dict[str, Tuple[VariableType, BANDMATH_VALUE_TYPE]]]:
    '''
    This function is used to unroll the serialized variables into the original variables and if there are batch variables
    make all of the filepaths into the appropriate structure in wiser. The serialized variables are in the class SerializedForm.
    We extracted the deserialized data from the SerializedForm class by first getting the class that we want to create using
    SerializedForm.get_serializable_class(). Then we can get the serialize_value and metadata from the SerializedForm and pass
    this into the deserialize_into_class function of the class that we got from SerializedForm.get_serializable_class().

    Args:
        serialized_variables: A dictionary of variables that have been serialized.
        filepaths: A list of filepaths that are used to load the batch variables. If there are no filepaths, filepaths is None

    Returns:
        A list of dictionaries of variables that have been unrolled. We can pass these variables into eval_full_bandmath_expr
        and it will evaluate the bandmath expression for each of the variables in the list.
    '''
    # This just lets us use serialized_form_to_variable without changing anything to this code
    if len(filepaths) == 0:
        filepaths = [None]

    loader = RasterDataLoader()
    prepared_variables = []
    for filepath in filepaths:
        single_batch_variables = {}
        for var_name, var_tuple in serialized_variables.items():
            var_type = var_tuple[0]
            var_value = var_tuple[1]
            single_batch_variables.update(serialized_form_to_variable(var_name, var_type, var_value, loader, filepath))
        prepared_variables.append(single_batch_variables)
    return prepared_variables

def eval_all_bandmath_expr(filepaths: List[str], bandmath_expr: str, expr_info: BandMathExprInfo, result_name: str, cache: DataCache,
                        serialized_variables: Dict[str, Tuple[VariableType, Union[SerializedForm, str, bool]]],
                        lower_functions: Dict[str, BandMathFunction], number_of_intermediates: int, tree: lark.ParseTree,
                        use_synchronous_method: bool, child_conn: mp_conn.Connection, return_queue: mp.Queue):
    loader = RasterDataLoader()
    # This case is if we are doing batch processing
    if filepaths:
        outputs = []
        count = 0
        total = len(filepaths)
        for filepath in filepaths:
            # First we get the result name
            base = os.path.basename(filepath)
            name, ext = os.path.splitext(base)
            new_result_name = f"{name}{result_name}"
            try:
                count += 1
                child_conn.send(["progress", {"Numerator": count, "Denominator": total, "Status": "Running"}])
                # Second we deserialize all of the variables
                current_variables = {}
                for var_name, var_tuple in serialized_variables.items():
                    var_type = var_tuple[0]
                    var_value = var_tuple[1]
                    current_variables.update(serialized_form_to_variable(var_name, var_type, var_value, loader, filepath))
                # Third we get the proper BandMathExprInfo
                current_expr_info = bandmath.get_bandmath_expr_info(bandmath_expr, current_variables, lower_functions)
                result = eval_singular_bandmath_expr(expr_info=current_expr_info, result_name=new_result_name, cache=cache,
                                                     lower_variables=current_variables, lower_functions=lower_functions,
                                                     number_of_intermediates=number_of_intermediates, tree=tree,
                                                     use_synchronous_method=use_synchronous_method, child_conn=child_conn)
                outputs.append(result)
                child_conn.send(["error", {"Result Name": result_name, "Message": None, "Traceback": None}])
            except Exception as e:
                child_conn.send(["error", {"Result Name": result_name, "Message": str(e), "Traceback": traceback.format_exc()}])
                outputs.append((None, None, new_result_name, None))
    
        child_conn.send(["progress", {"Numerator": count, "Denominator": total, "Status": "Finished"}])
        serialized_results: List[Tuple[VariableType, SerializedForm, str, BandMathExprInfo]] = []
        for result_type, result_value, result_name, result_expr_info in outputs:
            if isinstance(result_value, Serializable):
                serialized_results.append((result_type, result_value.get_serialized_form(), result_name, result_expr_info))
            else:
                serialized_results.append((result_type, result_value, result_name, result_expr_info))
        return_queue.put(serialized_results)
    else:
        new_result_name = result_name
        single_batch_variables = {}
        for var_name, var_tuple in serialized_variables.items():
            var_type = var_tuple[0]
            var_value = var_tuple[1]
            single_batch_variables.update(serialized_form_to_variable(var_name, var_type, var_value, loader))
        expr_info = bandmath.get_bandmath_expr_info(bandmath_expr, single_batch_variables, lower_functions)
        result = eval_singular_bandmath_expr(expr_info=expr_info, result_name=new_result_name, cache=cache,
                                            lower_variables=single_batch_variables, lower_functions=lower_functions,
                                            number_of_intermediates=number_of_intermediates, tree=tree,
                                            use_synchronous_method=use_synchronous_method, child_conn=child_conn)
        serialized_result = None
        result_type, result_value, result_name, result_expr_info = result
        if isinstance(result_value, Serializable):
            serialized_result = (result_type, result_value.get_serialized_form(), result_name, result_expr_info)
        else:
            serialized_result = (result_type, result_value, result_name, result_expr_info)

        return_queue.put([serialized_result])


def eval_singular_bandmath_expr(expr_info: BandMathExprInfo, result_name: str, cache: DataCache,
            lower_variables: Dict[str, Tuple[VariableType, BANDMATH_VALUE_TYPE]],
            lower_functions: Dict[str, BandMathFunction], number_of_intermediates: int, tree: lark.ParseTree,
            use_synchronous_method = True, child_conn: mp_conn.Connection = None \
        ) -> List[Tuple[Union[VariableType, RasterDataSet.__class__],
                        Union[np.ndarray, RasterDataSet],
                        str,
                        BandMathExprInfo]]:
    '''
    This function evaluates one singular bandmath expression

    Returns:
    - The first element in the tuple is the variable type or the RasterDataSet
        class. The second element is the actual value, which is either the numpy array or the
        RasterDataSet. The third element is the name of the resulting dataset. The fourth element
        is the expr_info for that dataset.
    '''
    gdal_type = np_dtype_to_gdal(np.dtype(expr_info.elem_type))
    
    max_chunking_bytes, should_chunk = max_bytes_to_chunk(expr_info.result_size()*number_of_intermediates)
    logger.debug(f"Max chunking bytes: {max_chunking_bytes}")

    spectral_metadata = expr_info.spectral_metadata_source
    data_ignore_value = DEFAULT_IGNORE_VALUE
    if spectral_metadata is not None and spectral_metadata.get_data_ignore_value() is not None:
        data_ignore_value = spectral_metadata.get_data_ignore_value()

    if not use_synchronous_method or \
    (expr_info.result_type == VariableType.IMAGE_CUBE and should_chunk):
        try:
            error = None
            eval = BandMathEvaluatorAsync(lower_variables, lower_functions, expr_info.shape)
            bands = 1
            lines = 1
            samples = 1
            if len(expr_info.shape) == 2:
                lines, samples = expr_info.shape
            elif len(expr_info.shape) == 3:
                bands, lines, samples = expr_info.shape
            else:
                raise RuntimeError(f"expr_info shape is neither 2 or 3, its {expr_info.shape}")
            # Gets the correct file path to make our temporary file
            result_path = get_unused_file_path_in_folder(TEMP_FOLDER_PATH, result_name)
            folder_path = os.path.dirname(result_path)
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            
            out_dataset_gdal = gdal.GetDriverByName('ENVI').Create(result_path, samples, lines, bands, gdal_type)
            # We declare the dataset write after so if any errors occur below,
            # the file gets destroyed (which happens in del of RasterDataSet)
            out_dataset = RasterDataLoader().dataset_from_gdal_dataset(out_dataset_gdal, cache)
            # We NO LONGER set the save state here. We must set it in the process that we pass
            # this piece of data to. If we set it here to IN_DISK_NOT_SAVED, then the garbage
            # collector will delete the underlying dataset when this process ends.
            out_dataset.set_dirty()
            
            # Based on memory limits (currently set in constants,, but we could make it more adjustable)
            # find the number of bands that we can access without exceeding it
            bytes_per_element = np.dtype(expr_info.elem_type).itemsize if expr_info.elem_type is not None else SCALAR_BYTES
            max_bytes = max_chunking_bytes/bytes_per_element
            max_bytes_per_intermediate = max_bytes / number_of_intermediates
            num_bands = int(np.floor(max_bytes_per_intermediate / (lines*samples)))
            num_bands = 1 if num_bands < 1 else num_bands

            writing_futures = []
            for band_index in range(0, bands, num_bands):
                band_index_list_current = [band for band in range(band_index, band_index+num_bands) if band < bands]
                band_index_list_next = [band for band in range(band_index+num_bands, band_index+2*num_bands) if band < bands]
                # print(f"Min: {min(band_index_list_current)} | Max: {max(band_index_list_current)}")
                
                eval.index_list_current = band_index_list_current
                eval.index_list_next = band_index_list_next
                
                result_value = eval.transform(tree)
                if isinstance(result_value, (asyncio.Future, Coroutine)):
                    result_value = asyncio.run_coroutine_threadsafe(result_value, eval._event_loop).result()
                res = result_value.value
                
                future = eval._write_thread_pool.submit(write_raster_to_dataset, \
                                                    out_dataset_gdal, band_index_list_current, \
                                                    res, gdal_type, default_ignore_value=data_ignore_value)
                writing_futures.append(future)
            concurrent.futures.wait(writing_futures)
        except BaseException as e:
            if eval is not None:
                eval.stop()
            raise e
        finally:
            eval.stop()
        return (RasterDataSet, out_dataset, result_name, expr_info)
    else:
        error = None
        try:
            eval = BandMathEvaluator(lower_variables, lower_functions)
            result_value = eval.transform(tree)
            res = result_value.value
        except BaseException as e:
            if eval:
                eval.stop()
            raise e
        finally:
            eval.stop()
        return (result_value.type, result_value.value, result_name, expr_info)



def eval_full_bandmath_expr(expr_info_list: List[BandMathExprInfo], result_names_list: List[str], cache: DataCache,
            prepared_variables_list: List[Dict[str, Tuple[VariableType, BANDMATH_VALUE_TYPE]]],
            lower_functions: Dict[str, BandMathFunction], number_of_intermediates: int, tree: lark.ParseTree,
            use_synchronous_method = True, child_conn: mp_conn.Connection = None \
        ) -> List[Tuple[Union[VariableType, RasterDataSet.__class__],
                        Union[np.ndarray, RasterDataSet],
                        str,
                        BandMathExprInfo]]:
    '''
    This function is used to evaluate one band math expression. Now this expression may or may not be 
    an expression that has batching. If it does, then we will have to do the batching logic here.

    Return
        - A list of tuples. The first element in the tuple is the variable type or the RasterDataSet
        class. The second element is the actual value, which is either the numpy array or the
        RasterDataSet. The third element is the name of the resulting dataset. The fourth element
        is the expr_info for that dataset.
    '''
    assert len(expr_info_list) == len(prepared_variables_list), "The number of expr_info_list and prepared_variables_list must be the same"
    count = 0
    outputs: List[Tuple[RasterDataSet.__class__, RasterDataSet, str, BandMathExprInfo]] = []
    for lower_variables, expr_info, result_name in zip(prepared_variables_list, expr_info_list, result_names_list):
        count += 1
        child_conn.send(["progress", {"Numerator": count, "Denominator": len(prepared_variables_list), "Status": "Running"}])
        gdal_type = np_dtype_to_gdal(np.dtype(expr_info.elem_type))
        
        max_chunking_bytes, should_chunk = max_bytes_to_chunk(expr_info.result_size()*number_of_intermediates)
        logger.debug(f"Max chunking bytes: {max_chunking_bytes}")

        spectral_metadata = expr_info.spectral_metadata_source
        data_ignore_value = DEFAULT_IGNORE_VALUE
        if spectral_metadata is not None and spectral_metadata.get_data_ignore_value() is not None:
            data_ignore_value = spectral_metadata.get_data_ignore_value()

        if not use_synchronous_method or \
        (expr_info.result_type == VariableType.IMAGE_CUBE and should_chunk):
            try:
                error = None
                eval = BandMathEvaluatorAsync(lower_variables, lower_functions, expr_info.shape)
                bands = 1
                lines = 1
                samples = 1
                if len(expr_info.shape) == 2:
                    lines, samples = expr_info.shape
                elif len(expr_info.shape) == 3:
                    bands, lines, samples = expr_info.shape
                else:
                    raise RuntimeError(f"expr_info shape is neither 2 or 3, its {expr_info.shape}")
                # Gets the correct file path to make our temporary file
                result_path = get_unused_file_path_in_folder(TEMP_FOLDER_PATH, result_name)
                folder_path = os.path.dirname(result_path)
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)
                
                out_dataset_gdal = gdal.GetDriverByName('ENVI').Create(result_path, samples, lines, bands, gdal_type)
                # We declare the dataset write after so if any errors occur below,
                # the file gets destroyed (which happens in del of RasterDataSet)
                out_dataset = RasterDataLoader().dataset_from_gdal_dataset(out_dataset_gdal, cache)
                # We NO LONGER set the save state here. We must set it in the process that we pass
                # this piece of data to. If we set it here to IN_DISK_NOT_SAVED, then the garbage
                # collector will delete the underlying dataset when this process ends.
                out_dataset.set_dirty()
                
                # Based on memory limits (currently set in constants,, but we could make it more adjustable)
                # find the number of bands that we can access without exceeding it
                bytes_per_element = np.dtype(expr_info.elem_type).itemsize if expr_info.elem_type is not None else SCALAR_BYTES
                max_bytes = max_chunking_bytes/bytes_per_element
                max_bytes_per_intermediate = max_bytes / number_of_intermediates
                num_bands = int(np.floor(max_bytes_per_intermediate / (lines*samples)))
                num_bands = 1 if num_bands < 1 else num_bands

                writing_futures = []
                for band_index in range(0, bands, num_bands):
                    band_index_list_current = [band for band in range(band_index, band_index+num_bands) if band < bands]
                    band_index_list_next = [band for band in range(band_index+num_bands, band_index+2*num_bands) if band < bands]
                    # print(f"Min: {min(band_index_list_current)} | Max: {max(band_index_list_current)}")
                    
                    eval.index_list_current = band_index_list_current
                    eval.index_list_next = band_index_list_next
                    
                    result_value = eval.transform(tree)
                    if isinstance(result_value, (asyncio.Future, Coroutine)):
                        result_value = asyncio.run_coroutine_threadsafe(result_value, eval._event_loop).result()
                    res = result_value.value
                    
                    future = eval._write_thread_pool.submit(write_raster_to_dataset, \
                                                        out_dataset_gdal, band_index_list_current, \
                                                        res, gdal_type, default_ignore_value=data_ignore_value)
                    writing_futures.append(future)
                concurrent.futures.wait(writing_futures)
            except BaseException as e:
                error = e
                if eval is not None:
                    eval.stop()
            finally:
                eval.stop()
            if error is None:
                child_conn.send(["error", {"Result Name": result_name, "Message": None, "Traceback": None}])
                outputs.append((RasterDataSet, out_dataset, result_name, expr_info))
            else:
                child_conn.send(["error", {"Result Name": result_name, "Message": str(error), "Traceback": traceback.format_exc()}])
                outputs.append((None, None, result_name, expr_info))
        else:
            error = None
            try:
                eval = BandMathEvaluator(lower_variables, lower_functions)
                result_value = eval.transform(tree)
                res = result_value.value
            except BaseException as e:
                error = e
                if eval:
                    eval.stop()
            finally:
                eval.stop()
            if error is None:
                child_conn.send(["error", {"Result Name": result_name, "Message": None, "Traceback": None}])
                outputs.append((result_value.type, result_value.value, result_name, expr_info))
            else:
                child_conn.send(["error", {"Result Name": result_name, "Message": str(error), "Traceback": traceback.format_exc()}])
                outputs.append((None, None, result_name, expr_info))


    child_conn.send(["progress", {"Numerator": count, "Denominator": len(prepared_variables_list), "Status": "Finished"}])
    return outputs
