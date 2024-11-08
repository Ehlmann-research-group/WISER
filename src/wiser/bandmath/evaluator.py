import os
import logging
import inspect

from typing import Any, Callable, Dict, Tuple, Coroutine

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

from wiser.raster.dataset import RasterDataSet

from osgeo import gdal

from .builtins import (
    OperatorCompare,
    OperatorAdd, OperatorSubtract, OperatorMultiply, OperatorDivide,
    OperatorUnaryNegate, OperatorPower,
    )

from wiser.raster.loader import RasterDataLoader
from wiser.raster.dataset_impl import SaveState

from .builtins.constants import SCALAR_BYTES, NUM_WRITERS, DEFAULT_IGNORE_VALUE, NUM_READERS, LHS_KEY, RHS_KEY

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
        # task_list = [child async for child in self._transform_children(tree.children)]
        children_tasks = [asyncio.create_task(self._transform_children(tree.children))]# for c in tree.children] #  [child async for child in self._transform_children(tree.children)]
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
    def __init__(self, variables: Dict[str, Tuple[VariableType, Any]],
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
        print("Eval operator event loop and thread cleaned up")

class BandMathEvaluator(lark.visitors.Transformer):
    '''
    A Lark Transformer for evaluating band-math expressions.
    '''
    def __init__(self, variables: Dict[str, Tuple[VariableType, Any]],
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
        print("Eval operator event loop cleaned up")

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

    id_assigner = UniqueIDAssigner()
    id_assigner.visit(tree)

    numInterFinder = NumberOfIntermediatesFinder(lower_variables, lower_functions, expr_info.shape)
    numInterFinder.transform(tree)
    number_of_intermediates = numInterFinder.get_max_intermediates()
    logger.debug(f'Number of intermediates: {number_of_intermediates}')

    gdal_type = np_dtype_to_gdal(np.dtype(expr_info.elem_type))
    
    max_chunking_bytes = max_bytes_to_chunk(expr_info.result_size()*number_of_intermediates)
    logger.debug(f"Max chunking bytes: {max_chunking_bytes}")

    if expr_info.result_type == VariableType.IMAGE_CUBE and max_chunking_bytes is not None and not use_old_method:
        try:
            print(f"New method")
            eval = BandMathEvaluatorAsync(lower_variables, lower_functions, expr_info.shape)

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
                assert (res.shape[0] == out_dataset_gdal.RasterXSize, \
                        res.shape[1] == out_dataset_gdal.RasterYSize)
                
                future = eval._write_thread_pool.submit(write_raster_to_dataset, \
                                                    out_dataset_gdal, band_index_list_current, \
                                                    res, gdal_type)
                writing_futures.append(future)
            concurrent.futures.wait(writing_futures)
        except BaseException as e:
            if eval is not None:
                eval.stop()
            raise e
        finally:
            eval.stop()
        correct_data_ignore_val = get_valid_ignore_value(out_dataset_gdal, DEFAULT_IGNORE_VALUE)
        out_dataset.set_data_ignore_value(correct_data_ignore_val)
        return (RasterDataSet, out_dataset)
    else:
        print(f"Old method")
        try:
            eval = BandMathEvaluator(lower_variables, lower_functions)
            result_value = eval.transform(tree)
            res = result_value.value
        except BaseException as e:
            eval.stop()
            raise e
        finally:
            eval.stop()
        return (result_value.type, result_value.value)
