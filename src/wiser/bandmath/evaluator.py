import os
import logging

from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Coroutine

import lark
from lark import Visitor, Tree, Token, v_args, Discard
from lark.exceptions import VisitError, GrammarError
import inspect
import numpy as np

import atexit

import queue
import asyncio
import threading
import concurrent.futures

from .types import VariableType, BandMathValue, BandMathEvalError, BandMathExprInfo
from .functions import BandMathFunction, get_builtin_functions
from .utils import TEMP_FOLDER_PATH

from wiser.raster.dataset import RasterDataSet

from osgeo import gdal

from .builtins import (
    OperatorCompare,
    OperatorAdd, OperatorSubtract, OperatorMultiply, OperatorDivide,
    OperatorUnaryNegate, OperatorPower, OperatorAddOrig
    )

from wiser.raster.loader import RasterDataLoader
from wiser.raster.dataset_impl import SaveState

from .builtins.constants import MAX_RAM_BYTES, SCALAR_BYTES, NUM_READERS, \
    NUM_PROCESSORS, NUM_WRITERS, LHS_KEY, RHS_KEY

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

# Define the Visitor class to traverse the tree
class PositionVisitor(Visitor):
    def __init__(self):
        # Track if this is the first node being visited
        self.is_root = True
    def _update_metadata(self, node, rule_name):
        # Iterate over the children of the node, which is a Tree object
        for i, child in enumerate(node.children):
            # Only add metadata if the child has the 'meta' attribute (typically for Tree objects)
            if hasattr(child, 'meta'):
                if i == 0:
                    child.meta.position = 'LEFT'
                else:
                    child.meta.position = 'RIGHT'
            
            # Add rule_name only if it's a Tree node, not a Token
            if isinstance(child, Tree):
                child.rule_name = rule_name

            # Mark this node as root if it's the first node visited
        if self.is_root:
            node.meta.position = 'ROOT'
            self.is_root = False

        return node

    def comparison(self, node):
        return self._update_metadata(node, 'comparison')

    def add_expr(self, node):
        return self._update_metadata(node, 'add_expr')

    def mul_expr(self, node):
        return self._update_metadata(node, 'mul_expr')

    def unary_negate_expr(self, node):
        return self._update_metadata(node, 'unary_negate_expr')

    def power_expr(self, node):
        return self._update_metadata(node, 'power_expr')

logger = logging.getLogger(__name__)

def trace_lock_event(frame, event, arg):
    if event == 'lock':
        print(f"Thread {threading.current_thread().name} is attempting to acquire a lock")
    elif event == 'acquire':
        print(f"Thread {threading.current_thread().name} acquired a lock")
    elif event == 'release':
        print(f"Thread {threading.current_thread().name} released a lock")

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

    @v_args(meta=True)
    def comparison(self, meta, args):
        logger.debug(' * comparison')
        # if node_id:
        #     print(f"Node id <: {node_id}")
        lhs = args[0]
        oper = args[1]
        rhs = args[2]
        if node_id not in self._read_data_queue_dict:
            self._read_data_queue_dict[node_id] = queue.Queue()
        # It is okay if we don't want to use index with bands or spectrum 
        # because in each operator, index is only used with image cubes
        return OperatorCompare(oper).apply([lhs, rhs], self.index_list_current)

    def update_max_intermediates(self, lhs, rhs):
        current_intermediates = 0
        if isinstance(lhs, BandMathValue) and isinstance(rhs, BandMathValue):
            # Lhs and rhs should never be an intermediate
            if lhs.is_intermediate and rhs.is_intermediate:
                self.decrement_interm_running_total()
            elif lhs.is_intermediate or rhs.is_intermediate:
                self.decrement_interm_running_total()
            else:
                if lhs.type == VariableType.IMAGE_CUBE and rhs.type == VariableType.IMAGE_CUBE:
                    self.increment_interm_running_total()
                    self.increment_interm_running_total()
                elif lhs.type == VariableType.IMAGE_CUBE or rhs.type == VariableType.IMAGE_CUBE:
                    self.increment_interm_running_total()
        # Then one of them is an int
        elif isinstance(lhs, BandMathValue) and isinstance(rhs, int):
            if lhs.type != VariableType.IMAGE_CUBE:
                self.increment_interm_running_total()

        elif isinstance(lhs, BandMathValue) or isinstance(rhs, BandMathValue):
            x = 1
        if isinstance(lhs, BandMathValue):
            if lhs.is_intermediate:
                self.decrement_interm_running_total()
            elif lhs.type == VariableType.IMAGE_CUBE:
                self.increment_interm_running_total()
        if isinstance(rhs, BandMathValue):
            if rhs.is_intermediate:
                self.decrement_interm_running_total()
            elif rhs.type == VariableType.IMAGE_CUBE:
                self.increment_interm_running_total()
        else:
            assert(isinstance(lhs, int))
            assert(isinstance(rhs, int))
            print(f"About to decrement running total, current interm running total interm: {self._intermediate_running_total}")
            self.decrement_interm_running_total()
            self.decrement_interm_running_total()
            # print(f"Decrementing running total, current running total amt: {self._intermediate_running_total}")
        # print(f"Current max intermediates: {self._max_intermediates} at node: {node_id}")


    @v_args(meta=True)
    def add_expr(self, meta, values):
        '''
        Implementation of addition and subtraction operations in the
        transformer.
        '''
        logger.debug(' * add_expr')
        # print(f"!!!!!!!!!!!!VALUES!!!!!!!! \n {values}")
        # print("====New add expr===")
        child_type = getattr(meta, 'position', None)
        node_id = getattr(meta, 'unique_id', None)
        # if node_id:
        #     print(f"node_id: {node_id}")
        # if child_type:
        #     print(f"child_type +: {child_type}")
        lhs = values[0]
        oper = values[1]
        rhs = values[2]

        self.update_max_intermediates(lhs, rhs)
        return self._max_intermediates

        # raise RuntimeError(f'Unexpected operator {oper}')


    @v_args(meta=True)
    def mul_expr(self, meta, args):
        '''
        Implementation of multiplication and division operations in the
        transformer.
        '''
        logger.debug(' * mul_expr')
        node_id = getattr(meta, 'unique_id', None)
        if node_id not in self._read_data_queue_dict:
            self._read_data_queue_dict[node_id] = queue.Queue()
        lhs = args[0]
        oper = args[1]
        rhs = args[2]

        if oper == '*':
            return OperatorMultiply().apply([lhs, rhs], self.index_list_current)

        elif oper == '/':
            return OperatorDivide().apply([lhs, rhs], self.index_list_current)

        raise RuntimeError(f'Unexpected operator {oper}')


    @v_args(meta=True)
    def power_expr(self, meta, args):
        '''
        Implementation of power operation in the transformer.
        '''
        logger.debug(' * power_expr')
        node_id = getattr(meta, 'unique_id', None)
        if node_id not in self._read_data_queue_dict:
            self._read_data_queue_dict[node_id] = queue.Queue()
        # if node_id:
        #     print(f"Node id **: {node_id}")
        return OperatorPower().apply([args[0], args[1]], self.index_list_current)


    @v_args(meta=True)
    def unary_negate_expr(self, meta, args):
        '''
        Implementation of unary negation in the transformer.
        '''
        logger.debug(' * unary_negate_expr')
        node_id = getattr(meta, 'unique_id', None)
        if node_id not in self._read_data_queue_dict:
            self._read_data_queue_dict[node_id] = queue.Queue()
        # if node_id:
        #     print(f"Node id -: {node_id}")
        # args[0] is the '-' character
        return OperatorUnaryNegate().apply([args[1]], self.index_list_current)


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
            self._read_thread_pool_rhs = concurrent.futures.ThreadPoolExecutor(max_workers=NUM_READERS)
            self._write_thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=NUM_WRITERS)
            self._event_loop = asyncio.new_event_loop()
            self._loop_thread = threading.Thread(target=self._event_loop.run_forever, daemon=False)
            self._loop_thread.start()
        else:
            self._read_data_queue = None
            self._write_data_queue = None

        # Dictionary that maps position in tree to queue
        # position so we don't have to have different queues for
        # different trees. The node in the tree wil be able to 
        # access the data it needs from the dictionary. The 
        # data will be the queue and the thread or process pool executor

    def get_node_id(self, node_meta):
        '''
        Generates a unique ID for a given node based on its meta attribute (position in the tree).
        '''
        # Create a unique key based on the line and column position in the source
        node_key = (node_meta.line, node_meta.column)
        
        # If the node doesn't have an ID, create one and store it
        if node_key not in self.node_ids:
            self.node_ids[node_key] = len(self.node_ids) + 1

        return self.node_ids[node_key]

    @v_args(meta=True)
    def comparison(self, meta, args):
        logger.debug(' * comparison')
        node_id = getattr(meta, 'unique_id', None)
        if node_id not in self._read_data_queue_dict:
            self._read_data_queue_dict[node_id] = queue.Queue()
        # if node_id:
        #     print(f"Node id <: {node_id}")
        lhs = args[0]
        oper = args[1]
        rhs = args[2]
        if node_id not in self._read_data_queue_dict:
            self._read_data_queue_dict[node_id] = queue.Queue()
        # It is okay if we don't want to use index with bands or spectrum 
        # because in each operator, index is only used with image cubes
        return OperatorCompare(oper).apply([lhs, rhs], self.index_list_current)

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
        # if node_id:
        #     print(f"Node id <: {node_id}")

        lhs = values[0]
        oper = values[1]
        rhs = values[2]
        # print(f"type(lhs): {type(lhs)} for node {node_id}")
        # print(f"type(rhs): {type(rhs)} for node {node_id}")

        if oper == '+':
            # Schedule this operation as a background task
            addition_task = asyncio.ensure_future(OperatorAdd().apply(
                [lhs, rhs],
                self.index_list_current,
                self.index_list_next,
                self._read_data_queue_dict[node_id],
                self._read_thread_pool,
                self._event_loop,
            ))
            return await addition_task

        elif oper == '-':
            return OperatorSubtract().apply([lhs, rhs], self.index_list_current)

        raise RuntimeError(f'Unexpected operator {oper}')


    @v_args(meta=True)
    def mul_expr(self, meta, args):
        '''
        Implementation of multiplication and division operations in the
        transformer.
        '''
        logger.debug(' * mul_expr')
        node_id = getattr(meta, 'unique_id', None)
        if node_id not in self._read_data_queue_dict:
            self._read_data_queue_dict[node_id] = queue.Queue()
        # if node_id:
        #     print(f"Node id *: {node_id}")
        lhs = args[0]
        oper = args[1]
        rhs = args[2]

        if oper == '*':
            return OperatorMultiply().apply([lhs, rhs], self.index_list_current)

        elif oper == '/':
            return OperatorDivide().apply([lhs, rhs], self.index_list_current)

        raise RuntimeError(f'Unexpected operator {oper}')


    @v_args(meta=True)
    def power_expr(self, meta, args):
        '''
        Implementation of power operation in the transformer.
        '''
        logger.debug(' * power_expr')
        node_id = getattr(meta, 'unique_id', None)
        if node_id not in self._read_data_queue_dict:
            self._read_data_queue_dict[node_id] = queue.Queue()
        # if node_id:
        #     print(f"Node id **: {node_id}")
        return OperatorPower().apply([args[0], args[1]], self.index_list_current)


    @v_args(meta=True)
    def unary_negate_expr(self, meta, args):
        '''
        Implementation of unary negation in the transformer.
        '''
        logger.debug(' * unary_negate_expr')
        node_id = getattr(meta, 'unique_id', None)
        if node_id not in self._read_data_queue_dict:
            self._read_data_queue_dict[node_id] = queue.Queue()
        # if node_id:
        #     print(f"Node id -: {node_id}")
        # args[0] is the '-' character
        return OperatorUnaryNegate().apply([args[1]], self.index_list_current)


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

class BandMathEvaluatorSync(lark.visitors.Transformer):
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

    def get_node_id(self, node_meta):
        '''
        Generates a unique ID for a given node based on its meta attribute (position in the tree).
        '''
        # Create a unique key based on the line and column position in the source
        node_key = (node_meta.line, node_meta.column)
        
        # If the node doesn't have an ID, create one and store it
        if node_key not in self.node_ids:
            self.node_ids[node_key] = len(self.node_ids) + 1

        return self.node_ids[node_key]

    @v_args(meta=True)
    def comparison(self, meta, args):
        logger.debug(' * comparison')
        node_id = getattr(meta, 'unique_id', None)
        if node_id not in self._read_data_queue_dict:
            self._read_data_queue_dict[node_id] = queue.Queue()
        # if node_id:
        #     print(f"Node id <: {node_id}")
        lhs = args[0]
        oper = args[1]
        rhs = args[2]
        if node_id not in self._read_data_queue_dict:
            self._read_data_queue_dict[node_id] = queue.Queue()
        # It is okay if we don't want to use index with bands or spectrum 
        # because in each operator, index is only used with image cubes
        return OperatorCompare(oper).apply([lhs, rhs], self.index_list_current)

    @v_args(meta=True)
    def add_expr(self, meta, values):
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
            return asyncio.run_coroutine_threadsafe(OperatorAdd().apply([lhs, rhs], self.index_list_current, self.index_list_next,
                                        self._read_data_queue_dict[node_id], self._read_thread_pool, \
                                        event_loop=self._event_loop, node_id=node_id), loop=self._event_loop).result()

        elif oper == '-':
            return OperatorSubtract().apply([lhs, rhs], self.index_list_current)

        raise RuntimeError(f'Unexpected operator {oper}')


    @v_args(meta=True)
    def mul_expr(self, meta, args):
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
            return asyncio.run_coroutine_threadsafe(OperatorMultiply().apply([lhs, rhs], self.index_list_current, self.index_list_next,
                                        self._read_data_queue_dict[node_id], self._read_thread_pool, \
                                        event_loop=self._event_loop), loop=self._event_loop).result()

        elif oper == '/':
            return asyncio.run_coroutine_threadsafe(OperatorDivide().apply([lhs, rhs], self.index_list_current, self.index_list_next,
                                        self._read_data_queue_dict[node_id], self._read_thread_pool, \
                                        event_loop=self._event_loop), loop=self._event_loop).result()

        raise RuntimeError(f'Unexpected operator {oper}')


    @v_args(meta=True)
    def power_expr(self, meta, args):
        '''
        Implementation of power operation in the transformer.
        '''
        logger.debug(' * power_expr')
        node_id = getattr(meta, 'unique_id', None)
        if node_id not in self._read_data_queue_dict:
            self._read_data_queue_dict[node_id] = queue.Queue()
        # if node_id:
        #     print(f"Node id **: {node_id}")
        return OperatorPower().apply([args[0], args[1]], self.index_list_current)


    @v_args(meta=True)
    def unary_negate_expr(self, meta, args):
        '''
        Implementation of unary negation in the transformer.
        '''
        logger.debug(' * unary_negate_expr')
        node_id = getattr(meta, 'unique_id', None)
        if node_id not in self._read_data_queue_dict:
            self._read_data_queue_dict[node_id] = queue.Queue()
        # if node_id:
        #     print(f"Node id -: {node_id}")
        # args[0] is the '-' character
        return OperatorUnaryNegate().apply([args[1]], self.index_list_current)


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
        print("!!!!!!!!!!!!!!!!!!!Destructor called, cleaning up event loop and thread...!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        self.stop()  # Ensure the loop and thread are stopped
        print("Event loop and thread cleaned up")

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
            return OperatorAddOrig().apply([lhs, rhs])

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

    print(f"GDAL Python Version: {gdal.__version__}")
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
    '''
    Okay so I think to make this parallel, in the tree when a data piece is retrieve we actually retrieve that data piece from the queue. But where is the queue? The queue could be stored in the evaluator and passed
    in to any of the Operators. Also we could have a processor class in BandMathEvaluator that spawns a ProcessPoolExecutor where we can run the operations on. Everytime a node of the tree is processed, we will add that Operator
    to the process, maybe wrapped in a function. Then writing to disk we can simply do as is done in the chatpgt code.
    '''
    logger.debug('Beginning band-math evaluation')
    # child_assigner = PositionVisitor()
    # child_assigner.visit(tree)
    id_assigner = UniqueIDAssigner()
    id_assigner.visit(tree)
    print(f"TREE: \n")
    # print_tree_with_positions(tree)
    # print('========')
    print_tree_with_meta(tree)

    numInterFinder = NumberOfIntermediatesFinder(lower_variables, lower_functions, expr_info.shape)
    number_of_intermediates = numInterFinder.transform(tree)
    print(f"Number of intermediates: {number_of_intermediates}")
    # print(f"TREE: \n {tree}")
    # print_tree_with_positions(tree)
    # print_tree_with_meta(tree)

    def write_band(out_dataset_gdal, gdal_band_index, band_index, res):
        band_to_write = None
        if len(band_index_list_current) == 1:
            band_to_write = np.squeeze(res)
        else:
            band_to_write = res[gdal_band_index-band_index]
        band = out_dataset_gdal.GetRasterBand(gdal_band_index+1)
        band.WriteArray(band_to_write)
        band.FlushCache()
        return True

    def write_raster(out_dataset_gdal, band_index_list_current, result):
        print("[[[[[[[[[[[[[[[[[[[ABOUT TO WRITE]]]]]]]]]]]]]]]]]]]")
        gdal_band_list_current = [band+1 for band in band_index_list_current]
        # Write queue data to be written all at once
        out_dataset_gdal.WriteRaster(
            0, 0, out_dataset_gdal.RasterXSize, out_dataset_gdal.RasterYSize,
            result.tobytes(),
            buf_xsize = out_dataset_gdal.RasterXSize, buf_ysize=out_dataset_gdal.RasterYSize,
            buf_type=gdal.GDT_Float32,
            band_list=gdal_band_list_current
        )
        out_dataset_gdal.FlushCache()
        print("(((((((((((((((((((((FINISHED FLUSHING)))))))))))))))))))))")

    from memory_profiler import memory_usage
    if expr_info.result_type == VariableType.IMAGE_CUBE and not use_old_method:
        eval = None
        try:
            eval = BandMathEvaluatorSync(lower_variables, lower_functions, expr_info.shape)

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
            out_dataset_gdal = gdal.GetDriverByName('ENVI').Create(result_path, samples, lines, bands, gdal.GDT_Float32)
            # out_dataset_gdal.FlushCache()
            # out_dataset_gdal = None

            # flags = gdal.OF_RASTER | gdal.OF_READONLY
            # out_dataset_gdal = gdal.OpenEx(result_path, flags)
            # if out_dataset_gdal is not None:
            #     is_thread_safe = out_dataset_gdal.GetMetadataItem("THREAD_SAFE", "OPEN_CONFIG")
            #     print(f"GDAL Dataset is thread-safe? {is_thread_safe}")
            # else:
            #     print("Failed to reopen dataset with GDAL_OF_THREADSAFE flag.")
            bytes_per_scalar = SCALAR_BYTES
            max_bytes = MAX_RAM_BYTES/bytes_per_scalar
            max_bytes_per_intermediate = max_bytes / 2
            num_bands = int(np.floor(max_bytes_per_intermediate / (lines*samples)))
            writing_futures = []
            memory_before = memory_usage()[0]
            max_memory_used = 0
            for band_index in range(0, bands, num_bands):
                band_index_list_current = [band for band in range(band_index, band_index+num_bands) if band < bands]
                band_index_list_next = [band for band in range(band_index+num_bands, band_index+2*num_bands) if band < bands]
                eval.index_list_current = band_index_list_current
                eval.index_list_next = band_index_list_next
                print(f"Max/min of band_index_list_next| min: {min(band_index_list_current)} & len: {len(band_index_list_current)}, max: {max(band_index_list_current)}  & len: {len(band_index_list_next)}")
                result_value = eval.transform(tree)
                memory_usage_during = memory_usage()[0]
                curr_memory_used = memory_usage_during-memory_before
                if curr_memory_used > max_memory_used:
                    max_memory_used = curr_memory_used
                    print(f"==========NEW MAX MEMORY USED: {max_memory_used} MB============")
                print(f';;;;;;;;;;;;;;; type of result_value before: {type(result_value)}')
                if isinstance(result_value, (asyncio.Future, Coroutine)):
                    result_value = asyncio.run_coroutine_threadsafe(eval.transform(tree), eval._event_loop).result()
                # print(f';;;;;;;;;;;;;;; type of result_vZalue after: {type(result_value)}')
                # result_value = result_value
                res = result_value.value
                # print("---------------------------EVAL RESULTS---------------------------")
                # print(f"eval res: \n {res}")
                if isinstance(result_value.value, np.ma.MaskedArray):
                    if not np.issubdtype(res.dtype, np.floating):
                        res = res.astype(np.float32)
                    res[result_value.value.mask] = np.nan

                # once everything is returned, then we add the result
                # to the queue to be written.
                # We add a function to the 
                # thread pool executor (that we can just define in that function's
                # if statement) that pops stuff from the to-be-written queue
                # and then writes everything to disk  asynchronously

                # print("Writing")
                assert (res.shape[0] == out_dataset_gdal.RasterXSize, \
                        res.shape[1] == out_dataset_gdal.RasterYSize)
                # # Write queue data to be written all at once
                # gdal_band_list_current = [band+1 for band in band_index_list_current]
                # out_dataset_gdal.WriteRaster(
                #     0, 0, out_dataset_gdal.RasterXSize, out_dataset_gdal.RasterYSize,
                #     res.tobytes(),
                #     buf_xsize = out_dataset_gdal.RasterXSize, buf_ysize=out_dataset_gdal.RasterYSize,
                #     buf_type=gdal.GDT_Float32,
                #     band_list=gdal_band_list_current
                # )
                # out_dataset_gdal.FlushCache()

                # future = eval._event_loop.run_in_executor(eval._write_thread_pool, write_raster, \
                #                                     out_dataset_gdal, band_index_list_current, res)
                future = eval._write_thread_pool.submit(write_raster, \
                                                    out_dataset_gdal, band_index_list_current, \
                                                    res)
                writing_futures.append(future)
            concurrent.futures.wait(writing_futures)
            # for future in writing_futures:
            #     print("waiting for future")
            #     waiting_result = future.result()
            # concurrent.futures.wait(writing_futures)
            # eval._event_loop.run_until_complete(asyncio.gather(*writing_futures))
        except BaseException as e:
            if eval is not None:
                eval.stop()
                raise e
        finally:
            print(f"==========NEW MAX MEMORY USED: THROUGHOUT PROCESS {max_memory_used} MB============")
        concurrent.futures.wait(writing_futures)
        print(f"DONE WRITING FUTURES")
        # print(f"---------------------------Out dataset data:---------------------------")
        # print(f"{out_dataset_gdal.ReadAsArray()}")
            # Loop through band index list and add a write task to the executor
            # for gdal_band_index in band_index_list_current:
            #     future = eval._write_thread_pool.submit(write_band, \
            #                                      out_dataset_gdal, gdal_band_index, band_index, res)
            #     writing_futures.append(future)
            # concurrent.futures.wait(writing_futures)
                # band_to_write = None
                # if len(band_index_list_current) == 1:
                #     band_to_write = np.squeeze(res)
                # else:
                #     band_to_write = res[gdal_band_index-band_index]
                # band = out_dataset_gdal.GetRasterBand(gdal_band_index+1)
                # band.WriteArray(band_to_write)
                # band.FlushCache()
        eval.stop()
        out_dataset = RasterDataLoader().dataset_from_gdal_dataset(out_dataset_gdal)
        out_dataset.set_save_state(SaveState.IN_DISK_NOT_SAVED)
        out_dataset.set_dirty()

        return (RasterDataSet, out_dataset)
    else:
        print("OLD METHOD")
        try:
            memory_before = memory_usage()[0]
            eval = BandMathEvaluator(lower_variables, lower_functions)
            result_value = eval.transform(tree)
            memory_usage_during = memory_usage()[0]
            memory_used = memory_usage_during-memory_before
            print(f"OLD METHOD MEMORY USED: {memory_used}")
            if isinstance(result_value, Coroutine): 
                result_value = \
                    asyncio.run_coroutine_threadsafe(result_value, eval._event_loop).result()
        except BaseException as e:
            raise e
        return (result_value.type, result_value.value)

def print_tree_with_meta(tree, indent=0):
    indent_str = "  " * indent
    if isinstance(tree, Tree):
        # Print the node type and its meta information if present
        meta_info = ""
        if hasattr(tree, 'meta') and tree.meta is not None:
            meta_info = f"(unique_id: {getattr(tree.meta, 'unique_id', 'N/A')})"
        print(f"{indent_str}{tree.data} {meta_info}")
        # Recursively print children nodes
        for child in tree.children:
            print_tree_with_meta(child, indent + 1)
    else:
        # If it's a terminal node (e.g., a token), print its value and its meta if available
        meta_info = ""
        if hasattr(tree, 'unique_id') or hasattr(tree, 'LEFT'):
            meta_info = f"(unique_id: {getattr(tree, 'unique_id', 'N/A')})"
        print(f"{indent_str}{tree} {meta_info} (Terminal)")
        
def print_tree_with_positions(node, indent=0):
    """
    Recursively prints the tree with position metadata for each node.
    """
    # Determine the metadata for the current node
    meta_info = ""
    if hasattr(node, 'meta') and hasattr(node.meta, 'position'):
        meta_info = f" [position={node.meta.position}]"
        
    # Display node information
    if isinstance(node, Tree):
        rule_info = f" ({node.rule_name})" if hasattr(node, 'rule_name') else ""
        print("  " * indent + f"Tree: {node.data}{rule_info}{meta_info}")
        
        # Print children nodes
        for child in node.children:
            print_tree_with_positions(child, indent + 1)
    elif isinstance(node, Token):
        print("  " * indent + f"Token: {node.type}='{node}'{meta_info}")
