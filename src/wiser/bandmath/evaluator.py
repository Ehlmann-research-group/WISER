import os
import logging
import inspect

from typing import (
    Callable,
    Dict,
    Tuple,
    Coroutine,
    Union,
    List,
    TYPE_CHECKING,
    Optional,
    Iterator,
)
from dataclasses import dataclass, field

import lark
from lark import Visitor, Tree, Token, v_args, ParseTree
from lark.exceptions import VisitError, GrammarError
import numpy as np

import queue
import asyncio
import threading
import concurrent.futures

from .types import VariableType, BandMathValue, BandMathEvalError, BandMathExprInfo
from .functions import BandMathFunction, get_builtin_functions
from .utils import (
    TEMP_FOLDER_PATH,
    get_unused_file_path_in_folder,
    np_dtype_to_gdal,
    write_raster_to_dataset,
    max_bytes_to_chunk,
    BandMathResultInfo,
)

from wiser import bandmath
from wiser.bandmath.types import BANDMATH_VALUE_TYPE, BANDMATH_SERIALIZED_TYPE

from wiser.raster.serializable import Serializable, SerializedForm
from wiser.raster.data_cache import DataCache
from wiser.raster.dataset import (
    RasterDataSet,
    RasterDataBatchBand,
    RasterDataDynamicBand,
    RasterBand,
)
from wiser.raster.loader import RasterDataLoader
from wiser.raster.serializable import BasicValueSerialized
from wiser.raster.spectrum import Spectrum

from wiser.gui.subprocessing_manager import ProcessManager

from osgeo import gdal
import multiprocessing as mp
import multiprocessing.connection as mp_conn

from .builtins import (
    OperatorCompare,
    OperatorAdd,
    OperatorSubtract,
    OperatorMultiply,
    OperatorDivide,
    OperatorUnaryNegate,
    OperatorPower,
)


from .builtins.constants import (
    SCALAR_BYTES,
    NUM_WRITERS,
    DEFAULT_IGNORE_VALUE,
    NUM_READERS,
    LHS_KEY,
    RHS_KEY,
)

import traceback

if TYPE_CHECKING:
    from wiser.gui.app_state import ApplicationState
    from concurrent.futures import Future

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
            return await self.__default__(
                tree.data, children, tree.meta
            )  # Ensure we await the default method if overridden
        else:
            try:
                wrapper = getattr(f, "visit_wrapper", None)
                if wrapper is not None:
                    return await f.visit_wrapper(f, tree.data, children, tree.meta)
                else:
                    # Check if the transformation method is async or sync
                    res = f(children)
                    if inspect.isawaitable(res):
                        return await res
                    else:
                        return res
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
            return await self.__default_token__(
                token
            )  # Ensure we await the default token method if overridden
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
                child_tasks.append(
                    asyncio.create_task(asyncio.sleep(0, c))
                )  # Wrap raw values as completed tasks

        # Await all child tasks concurrently and gather results into a list
        transformed_children = await asyncio.gather(*child_tasks)
        return transformed_children

    async def _transform_tree(self, tree):
        """
        Asynchronously transform a tree node.
        This function recursively transforms the children first, and then calls the
        transformation method for the node.
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
    """
    A Lark Transformer for evaluating band-math expressions.
    """

    def __init__(
        self,
        variables: Dict[str, Tuple[VariableType, BANDMATH_VALUE_TYPE]],
        functions: Dict[str, Callable],
        shape: Tuple[int, int, int] = None,
        use_async_io=True,
    ):
        self._variables = variables
        self._functions = functions
        self.index_list_current = None
        self.index_list_next = None
        self._shape = shape
        if use_async_io:
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
        logger.debug(" * comparison")
        node_id = getattr(meta, "unique_id", None)
        if node_id not in self._read_data_queue_dict:
            self._read_data_queue_dict[node_id] = {}
            self._read_data_queue_dict[node_id][LHS_KEY] = queue.Queue()
            self._read_data_queue_dict[node_id][RHS_KEY] = queue.Queue()

        lhs = args[0]
        oper = args[1]
        rhs = args[2]

        # Schedule this operation as a background task
        addition_task = asyncio.ensure_future(
            OperatorCompare(oper).apply(
                [lhs, rhs],
                self.index_list_current,
                self.index_list_next,
                self._read_data_queue_dict[node_id],
                self._read_thread_pool,
                self._event_loop,
                node_id,
            )
        )
        return await addition_task

    @v_args(meta=True)
    async def add_expr(self, meta, values):
        """
        Implementation of addition and subtraction operations in the
        transformer.
        """
        logger.debug(" * add_expr")
        node_id = getattr(meta, "unique_id", None)
        if node_id not in self._read_data_queue_dict:
            self._read_data_queue_dict[node_id] = {}
            self._read_data_queue_dict[node_id][LHS_KEY] = queue.Queue()
            self._read_data_queue_dict[node_id][RHS_KEY] = queue.Queue()

        lhs = values[0]
        oper = values[1]
        rhs = values[2]

        if oper == "+":
            # Schedule this operation as a background task
            addition_task = asyncio.ensure_future(
                OperatorAdd().apply(
                    [lhs, rhs],
                    self.index_list_current,
                    self.index_list_next,
                    self._read_data_queue_dict[node_id],
                    self._read_thread_pool,
                    self._event_loop,
                    node_id,
                )
            )
            return await addition_task

        elif oper == "-":
            # Schedule this operation as a background task
            addition_task = asyncio.ensure_future(
                OperatorSubtract().apply(
                    [lhs, rhs],
                    self.index_list_current,
                    self.index_list_next,
                    self._read_data_queue_dict[node_id],
                    self._read_thread_pool,
                    self._event_loop,
                    node_id,
                )
            )
            return await addition_task

        raise RuntimeError(f"Unexpected operator {oper}")

    @v_args(meta=True)
    async def mul_expr(self, meta, args):
        """
        Implementation of multiplication and division operations in the
        transformer.
        """
        logger.debug(" * mul_expr")
        node_id = getattr(meta, "unique_id", None)
        if node_id not in self._read_data_queue_dict:
            self._read_data_queue_dict[node_id] = {}
            self._read_data_queue_dict[node_id][LHS_KEY] = queue.Queue()
            self._read_data_queue_dict[node_id][RHS_KEY] = queue.Queue()

        lhs = args[0]
        oper = args[1]
        rhs = args[2]

        if oper == "*":
            # Schedule this operation as a background task
            addition_task = asyncio.ensure_future(
                OperatorMultiply().apply(
                    [lhs, rhs],
                    self.index_list_current,
                    self.index_list_next,
                    self._read_data_queue_dict[node_id],
                    self._read_thread_pool,
                    self._event_loop,
                    node_id,
                )
            )
            return await addition_task

        elif oper == "/":
            # Schedule this operation as a background task
            addition_task = asyncio.ensure_future(
                OperatorDivide().apply(
                    [lhs, rhs],
                    self.index_list_current,
                    self.index_list_next,
                    self._read_data_queue_dict[node_id],
                    self._read_thread_pool,
                    self._event_loop,
                    node_id,
                )
            )
            return await addition_task

        raise RuntimeError(f"Unexpected operator {oper}")

    @v_args(meta=True)
    async def power_expr(self, meta, args):
        """
        Implementation of power operation in the transformer.
        """
        logger.debug(" * power_expr")
        node_id = getattr(meta, "unique_id", None)
        if node_id not in self._read_data_queue_dict:
            self._read_data_queue_dict[node_id] = {}
            self._read_data_queue_dict[node_id][LHS_KEY] = queue.Queue()
            self._read_data_queue_dict[node_id][RHS_KEY] = queue.Queue()

        addition_task = asyncio.ensure_future(
            OperatorPower().apply(
                [args[0], args[1]],
                self.index_list_current,
                self.index_list_next,
                self._read_data_queue_dict[node_id],
                self._read_thread_pool,
                self._event_loop,
                node_id,
            )
        )
        return await addition_task

    @v_args(meta=True)
    async def unary_negate_expr(self, meta, args):
        """
        Implementation of unary negation in the transformer.
        """
        logger.debug(" * unary_negate_expr")
        node_id = getattr(meta, "unique_id", None)
        if node_id not in self._read_data_queue_dict:
            self._read_data_queue_dict[node_id] = {}
            self._read_data_queue_dict[node_id][LHS_KEY] = queue.Queue()

        addition_task = asyncio.ensure_future(
            OperatorUnaryNegate().apply(
                [args[1]],
                self.index_list_current,
                self.index_list_next,
                self._read_data_queue_dict[node_id],
                self._read_thread_pool,
                self._event_loop,
                node_id,
            )
        )
        return await addition_task

    def true(self, args):
        """Returns a BandMathValue of True."""
        logger.debug(" * true")
        return BandMathValue(VariableType.BOOLEAN, True, computed=False)

    def false(self, args):
        """Returns a BandMathValue of False."""
        logger.debug(" * false")
        return BandMathValue(VariableType.BOOLEAN, False, computed=False)

    def number(self, args):
        """Returns a BandMathValue containing a specific number."""
        logger.debug(f" * number {args[0]}")
        return args[0]

    def string(self, args):
        """Returns a BandMathValue containing a specific string."""
        logger.debug(f' * string "{args[0]}"')
        return args[0]

    def variable(self, args) -> BandMathValue:
        """
        Returns a BandMathValue containing the value of the specified variable.
        """
        logger.debug(" * variable")
        name = args[0]
        if name not in self._variables or self._variables[name][1] is None:
            raise BandMathEvalError(f'Variable "{name}" is unspecified')
        (type, value) = self._variables[name]
        return BandMathValue(type, value, computed=False)

    def named_expression(self, args) -> BandMathValue:
        """
        Named expressions can appear in function arguments.
        """
        logger.debug(" * named_expression")
        # The first argument is the name, and the second argument is a
        # BandMathValue object holding the result of the expression evaluation.
        # Set the name and return the object.
        value = args[1]
        value.set_name(args[0])
        return value

    def function(self, args) -> BandMathValue:
        """
        Calls the function named in args[0], passing it args[1:], and returns
        the result as a BandMathValue.
        """
        logger.debug(" * function")
        func_name = args[0]
        func_args = args[1:]

        has_named_args = False
        for fa in func_args:
            if fa.name is None:
                if has_named_args:
                    raise BandMathEvalError(
                        "Named arguments must be " "specified after all positional arguments"
                    )
            else:
                has_named_args = True

        if func_name not in self._functions:
            raise BandMathEvalError(f'Unrecognized function "{func_name}"')

        func_impl = self._functions[func_name]
        return func_impl.apply(func_args)

    def NAME(self, token) -> str:
        """
        Parse a token as a string variable name.  The variable name is converted
        to lowercase.
        """
        logger.debug(" * NAME")
        return str(token).lower()

    def NUMBER(self, token) -> BandMathValue:
        """
        Parse a token as a number.  The number is represented as a Python float,
        and is wrapped in a BandMathValue object.
        """
        logger.debug(" * NUMBER")
        return BandMathValue(VariableType.NUMBER, float(token), computed=False)

    def STRING(self, token) -> str:
        """
        Parse a token as a string literal.  The variable name is converted
        to lowercase.
        """
        logger.debug(" * STRING")
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
    """
    A Lark Transformer for evaluating band-math expressions.
    """

    def __init__(
        self,
        variables: Dict[str, Tuple[VariableType, BANDMATH_VALUE_TYPE]],
        functions: Dict[str, Callable],
    ):
        self._variables = variables
        self._functions = functions
        self.index_list = None
        self._event_loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(target=self._event_loop.run_forever, daemon=False)
        self._loop_thread.start()

    def comparison(self, args):
        logger.debug(" * comparison")
        lhs = args[0]
        oper = args[1]
        rhs = args[2]

        # Since we do not await the future, this is effectively synchronous
        future = asyncio.run_coroutine_threadsafe(
            OperatorCompare(oper).apply([lhs, rhs], self.index_list), self._event_loop
        )
        return future.result()

    def add_expr(self, values):
        """
        Implementation of addition and subtraction operations in the
        transformer.
        """
        logger.debug(" * add_expr")
        lhs = values[0]
        oper = values[1]
        rhs = values[2]

        # Since we do not await the future, this is effectively synchronous
        if oper == "+":
            future = asyncio.run_coroutine_threadsafe(
                OperatorAdd().apply([lhs, rhs], self.index_list), self._event_loop
            )
            return future.result()

        elif oper == "-":
            future = asyncio.run_coroutine_threadsafe(
                OperatorSubtract().apply([lhs, rhs], self.index_list), self._event_loop
            )
            return future.result()

        raise RuntimeError(f"Unexpected operator {oper}")

    def mul_expr(self, args):
        """
        Implementation of multiplication and division operations in the
        transformer.
        """
        logger.debug(" * mul_expr")
        lhs = args[0]
        oper = args[1]
        rhs = args[2]

        # Since we do not await the future, this is effectively synchronous
        if oper == "*":
            future = asyncio.run_coroutine_threadsafe(
                OperatorMultiply().apply([lhs, rhs], self.index_list), self._event_loop
            )
            return future.result()

        elif oper == "/":
            future = asyncio.run_coroutine_threadsafe(
                OperatorDivide().apply([lhs, rhs], self.index_list), self._event_loop
            )
            return future.result()

        raise RuntimeError(f"Unexpected operator {oper}")

    def power_expr(self, args):
        """
        Implementation of power operation in the transformer.
        """
        logger.debug(" * power_expr")

        # Since we do not await the future, this is effectively synchronous
        future = asyncio.run_coroutine_threadsafe(
            OperatorPower().apply([args[0], args[1]], self.index_list), self._event_loop
        )
        return future.result()

    def unary_negate_expr(self, args):
        """
        Implementation of unary negation in the transformer.
        """
        logger.debug(" * unary_negate_expr")
        # args[0] is the '-' character

        # Since we do not await the future, this is effectively synchronous
        future = asyncio.run_coroutine_threadsafe(
            OperatorUnaryNegate().apply([args[1]], self.index_list), self._event_loop
        )
        return future.result()

    def true(self, args):
        """Returns a BandMathValue of True."""
        logger.debug(" * true")
        return BandMathValue(VariableType.BOOLEAN, True, computed=False)

    def false(self, args):
        """Returns a BandMathValue of False."""
        logger.debug(" * false")
        return BandMathValue(VariableType.BOOLEAN, False, computed=False)

    def number(self, args):
        """Returns a BandMathValue containing a specific number."""
        logger.debug(f" * number {args[0]}")
        return args[0]

    def string(self, args):
        """Returns a BandMathValue containing a specific string."""
        logger.debug(f' * string "{args[0]}"')
        return args[0]

    def variable(self, args) -> BandMathValue:
        """
        Returns a BandMathValue containing the value of the specified variable.
        """
        logger.debug(" * variable")
        name = args[0]
        if name not in self._variables or self._variables[name][1] is None:
            raise BandMathEvalError(f'Variable "{name}" is unspecified')

        (type, value) = self._variables[name]
        return BandMathValue(type, value, computed=False)

    def named_expression(self, args) -> BandMathValue:
        """
        Named expressions can appear in function arguments.
        """
        logger.debug(" * named_expression")
        # The first argument is the name, and the second argument is a
        # BandMathValue object holding the result of the expression evaluation.
        # Set the name and return the object.
        value = args[1]
        value.set_name(args[0])
        return value

    def function(self, args) -> BandMathValue:
        """
        Calls the function named in args[0], passing it args[1:], and returns
        the result as a BandMathValue.
        """
        logger.debug(" * function")
        func_name = args[0]
        func_args = args[1:]

        has_named_args = False
        for fa in func_args:
            if fa.name is None:
                if has_named_args:
                    raise BandMathEvalError(
                        "Named arguments must be " "specified after all positional arguments"
                    )
            else:
                has_named_args = True

        if func_name not in self._functions:
            raise BandMathEvalError(f'Unrecognized function "{func_name}"')

        func_impl = self._functions[func_name]
        return func_impl.apply(func_args)

    def NAME(self, token) -> str:
        """
        Parse a token as a string variable name.  The variable name is converted
        to lowercase.
        """
        logger.debug(" * NAME")
        return str(token).lower()

    def NUMBER(self, token) -> BandMathValue:
        """
        Parse a token as a number.  The number is represented as a Python float,
        and is wrapped in a BandMathValue object.
        """
        logger.debug(" * NUMBER")
        return BandMathValue(
            VariableType.NUMBER,
            float(token),
            computed=False,
        )

    def STRING(self, token) -> str:
        """
        Parse a token as a string literal.  The variable name is converted
        to lowercase.
        """
        logger.debug(" * STRING")
        # Chop the quotes off of the string value
        return str(token)[1:-1]

    def stop(self):
        """Gracefully stop the event loop and wait for the thread to finish."""

        if hasattr(self, "_event_loop") and self._event_loop.is_running():
            self._event_loop.call_soon_threadsafe(self._event_loop.stop)  # Safely stop the loop
        if hasattr(self, "_loop_thread"):
            self._loop_thread.join()

    def __del__(self):
        self.stop()  # Ensure the loop and thread are stopped


class NumberOfIntermediatesFinder(BandMathEvaluator):
    """
    A Lark Transformer for evaluating band-math expressions.
    """

    def __init__(
        self,
        variables: Dict[str, Tuple[VariableType, BANDMATH_VALUE_TYPE]],
        functions: Dict[str, Callable],
    ):
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
                # Because lhs is new since it is an image cube bandmath value
                self.increment_interm_running_total()
                self.decrement_interm_running_total()
                has_intermediate = 1
            elif lhs.type == VariableType.IMAGE_CUBE:
                # We don't decrement because we aren't combining two values
                self.increment_interm_running_total()
                has_intermediate = 1
        elif isinstance(lhs, int) and isinstance(rhs, BandMathValue):
            if rhs.type == VariableType.IMAGE_CUBE and lhs > 0:
                self.increment_interm_running_total()
                self.decrement_interm_running_total()
                has_intermediate = 1
            elif rhs.type == VariableType.IMAGE_CUBE:
                # We don't decrement because we aren't combining two values
                self.increment_interm_running_total()
                has_intermediate = 1
        elif isinstance(lhs, int) and isinstance(rhs, int):
            if lhs > 0 and rhs > 0:
                self.decrement_interm_running_total()
                has_intermediate = 1
        else:
            raise TypeError(f" Got wrong type in either argument. Arg1 {lhs}, arg2: {rhs}")

        return has_intermediate

    def comparison(self, args):
        logger.debug(" * comparison")
        lhs = args[0]
        # oper = args[1]
        rhs = args[2]
        return self.find_current_interm_and_update_max(lhs, rhs)

    def add_expr(self, values):
        """
        Implementation of addition and subtraction operations in the
        transformer.
        """
        logger.debug(" * add_expr")
        lhs = values[0]
        oper = values[1]
        rhs = values[2]

        if oper != "+" and oper != "-":
            raise RuntimeError(f"Unexpected operator {oper}")

        return self.find_current_interm_and_update_max(lhs, rhs)

    def mul_expr(self, args):
        """
        Implementation of multiplication and division operations in the
        transformer.
        """
        logger.debug(" * mul_expr")
        lhs = args[0]
        oper = args[1]
        rhs = args[2]

        if oper != "*" and oper != "/":
            raise RuntimeError(f"Unexpected operator {oper}")

        return self.find_current_interm_and_update_max(lhs, rhs)

    def power_expr(self, args):
        """
        Implementation of power operation in the transformer.
        """
        logger.debug(" * power_expr")
        return self.find_current_interm_and_update_max(args[0], args[1])

    def unary_negate_expr(self, args):
        """
        Implementation of unary negation in the transformer.
        """
        logger.debug(" * unary_negate_expr")
        return self.find_current_interm_and_update_max(args[1], 0)

    def function(self, args) -> BandMathValue:
        """
        Currently, we just treat these functions like the basic +,-,/,*

        TODO (Joshua G-K): Create a way to use the function's analyze method
        to get the output type of the lhs and rhs side. Then use this in
        find_current_interm_and_update_max.
        """
        logger.debug(" * function")
        func_name = args[0]
        func_args = args[1:]

        has_named_args = False
        for fa in func_args:
            if fa.name is None:
                if has_named_args:
                    raise BandMathEvalError(
                        "Named arguments must be " "specified after all positional arguments"
                    )
            else:
                has_named_args = True

        if func_name not in self._functions:
            raise BandMathEvalError(f'Unrecognized function "{func_name}"')

        def make_bandmath_exprs_from_values(values: List[BandMathValue]) -> List[BandMathExprInfo]:
            """Build BandMathExprInfo (type/shape/elem_type) for each BandMathValue."""
            exprs: List[BandMathExprInfo] = []
            for v in values:
                info = BandMathExprInfo(result_type=v.type)
                if v.type in (
                    VariableType.IMAGE_CUBE,
                    VariableType.IMAGE_BAND,
                    VariableType.SPECTRUM,
                ) or isinstance(v.value, (np.ndarray, BasicValueSerialized)):
                    info.shape = v.get_shape()
                    info.elem_type = v.get_elem_type()
                exprs.append(info)
            return exprs

        # Calculate how many intermediates this function needs
        expr_infos = make_bandmath_exprs_from_values(func_args)
        increment_counter = 0
        for i in range(len(expr_infos)):
            expr_info = expr_infos[i]
            if expr_info.result_type in (VariableType.IMAGE_CUBE, VariableType.IMAGE_CUBE_DATASET):
                self.increment_interm_running_total()
                increment_counter += 1

        for i in range(increment_counter):
            self.decrement_interm_running_total()

        func_impl = self._functions[func_name]
        expr_info_output: "BandMathExprInfo" = func_impl.analyze(expr_infos)
        has_intermediate = 0
        if expr_info_output.result_type:
            self.increment_interm_running_total()
            has_intermediate = 1

        return has_intermediate


@dataclass(frozen=True)
class SingleBandMathJob:
    """A fully-resolved job for evaluating bandmath once (single target)."""

    bandmath_expr: str
    expr_info: "BandMathExprInfo"
    result_name: str
    cache: "DataCache"
    lower_variables: Dict[str, Tuple["VariableType", "BANDMATH_VALUE_TYPE"]]
    lower_functions: Dict[str, "BandMathFunction"]
    number_of_intermediates: int
    tree: lark.ParseTree
    use_synchronous_method: bool
    subdataset_name: str = ""
    filepath: Optional[str] = None  # for debugging / traceability


@dataclass
class BandMathJob:
    """
    Represents all information required to evaluate a band-math expression.

    A BandMathJob may represent either:
    - a *single* band-math evaluation (non-batch), or
    - a *batch* evaluation over multiple filepaths.

    The job is iterable. Iteration yields fully-resolved SingleBandMathJob
    instances, one per evaluation target:

    - For batch jobs, iteration yields one SingleBandMathJob per filepath.
    - For single jobs, iteration yields exactly one SingleBandMathJob.

    Batch vs single behavior is determined at construction time based on
    `serialized_variables` using `is_batch_job(...)`.

    Invariants:
        - If `filepaths` is non-empty, the job represents a batch job.
        - If `filepaths` is empty, the job represents a single job.
        - Iterating always yields at least one SingleBandMathJob.

    Each yielded SingleBandMathJob contains:
        - deserialized variables for the specific target (filepath-aware)
        - a correctly computed BandMathExprInfo for those variables
        - a final result name (prefixed by filename for batch jobs)

    This class performs no computation itself. It is responsible only for:
        - determining batch vs single execution
        - resolving per-target variables and metadata
        - producing SingleBandMathJob units suitable for execution

    Intended usage:
        job = BandMathJob(...)
        for single_job in job:
            result = eval_singular_bandmath_expr(...)

        \# or, when single-only behavior is required:
        single_job = get_single_bandmath_job(job)
    """

    bandmath_expr: str
    expr_info: "BandMathExprInfo"  # kept for compatibility; per-target expr_info is recomputed
    result_name: str
    cache: "DataCache"
    serialized_variables: Dict[str, Tuple["VariableType", "SerializedForm"]]
    lower_functions: Dict[str, "BandMathFunction"]
    number_of_intermediates: int
    tree: lark.ParseTree
    use_synchronous_method: bool
    subdataset_name: str = ""

    # Internal / derived
    loader: "RasterDataLoader" = field(default_factory=lambda: RasterDataLoader(), init=False)
    is_batch: bool = field(init=False)
    filepaths: List[str] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        # Determine if this is a batch job and capture filepaths if so.
        self.is_batch = is_batch_job(self.serialized_variables)
        self.filepaths = get_batch_filepaths(self.serialized_variables) if self.is_batch else []

    def get_single_bandmath_job(self: "BandMathJob") -> SingleBandMathJob:
        """
        Return the single resolved bandmath job from a BandMathJob.

        This function may only be called when the BandMathJob represents a
        non-batch (single) job. If the job represents a batch job (i.e. has
        filepaths), a ValueError is raised.

        Args:
            job: A BandMathJob expected to represent a single bandmath evaluation.

        Returns:
            A SingleBandMathJob instance containing fully deserialized variables,
            computed expr_info, and the final result name.

        Raises:
            ValueError: If the BandMathJob represents a batch job.
        """
        if self.filepaths:
            raise ValueError(
                "get_single_bandmath_job() may only be called when BandMathJob "
                "represents a single (non-batch) bandmath job."
            )

        # BandMathJob is iterable by design; for a single job it yields exactly one item
        return next(iter(self))

    def __iter__(self) -> Iterator[SingleBandMathJob]:
        """
        Yield SingleBandMathJob instances.

        - Batch: one per filepath (variables deserialized per-file; expr_info per-file; result_name
          prefixed by filename stem).
        - Single: exactly one job (filepath=None; variables deserialized without filepath; expr_info
          computed once; result_name unchanged).
        """
        targets: List[Optional[str]] = self.filepaths if self.filepaths else [None]

        for filepath in targets:
            # Build per-target result name
            if filepath is None:
                new_result_name = self.result_name
            else:
                base = os.path.basename(filepath)
                name, _ext = os.path.splitext(base)
                new_result_name = f"{name}{self.result_name}"

            # Deserialize variables for this target
            if filepath is None:
                # Single job: deserialize without a filepath context
                lower_variables = deserialize_bandmath_variables(
                    serialized_variables=self.serialized_variables,
                    subdataset_name=self.subdataset_name,
                    filepath=None,
                    loader=self.loader,
                )
            else:
                # Batch job: deserialize with filepath context
                lower_variables = deserialize_bandmath_variables(
                    serialized_variables=self.serialized_variables,
                    subdataset_name=self.subdataset_name,
                    filepath=filepath,
                    loader=self.loader,
                )

            # Compute the correct expr_info for these variables (batch-safe)
            current_expr_info = bandmath.get_bandmath_expr_info(
                self.bandmath_expr, lower_variables, self.lower_functions
            )

            yield SingleBandMathJob(
                bandmath_expr=self.bandmath_expr,
                expr_info=current_expr_info,
                result_name=new_result_name,
                cache=self.cache,
                lower_variables=lower_variables,
                lower_functions=self.lower_functions,
                number_of_intermediates=self.number_of_intermediates,
                tree=self.tree,
                use_synchronous_method=self.use_synchronous_method,
                subdataset_name=self.subdataset_name,
                filepath=filepath,
            )


# region Bandmath Evaluation


def start_bandmath_evaluation(
    bandmath_expr: str,
    expr_info: BandMathExprInfo,
    result_name: str,
    cache: DataCache,
    variables: Dict[str, Tuple[VariableType, Serializable]],
    functions: Dict[str, BandMathFunction] = None,
    subdataset_name: str = "",
    succeeded_callback: Callable = lambda _: None,
    status_callback: Callable = lambda _: None,
    error_callback: Callable = lambda _: None,
    started_callback: Callable = lambda _: None,
    cancelled_callback: Callable = lambda _: None,
    app_state: "ApplicationState" = None,
    use_synchronous_method=True,
) -> ProcessManager:
    """
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
    """

    # Just to be defensive against potentially bad inputs, make sure all names
    # of variables and functions are lowercase.
    # TODO(donnie):  Can also make sure they are valid, trimmed of whitespace,
    #     etc.

    lower_variables: Dict[str, Tuple[VariableType, Serializable]] = {}
    for name, value in variables.items():
        lower_variables[name.lower()] = value

    lower_functions = get_builtin_functions()
    if functions:
        for name, function in functions.items():
            lower_functions[name.lower()] = function

    parser = lark.Lark.open("bandmath.lark", rel_to=__file__, start="expression", propagate_positions=True)
    tree = parser.parse(bandmath_expr)

    logger.info(f"Band-math parse tree:\n{tree.pretty()}")
    logger.debug("Beginning band-math evaluation")

    id_assigner = UniqueIDAssigner()
    id_assigner.visit(tree)

    numInterFinder = NumberOfIntermediatesFinder(lower_variables, lower_functions)
    numInterFinder.transform(tree)
    number_of_intermediates = numInterFinder.get_max_intermediates()
    number_of_intermediates += 1
    logger.debug(f"Number of intermediates: {number_of_intermediates}")

    # We must serialize RasterDataSet, RasterBand, and Spectrum objects because they
    # could have an underlying gdal or osgeo object that can't be pickled
    serialized_variables = serialize_bandmath_variables(lower_variables)

    bandmath_job_data = BandMathJob(
        bandmath_expr=bandmath_expr,
        expr_info=expr_info,
        result_name=result_name,
        cache=None,
        serialized_variables=serialized_variables,
        lower_functions=lower_functions,
        number_of_intermediates=number_of_intermediates,
        tree=tree,
        use_synchronous_method=use_synchronous_method,
        subdataset_name=subdataset_name,
    )

    kwargs = {"bandmath_job_data": bandmath_job_data}

    process_manager = ProcessManager(bandmath_subprocess_entrypoint, kwargs)
    if app_state:
        app_state.add_running_process(process_manager)

    task = process_manager.get_task()
    task.cancelled.connect(cancelled_callback)
    # The started slot is passed the task
    task.started.connect(started_callback)
    # The error slot is passed the process_manager's task
    task.error.connect(error_callback)
    # The progress slot is passed the message that bandmath_subprocess_entrypoint
    # sends over the pipe
    task.status.connect(status_callback)
    task.succeeded.connect(lambda task: succeeded_callback(task.get_result()))
    process_manager.start_task()
    return process_manager


def bandmath_subprocess_entrypoint(
    bandmath_job_data: BandMathJob,
    child_conn: mp_conn.Connection,
    return_queue: mp.Queue,
):
    eval_bandmath_expressions(
        bandmath_job_data=bandmath_job_data,
        child_conn=child_conn,
        return_queue=return_queue,
    )


def eval_bandmath_expressions(
    bandmath_job_data: BandMathJob,
    child_conn: mp_conn.Connection,
    return_queue: mp.Queue,
):
    # This case is if we are doing batch processing
    if bandmath_job_data.is_batch:
        eval_bandmath_batch(
            bandmath_job_data,
            child_conn=child_conn,
            return_queue=return_queue,
        )
    else:
        update_progress_child_conn(child_conn=child_conn, numerator=1, denominator=1, status="Running")
        single_bandmath_job = bandmath_job_data.get_single_bandmath_job()
        result = eval_singular_bandmath_expr(
            single_bandmath_job=single_bandmath_job,
        )
        serialized_result = serialize_bandmath_results([result])
        update_progress_child_conn(child_conn=child_conn, numerator=1, denominator=1, status="Finished")
        return_queue.put(serialized_result)


def eval_bandmath_batch(
    bandmath_job_data: BandMathJob,
    child_conn: mp_conn.Connection,
    return_queue: mp.Queue,
):
    outputs = []
    count = 0
    total = len(bandmath_job_data.filepaths)
    for single_bandmath_job in bandmath_job_data:
        try:
            count += 1
            update_progress_child_conn(
                child_conn=child_conn, numerator=count, denominator=total, status="Running"
            )
            # Then we calculate the result and serialize it
            result = eval_singular_bandmath_expr(
                single_bandmath_job,
            )
            outputs.append(result)
            send_error_child_conn(
                child_conn=child_conn,
                result_name=single_bandmath_job.result_name,
                message=None,
                traceback_str=None,
            )
        except Exception as e:
            send_error_child_conn(
                child_conn=child_conn,
                result_name=single_bandmath_job.result_name,
                message=str(e),
                traceback_str=traceback.format_exc(),
            )
            outputs.append((None, e, traceback.format_exc(), None))

    update_progress_child_conn(child_conn=child_conn, numerator=count, denominator=total, status="Finished")

    serialized_results = serialize_bandmath_results(outputs)
    return_queue.put(serialized_results)


def eval_singular_bandmath_expr(
    single_bandmath_job: SingleBandMathJob,
) -> Tuple[
    Union[VariableType, RasterDataSet.__class__],
    Union[np.ndarray, RasterDataSet],
    str,
    BandMathExprInfo,
]:
    """
    This function evaluates one singular bandmath expression

    Returns:
    - The first element in the tuple is the variable type or the RasterDataSet
        class. The second element is the actual value, which is either the numpy array or the
        RasterDataSet. The third element is the name of the resulting dataset. The fourth element
        is the expr_info for that dataset.
    """
    # Extract variables from current bandmath job
    expr_info = single_bandmath_job.expr_info
    number_of_intermediates = single_bandmath_job.number_of_intermediates
    subdataset_name = single_bandmath_job.subdataset_name
    use_synchronous_method = single_bandmath_job.use_synchronous_method
    result_name = single_bandmath_job.result_name
    if subdataset_name:
        result_name = f"{subdataset_name}_{result_name}"

    max_chunking_bytes, should_chunk = max_bytes_to_chunk(expr_info.result_size() * number_of_intermediates)
    logger.debug(f"Max chunking bytes: {max_chunking_bytes}")
    # Either we explicitly say we don't want to use the synchronous method or we decide to chunk based
    # on how big the image cube is.
    if not use_synchronous_method or (expr_info.result_type == VariableType.IMAGE_CUBE and should_chunk):
        return eval_singular_bandmath_expr_async(
            single_bandmath_job=single_bandmath_job,
            max_chunking_bytes=max_chunking_bytes,
        )
    else:
        return eval_singular_bandmath_expr_sync(single_bandmath_job=single_bandmath_job)


def eval_singular_bandmath_expr_async(
    single_bandmath_job: SingleBandMathJob,
    max_chunking_bytes: Union[float, int],
) -> Tuple[
    Union[VariableType],
    Union[np.ndarray, RasterDataSet],
    str,
    BandMathExprInfo,
]:
    """
    Evaluate a single band-math expression asynchronously using chunked execution.

    This function evaluates the band-math expression described by a
    SingleBandMathJob using an asynchronous evaluator. For image-cube results,
    the output is computed in band windows to respect memory constraints, with
    asynchronous I/O read-ahead and write-back. Intermediate results are written
    incrementally to an on-disk GDAL dataset.

    Args:
        single_bandmath_job: Fully-resolved band-math job containing deserialized
            variables, expression metadata, and execution configuration.
        max_chunking_bytes: Maximum number of bytes available for chunked
            evaluation.

    Returns:
        A tuple of:
            - the result variable type (or RasterDataSet class),
            - the resulting value (RasterDataSet for image outputs),
            - the result dataset name,
            - and the BandMathExprInfo describing the result.

    Raises:
        Exception: Propagates any exception raised during evaluation or I/O.
    """
    expr_info = single_bandmath_job.expr_info
    number_of_intermediates = single_bandmath_job.number_of_intermediates
    subdataset_name = single_bandmath_job.subdataset_name
    lower_variables = single_bandmath_job.lower_variables
    lower_functions = single_bandmath_job.lower_functions
    cache = single_bandmath_job.cache
    tree = single_bandmath_job.tree
    result_name = single_bandmath_job.result_name
    if subdataset_name:
        result_name = f"{subdataset_name}_{result_name}"

    # Get metadata
    gdal_type, data_ignore_value = extract_expression_metadata(expr_info)

    # Evaluate
    try:
        evaluator = BandMathEvaluatorAsync(lower_variables, lower_functions, expr_info.shape)
        out_dataset, out_dataset_gdal, bands, lines, samples = create_output_dataset(
            expr_info=expr_info,
            gdal_type=gdal_type,
            result_name=result_name,
            cache=cache,
            temp_folder_path=TEMP_FOLDER_PATH,
        )

        # Based on memory limits (currently set in constants, but we could make it more adjustable)
        # find the number of bands that we can access without exceeding it
        num_bands = compute_bands_per_chunk(
            max_chunking_bytes, expr_info, number_of_intermediates, lines, samples
        )

        futures = []
        for current_bands, next_bands in iter_band_windows(bands, num_bands):
            # print(f"Min: {min(current_bands)} | Max: {max(current_bands)}")

            arr = evaluate_band_window(evaluator, tree, current_bands, next_bands)

            future = submit_raster_write(
                evaluator, out_dataset_gdal, current_bands, arr, gdal_type, data_ignore_value
            )
            futures.append(future)
        wait_for_all_futures(futures)
    except BaseException as e:
        if evaluator is not None:
            evaluator.stop()
        raise e
    finally:
        evaluator.stop()
    return (VariableType.IMAGE_CUBE_DATASET, out_dataset, result_name, expr_info)


def eval_singular_bandmath_expr_sync(
    single_bandmath_job: SingleBandMathJob,
) -> Tuple[
    Union[VariableType],
    Union[np.ndarray, RasterDataSet],
    str,
    BandMathExprInfo,
]:
    """
    Evaluate a single band-math expression synchronously.

    Args:
        single_bandmath_job: Fully-resolved band-math job containing deserialized
            variables, the parsed expression tree, and result metadata.

    Returns:
        A tuple of:
            - the result variable type (or RasterDataSet class),
            - the evaluated result value,
            - the result name,
            - and the BandMathExprInfo describing the result.

    Raises:
        Exception: Propagates any exception raised during evaluation.
    """
    try:
        eval = BandMathEvaluator(
            single_bandmath_job.lower_variables,
            single_bandmath_job.lower_functions,
        )
        result_value = eval.transform(single_bandmath_job.tree)
    except BaseException as e:
        if eval:
            eval.stop()
        raise e
    finally:
        eval.stop()
    return (
        result_value.type,
        result_value.value,
        single_bandmath_job.result_name,
        single_bandmath_job.expr_info,
    )


# region Helpers

# Serialization and deserialization helpers


def serialize_bandmath_variables(
    variables: Dict[str, Tuple[VariableType, BANDMATH_VALUE_TYPE]],
) -> Dict[str, Tuple[VariableType, SerializedForm]]:
    """
    This function serializes the 'variables' and 'functions' dictionaries into a
    format that can be passed to a subprocess. In the subprocess, we will
    deserialize these variables and functions, then pass them to the
    `start_bandmath_evaluation` function.
    """
    variables_serialized = {}
    for var_name, var_tuple in variables.items():
        var_type = var_tuple[0]
        var_value = var_tuple[1]
        if isinstance(var_value, Serializable):
            variables_serialized[var_name] = (var_type, var_value.get_serialized_form())
        else:
            variables_serialized[var_name] = (var_type, BasicValueSerialized(var_value).get_serialized_form())
    return variables_serialized


def serialized_form_to_variable(
    var_name: str,
    var_type: VariableType,
    var_value: SerializedForm,
    loader: RasterDataLoader,
    filepath: str = None,
    subdataset_name: str = "",
) -> Dict[str, Tuple[VariableType, BANDMATH_VALUE_TYPE]]:
    """
    This function is used to convert a serialized form of an object back into the original object.

    Args:
        var_name (str):
            The name of the variable.
        var_type (VariableType):
            The type of the variable.
        var_value (Union[SerializedForm]):
            The serialized form of the variable.
        loader (RasterDataLoader):
            The raster data loader to use for loading data from file.
        filepath (str, optional):
            The filepath to load data from for this variable. This is variable specific.
        subdataset_name (str, optional):
            The name of the GDAL subdataset to load data from. This is specified when the user
            selects an input folder and this subdataset name is used on all datasets in that folder.
    Returns:
        Dict[str, Tuple[VariableType, BANDMATH_VALUE_TYPE]]: A dictionary with the variable name as
        the key and a tuple of the variable type and the deserialized object as the value.
    """
    assert isinstance(var_value, SerializedForm), "The argument var_value is not a SerializedForm"
    if var_type == VariableType.IMAGE_CUBE:
        obj = var_value.get_serializable_class().deserialize_into_class(var_value)
        return {var_name: (var_type, obj)}
    # At this point, even though the type is image cube batch, we are loading a filepath
    elif var_type == VariableType.IMAGE_CUBE_BATCH:
        assert isinstance(
            var_value.get_serialize_value(), str
        ), "Image Cube Batch variables should be strings"
        assert filepath is not None, "Filepath is required for Image Cube Batch variables"
        dataset = loader.load_from_file(path=filepath, subdataset_name=subdataset_name, interactive=False)[0]
        return {var_name: (VariableType.IMAGE_CUBE, dataset)}

    elif var_type == VariableType.IMAGE_BAND:
        obj = var_value.get_serializable_class().deserialize_into_class(var_value)
        return {var_name: (var_type, obj)}

    elif var_type == VariableType.IMAGE_BAND_BATCH:
        assert filepath is not None, "Filepath is required for Image Band Batch variables"
        assert (
            "band_index" in var_value.get_metadata() and var_value.get_metadata()["band_index"] is not None
        ) or (
            "wavelength_value" in var_value.get_metadata()
            and var_value.get_metadata()["wavelength_value"] is not None
        ), "Band index or wavelength value is required for Image Band Batch variables"
        serializable_class = var_value.get_serializable_class()
        # This should never occur, but if it does we make it a RasterDataDynamicBand
        if issubclass(serializable_class, RasterDataBatchBand):
            band_index = var_value.get_metadata().get("band_index", None)
            wavelength_value = var_value.get_metadata().get("wavelength_value", None)
            wavelength_units = var_value.get_metadata().get("wavelength_units", None)
            epsilon = var_value.get_metadata().get("epsilon", None)
            dataset = loader.load_from_file(path=filepath, subdataset_name=subdataset_name)
            band = RasterDataDynamicBand(
                dataset,
                band_index=band_index,
                wavelength_value=wavelength_value,
                wavelength_units=wavelength_units,
                epsilon=epsilon,
            )
        else:
            serialize_metadata = var_value.get_metadata()
            serialize_metadata.update({"filepath": filepath})
            band = serializable_class.deserialize_into_class(var_value)
        return {var_name: (VariableType.IMAGE_BAND, band)}

    elif var_type == VariableType.SPECTRUM:
        obj = var_value.get_serializable_class().deserialize_into_class(var_value)
        return {var_name: (var_type, obj)}

    # This should only be reached in testing
    elif var_type == VariableType.NUMBER or var_type == VariableType.BOOLEAN:
        return {var_name: (var_type, var_value.get_serialize_value())}

    else:
        raise ValueError(f"Unsupported variable type: {var_type}")


def deserialize_bandmath_variables(
    serialized_variables: Dict[str, Tuple[VariableType, SerializedForm]],
    subdataset_name: str = "",
    filepath: str = None,
    loader: RasterDataLoader = None,
) -> Dict[str, Tuple[VariableType, BANDMATH_VALUE_TYPE]]:
    """
    This function deserializes the bandmath variables from their serialized form.
    """
    if loader is None:
        loader = RasterDataLoader()
    deserialized_variables = {}
    for var_name, var_tuple in serialized_variables.items():
        var_type = var_tuple[0]
        var_value = var_tuple[1]
        if isinstance(var_value, SerializedForm):
            deserialized_variables.update(
                serialized_form_to_variable(
                    var_name,
                    var_type,
                    var_value,
                    loader,
                    filepath=filepath,
                    subdataset_name=subdataset_name,
                )
            )
        else:
            raise ValueError("Variable value is not a SerializedForm")
    return deserialized_variables


def serialize_bandmath_results(
    results: List[
        Tuple[
            Union[VariableType, RasterDataSet.__class__],
            Union[np.ndarray, RasterDataSet],
            str,
            BandMathExprInfo,
        ]
    ],
) -> List[BandMathResultInfo]:
    """
    This function serializes the bandmath results and sends them to the return queue.
    """
    serialized_results: List[BandMathResultInfo] = []
    for result_type, result_value, result_name, result_expr_info in results:
        if isinstance(result_value, Serializable):
            serialized_results.append(
                (
                    result_type,
                    result_value.get_serialized_form(),
                    result_name,
                    result_expr_info,
                )
            )
        else:
            serialized_results.append((result_type, result_value, result_name, result_expr_info))
    return serialized_results


# Misc helpers


def extract_expression_metadata(expr_info: "BandMathExprInfo") -> Tuple[int, Union[float, int]]:
    """
    Extract output metadata required for band-math evaluation.

    This function determines the GDAL data type for the expression result and
    resolves the data ignore value, preferring the value defined in the
    expression's spectral metadata when available.

    Args:
        expr_info: BandMathExprInfo describing the result element type and
            associated spectral metadata.

    Returns:
        A tuple of (gdal_type, spectral_metadata, data_ignore_value), where:
            - gdal_type is the GDAL data type corresponding to the expression
              element type.
            - data_ignore_value is the value used to fill masked or invalid data.
    """
    gdal_type = np_dtype_to_gdal(np.dtype(expr_info.elem_type))

    spectral_metadata = expr_info.spectral_metadata_source
    data_ignore_value = DEFAULT_IGNORE_VALUE
    if spectral_metadata is not None and spectral_metadata.get_data_ignore_value() is not None:
        data_ignore_value = spectral_metadata.get_data_ignore_value()

    return gdal_type, data_ignore_value


def wait_for_all_futures(futures: "Future"):
    """Block until all futures have completed."""
    concurrent.futures.wait(futures)


def submit_raster_write(
    evaluator: BandMathEvaluatorAsync,
    gdal_dataset: gdal.Dataset,
    current_bands: List[int],
    arr: np.ndarray,
    gdal_type: int,
    data_ignore: Union[float, int] = DEFAULT_IGNORE_VALUE,
) -> "Future":
    """
    Submit an asynchronous write of evaluated band data to a GDAL dataset.

    Args:
        evaluator: Asynchronous band-math evaluator providing the write thread pool.
        gdal_dataset: GDAL dataset to write the raster data into.
        current_bands: List of band indices corresponding to the data in `arr`.
        arr: NumPy array containing evaluated band data for the current window.
        gdal_type: GDAL data type for writing the raster values.
        data_ignore: Value used to fill masked elements before writing.

    Returns:
        A Future representing the asynchronous write operation.
    """
    return evaluator._write_thread_pool.submit(
        write_raster_to_dataset,
        gdal_dataset,
        current_bands,
        arr,
        gdal_type,
        default_ignore_value=data_ignore,
    )


def evaluate_band_window(
    evaluator: BandMathEvaluatorAsync,
    tree: ParseTree,
    current_bands: List[int],
    next_bands: List[int],
) -> np.ndarray:
    """
    Evaluate a single band window of a band-math expression.

    This function evaluates the band-math expression over the specified
    window of band indices. The `next_bands` argument is used by the
    evaluator to perform asynchronous I/O read-ahead for upcoming band
    windows. The resulting values for the current window are returned as
    a NumPy array.

    As a side effect, this function configures the evaluator's internal
    state by setting the current and next band index lists prior to
    execution.

    Args:
        evaluator: Asynchronous band-math evaluator used to execute the
            expression.
        tree: Parsed band-math expression tree.
        current_bands: List of band indices to evaluate in the current
            window.
        next_bands: List of band indices that will be evaluated next, used
            for asynchronous I/O read-ahead.

    Returns:
        A NumPy array containing the evaluated result for the current band
        window.
    """
    evaluator.index_list_current = current_bands
    evaluator.index_list_next = next_bands

    result_value = evaluator.transform(tree)
    if isinstance(result_value, (asyncio.Future, Coroutine)):
        result_value = asyncio.run_coroutine_threadsafe(result_value, evaluator._event_loop).result()
    res = result_value.value
    res = extract_array(res)
    return res


def iter_band_windows(bands: int, num_bands_per_iter: int):
    """
    Yield successive band index windows for chunked band-math evaluation.

    For each iteration, this generator yields a tuple of:
    - the current band indices to evaluate, and
    - the next band indices, used for asynchronous I/O read-ahead.

    Args:
        bands: Total number of bands in the dataset.
        num_bands_per_iter: Number of bands to include in each evaluation window.

    Yields:
        Tuples of (current_bands, next_bands), where each element is a list
        of band indices.
    """
    for band_index in range(0, bands, num_bands_per_iter):
        start = band_index
        end = min(bands, start + num_bands_per_iter)
        current_bands = [band for band in range(start, end)]
        next_start = band_index + num_bands_per_iter
        next_end = min(bands, band_index + 2 * num_bands_per_iter)
        next_bands = [band for band in range(next_start, next_end)]
        yield current_bands, next_bands


def compute_bands_per_chunk(max_bytes, expr_info, num_intermediates, lines, samples):
    """
    Compute the number of bands that can be processed per chunk under memory constraints.

    Args:
        max_bytes: Maximum number of bytes allowed for processing a chunk.
        expr_info: BandMathExprInfo describing the result element type.
        num_intermediates: Number of intermediate arrays used during evaluation.
        lines: Number of lines (rows) in the output array.
        samples: Number of samples (columns) in the output array.

    Returns:
        The number of bands that can be processed per chunk. Always at least 1.
    """
    bytes_per_element = (
        np.dtype(expr_info.elem_type).itemsize if expr_info.elem_type is not None else SCALAR_BYTES
    )
    max_bytes = max_bytes / bytes_per_element
    max_bytes_per_intermediate = max_bytes / num_intermediates
    num_bands = int(np.floor(max_bytes_per_intermediate / (lines * samples)))
    num_bands = 1 if num_bands < 1 else num_bands
    return num_bands


def get_batch_filepaths(
    serialized_variables: Dict[str, Tuple[VariableType, SerializedForm]],
) -> List[str]:
    filepaths = []
    # We need to check if we are doing batching or not. If we are, then we need to
    # make a list of the filepaths and then we need to make a list of the variables.
    for _, var_tuple in serialized_variables.items():
        var_type = var_tuple[0]
        var_value = var_tuple[1]

        # All batch variables
        if var_type == VariableType.IMAGE_CUBE_BATCH:
            folder_path = var_value.get_serialize_value()
            filepaths = get_unique_filepaths(folder_path)
            break
        elif var_type == VariableType.IMAGE_BAND_BATCH:
            folder_path = var_value.get_serialize_value()
            filepaths = get_unique_filepaths(folder_path)
            break
    return filepaths


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


def is_batch_job(
    serialized_variables: Dict[str, Tuple[VariableType, SerializedForm]],
) -> bool:
    """
    This function is used to decide if we are doing batching or not. It checks
    to see if any of the variable types are batch types.
    """
    for _, var_tuple in serialized_variables.items():
        var_type = var_tuple[0]
        if var_type == VariableType.IMAGE_CUBE_BATCH or var_type == VariableType.IMAGE_BAND_BATCH:
            return True
    return False


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


def prepare_expr_info(
    bandmath_expr: str,
    variables_list: List[Dict[str, Tuple[VariableType, Union[SerializedForm, str, bool]]]],
    functions: Dict[str, BandMathFunction],
) -> List[BandMathExprInfo]:
    """
    This function is used to expand expr_info to include expression info for each of the variable dictionaries
    in the list
    """
    # Go through each of the variables in the list and get the expr_info for each of them
    expr_info_list = []
    for variables in variables_list:
        expr_info_list.append(bandmath.get_bandmath_expr_info(bandmath_expr, variables, functions))
    return expr_info_list


def extract_array(data) -> np.ndarray:
    """
    Extracts the numpy array from the data, the data can be either a numpy array,
    or a RasterDataBand or RasterDataSet or a Spectrum.
    """
    # If the value is already a NumPy array, we are done!
    if isinstance(data, np.ndarray):
        return data
    elif isinstance(data, RasterDataSet):
        return data.get_image_data()
    elif isinstance(data, RasterBand):
        return data.get_data()
    elif isinstance(data, Spectrum):
        return data.get_spectrum()
    raise TypeError(f"Unsupported data type for extraction into numpy array: {type(data)}")


def create_output_dataset(
    *,
    expr_info: "BandMathExprInfo",
    gdal_type: int,
    result_name: str,
    cache: "DataCache",
    temp_folder_path: str = "TEMP_FOLDER_PATH",
) -> Tuple["RasterDataSet", "gdal.Dataset", str, int, int, int, "VariableType"]:
    """
    Create an on-disk output dataset for bandmath evaluation.

    This function encapsulates the dataset-creation portion of the async bandmath
    execution path:

    - Computes the final output name (prefixing with subdataset_name if provided).
    - Determines GDAL output type based on expr_info.elem_type.
    - Computes output shape (bands, lines, samples) from expr_info.shape.
    - Allocates a unique ENVI dataset on disk in TEMP_FOLDER_PATH.
    - Wraps the GDAL dataset in a RasterDataSet bound to the provided cache.
    - Marks the RasterDataSet dirty (so the file isn't deleted when this process exits).

    Args:
        expr_info: BandMathExprInfo describing result type, shape, and element dtype.
        result_name: Base name for the result dataset.
        cache: DataCache used to construct the RasterDataSet wrapper.
        subdataset_name: Optional prefix added to result_name (e.g. "<subdataset>_<result>").
        temp_folder_path: Folder where temporary ENVI datasets should be created. In your
            codebase this is typically TEMP_FOLDER_PATH; pass that constant in.

    Returns:
        A tuple of:
            (out_dataset, out_dataset_gdal, final_result_name, bands, lines, samples, gdal_type)

        where:
            - out_dataset is a RasterDataSet wrapper
            - out_dataset_gdal is the underlying GDAL dataset
            - final_result_name includes any subdataset prefix
            - (bands, lines, samples) are output dimensions
            - gdal_type is the GDAL data type used to create the dataset

    Raises:
        RuntimeError: If expr_info.shape is not length 2 or 3.
        OSError: If the output folder cannot be created.
        Exception: If GDAL dataset creation fails.
    """
    # Determine output dimensions
    bands = 1
    lines = 1
    samples = 1
    if len(expr_info.shape) == 2:
        lines, samples = expr_info.shape
    elif len(expr_info.shape) == 3:
        bands, lines, samples = expr_info.shape
    else:
        raise RuntimeError(f"expr_info shape is neither 2 or 3, its {expr_info.shape}")

    # Allocate unique file path and ensure folder exists
    result_path = get_unused_file_path_in_folder(temp_folder_path, result_name)
    folder_path = os.path.dirname(result_path)
    if folder_path and not os.path.exists(folder_path):
        os.makedirs(folder_path, exist_ok=True)

    # Create the GDAL dataset on disk (ENVI)
    out_dataset_gdal = gdal.GetDriverByName("ENVI").Create(result_path, samples, lines, bands, gdal_type)

    # Wrap into RasterDataSet + mark dirty (prevents GC cleanup in subprocess exit cases)
    out_dataset = RasterDataLoader().dataset_from_gdal_dataset(out_dataset_gdal, cache)
    out_dataset.set_dirty()

    return out_dataset, out_dataset_gdal, bands, lines, samples


def update_progress_child_conn(
    child_conn: mp_conn.Connection,
    numerator: int,
    denominator: int,
    status: str,
):
    """
    Send a progress update to the parent process via the child connection.
    """
    child_conn.send(
        [
            "progress",
            {"Numerator": numerator, "Denominator": denominator, "Status": status},
        ]
    )


def send_error_child_conn(
    child_conn: mp_conn.Connection,
    result_name: str,
    message: str,
    traceback_str: str,
):
    """
    Send an error message to the parent process via the child connection.
    """
    child_conn.send(
        [
            "error",
            {
                "Result Name": result_name,
                "Message": message,
                "Traceback": traceback_str,
            },
        ]
    )
