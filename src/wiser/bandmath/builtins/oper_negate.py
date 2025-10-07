from typing import List

import queue
from concurrent.futures import ThreadPoolExecutor
import asyncio

from wiser.bandmath import VariableType, BandMathValue, BandMathExprInfo
from wiser.bandmath.functions import BandMathFunction

from ..utils import get_lhs_value_async

class OperatorUnaryNegate(BandMathFunction):
    '''
    Unary negation operator.
    '''

    def _report_type_error(self, operand_type):
        raise TypeError(f'Don\'t know how to unary-negate {operand_type}')


    def analyze(self, infos: List[BandMathExprInfo]):

        if len(infos) != 1:
            raise Exception('Unary negation requires exactly one operand')

        arg = infos[0]

        # Make sure the input type is compatible with unary negation
        if arg.result_type not in [VariableType.IMAGE_CUBE,
            VariableType.IMAGE_BAND, VariableType.SPECTRUM, VariableType.NUMBER,
            VariableType.IMAGE_CUBE_BATCH, VariableType.IMAGE_BAND_BATCH]:
            self._report_type_error(arg.result_type)

        # Unary negation returns the same kind of input that it is given.
        # The metadata-source of the result will also be the same as the input.
        return arg


    async def apply(self, args: List[BandMathValue], index_list_current: List[int] = None, \
              index_list_next: List[int] = None, read_task_queue: queue.Queue = None, \
              read_thread_pool: ThreadPoolExecutor = None, \
                event_loop: asyncio.AbstractEventLoop = None, node_id: int = None):
        '''
        Perform unary negation on the argument and return the result.
        '''
        if len(args) != 1:
            raise Exception('Unary negation requires exactly one operand')

        arg = args[0]

        if arg.type == VariableType.NUMBER:
            return BandMathValue(VariableType.NUMBER, -arg.value)
        elif arg.type == VariableType.IMAGE_CUBE and index_list_current is not None:
            if isinstance(index_list_current, int):
                index_list_current = [index_list_current]
            if isinstance(index_list_next, int):
                index_list_next = [index_list_next]
            arr = await get_lhs_value_async(arg, index_list_current, index_list_next, \
                                        read_task_queue, read_thread_pool, event_loop)
            result_arr = -arr
            return BandMathValue(arg.type, result_arr)
        elif arg.type in [VariableType.IMAGE_CUBE,
                          VariableType.IMAGE_BAND,
                          VariableType.SPECTRUM]:
            if index_list_current is not None:
                if isinstance(index_list_current, int):
                    index_list_current = [index_list_current]
                arr = arg.as_numpy_array_by_bands(index_list_current)
            else:
                arr = arg.as_numpy_array()
            result_arr = -arr
            return BandMathValue(arg.type, result_arr)

        self._report_type_error(arg.type)
