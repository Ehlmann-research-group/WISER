from typing import List

import numpy as np

import queue
from concurrent.futures import ThreadPoolExecutor
import asyncio

from wiser.bandmath import VariableType, BandMathValue, BandMathExprInfo
from wiser.bandmath.functions import BandMathFunction

import time

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
            VariableType.IMAGE_BAND, VariableType.SPECTRUM, VariableType.NUMBER]:
            self._report_type_error(arg.result_type)

        # Unary negation returns the same kind of input that it is given.
        # The metadata-source of the result will also be the same as the input.
        return arg


    def apply(self, args: List[BandMathValue], index_list: List[int]):
        '''
        Perform unary negation on the argument and return the result.
        '''
        if len(args) != 1:
            raise Exception('Unary negation requires exactly one operand')

        arg = args[0]

        if arg.type == VariableType.NUMBER:
            return BandMathValue(VariableType.NUMBER, -arg.value)
        elif arg.type == VariableType.IMAGE_CUBE:
            if isinstance(index_list, int):
                index_list = [index_list]
            arr = arg.as_numpy_array_by_bands(index_list)
            result_arr = -arr
            return BandMathValue(arg.type, result_arr)
        elif arg.type in [VariableType.IMAGE_BAND,
                          VariableType.SPECTRUM]:
            arr = arg.as_numpy_array_by_bands(index_list)
            result_arr = -arr
            return BandMathValue(arg.type, result_arr)

        self._report_type_error(arg.type)
