from typing import List

import numpy as np

from wiser.bandmath import VariableType, BandMathValue
from wiser.bandmath.functions import BandMathFunction


class OperatorUnaryNegate(BandMathFunction):
    '''
    Unary negation operator.
    '''

    def _report_type_error(self, operand_type):
        raise TypeError(f'Don\'t know how to unary-negate {operand_type}')


    def get_result_type(self, arg_types: List[VariableType]):

        arg_type = arg_types[0]

        if arg_type not in [VariableType.IMAGE_CUBE, VariableType.IMAGE_BAND, \
                            VariableType.SPECTRUM, VariableType.NUMBER]:
            self._report_type_error(arg_type)

        # Unary negation returns the same kind of value that it is given.
        return arg_type


    def apply(self, args: List[BandMathValue]):
        '''
        Perform unary negation on the argument and return the result.
        '''

        if len(args) != 1:
            raise Exception('Unary negation requires exactly one operand')

        arg = args[0]

        if arg.type == VariableType.NUMBER:
            return BandMathValue(VariableType.NUMBER, -arg.value)

        elif arg.type in [VariableType.IMAGE_CUBE, VariableType.IMAGE_BAND,
                          VariableType.SPECTRUM]:
            arr = arg.as_numpy_array()
            result_arr = -arr
            return BandMathValue(arg.type, result_arr)

        self._report_type_error(arg.type)
