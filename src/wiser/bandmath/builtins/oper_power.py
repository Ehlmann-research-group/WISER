from typing import List

from wiser.bandmath import VariableType, BandMathValue
from wiser.bandmath.functions import BandMathFunction


class OperatorPower(BandMathFunction):
    '''
    Binary power/root operator.
    '''

    def _report_type_error(self, lhs_type, rhs_type):
        raise TypeError(f'Operands {lhs_type} and {rhs_type} not compatible for **')


    def get_result_type(self, arg_types: List[VariableType]):
        raise NotImplementedError()

    def apply(self, args: List[BandMathValue]):
        '''
        Add the LHS and RHS and return the result.
        '''

        if len(args) != 2:
            raise Exception('** requires exactly two arguments')

        lhs = args[0]
        rhs = args[1]

        raise NotImplementedError()
