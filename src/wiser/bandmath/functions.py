from typing import Dict

import numpy as np

from .types import BandMathValue, BandMathFunction
from .builtins.oper_builtin import OperatorTrigFunction, OperatorTrigFunctionTwoArgs, OperatorDotProduct

def get_builtin_functions() -> Dict[str, BandMathFunction]:
    '''
    This function returns a dictionary of built-in functions supported by the
    band-math evaluator.
    '''
    return {
        'tan': OperatorTrigFunction(np.ma.tan),
        'sin': OperatorTrigFunction(np.ma.sin),
        'cos': OperatorTrigFunction(np.ma.cos),
        'arctan2': OperatorTrigFunctionTwoArgs(np.ma.arctan2),
        'arctan': OperatorTrigFunction(np.ma.arctan),
        'arcsin': OperatorTrigFunction(np.ma.arcsin),
        'arccos': OperatorTrigFunction(np.ma.arccos),
        'dotprod': OperatorDotProduct(),
    }

def verify_function_args(args):
    '''
    This helper function checks the arguments passed to a band-math function
    implementation.  It performs these checks:

    *   All arguments must be of type BandMathValue
    '''
    for arg in args:
        if not isinstance(arg, BandMathValue):
            raise TypeError('All arguments must be of type BandMathValue')
