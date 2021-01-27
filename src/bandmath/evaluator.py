from typing import Any, Callable, Dict, List, Set

import lark


class BandMathEvaluator(lark.visitors.Transformer):
    '''
    A Lark Transformer for evaluating band-math expressions.
    '''
    def __init__(self, variables: Dict[str, Any], functions: Dict[str, Callable]):
        self._variables = variables
        self._functions = functions

    def comparison(self, args):
        print(f'TODO:  comparison({args})')

    def add_oper(self, args):
        print(f'TODO:  add_oper({args})')

    def mul_oper(self, args):
        print(f'TODO:  mul_oper({args})')

    def unary_op(self, args):
        print(f'TODO:  unary_op({args})')

    def variable(self, args):
        print(f'variable({args})')
        return self._variables[args[0]]

    def function(self, args):
        print(f'TODO:  function({args})')

    def NAME(self, token):
        return str(token)

    def NUMBER(self, token):
        return float(token)


def eval_bandmath_expr(bandmath_expr: str, variables: Dict[str, Any],
        functions: Dict[str, Callable]):
    # TODO(donnie):  Implement
    return 0
