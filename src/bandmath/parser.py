import enum

from typing import Dict, List, Set

import lark


class VariableType(enum.Enum):
    '''
    Types of variables that are supported by the band-math functionality.
    '''

    IMAGE_CUBE = 1

    IMAGE_BAND = 2

    SPECTRUM = 3

    REGION_OF_INTEREST = 4


class VariableCollector(lark.visitors.Visitor):
    '''
    A Lark tree visitor that collects all variable names in an expression.
    '''
    def __init__(self):
        self.variables: Set[str] = set()

    def variable(self, tree):
        var = str(tree.children[0]).lower()
        self.variables.add(var)


def get_bandmath_variables(bandmath_expr: str) -> Set[str]:
    '''
    Parses the specified band-math expression, and returns a set of all
    variables found in the expression.  All variable names are converted to
    lowercase.
    '''
    parser = lark.Lark.open('bandmath.lark', rel_to=__file__, start='expression')
    tree = parser.parse(bandmath_expr)
    collector = VariableCollector()
    collector.visit(tree)
    return collector.variables


def verify_bandmath_expr(bandmath_expr: str, bindings: Dict[str, VariableType]) -> List[str]:
    return []
