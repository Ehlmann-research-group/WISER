from typing import Set

import lark


class VariableCollector(lark.visitors.Visitor):
    '''
    A Lark tree visitor that collects all variable names in an expression.
    '''
    def __init__(self):
        self.variables: Set[str] = set()

    def variable(self, tree):
        var = str(tree.children[0]).lower()
        self.variables.add(var)


def get_variables(bandmath_expr: str) -> Set[str]:
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
