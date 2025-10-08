from typing import Dict, List, Set

import lark

from .types import VariableType


class VariableCollector(lark.visitors.Visitor):
    """
    A Lark tree visitor that collects all variable names in an expression.
    """

    def __init__(self):
        self.variables: Set[str] = set()

    def variable(self, tree):
        var = str(tree.children[0]).lower()
        self.variables.add(var)


def parse_bandmath(bandmath_expr: str) -> lark.Tree:
    """
    Parses the specified band-math expression, and returns the Lark parse tree.
    """
    parser = lark.Lark.open("bandmath.lark", rel_to=__file__, start="expression")
    return parser.parse(bandmath_expr)


def bandmath_parses(bandmath_expr: str) -> bool:
    """
    Returns True if the band-math expression parses, or False if it does not
    parse.
    """
    try:
        tree = parse_bandmath(bandmath_expr)  # noqa: F841
        return True

    except:
        return False


def get_bandmath_variables(bandmath_expr: str) -> Set[str]:
    """
    Parses the specified band-math expression, and returns a set of all
    variables found in the expression.  All variable names are converted to
    lowercase.
    """
    tree = parse_bandmath(bandmath_expr)
    collector = VariableCollector()
    collector.visit(tree)
    return collector.variables


def verify_bandmath_expr(bandmath_expr: str, bindings: Dict[str, VariableType]) -> List[str]:
    return []
