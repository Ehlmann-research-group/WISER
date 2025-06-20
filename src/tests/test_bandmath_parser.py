import unittest


from wiser import bandmath


class TestBandmathParser(unittest.TestCase):
    """
    Exercise code in the bandmath.parser module.
    """

    def test_bandmath_getvars_literal_expr(self):
        """
        Getting variables from an expression only containing literals should
        produce an empty set.
        """
        vars = bandmath.get_bandmath_variables("15*(6-3)")
        self.assertTrue(len(vars) == 0)

    def test_bandmath_getvars_multiple_vars(self):
        """
        Getting variables from an expression reports all variables, and converts
        everything to lowercase.
        """
        vars = bandmath.get_bandmath_variables("b1*b2/B3+b4-B5")
        self.assertTrue(len(vars) == 5)
        self.assertTrue("b1" in vars)
        self.assertTrue("b2" in vars)
        self.assertTrue("b3" in vars)
        self.assertTrue("b4" in vars)
        self.assertTrue("b5" in vars)
        self.assertTrue("B3" not in vars)
        self.assertTrue("B5" not in vars)

    def test_bandmath_getvars_repeated_variables(self):
        """
        Getting variables from an expression reports each variable only once.
        """
        vars = bandmath.get_bandmath_variables("((B1-B2)/(B1+B2))*((B1-B3)/(B1+B3))")
        self.assertTrue(len(vars) == 3)
        self.assertTrue("b1" in vars)
        self.assertTrue("b2" in vars)
        self.assertTrue("b3" in vars)
