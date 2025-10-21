import unittest

import util


class TestUtil(unittest.TestCase):
    def test_closest_value(self):
        # Some simple cases with floating point values
        self.assertEqual(util.closest_value([1.0, 2.0, 3.0], 0.0), 1.0)
        self.assertEqual(util.closest_value([1.0, 2.0, 3.0], 1.0), 1.0)
        self.assertEqual(util.closest_value([1.0, 2.0, 3.0], 2.0), 2.0)
        self.assertEqual(util.closest_value([1.0, 2.0, 3.0], 3.0), 3.0)
        self.assertEqual(util.closest_value([1.0, 2.0, 3.0], 4.0), 3.0)

        # Some simple cases with integer values
        self.assertEqual(util.closest_value([1, 2, 3], 0), 1)
        self.assertEqual(util.closest_value([1, 2, 3], 1), 1)
        self.assertEqual(util.closest_value([1, 2, 3], 2), 2)
        self.assertEqual(util.closest_value([1, 2, 3], 3), 3)
        self.assertEqual(util.closest_value([1, 2, 3], 4), 3)

        # Some cases with mixed types
        self.assertEqual(util.closest_value([1.0, 2, 3.0], 2.0), 2)
        self.assertEqual(util.closest_value([1.0, 2, 3.0], 1.0), 1.0)
        self.assertEqual(util.closest_value([1.0, 2, 3.0], 3), 3.0)

        # Some edge cases
        self.assertIsNone(util.closest_value([], 123))
        self.assertEqual(util.closest_value([123], 123), 123)
        self.assertEqual(util.closest_value([123], -800), 123)


if __name__ == "__main__":
    unittest.main()
