import unittest
from PySide2.QtCore import *

import tests.context
from wiser.gui.spectrum_plot import generate_ticks

import pytest

pytestmark = [
    pytest.mark.unit,
    pytest.mark.smoke,
]


class TestGuiSpectrumPlot(unittest.TestCase):
    """
    Exercise code in the gui.spectrum_plot module.
    """

    # ======================================================
    # gui.spectrum_plot.generate_ticks()

    def test_generate_ticks_with_endpoints(self):
        expected = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
        ticks = generate_ticks(0, 1, 0.2)
        self.assertEqual(len(expected), len(ticks))
        for i in range(len(expected)):
            self.assertAlmostEqual(ticks[i], expected[i])

    def test_generate_ticks_no_endpoints(self):
        expected = [0.2, 0.4, 0.6, 0.8]
        ticks = generate_ticks(0.1, 0.9, 0.2)
        self.assertEqual(len(expected), len(ticks))
        for i in range(len(expected)):
            self.assertAlmostEqual(ticks[i], expected[i])

    def test_generate_ticks_no_ticks(self):
        ticks = generate_ticks(0.1, 0.9, 100)
        self.assertEqual(ticks, [])
