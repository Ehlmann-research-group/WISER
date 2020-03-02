# stretch.py -- applying modifications to the display image

import numpy as np
from PySide2.QtCore import QObject, Signal

class StretchBase(QObject):
    # Base class for stretch objects

    name = "Base"

    def apply(self, input):
        return input

class StretchLinear(StretchBase):
    """ Linear stretches """
    _slope = 1.
    _offset = 0.
    _lower = 0.
    _upper = 1.

    # Constructor
    def __init__(self):
        StretchBase.__init__(self)
        self._slope = 1.
        self._offset = 0.
        self._lower = 0.
        self._upper = 1.
        self.name = "Linear"
    
    # Signals
    lowerChanged = Signal(int)
    upperChanged = Signal(int)
    
    def apply(self, a):
        a *= self._slope
        a += self._offset
        np.clip(a, 0., 1., out=a)
        return a
    
    def lower(self):
        return self._lower

    def upper(self):
        return self._upper
    
    def find_limit(self, targetCount: int, bins, doLower: bool) -> int:
        if doLower:
            range_lower = 0
            range_upper = len(bins)-1
            step = 1
        else:
            range_lower = len(bins)-1
            range_upper = 0
            step = -1
        sum = 0
        for bin in range(range_lower, range_upper, step):
            sum += bins[bin]
            if sum > targetCount:
                if doLower:
                    self.lowerChanged.emit(bin * 100 / len(bins))
                else:
                    self.upperChanged.emit(bin * 100 / len(bins))
                return bin
        return range_lower
    
    def calculate_from_limits(self, lower: int, upper: int, range_max: int):
        self._lower = lower / float(range_max)
        self._upper = upper / float(range_max)
        if upper == lower:
            self._slope = 1.
        else:
            self._slope = 1. / (self._upper - self._lower)

        self._offset = -lower * self._slope

    def calculate_from_pct(self, pixels, bins, pct):
        targetCount = (pct / 2.) * pixels
        lower = self.find_limit(targetCount, bins, True)
        upper = self.find_limit(targetCount, bins, False)
        self.calculate_from_limits(lower, upper, len(bins))
        print("Pixels: {}, Lower: {}, Upper: {}, slope: {}, offset: {}".format(pixels, lower, upper, self._slope, self._offset))

class StretchHistEqualize(StretchBase):
    """ Histogram Equalization Stretches """
    _cdf = None
    _histo_edges = None

    # Constructor
    def __init__(self):
        StretchBase.__init__(self)
        self._cdf = None
        self._histo_edges = None
        self.name = "Equalize"

    def apply(self, a: np.array):
        a = np.interp(a, self._histo_edges[:-1], self._cdf)
        return a
    
    def calculate(self, bins: np.array, edges: np.array):
        self._histo_edges = edges
        # First, calculate a density probability histogram from the counts version
        # (mimics the handling of density in numpy's histogram() implementation)
        db = np.array(np.diff(edges), float)
        density_bins = bins/db/bins.sum()
        # Now calculate a cumulative distribution function and normalize it
        self._cdf = density_bins.cumsum()
        self._cdf /= self._cdf[-1]
        
