# stretch.py -- applying modifications to the display image

import numpy as np
from PySide2.QtCore import QObject, Signal

class StretchBase(QObject):
    # Base class for stretch objects

    def apply(self, input):
        out = input.copy() # placeholder to show the ins and outs
        return out

class StretchLinear(StretchBase):
    """ Linear stretches """
    _slope = 1.
    _offset = 0.

    # Constructor
    def __init__(self):
        StretchBase.__init__(self)
        self._slope = 1.
        self._offset = 0.
    
    # Signals
    lowerChanged = Signal(int)
    upperChanged = Signal(int)
    
    def apply(self, a):
        out = a.copy()
        out *= self._slope
        out += self._offset
        np.clip(out, 0., 1., out=out)
        return out
    
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
        if upper == lower:
            self._slope = 1.
        else:
            self._slope = 1. / ((upper - lower) / float(range_max))

        self._offset = -lower * self._slope

    def calculate_from_pct(self, pixels, bins, pct):
        targetCount = (pct / 2.) * pixels
        lower = self.find_limit(targetCount, bins, True)
        upper = self.find_limit(targetCount, bins, False)
        self.calculate_from_limits(lower, upper, len(bins))
        print("Pixels: {}, Lower: {}, Upper: {}, slope: {}, offset: {}".format(pixels, lower, upper, self._slope, self._offset))
