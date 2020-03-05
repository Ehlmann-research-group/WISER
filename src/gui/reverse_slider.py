# reverse_slider.py

from PySide2.QtWidgets import QSlider

class QReverseSlider(QSlider):
    """
    A class that inverts the reverses scheme of sliders on certain
    platforms (Mac and Qt native, but not Windows), so that the
    slider's groove is colored to the right of the control rather
    than to the left.
    """

    def __init__(self, obj):
        if isinstance(obj, QSlider):
            super().__init__(obj.parent())
            self.setObjectName(obj.objectName())
            self.setGeometry(obj.geometry())
            self.setCursor(obj.cursor())
            self.setMinimum(obj.minimum())
            self.setMaximum(obj.maximum())
            self.setPageStep(obj.pageStep())
            self.setValue(obj.value())
            self.setSliderPosition(obj.sliderPosition())
            self.setOrientation(obj.orientation())
            self.setInvertedAppearance(obj.invertedAppearance())
            self.setInvertedControls(obj.invertedControls())
        else:
            super().__init__(obj)
        
    def value(self):
        return super().maximum() - super().value()
    
    def setValue(self, value):
        super().setValue(super().maximum() - value)
    
    def sliderPosition(self):
        return super().maximum() - super().sliderPosition()
    
    def setSliderPosition(self, value):
        super().setSliderPosition(super().maximum() - value)
    
    def __eq__(self, value):
        return super().__eq__(super().maximum() - value)
    
    def __ge__(self, value):
        return super().__ge__(super().maximum() - value)
    
    def __gt__(self, value):
        return super().__gt__(super().maximum() - value)
    
    def __le__(self, value):
        return super().__le__(super().maximum() - value)
    
    def __lt__(self, value):
        return super().__lt__(super().maximum() - value)
    
    def __ne__(self, value):
        return super().__ne__(super().maximum() - value)
    
    def triggerAction(self, action):
        if action == SliderSingleStepAdd:
            super().triggerAction(SliderSingleStepSub)
        elif action == SliderSingleStepSub:
            super().triggerAction(SliderSingleStepAdd)
        elif action == SliderPageStepAdd:
            super().triggerAction(SliderPageStepSub)
        elif action == SliderPageStepSub:
            super().triggerAction(SliderPageStepAdd)
        elif action == SliderToMinimum:
            super().triggerAction(SliderToMaximum)
        elif action == SliderToMaximum:
            super().triggerAction(SliderToMinimum)
        else:
            super().triggerAction(action)
    
