import sys

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

from .constants import ImageColors


class RGBSelector(QDialog):
    '''
    For multispectral / hyperspectral raster data, this selector provides a
    simple UI for selecting what bands are displayed as the red / green / blue
    components in a display image.
    '''

    # TODO(donnie):  Should be Signal(ImageColors, int), but causes PySide2 to
    #     crash at startup.
    band_changed = Signal(int, int)

    def __init__(self, parent=None):
        super().__init__(parent=parent)

        self.slider_red = QSlider()
        self.slider_red.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        self.slider_red.sliderMoved.connect(self.red_slider_value_changed)
        self.slider_red.sliderReleased.connect(self.red_slider_value_changed)

        self.slider_grn = QSlider()
        self.slider_grn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        self.slider_grn.sliderMoved.connect(self.green_slider_value_changed)
        self.slider_grn.sliderReleased.connect(self.green_slider_value_changed)

        self.slider_blu = QSlider()
        self.slider_blu.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        self.slider_blu.sliderMoved.connect(self.blue_slider_value_changed)
        self.slider_blu.sliderReleased.connect(self.blue_slider_value_changed)

        self.label_red = QLabel(self.tr('R'))
        self.label_red.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.label_red.setBuddy(self.slider_red)

        self.label_grn = QLabel(self.tr('G'))
        self.label_red.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.label_grn.setBuddy(self.slider_grn)

        self.label_blu = QLabel(self.tr('B'))
        self.label_red.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.label_blu.setBuddy(self.slider_blu)

        # self.cb_grayscale = QCheckBox(self.tr('Grayscale'))

        layout = QGridLayout()
        layout.setContentsMargins(QMargins(0, 0, 0, 0))

        layout.addWidget(self.slider_red, 0, 0)
        layout.addWidget(self.slider_grn, 0, 1)
        layout.addWidget(self.slider_blu, 0, 2)

        layout.addWidget(self.label_red, 1, 0)
        layout.addWidget(self.label_grn, 1, 1)
        layout.addWidget(self.label_blu, 1, 2)

        # layout.addWidget(self.cb_grayscale, 2, 0, 1, 3)

        self.setLayout(layout)

        # Set some default values
        self.set_slider_ranges(100)
        self.set_slider_values(0, 0, 0)


    def set_slider_ranges(self, num_bands):
        '''
        Configure each slider to select a band-index value in the range
        [0, num_bands - 1].  Each band's slider is also configured to have a
        "single step" size of 1 and a "page step" size of num_bands / 10.
        '''
        for s in [self.slider_red, self.slider_grn, self.slider_blu]:
            s.setRange(0, num_bands - 1)
            s.setSingleStep(1)
            s.setPageStep(num_bands // 10)  # Integer division


    def set_slider_values(self, red_band, green_band, blue_band):
        self.slider_red.setValue(red_band)
        self.slider_grn.setValue(green_band)
        self.slider_blu.setValue(blue_band)

    @Slot()
    def red_slider_value_changed(self):
        self.band_changed.emit(ImageColors.RED, self.slider_red.value())

    @Slot()
    def green_slider_value_changed(self):
        self.band_changed.emit(ImageColors.GREEN, self.slider_grn.value())

    @Slot()
    def blue_slider_value_changed(self):
        self.band_changed.emit(ImageColors.BLUE, self.slider_blu.value())


if __name__ == '__main__':
    app = QApplication(sys.argv)

    ui = RGBSelector()
    ui.show()

    sys.exit(app.exec_())
