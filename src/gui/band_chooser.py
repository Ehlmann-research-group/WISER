from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *


class BandChooser(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent=parent)

        # Internal State

        self._dataset = None
        self._display_bands = []

        # UI Widgets

        self._rb_rgb_display = QRadioButton(self.tr('Red/Green/Blue Display'), self)
        self._rb_grayscale_display = QRadioButton(self.tr('Grayscale Display'), self)

        self._lbl_band_red = QLabel(self.tr('Red'))
        self._lbl_band_grn = QLabel(self.tr('Green'))
        self._lbl_band_blu = QLabel(self.tr('Blue'))

        self._cbox_band_red = QComboBox()
        self._cbox_band_grn = QComboBox()
        self._cbox_band_blu = QComboBox()

        self._lbl_band_grayscale = QLabel(self.tr('Grayscale'))
        self._cbox_band_grayscale = QComboBox()

        # Widget Layout

        layout = QGridLayout(self)

        layout.addWidget(self._rb_rgb_display, 0, 0, columnSpan=2)

        layout.addWidget(self._lbl_band_red, 1, 0)
        layout.addWidget(self._cbox_band_red, 1, 1)

        layout.addWidget(self._lbl_band_grn, 2, 0)
        layout.addWidget(self._cbox_band_grn, 2, 1)

        layout.addWidget(self._lbl_band_blu, 3, 0)
        layout.addWidget(self._cbox_band_blu, 3, 1)

        layout.addWidget(self._rb_grayscale_display, 4, 0, columnSpan=2)

        layout.addWidget(self._lbl_band_grayscale, 5, 0)
        layout.addWidget(self._cbox_band_grayscale, 5, 1)

        self.setLayout(layout)

        # TODO:  ENABLE/DISABLE WIDGETS IF DATASET NOT PRESENT


    def _populate_comboboxes(self):
        boxes = [
            self._cbox_band_red, self._cbox_band_grn, self._cbox_band_blu,
            self._cbox_band_grayscale
        ]

        # Clear all combo-boxes
        for box in boxes:
            box.clear()

        if self._dataset is None:
            return

        # Populate all of the combo-boxes with the band list
        bands = self._dataset.band_list()

        items = [b['description'] for b in bands]
        for box in boxes:
            box.addItems(items)


    def set_dataset(self, dataset):
        self._dataset = dataset
        self._populate_comboboxes()

        # TODO:  ENABLE/DISABLE WIDGETS IF DATASET NOT PRESENT


    def set_display_bands(self, display_bands):
        if len(display_bands) not in [1, 3]:
            raise ValueError(f'Invalid number of display bands specified:  {display_bands}')

        self._display_bands = display_bands

        # Choose the display bands in the corresponding combo-boxes

        if len(self._display_bands) == 3:
            # RGB display

            self._rb_rgb_display.setChecked(True)
            self._cbox_band_red.setCurrentIndex(self._display_bands[0])
            self._cbox_band_grn.setCurrentIndex(self._display_bands[1])
            self._cbox_band_blu.setCurrentIndex(self._display_bands[2])

        else:
            assert len(self._display_bands) == 1
            self._rb_grayscale_display.setChecked(True)
            self._cbox_band_grayscale.setCurrentIndex(self._display_bands[0])


    def _update_enabled_state(self):
        if self._dataset is not None:
            # Enable / disable widgets based on selection.
            if self._rb_rgb_display.isChecked():
                self._cbox_band_red.setEnabled(True)
                self._cbox_band_grn.setEnabled(True)
                self._cbox_band_blu.setEnabled(True)
                self._cbox_band_grayscale.setEnabled(False)

            else:
                self._cbox_band_red.setEnabled(False)
                self._cbox_band_grn.setEnabled(False)
                self._cbox_band_blu.setEnabled(False)
                self._cbox_band_grayscale.setEnabled(True)
        else:
            # No dataset.
            self._cbox_band_red.setEnabled(False)
            self._cbox_band_grn.setEnabled(False)
            self._cbox_band_blu.setEnabled(False)
            self._cbox_band_grayscale.setEnabled(False)


    def get_display_bands(self):
        if self._dataset is None:
            return None

        if self._rb_rgb_display.isChecked():
            return [self._cbox_band_red.currentIndex(),
                    self._cbox_band_grn.currentIndex(),
                    self._cbox_band_blu.currentIndex()]

        else:
            assert self._rb_grayscale_display.isChecked()
            return [self._cbox_band_grayscale.currentIndex()]


class BandChooserDialog(QDialog):
    def __init__(self, dataset, display_bands, parent=None):
        super().__init__(parent=parent)
        self.setWindowTitle(self.tr('Choose Bands to Display'))

        layout = QVBoxLayout(self)

        self._band_chooser = BandChooser(self)
        self._band_chooser.set_dataset(dataset)
        self._band_chooser.set_display_bands(display_bands)

        layout.addWidget(self._band_chooser)

        # Dialog buttons - hook to built-in QDialog functions
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel,
            Qt.Horizontal, self)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout.addWidget(buttons)

    def get_display_bands(self):
        return self._band_chooser.get_display_bands()


class BandChooserAction(QWidgetAction):
    '''
    This class is used to pop up the Band Chooser widget from a toolbar button.
    '''

    def createWidget(self, parent):
        return BandChooser(parent=parent)
