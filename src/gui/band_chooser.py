from typing import List

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

from .generated.band_chooser_ui import Ui_BandChooserDialog

from raster.dataset import RasterDataSet, find_truecolor_bands


class BandChooserDialog(QDialog):
    '''
    A dialog that shows various important system details in a plain text box,
    so that the user can see what libraries and versions the app is using.
    There is also a "copy to clipboard" button to simplify bug reporting.
    '''

    def __init__(self, dataset: RasterDataSet, display_bands: List[int], parent=None):
        super().__init__(parent=parent)

        if len(display_bands) not in [1, 3]:
            raise ValueError('display_bands must be either 1 element or 3 elements')

        self._dataset = dataset
        self._display_bands = display_bands

        # Set up the UI state
        self._ui = Ui_BandChooserDialog()
        self._ui.setupUi(self)

        # Configure UI components based on incoming data set and display bands

        for combobox in [self._ui.cbox_red_band, self._ui.cbox_green_band,
                         self._ui.cbox_blue_band, self._ui.cbox_gray_band]:
            self._populate_combobox(combobox)

        if len(display_bands) == 3:
            self._ui.rbtn_rgb.setChecked(True)
            self._set_rgb_bands(display_bands)
            self._ui.config_stack.setCurrentWidget(self._ui.config_rgb)

        elif len(display_bands) == 1:
            self._ui.rbtn_grayscale.setChecked(True)
            self._set_grayscale_bands(display_bands)
            self._ui.config_stack.setCurrentWidget(self._ui.config_grayscale)

        self._configure_buttons()

        self._ui.chk_apply_globally.setChecked(True)

        # Hook up event handlers

        self._ui.rbtn_rgb.toggled.connect(self._on_rgb_toggled)
        self._ui.rbtn_grayscale.toggled.connect(self._on_grayscale_toggled)

        self._ui.btn_rgb_choose_defaults.clicked.connect(self._on_rgb_choose_default_bands)
        self._ui.btn_rgb_choose_visible.clicked.connect(self._on_rgb_choose_visible_bands)

        self._ui.btn_gray_choose_defaults.clicked.connect(self._on_grayscale_choose_default_bands)


    def _populate_combobox(self, combobox):
        '''
        Populate the specified combo-box with the band information from the
        data set.
        '''
        combobox.clear()

        if self._dataset is None:
            return

        # Get the band list from the dataset, and get a string description for
        # each band
        bands = self._dataset.band_list()
        items = []
        for b in bands:
            desc = b['description']
            if len(desc) == 0:
                desc = f'Band {b["index"]}'

            items.append(desc)

        # Add all the band descriptions to the combobox.
        combobox.addItems(items)

        if self._dataset.has_wavelengths():
            # Since the dataset has wavelengths, set the combobox's alignment
            # to right-aligned to make it prettier.
            for i in range(len(items)):
                combobox.setItemData(i, Qt.AlignRight, role=Qt.TextAlignmentRole)


    def _set_rgb_bands(self, display_bands: List[int]):
        assert(len(display_bands) == 3)
        self._ui.cbox_red_band.setCurrentIndex(display_bands[0])
        self._ui.cbox_green_band.setCurrentIndex(display_bands[1])
        self._ui.cbox_blue_band.setCurrentIndex(display_bands[2])


    def _set_grayscale_bands(self, display_bands: List[int]):
        assert(len(display_bands) == 1)
        self._ui.cbox_gray_band.setCurrentIndex(display_bands[0])


    def _configure_buttons(self):
        '''
        Configure whether the "choose visible-light bands" and "choose default
        bands" buttons are enabled or not, based on whether the data set has
        visible-light bands and/or default bands.
        '''

        default_bands = self._dataset.default_display_bands()

        self._ui.btn_rgb_choose_defaults.setEnabled(
            default_bands is not None and len(default_bands) == 3)

        self._ui.btn_gray_choose_defaults.setEnabled(
            default_bands is not None and len(default_bands) == 1)

        truecolor_bands = find_truecolor_bands(self._dataset)
        self._ui.btn_rgb_choose_visible.setEnabled(truecolor_bands is not None)



    def _on_rgb_toggled(self, checked):
        if checked:
            self._ui.config_stack.setCurrentWidget(self._ui.config_rgb)


    def _on_grayscale_toggled(self, checked):
        if checked:
            self._ui.config_stack.setCurrentWidget(self._ui.config_grayscale)


    def _on_rgb_choose_visible_bands(self, checked):
        pass


    def _on_rgb_choose_default_bands(self, checked):
        self._set_rgb_bands(self._dataset.default_display_bands())


    def _on_grayscale_choose_default_bands(self, checked):
        self._set_grayscale_bands(self._dataset.default_display_bands())


    def get_display_bands(self):
        if self._dataset is None:
            return None

        if self._ui.rbtn_rgb.isChecked():
            return (self._ui.cbox_red_band.currentIndex(),
                    self._ui.cbox_green_band.currentIndex(),
                    self._ui.cbox_blue_band.currentIndex(),)
        else:
            assert self._ui.rbtn_grayscale.isChecked()
            return (self._ui.cbox_gray_band.currentIndex(),)


    def apply_globally(self):
        return self._ui.chk_apply_globally.isChecked()
