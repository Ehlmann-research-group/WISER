from typing import List, Optional, Tuple

import numpy as np

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

from astropy import units as u
import matplotlib.pyplot as plt
from matplotlib import cm

from .generated.band_chooser_ui import Ui_BandChooserDialog

from .app_state import ApplicationState

from wiser.raster.dataset import RasterDataSet, find_truecolor_bands


class BandChooserDialog(QDialog):
    """
    A dialog that shows various important system details in a plain text box,
    so that the user can see what libraries and versions the app is using.
    There is also a "copy to clipboard" button to simplify bug reporting.
    """

    def __init__(
        self,
        app_state: ApplicationState,
        dataset: RasterDataSet,
        display_bands: List[int],
        colormap: Optional[str] = None,
        can_apply_global: bool = True,
        parent=None,
    ):
        super().__init__(parent=parent)
        if len(display_bands) not in [1, 3]:
            raise ValueError("display_bands must be either 1 element or 3 elements")

        self._app_state = app_state
        self._dataset = dataset
        self._display_bands = display_bands

        # Set up the UI state
        self._ui = Ui_BandChooserDialog()
        self._ui.setupUi(self)

        # Configure UI components based on incoming data set and display bands

        for combobox in [
            self._ui.cbox_red_band,
            self._ui.cbox_green_band,
            self._ui.cbox_blue_band,
            self._ui.cbox_gray_band,
        ]:
            self._populate_combobox(combobox)

        # Populate the list of colormaps based on what matplotlib reports

        for cmap in plt.colormaps():
            self._ui.cbox_colormap_name.addItem(cmap)

        # Set up the UI based on whether we are in RGB or grayscale mode

        if len(display_bands) == 3:
            self._ui.rbtn_rgb.setChecked(True)
            self._set_rgb_bands(display_bands)
            self._ui.config_stack.setCurrentWidget(self._ui.config_rgb)

        elif len(display_bands) == 1:
            self._ui.rbtn_grayscale.setChecked(True)
            self._set_grayscale_bands(display_bands)
            self._ui.config_stack.setCurrentWidget(self._ui.config_grayscale)

            if colormap is not None:
                index = self._ui.cbox_colormap_name.findText(colormap)
                if index != -1:
                    self._ui.cbox_colormap_name.setCurrentIndex(index)

        self._configure_buttons()

        # Configure the colormap portion of the dialog.
        self._ui.lbl_colormap_display.setScaledContents(True)
        self._ui.chk_use_colormap.setChecked(colormap is not None)
        self._on_grayscale_use_colormap(colormap is not None)
        self._on_grayscale_choose_colormap(-1)  # The argument here is ignored

        if can_apply_global:
            self._ui.chk_apply_globally.setChecked(True)
        else:
            self._ui.chk_apply_globally.setEnabled(False)

        # Hook up event handlers

        self._ui.rbtn_rgb.toggled.connect(self._on_rgb_toggled)
        self._ui.rbtn_grayscale.toggled.connect(self._on_grayscale_toggled)

        self._ui.btn_rgb_choose_defaults.clicked.connect(
            self._on_rgb_choose_default_bands
        )
        self._ui.btn_rgb_choose_visible.clicked.connect(
            self._on_rgb_choose_visible_bands
        )

        self._ui.btn_gray_choose_defaults.clicked.connect(
            self._on_grayscale_choose_default_bands
        )

        self._ui.chk_use_colormap.clicked.connect(self._on_grayscale_use_colormap)
        self._ui.cbox_colormap_name.activated.connect(
            self._on_grayscale_choose_colormap
        )

    def _populate_combobox(self, combobox):
        """
        Populate the specified combo-box with the band information from the
        data set.
        """
        combobox.clear()

        if self._dataset is None:
            return

        # Get the band list from the dataset, and get a string description for
        # each band
        bands = self._dataset.band_list()
        items = []
        for b in bands:
            desc = b["description"]
            if desc:
                desc = f'Band {b["index"]}: {desc}'
            else:
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
        assert len(display_bands) == 3
        self._ui.cbox_red_band.setCurrentIndex(display_bands[0])
        self._ui.cbox_green_band.setCurrentIndex(display_bands[1])
        self._ui.cbox_blue_band.setCurrentIndex(display_bands[2])

    def _set_grayscale_bands(self, display_bands: List[int]):
        assert len(display_bands) == 1
        self._ui.cbox_gray_band.setCurrentIndex(display_bands[0])

    def _get_truecolor_bands(self) -> Optional[Tuple[int, int, int]]:
        """
        A helper function that tries to find the truecolor bands of the current
        dataset, using the app-state's current definitions of
        "red", "green" and "blue" wavelengths.
        """
        return find_truecolor_bands(
            self._dataset,
            red=self._app_state.get_config("general.red_wavelength_nm") * u.nm,
            green=self._app_state.get_config("general.green_wavelength_nm") * u.nm,
            blue=self._app_state.get_config("general.blue_wavelength_nm") * u.nm,
        )

    def _configure_buttons(self):
        """
        Configure whether the "choose visible-light bands" and "choose default
        bands" buttons are enabled or not, based on whether the data set has
        visible-light bands and/or default bands.
        """

        default_bands = self._dataset.default_display_bands()
        # print(f'band chooser:  default bands = {default_bands}')

        self._ui.btn_rgb_choose_defaults.setEnabled(
            default_bands is not None and len(default_bands) == 3
        )

        self._ui.btn_gray_choose_defaults.setEnabled(
            default_bands is not None and len(default_bands) == 1
        )

        truecolor_bands = self._get_truecolor_bands()
        self._ui.btn_rgb_choose_visible.setEnabled(truecolor_bands is not None)

    def _on_rgb_toggled(self, checked):
        if checked:
            self._ui.config_stack.setCurrentWidget(self._ui.config_rgb)

    def _on_grayscale_toggled(self, checked):
        if checked:
            self._ui.config_stack.setCurrentWidget(self._ui.config_grayscale)

    def _on_rgb_choose_visible_bands(self, checked):
        self._set_rgb_bands(self._get_truecolor_bands())

    def _on_rgb_choose_default_bands(self, checked: bool):
        self._set_rgb_bands(self._dataset.default_display_bands())

    def _on_grayscale_choose_default_bands(self, checked: bool):
        self._set_grayscale_bands(self._dataset.default_display_bands())

    def _on_grayscale_use_colormap(self, checked: bool):
        self._ui.cbox_colormap_name.setEnabled(checked)
        self._ui.lbl_colormap_display.setEnabled(checked)

    def _on_grayscale_choose_colormap(self, index: int):
        """
        Generate an image to display the currently selected colormap.
        """

        cm_name = self._ui.cbox_colormap_name.currentText()
        cmap = cm.get_cmap(cm_name, 256)
        img = QImage(cmap.N, 24, QImage.Format_RGB32)

        for x in range(cmap.N):
            rgba = cmap(x, bytes=True)
            for y in range(img.height()):
                rgb_val = np.uint32(0)
                rgb_val |= rgba[0]
                rgb_val = rgb_val << 8
                rgb_val |= rgba[1]
                rgb_val = rgb_val << 8
                rgb_val |= rgba[2]
                rgb_val |= 0xFF000000
                img.setPixel(x, y, rgb_val)

        self._ui.lbl_colormap_display.setPixmap(QPixmap.fromImage(img))

    def get_display_bands(self):
        if self._dataset is None:
            return None

        if self._ui.rbtn_rgb.isChecked():
            return (
                self._ui.cbox_red_band.currentIndex(),
                self._ui.cbox_green_band.currentIndex(),
                self._ui.cbox_blue_band.currentIndex(),
            )
        else:
            assert self._ui.rbtn_grayscale.isChecked()
            return (self._ui.cbox_gray_band.currentIndex(),)

    def apply_globally(self):
        return self._ui.chk_apply_globally.isChecked()

    def use_colormap(self):
        return (
            self._ui.rbtn_grayscale.isChecked()
            and self._ui.chk_use_colormap.isChecked()
        )

    def get_colormap_name(self) -> Optional[str]:
        """
        If a colormap should be used (i.e. the image is to be displayed in
        grayscale and the user chose to use a colormap), then this method
        returns the name of the colormap as registered with matplotlib.

        Otherwise this method returns ``None``.
        """
        cm_name: Optional[str] = None
        if self.use_colormap():
            cm_name = self._ui.cbox_colormap_name.currentText()

        return cm_name
