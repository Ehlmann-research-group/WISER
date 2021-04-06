import math

from typing import Optional

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

from astropy import units as u

from .generated.app_config_ui import Ui_AppConfigDialog

from .app_state import ApplicationState


class AppConfigDialog(QDialog):
    '''
    This dialog provides configuration options for the spectrum plot component,
    and for spectrum collection.
    '''

    def __init__(self, app_state, parent=None):
        super().__init__(parent=parent)
        self._ui = Ui_AppConfigDialog()
        self._ui.setupUi(self)

        self._app_state = app_state

        self._init_general_tab()
        self._init_plugins_tab()


    def _init_general_tab(self):
        #==============================
        # Error-Reporting group-box

        self._ui.ckbox_online_bug_reporting.setChecked(
            self._app_state.get_config('general.online_bug_reporting'))

        #==============================
        # Visible-Light group-box

        self._ui.ledit_red_wavelength.setValidator(QIntValidator())
        self._ui.ledit_green_wavelength.setValidator(QIntValidator())
        self._ui.ledit_blue_wavelength.setValidator(QIntValidator())

        red = self._app_state.get_config('general.red_wavelength_nm')
        green = self._app_state.get_config('general.green_wavelength_nm')
        blue = self._app_state.get_config('general.blue_wavelength_nm')

        self._ui.ledit_red_wavelength.setText(f'{red}')
        self._ui.ledit_green_wavelength.setText(f'{green}')
        self._ui.ledit_blue_wavelength.setText(f'{blue}')

        #==============================
        # Raster Display group-box

        self._ui.ledit_viewport_highlight_color.setText(
            self._app_state.get_config('raster.viewport_highlight_color'))

        self._ui.cbox_pixel_cursor_type.addItem(self.tr('Crosshair'), 'SMALL_CROSS')
        self._ui.cbox_pixel_cursor_type.addItem(self.tr('Large crosshair'), 'LARGE_CROSS')
        self._ui.cbox_pixel_cursor_type.addItem(self.tr('Crosshair with box'), 'SMALL_CROSS_BOX')

        self._ui.ledit_pixel_cursor_color.setText(
            self._app_state.get_config('raster.pixel_cursor_color'))

        self._ui.btn_viewport_highlight_color.clicked.connect(self._on_choose_viewport_highlight_color)
        self._ui.btn_pixel_cursor_color.clicked.connect(self._on_choose_pixel_cursor_color)

        # Fetch the cursor type as a string
        cursor = self._app_state.get_config('raster.pixel_cursor_type')
        index = self._ui.cbox_pixel_cursor_type.findData(cursor)
        if index == -1:
            index = 0
        self._ui.cbox_pixel_cursor_type.setCurrentIndex(index)

        #==============================
        # New Spectra group-box

        self._ui.ledit_aavg_x.setValidator(QIntValidator(1, 99))
        self._ui.ledit_aavg_y.setValidator(QIntValidator(1, 99))

        self._ui.ledit_aavg_x.setText(
            str(self._app_state.get_config('spectra.default_area_avg_x')))
        self._ui.ledit_aavg_y.setText(
            str(self._app_state.get_config('spectra.default_area_avg_y')))

        self._ui.cbox_default_avg_mode.addItem(self.tr('Mean'  ), 'MEAN')
        self._ui.cbox_default_avg_mode.addItem(self.tr('Median'), 'MEDIAN')

        # Fetch the mode as a string
        mode = self._app_state.get_config('spectra.default_area_avg_mode')
        index = self._ui.cbox_default_avg_mode.findData(mode)
        if index == -1:
            index = 0
        self._ui.cbox_default_avg_mode.setCurrentIndex(index)


    def _init_plugins_tab(self):
        plugin_paths = self._app_state.get_config('plugin_paths')
        for p in plugin_paths:
            item = QListWidgetItem(p)
            item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            self._ui.list_plugin_paths.addItem(item)

        self._ui.btn_del_plugin_path.setEnabled(False)

        self._ui.list_plugin_paths.itemSelectionChanged.connect(self._on_plugin_path_selection_changed)
        self._ui.btn_add_plugin_path.clicked.connect(self._on_add_plugin_path)
        self._ui.btn_del_plugin_path.clicked.connect(self._on_del_plugin_path)

    def _on_choose_viewport_highlight_color(self, checked):
        initial_color = QColor(self._ui.ledit_viewport_highlight_color.text())
        color = QColorDialog.getColor(parent=self, initial=initial_color)
        if color.isValid():
            self._ui.ledit_viewport_highlight_color.setText(color.name())


    def _on_choose_pixel_cursor_color(self, checked):
        initial_color = QColor(self._ui.ledit_pixel_cursor_color.text())
        color = QColorDialog.getColor(parent=self, initial=initial_color)
        if color.isValid():
            self._ui.ledit_pixel_cursor_color.setText(color.name())


    def _on_plugin_path_selection_changed(self):
        # Enable or disable the "Remove path" button based on whether something
        # is actually selected.
        self._ui.btn_del_plugin_path.setEnabled(
            len(self._ui.list_plugin_paths.selectedItems()) > 0)

    def _on_add_plugin_path(self, checked=False):
        path = QFileDialog.getExistingDirectory(parent=self,
            caption=self.tr('Choose plugin path'))

        if path:
            # Make sure the path isn't already in the list of paths.
            for i in range(self._ui.list_plugin_paths.count()):
                item = self._ui.list_plugin_paths.item(i)
                if item.text() == path:
                    QMessageBox.information(self,
                        self.tr('Path already in plugin paths'),
                        self.tr('Path "{0}" already in plugin-paths list').format(path))
                    return

            # Add the path to the list widget.
            item = QListWidgetItem(path)
            item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            self._ui.list_plugin_paths.addItem(item)

    def _on_del_plugin_path(self, checked=False):
        # Find the path or paths to be removed

        paths: List[Tuple[int, str]] = []
        for i in range(self._ui.list_plugin_paths.count()):
            item = self._ui.list_plugin_paths.item(i)
            if item.isSelected():
                paths.append( (i, item.text()) )

        # Get user confirmation

        if len(paths) == 1:
            msg = self.tr('Remove this plugin path?') + f'\n\n{paths[0][1]}'
        else:
            msg = self.tr('Remove these plugin paths?') + \
                '\n' + '\n'.join([p[1] for p in paths])

        result = QMessageBox.question(self, self.tr('Remove plugin paths?'), msg)
        if result != QMessageBox.Yes:
            return  # User decided not to remove the path(s).

        # Delete all affected plugin paths.  Delete in decreasing index order,
        # so that indexes aren't shifted/invalidated by deleting lower-index
        # entries.
        indexes = [p[0] for p in paths]
        indexes.sort(reverse=True)
        for i in indexes:
            item = self._ui.list_plugin_paths.item(i)
            if item.isSelected():
                self._ui.list_plugin_paths.takeItem(i)

        # Now, no paths should be selected
        self._ui.btn_del_plugin_path.setEnabled(False)


    def accept(self):

        # Verify values

        aavg_x = int(self._ui.ledit_aavg_x.text())
        aavg_y = int(self._ui.ledit_aavg_y.text())

        if aavg_x % 2 != 1 or aavg_y % 2 != 1:
            QMessageBox.critical(self, self.tr('Default area-average values'),
                self.tr('Default area-average values must be odd.'), QMessageBox.Ok)
            return

        # Apply values

        #==============================
        # Error-Reporting group-box

        self._app_state.set_config('general.online_bug_reporting',
            self._ui.ckbox_online_bug_reporting.isChecked())

        #==============================
        # Visible-Light group-box

        self._app_state.set_config('general.red_wavelength_nm',
            int(self._ui.ledit_red_wavelength.text()))

        self._app_state.set_config('general.green_wavelength_nm',
            int(self._ui.ledit_green_wavelength.text()))

        self._app_state.set_config('general.blue_wavelength_nm',
            int(self._ui.ledit_blue_wavelength.text()))

        #==============================
        # Raster Display group-box

        self._app_state.set_config('raster.viewport_highlight_color',
            self._ui.ledit_viewport_highlight_color.text())

        cursor = self._ui.cbox_pixel_cursor_type.currentData()
        self._app_state.set_config('raster.pixel_cursor_type', cursor)

        self._app_state.set_config('raster.pixel_cursor_color',
            self._ui.ledit_pixel_cursor_color.text())

        #==============================
        # New Spectra group-box

        self._app_state.set_config('spectra.default_area_avg_x', aavg_x)
        self._app_state.set_config('spectra.default_area_avg_y', aavg_y)

        mode = self._ui.cbox_default_avg_mode.currentData()
        self._app_state.set_config('spectra.default_area_avg_mode', mode)

        #==============================
        # All done!

        super().accept()
