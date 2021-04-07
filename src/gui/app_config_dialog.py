import math
import sys

from typing import List, Optional, Tuple

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

from astropy import units as u

from .generated.app_config_ui import Ui_AppConfigDialog

from .app_state import ApplicationState

import plugins


def qlistwidget_to_list(list_widget: QListWidget) -> List[str]:
    result: List[str] = []
    for i in range(list_widget.count()):
        result.append(list_widget.item(i).text())

    return result


def qlistwidget_selections(list_widget: QListWidget) -> List[Tuple[int, str]]:
    result = []
    for i in range(list_widget.count()):
        item = list_widget.item(i)
        if item.isSelected():
            result.append( (i, item.text()) )

    return result


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

        # Duplicate the initial plugin-paths and plugin lists, so that we can
        # act based on the changes made by the user.  Also, duplicate the
        # Python system path, so that we can restore it when needed.
        self._initial_sys_path: List[str] = list(sys.path)
        self._initial_plugin_paths: List[str] = \
            list(self._app_state.get_config('plugin_paths'))
        self._initial_plugins: List[str] = \
            list(self._app_state.get_config('plugin_paths'))

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
        # Plugin paths

        plugin_paths = self._app_state.get_config('plugin_paths')
        for p in plugin_paths:
            item = QListWidgetItem(p)
            item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            self._ui.list_plugin_paths.addItem(item)

        self._ui.btn_del_plugin_path.setEnabled(False)

        self._ui.list_plugin_paths.itemSelectionChanged.connect(self._on_plugin_path_selection_changed)
        self._ui.btn_add_plugin_path.clicked.connect(self._on_add_plugin_path)
        self._ui.btn_del_plugin_path.clicked.connect(self._on_del_plugin_path)

        # Plugins

        plugins = self._app_state.get_config('plugins')
        for p in plugins:
            item = QListWidgetItem(p)
            item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            self._ui.list_plugins.addItem(item)

        self._ui.btn_del_plugin.setEnabled(False)

        self._ui.list_plugins.itemSelectionChanged.connect(self._on_plugin_selection_changed)
        self._ui.btn_add_plugin.clicked.connect(self._on_add_plugin)
        self._ui.btn_del_plugin.clicked.connect(self._on_del_plugin)
        self._ui.btn_verify_plugins.clicked.connect(self._on_verify_plugins)


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

    #========================================================================
    # PLUGIN PATH UI
    #========================================================================

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
            if path in qlistwidget_to_list(self._ui.list_plugin_paths):
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
        paths: List[Tuple[int, str]] = qlistwidget_selections(self._ui.list_plugin_paths)

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
        for i in sorted([p[0] for p in paths], reverse=True):
            self._ui.list_plugin_paths.takeItem(i)

        # Now, no paths should be selected
        self._ui.btn_del_plugin_path.setEnabled(False)

    #========================================================================
    # PLUGIN UI
    #========================================================================

    def _on_plugin_selection_changed(self):
        # Enable or disable the "Remove plugin" button based on whether
        # something is actually selected.
        self._ui.btn_del_plugin.setEnabled(
            len(self._ui.list_plugins.selectedItems()) > 0)

    def _on_add_plugin(self, checked=False):
        (plugin, success) = QInputDialog.getText(self,
            self.tr('Plugin class name'),
            self.tr('Enter fully-qualified name of plugin class'))

        if success:
            # Make sure the plugin isn't already in the list of plugins.
            if plugin in qlistwidget_to_list(self._ui.list_plugins):
                QMessageBox.information(self,
                    self.tr('Plugin already included'),
                    self.tr('Plugin "{0}" already in plugin list').format(plugin))
                return

            # Add the plugin to the list widget.
            item = QListWidgetItem(plugin)
            item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            self._ui.list_plugins.addItem(item)

    def _on_del_plugin(self, checked=False):
        # Find the plugin(s) to be removed
        plugins: List[Tuple[int, str]] = qlistwidget_selections(self._ui.list_plugins)

        # Get user confirmation
        if len(plugins) == 1:
            msg = self.tr('Remove this plugin?') + f'\n\n{plugins[0][1]}'
        else:
            msg = self.tr('Remove these plugins?') + \
                '\n' + '\n'.join([p[1] for p in plugins])

        result = QMessageBox.question(self, self.tr('Remove plugins?'), msg)
        if result != QMessageBox.Yes:
            return  # User decided not to remove the plugin(s).

        # Delete all affected plugins.  Delete in decreasing index order,
        # so that indexes aren't shifted/invalidated by deleting lower-index
        # entries.
        for i in sorted([p[0] for p in plugins], reverse=True):
            self._ui.list_plugins.takeItem(i)

        # Now, no plugins should be selected
        self._ui.btn_del_plugin.setEnabled(False)

    def _on_verify_plugins(self, checked=False):

        if self._ui.list_plugins.count() == 0:
            QMessageBox.information(self, self.tr('No plugins'),
                self.tr('No plugins have been specified.'))
            return

        # Create a new list of system paths with the updated plugin paths.

        paths = self._initial_sys_path[:]
        for p in self._initial_plugin_paths:
            try:
                paths.remove(p)
            except ValueError:
                pass

        for p in qlistwidget_to_list(self._ui.list_plugin_paths):
            if p not in paths:
                paths.append(p)

        sys.path = paths

        # Try to instantiate each plugin class, and verify that it is of the
        # correct type

        issues = []
        for p in qlistwidget_to_list(self._ui.list_plugins):
            try:
                inst = plugins.instantiate(p)

                if (not isinstance(inst, plugins.ToolsMenuPlugin) and
                    not isinstance(inst, plugins.ContextMenuPlugin) and
                    not isinstance(inst, plugins.BandMathPlugin)):
                    msg = self.tr('Class "{0}" isn\'t a recognized plugin type')
                    issues.append(msg.format(p))

            except Exception as e:
                msg = self.tr('Can\'t instantiate plugin "{0}":  {1}')
                issues.append(msg.format(p, e))

        if issues:
            QMessageBox.warning(self, self.tr('Plugin issues found'),
                self.tr('Found these plugin issues:') + '\n\n' +
                '\n'.join(issues))

        else:
            QMessageBox.information(self, self.tr('No plugin issues found'),
                self.tr('No plugin issues found!'))

        # Restore the original system path
        sys.path = self._initial_sys_path[:]

    #========================================================================
    # OTHER OPERATIONS
    #========================================================================

    def accept(self):

        #=======================================================================
        # Verify values

        #==============================
        # New Spectra group-box

        aavg_x = int(self._ui.ledit_aavg_x.text())
        aavg_y = int(self._ui.ledit_aavg_y.text())

        if aavg_x % 2 != 1 or aavg_y % 2 != 1:
            QMessageBox.critical(self, self.tr('Default area-average values'),
                self.tr('Default area-average values must be odd.'), QMessageBox.Ok)
            return

        #==============================
        # Plugin details group-box

        # TODO(donnie):  Validate that plugins can be loaded?

        #=======================================================================
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
        # Plugin details group-box

        plugin_paths = qlistwidget_to_list(self._ui.list_plugin_paths)
        plugins = qlistwidget_to_list(self._ui.list_plugins)

        #==============================
        # All done!

        super().accept()
