from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

from .generated.geo_coords_dialog_ui import Ui_GeoCoordsDialog

from .geo_coords_config import CoordinateDisplayConfig


class GeoCoordsDialog(QDialog):
    # Signal:  The user changed the configuration
    config_changed = Signal(int, object)

    # Signal:  Go to a specific coordinate (arg 2), using a specific
    #          spatial-reference-system (arg 1).
    goto_coord = Signal(object, tuple)


    def __init__(self, dataset, config, parent=None):
        super().__init__(parent=parent)

        # Set up the UI state
        self._ui = Ui_GeoCoordsDialog()
        self._ui.setupUi(self)

        self._dataset = dataset
        self._initial_config = config

        self._ui.ledit_axis_1.setValidator(QDoubleValidator())
        self._ui.ledit_axis_2.setValidator(QDoubleValidator())

        self._populate_crs_combobox(self._ui.cbox_display_coords)
        self._populate_crs_combobox(self._ui.cbox_goto_coords)

        self._ui.ckbox_deg_min_sec.setChecked(config.use_deg_min_sec)
        self._ui.ckbox_neg_degrees.setChecked(config.use_negative_degrees)

        self._update_goto_coords_ui()

        self._ui.cbox_display_coords.activated.connect(self._on_display_crs_changed)
        self._ui.cbox_goto_coords.activated.connect(self._on_goto_crs_changed)

        self._ui.ckbox_deg_min_sec.clicked.connect(self._on_display_ckbox_changed)
        self._ui.ckbox_neg_degrees.clicked.connect(self._on_display_ckbox_changed)



    def _populate_crs_combobox(self, combobox):
        def _add_cbox_item(text, spatial_ref):
            combobox.addItem(
                self.tr(text).format(spatial_ref.GetName()),
                spatial_ref)

        spatial_ref = self._dataset.get_spatial_ref()
        if spatial_ref is not None:
            if spatial_ref.IsProjected():
                _add_cbox_item('Projected CRS:  {0}', spatial_ref)
                geog_spatial_ref = spatial_ref.CloneGeogCS()
                _add_cbox_item('Geographic CRS:  {0}', geog_spatial_ref)

            elif spatial_ref.IsGeographic():
                _add_cbox_item('Geographic CRS:  {0}', spatial_ref)

            else:
                print(f'WARNING:  Unknown CRS type.\n{spatial_ref}')
                combobox.addItem(self.tr('Pixel'), None)
        else:
            combobox.addItem(self.tr('Pixel'), None)

        # Make sure to select the current configuration's spatial reference.
        if self._initial_config.spatial_ref is not None:
            for i in range(combobox.count()):
                if self._initial_config.spatial_ref.IsSame(combobox.itemData(i)):
                    combobox.setCurrentIndex(i)
                    break

    def _update_goto_coords_ui(self):
        spatial_ref = self._ui.cbox_goto_coords.currentData()
        if spatial_ref is not None:
            for i in range(spatial_ref.GetAxesCount()):
                print(f'Axis {i} name is "{spatial_ref.GetAxisName(None, i)}"')

            # TODO(donnie):  What to do if not 2 axes?
            self._ui.lbl_axis_1.setText(spatial_ref.GetAxisName(None, 0) + ':')
            self._ui.lbl_axis_2.setText(spatial_ref.GetAxisName(None, 1) + ':')

            unit_name = self.tr('unknown')
            if spatial_ref.IsProjected():
                unit_name = spatial_ref.GetLinearUnitsName()
            elif spatial_ref.IsGeographic():
                unit_name = spatial_ref.GetAngularUnitsName()

            self._ui.lbl_axis_units.setText(unit_name)


    def _get_current_config(self):
        config = CoordinateDisplayConfig(self._ui.cbox_display_coords.currentData())

        config.use_deg_min_sec = self._ui.ckbox_deg_min_sec.isChecked()
        config.use_negative_degrees = self._ui.ckbox_neg_degrees.isChecked()

        # TODO(donnie):  Get other config details
        return config


    def _on_display_crs_changed(self, index):
        self.config_changed.emit(self._dataset.get_id(), self._get_current_config())

    def _on_display_ckbox_changed(self, checked=False):
        self.config_changed.emit(self._dataset.get_id(), self._get_current_config())

    def _on_goto_crs_changed(self, index):
        self._update_goto_coords_ui()


    def reject(self):
        super().reject()
        self.config_changed.emit(self._dataset.get_id(), self._initial_config)
