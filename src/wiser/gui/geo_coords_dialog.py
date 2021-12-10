from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

from osgeo import gdal, osr

from .generated.geo_coords_dialog_ui import Ui_GeoCoordsDialog

from .geo_coords_config import *


class GeoCoordsDialog(QDialog):
    # Signal:  The user changed the configuration
    config_changed = Signal(int, object)

    # Signal:  Go to a specific coordinate (arg 2) on a specific dataset (arg 1).
    goto_coord = Signal(int, tuple)


    def __init__(self, dataset, config, parent=None):
        super().__init__(parent=parent)

        # Set up the UI state
        self._ui = Ui_GeoCoordsDialog()
        self._ui.setupUi(self)

        self._dataset = dataset
        self._initial_config = config

        # TODO(donnie):  Don't want to set a validator on these fields, if users
        #     want to enter in degrees/minutes/seconds.
        # self._ui.ledit_axis_1.setValidator(QDoubleValidator())
        # self._ui.ledit_axis_2.setValidator(QDoubleValidator())

        self._populate_crs_combobox(self._ui.cbox_display_coords)
        self._populate_crs_combobox(self._ui.cbox_goto_coords)

        self._ui.ckbox_deg_min_sec.setChecked(config.use_deg_min_sec)
        self._ui.ckbox_neg_degrees.setChecked(config.use_negative_degrees)

        self._update_goto_coords_ui()

        self._ui.cbox_display_coords.activated.connect(self._on_display_crs_changed)
        self._ui.cbox_goto_coords.activated.connect(self._on_goto_crs_changed)

        self._ui.ckbox_deg_min_sec.clicked.connect(self._on_display_ckbox_changed)
        self._ui.ckbox_neg_degrees.clicked.connect(self._on_display_ckbox_changed)

        self._ui.btn_goto_coord.clicked.connect(self._on_goto_coordinates)



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

            units = get_srs_units(spatial_ref)
            if units is None:
                units = self.tr('(unknown)')

            self._ui.lbl_axis_units.setText(units)


    def _get_current_config(self):
        config = CoordinateDisplayConfig(self._ui.cbox_display_coords.currentData())

        config.use_deg_min_sec = self._ui.ckbox_deg_min_sec.isChecked()
        config.use_negative_degrees = self._ui.ckbox_neg_degrees.isChecked()

        # TODO(donnie):  Get other config details
        return config


    def _on_display_crs_changed(self, index):
        config = self._get_current_config()
        self.config_changed.emit(self._dataset.get_id(), self._get_current_config())

        is_geo_crs = (config.spatial_ref is not None and
                      config.spatial_ref.IsGeographic())

        for w in [self._ui.ckbox_deg_min_sec, self._ui.ckbox_neg_degrees]:
            w.setEnabled(is_geo_crs)


    def _on_display_ckbox_changed(self, checked=False):
        self.config_changed.emit(self._dataset.get_id(), self._get_current_config())


    def _on_goto_crs_changed(self, index):
        self._update_goto_coords_ui()

    def _on_goto_coordinates(self, checked=False):
        print('TODO:  Goto coordinates ' +
              f'({self._ui.ledit_axis_1.text()}, {self._ui.ledit_axis_2.text()}), ' +
              f'SRS = \n{self._ui.cbox_goto_coords.currentData()}')

        ds_spatial_ref = self._dataset.get_spatial_ref()
        coords_ref = self._ui.cbox_goto_coords.currentData()

        # TODO(donnie):  This.
        geo_x = float(self._ui.ledit_axis_1.text())
        geo_y = float(self._ui.ledit_axis_2.text())
        print(f'Geo Coords = {geo_x}, {geo_y}')

        if not coords_ref.IsSame(ds_spatial_ref):
            xform2 = osr.CoordinateTransformation(coords_ref, ds_spatial_ref)
            print(f'Inv Geo Transform 2 = {xform2}')
            geo_coords = xform2.TransformPoint(geo_x, geo_y)
            # TransformPoint() returns a 3-tuple, so convert back to a 2-list.
            geo_x = geo_coords[0]
            geo_y = geo_coords[1]
            print(f'Geo Coords 2 = {geo_x}, {geo_y}')

        xform = self._dataset.get_geo_transform()
        print(f'Geo Transform = {xform}')
        xform = gdal.InvGeoTransform(xform)
        print(f'Inv Geo Transform = {xform}')
        result = gdal.ApplyGeoTransform(xform, geo_x, geo_y)
        pixel_x = result[0]
        pixel_y = result[1]

        print(f'Pixel Coords = {pixel_x}, {pixel_y}')

        if (pixel_x < 0 or pixel_x >= self._dataset.get_width() or
            pixel_y < 0 or pixel_y >= self._dataset.get_height()):
            QMessageBox.warning(self, self.tr('Coordinates Outside Image'),
                self.tr('The specified coordinates are outside the current image.'))

        else:
            self.goto_coord.emit(self._dataset.get_id(), (pixel_x, pixel_y) )


    def reject(self):
        super().reject()
        # Make sure to restore the initial configuration for coordinate display
        self.config_changed.emit(self._dataset.get_id(), self._initial_config)
