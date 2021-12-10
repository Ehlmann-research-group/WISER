from typing import Any, Dict, List, Optional, Tuple

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

from osgeo import gdal, osr

from .generated.image_coords_widget_ui import Ui_ImageCoordsWidget

from .geo_coords_config import CoordinateDisplayConfig
from .geo_coords_dialog import GeoCoordsDialog

from wiser.raster import RasterDataSet


def to_deg_min_sec(value):
    '''
    Convert a floating-point degrees value to a tuple containing
    [degrees, minutes, seconds].
    '''
    deg = int(value)

    value = (value - int(value)) * 60
    min = int(value)

    value = (value - int(value)) * 60
    sec = value

    return (deg, min, sec)

def to_deg_min_sec_str(value, axis):
    '''
    Convert a floating-point degrees value to a string
    '{degs}°{mins}\'{secs}"'.
    '''
    dms = to_deg_min_sec(value)
    return f'{dms[0]}\N{DEGREE SIGN}{dms[1]}\'{dms[2]:.2f}"{axis}'


class ImageCoordsWidget(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent=parent)

        # Set up the UI state
        self._ui = Ui_ImageCoordsWidget()
        self._ui.setupUi(self)

        self._ui.tbtn_geo_goto.setIcon(QIcon(':/icons/geo-coords.svg'))

        self._deg_min_sec = False

        # The current dataset and pixel-coordinates being shown in the widget.
        # We record these because when the config changes, we need to update
        # the displayed information.
        self._dataset: Optional[RasterDataSet] = None
        self._pixel_coords = None

        self._display_config: Dict[int, CoordinateDisplayConfig] = {}

        self._ui.tbtn_geo_goto.clicked.connect(self._on_geo_dialog)



    def _set_geo_visible(self, visible):
        for w in [self._ui.lbl_geo, self._ui.lbl_geo_coords, self._ui.tbtn_geo_goto]:
            w.setVisible(visible)

    def _set_pixel_coords_text(self, pixel_coords):
        self._ui.lbl_pixel_coords.setText(f'({pixel_coords[0]}, {pixel_coords[1]})')

    def _set_geo_coords_text(self, pixel_coords, dataset, config):
        self._ui.lbl_geo_coords.clear()
        if config.spatial_ref is None:
            return

        xform = dataset.get_geo_transform()
        ds_spatial_ref = dataset.get_spatial_ref()

        print(f'Geo Transform = {xform}')
        geo_coords = gdal.ApplyGeoTransform(xform, pixel_coords[0], pixel_coords[1])
        print(f'Geo Coords = {geo_coords}')

        if not config.spatial_ref.IsSame(ds_spatial_ref):
            xform2 = osr.CoordinateTransformation(ds_spatial_ref, config.spatial_ref)
            print(f'Geo Transform 2 = {xform2}')
            geo_coords = xform2.TransformPoint(geo_coords[0], geo_coords[1])
            # TransformPoint() returns a 3-tuple, so convert back to a 2-list.
            geo_coords = [geo_coords[0], geo_coords[1]]
            print(f'Geo Coords 2 = {geo_coords}')

        if config.spatial_ref.IsGeographic():
            # Latitude/longitude.  Longitude comes out first, so flip 'em.

            if config.spatial_ref.GetAngularUnitsName().lower().startswith('degree'):
                ns_axis = 'N'
                ew_axis = 'E'

                if not config.use_negative_degrees:
                    print('Fixing negative degree values')
                    if geo_coords[1] < 0:
                        geo_coords[1] = -geo_coords[1]
                        ns_axis = 'S'

                    if geo_coords[0] < 0:
                        geo_coords[0] = -geo_coords[0]
                        ew_axis = 'W'

                if config.use_deg_min_sec:
                    print('Going to degrees/minutes/seconds')
                    v1 = to_deg_min_sec_str(geo_coords[1], ns_axis)
                    v2 = to_deg_min_sec_str(geo_coords[0], ew_axis)

                else:
                    # Use straight degrees
                    v1 = f'{geo_coords[1]:.6f}\N{DEGREE SIGN}{ns_axis}'
                    v2 = f'{geo_coords[0]:.6f}\N{DEGREE SIGN}{ew_axis}'

            else:
                # Unknown units.
                v1 = f'{geo_coords[1]:.6f}'
                v2 = f'{geo_coords[0]:.6f}'

        else:
            # Northing/Easting.
            v1 = f'{geo_coords[0]:.2f}E'
            v2 = f'{geo_coords[1]:.2f}N'

        self._ui.lbl_geo_coords.setText(f'({v1}, {v2})')


    def _get_config_for_dataset(self, dataset) -> CoordinateDisplayConfig:
        ds_id = dataset.get_id()
        if ds_id not in self._display_config:
            # Generate a new configuration for the dataset.
            spatial_ref = dataset.get_spatial_ref()

            # We prefer the geographic CRS over the projected CRS
            if spatial_ref is not None and spatial_ref.IsProjected():
                spatial_ref = spatial_ref.CloneGeogCS()

            self._display_config[ds_id] = CoordinateDisplayConfig(spatial_ref)

        return self._display_config[ds_id]


    def update_coords(self, dataset, pixel_coords):
        self._dataset = dataset
        self._pixel_coords = pixel_coords
        self._update_internal()


    def _update_internal(self):
        if self._dataset is None or self._pixel_coords is None:
            self.setVisible(False)
            return

        self.setVisible(True)

        config = self._get_config_for_dataset(self._dataset)

        if config.spatial_ref is None:
            self._set_pixel_coords_text(self._pixel_coords)
            self._set_geo_visible(False)

        else:
            self._set_pixel_coords_text(self._pixel_coords)
            self._set_geo_coords_text(self._pixel_coords, self._dataset, config)
            self._set_geo_visible(True)


    def _on_geo_dialog(self):
        if self._dataset is None:
            return

        config = self._get_config_for_dataset(self._dataset)
        dialog = GeoCoordsDialog(self._dataset, config, parent=self)
        dialog.config_changed.connect(self._on_display_config_changed)
        dialog.goto_coord.connect(self._on_goto_coordinate)

        # corner = self.mapToGlobal(QPoint(self.width(), 0.0))
        # corner = QPoint(corner.x() - dialog.width(), corner.y() - dialog.height())
        # dialog.move(corner)

        dialog.exec()


    def _on_display_config_changed(self, ds_id, config):
        print(f'Received display-config changed notification:  {ds_id}, {config}')
        self._display_config[ds_id] = config
        self._update_internal()


    def _on_goto_coordinate(self, ds_id, coord, spatial_ref):
        print(f'TODO:  Go-to-coordinate {coord} with spatial reference:\n{spatial_ref}')
