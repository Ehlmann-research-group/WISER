import os
from typing import List, Optional, Dict, Tuple

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

from .generated.geo_referencer_dialog_ui import Ui_GeoReferencerDialog

from wiser.gui.app_state import ApplicationState
from wiser.gui.rasterview import RasterView
from wiser.gui.rasterpane import RasterPane
from wiser.gui.geo_reference_pane import GeoReferencerPane
from wiser.gui.geo_reference_task_delegate import \
    (GeoReferencerTaskDelegate, GroundControlPointPair, GroundControlPoint, \
     GroundControlPointCoordinate, PointSelectorType, PointSelector)
from wiser.gui.util import get_random_matplotlib_color, get_color_icon

from wiser.raster.dataset import RasterDataSet
from wiser.raster.dataset_impl import RasterDataImpl, GDALRasterDataImpl
from wiser.raster.utils import copy_metadata_to_gdal_dataset, set_data_ignore_of_gdal_dataset

from wiser.bandmath.utils import write_raster_to_dataset

from enum import IntEnum, Enum

from osgeo import gdal, osr, gdal_array

import numpy as np

from pyproj import CRS
from pyproj.database import get_authorities

AVAILABLE_AUTHORITIES = get_authorities()

from wiser.bandmath.builtins.constants import MAX_RAM_BYTES


class COLUMN_ID(IntEnum):
    ENABLED_COL = 0
    ID_COL = 1
    TARGET_X_COL = 2
    TARGET_Y_COL = 3
    REF_X_COL = 4
    REF_Y_COL = 5
    RESIDUAL_X_COL = 6
    RESIDUAL_Y_COL = 7
    COLOR_COL = 8
    REMOVAL_COL = 9

ID_PROPERTY = "ENTRY_ID"

# RESAMPLE_ALGORITHMS = {
#     "Nearest Neighbour": gdal.GRA_NearestNeighbour,
#     "Bilinear": gdal.GRA_Bilinear,
#     "Cubic": gdal.GRA_Cubic,
#     "Cubic Spline": gdal.GRA_CubicSpline,
#     "Lanczos": gdal.GRA_Lanczos,
#     "Average": gdal.GRA_Average,
#     "Mode": gdal.GRA_Mode
# }

RESAMPLE_ALGORITHMS = {name: getattr(gdal, name) for name in dir(gdal) if name.startswith("GRA_")}

class TRANSFORM_TYPES(Enum):
    POLY_1 = "Affine (Polynomial 1)"
    POLY_2 = "Polynomial 2"
    POLY_3 = "Polynomial 3"
    TPS = "Thin Plate Spline (TPS)"

min_points_per_transform = {
    TRANSFORM_TYPES.POLY_1: 3,
    TRANSFORM_TYPES.POLY_2: 6,
    TRANSFORM_TYPES.POLY_3: 10,
    TRANSFORM_TYPES.TPS: 10,
}

COMMON_SRS = {
    "WGS84 EPSG:4326": ('EPSG', 4326),
    "Web Mercator EPSG:3857": ('EPSG', 3857),
    "NAD83 / UTM zone 15N EPSG:26915": ('EPSG', 26915),
    # Add more as required by your application.
}

class GeoRefTableEntry:
    def __init__(self, gcp_pair: GroundControlPointPair, enabled: bool, id: int, residual_x: float, residual_y: float, color: str):
        self._gcp_pair = gcp_pair
        self._enabled = enabled
        self._id = id
        self._residual_x = residual_x
        self._residual_y = residual_y
        self._color = color  # Hex code for color

    # Getter and Setter for gcp_pair
    def get_gcp_pair(self) -> GroundControlPointPair:
        return self._gcp_pair

    def set_gcp_pair(self, gcp_pair: GroundControlPointPair):
        self._gcp_pair = gcp_pair

    # Getter and Setter for enabled
    def is_enabled(self) -> bool:
        return self._enabled

    def set_enabled(self, enabled: bool):
        self._enabled = enabled

    # Getter and Setter for id
    def get_id(self) -> int:
        return self._id

    def set_id(self, id: int):
        self._id = id

    # Getter and Setter for residuals
    def get_residual_x(self) -> float:
        return self._residual_x

    def set_residual_x(self, residual_x: float):
        self._residual_x = residual_x

    # Getter and Setter for residuals
    def get_residual_y(self) -> float:
        return self._residual_y

    def set_residual_y(self, residual_y: float):
        self._residual_y = residual_y

    # Getter and Setter for residuals
    def get_color(self) -> str:
        return self._color

    def set_color(self, color: str):
        self._color = color

    def replace_entry(self, newEntry: 'GeoRefTableEntry'):
        self.set_gcp_pair(newEntry.get_gcp_pair())
        self.set_enabled(newEntry.is_enabled())
        self.set_id(newEntry.get_id())
        self.set_residual_x(newEntry.get_residual_x())
        self.set_residual_y(newEntry.get_residual_y())

    def __str__(self):
        return (
        "=======================\n"
        f"gcp_pair: {self._gcp_pair}\n"
        f"id: {self._id}\n"
        f"enabled: {self._enabled}\n"
        f"residual-x: {self._residual_x}\n"
        f"residual-y: {self._residual_y}\n"
        "======================="
    )

class NumericDelegate(QStyledItemDelegate):
    def __init__(self, parent=None, minimum=0.0):
        super().__init__(parent)
        self._minimum = minimum

    def createEditor(self, parent, option, index):
        editor = QLineEdit(parent)
        # allow e.g. floats ≥0 with up to 10 decimals
        validator = QDoubleValidator(self._minimum, 1e10, 15, editor)
        validator.setNotation(QDoubleValidator.StandardNotation)
        editor.setValidator(validator)
        return editor


class GeoReferencerDialog(QDialog):

    gcp_pair_added = Signal(GroundControlPointPair)

    gcp_add_attempt = Signal(GroundControlPoint)

    key_press_event = Signal(QKeyEvent, PointSelector)

    def __init__(self, app_state: ApplicationState, main_view: RasterPane, parent=None):
        super().__init__(parent=parent)
        self._app_state = app_state
        self._main_view = main_view

        # Set up the UI state
        self._ui = Ui_GeoReferencerDialog()
        self._ui.setupUi(self)

        self._target_cbox = self._ui.cbox_target_dataset_chooser
        self._reference_cbox = self._ui.cbox_reference_dataset_chooser

        self._target_rasterpane = GeoReferencerPane(app_state=app_state, pane_type = PointSelectorType.TARGET_POINT_SELECTOR)
        self._reference_rasterpane = GeoReferencerPane(app_state=app_state, pane_type = PointSelectorType.REFERENCE_POINT_SELECTOR)
        self._georeferencer_task_delegate = GeoReferencerTaskDelegate(self._target_rasterpane,
                                                                      self._reference_rasterpane,
                                                                      self,
                                                                      app_state)
        self._target_rasterpane.set_task_delegate(self._georeferencer_task_delegate)
        self._reference_rasterpane.set_task_delegate(self._georeferencer_task_delegate)

        self.gcp_pair_added.connect(self._on_gcp_pair_added)

        self._table_entry_list: List[GeoRefTableEntry] = []

        self._curr_output_srs: str = None
        self._current_resample_alg = None
        self._current_transform_type: TRANSFORM_TYPES = None

        self._default_color_button: QPushButton = None

        self._manual_entry_spacer = None

        # Set up dataset choosers 
        self._init_dataset_choosers()
        self._init_rasterpanes()
        self._init_gcp_table()
        self._init_output_srs_cbox()
        self._init_interpolation_type_cbox()
        self._init_poly_order_cbox()
        self._init_file_saver()
        self._init_default_color_chooser()
        self._init_manual_ref_system_finder()
        self._init_manual_ref_point_enter()
        self._update_manual_ref_coord_chooser(None)

        self._warp_kwargs: Dict = None
        self._suppress_cell_changed: bool = False

    # region Initialization

    def _init_manual_ref_point_enter(self):
        lat_north_ledit = self._ui.ledit_lat_north
        lon_east_ledit  = self._ui.ledit_lon_east

        # Validator that only allows floating-point numbers (no strict range)
        float_validator = QDoubleValidator(self)
        float_validator.setNotation(QDoubleValidator.StandardNotation)
        # Optionally set a very wide range if you want to enforce some bounds:
        # float_validator.setRange(-1e12, 1e12, 8)

        lat_north_ledit.setValidator(float_validator)
        lon_east_ledit.setValidator(float_validator)

        lat_north_ledit.returnPressed.connect(self._on_ref_manual_ledit_enter)
        lon_east_ledit.returnPressed.connect(self._on_ref_manual_ledit_enter)

    def _init_manual_ref_system_finder(self):
        authority_cbox = self._ui.cbox_authority
        authority_cbox.clear()
        for auth in AVAILABLE_AUTHORITIES:
            authority_cbox.addItem(auth, auth)
        # Go through the list of strings in AVAILABLE_AUTHORITIES variable
        # and populate the authority cbox with them. The name and data should both be the string

        srs_code_ledit = self._ui.ledit_srs_code
        # Create an integer validator that only allows values ≥ 1
        int_validator = QIntValidator(1, 2147483647, self)

        # Attach it to the line edit
        srs_code_ledit.setValidator(int_validator)

        srs_to_choose_cbox = self._ui.cbox_choose_crs
        for name, srs in COMMON_SRS.items():
            srs_to_choose_cbox.addItem(name, srs)

        find_crs_btn = self._ui.btn_find_crs
        find_crs_btn.clicked.connect(self._on_find_crs)

    def _update_manual_ref_coord_chooser(self, dataset: Optional[RasterDataSet]):
        """
        We want to enable this when no data is selected and disable this when data is selected
        """
        if self._manual_entry_spacer is None:
            self._manual_entry_spacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)
        if dataset is None:
            # if not self._ui.widget_ref_image.isVisible():
            self._ui.vlayout_reference.addItem(self._manual_entry_spacer)
            # We want to make sure the manual chooser is being shown
            self._ui.widget_manual_entry.show()
            self._ui.widget_ref_image.hide()
        else:
            # We want to make sure it is not being shown 
            self._ui.widget_manual_entry.hide()
            self._ui.widget_ref_image.show()
            self._ui.vlayout_reference.removeItem(self._manual_entry_spacer)

    def _init_default_color_chooser(self):
        horizontal_layout = self._ui.hlayout_color_change
        self._default_color_button = QPushButton()
        self._default_color_button.clicked.connect(lambda checked: self._on_choose_default_color())
        self._initial_default_color = QColor("orange").name()
        color_icon = get_color_icon(self._initial_default_color)
        self._default_color_button.setIcon(color_icon)
        horizontal_layout.addWidget(self._default_color_button)

    def _init_file_saver(self):
        self._ui.btn_save_path.clicked.connect(self._on_choose_save_filename)

    def _init_output_srs_cbox(self):
        """Initialize the spatial reference combo box."""
        srs_cbox = self._ui.cbox_srs
        srs_cbox.activated.connect(self._on_switch_output_srs)
        srs_cbox.clear()
        # Use the friendly key (e.g., "WGS84") as the display text,
        # and store the corresponding SRS string (e.g., "EPSG:4326") as userData.
        if self._reference_rasterpane is not None and self._reference_rasterpane.get_rasterview().get_raster_data() is not None:
            ref_ds = self._reference_rasterpane.get_rasterview().get_raster_data()
            reference_srs_name = "Ref CRS: " + ref_ds.get_spatial_ref().GetName()
            reference_srs_code = ref_ds.get_spatial_ref().GetAuthorityCode(None)
            print(f"reference srs code: {reference_srs_code}")
            print(f"reference_srs_name: {reference_srs_name}")
            print(f"ref_ds.get_spatial_ref().GetAuthorityCode('PROJCS'): {ref_ds.get_spatial_ref().GetAuthorityCode('PROJCS')}")
            print(f"ref_ds.get_spatial_ref().GetAuthorityCode('GEOGCS'): {ref_ds.get_spatial_ref().GetAuthorityCode('GEOGCS')}")
            if reference_srs_code is None:
                self.set_message_text("Could not get an authority code for default dataset")
                ref_srs = ref_ds.get_spatial_ref()
                crs = CRS.from_wkt(ref_srs.ExportToWkt())
                if crs is not None:
                    auth_name, auth_code = crs.to_authority()
                    srs_cbox.addItem(reference_srs_name, (auth_name, int(auth_code)))
                print(f"crs.to_epsg(): {crs.to_epsg()}")
            else:
                srs_cbox.addItem(reference_srs_name, reference_srs_code)

        for name, srs in COMMON_SRS.items():
            srs_cbox.addItem(name, srs)

        self._on_switch_output_srs(srs_cbox.currentIndex())

    def _init_interpolation_type_cbox(self):
        """Initialize the interpolation type combo box using the GDAL resample constants."""
        interp_type_cbox = self._ui.cbox_interpolation
        interp_type_cbox.activated.connect(self._on_switch_resample_alg)
        interp_type_cbox.clear()
        # Sorting the keys gives a consistent order.
        for name in sorted(RESAMPLE_ALGORITHMS.keys()):
            # The display text is the name, and the actual GDAL constant is stored as userData.
            interp_type_cbox.addItem(name, RESAMPLE_ALGORITHMS[name])
        self._on_switch_resample_alg(0)  # Initializes the data to be the first displayed item

    def _init_poly_order_cbox(self):
        """Initialize the transformation type (polynomial order) combo box from the enum."""
        poly_order_cbox = self._ui.cbox_poly_order
        poly_order_cbox.activated.connect(self._on_switch_transform_type)
        poly_order_cbox.clear()
        # Iterate through each transformation type in the TRANSFORM_TYPES enum.
        for transform in TRANSFORM_TYPES:
            # Display the string (e.g., "Affine (Polynomial 1)") and store the enum member as userData.
            poly_order_cbox.addItem(transform.value, transform)
        self._on_switch_transform_type(0)

    def _init_gcp_table(self):
        table_widget = self._ui.table_gcps
        table_widget.setColumnCount(len(COLUMN_ID))
        headers = ["Enabled", "ID", "Target X", "Target Y", "Ref X", "Ref Y", "dX (Pix)", "dY (Pix)", "Color", "Remove"]
        table_widget.setHorizontalHeaderLabels(headers)

        # Do not use QHeaderView.Stretch here!!! It will cause a very hard to track down bug.
        table_widget.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self._target_x_col_delegate = NumericDelegate()
        self._target_y_col_delegate = NumericDelegate()
        table_widget.setItemDelegateForColumn(COLUMN_ID.TARGET_X_COL, self._target_x_col_delegate)
        table_widget.setItemDelegateForColumn(COLUMN_ID.TARGET_Y_COL, self._target_y_col_delegate)

        self._ref_x_col_delegate = NumericDelegate(minimum=-1e10)
        self._ref_y_col_delegate = NumericDelegate(minimum=-1e10)
        table_widget.setItemDelegateForColumn(COLUMN_ID.REF_X_COL, self._ref_x_col_delegate)
        table_widget.setItemDelegateForColumn(COLUMN_ID.REF_Y_COL, self._ref_y_col_delegate)

        table_widget.cellChanged.connect(self._on_cell_changed)

    def _init_dataset_choosers(self):
        self._target_cbox.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self._target_cbox.activated.connect(self._on_switch_target_dataset)
        self._update_target_dataset_chooser()

        self._reference_cbox.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self._reference_cbox.activated.connect(self._on_switch_reference_dataset)
        self._update_reference_dataset_chooser()

    def _init_rasterpanes(self):
        target_layout = QVBoxLayout(self._ui.widget_target_image)
        self._ui.widget_target_image.setLayout(target_layout)

        target_layout.addWidget(self._target_rasterpane)

        reference_layout = QVBoxLayout(self._ui.widget_ref_image)
        self._ui.widget_ref_image.setLayout(reference_layout)

        reference_layout.addWidget(self._reference_rasterpane)

    #========================
    # region Slots
    #========================

    def _on_ref_manual_ledit_enter(self):
        lat_north_str = self._ui.ledit_lat_north.text()
        lon_east_str = self._ui.ledit_lon_east.text()

        if lat_north_str == "" or lon_east_str == "":
            self.set_message_text("Ensure both Lat/North and Lon/East have valid values")
            return

        print(f"About to emit gcp add event")
        lat_north_value = float(lat_north_str)
        lon_east_value = float(lon_east_str)

        chosen_srs = self._get_manual_ref_chosen_crs()

        # Since we set SRS's to OAMS_TRADITIONAL_GIS_ORDER, we create gcp
        # in long/lat order 
        gcp = GroundControlPointCoordinate((lon_east_value, lat_north_value), \
                                           PointSelectorType.REFERENCE_POINT_SELECTOR, \
                                            srs=chosen_srs)
        self.gcp_add_attempt.emit(gcp)

    def _on_find_crs(self):
        authority_str = self._ui.cbox_authority.currentText()
        authority_code = self._ui.ledit_srs_code.text()

        # Use pyproj or gdal to use the authority name (authority_str) and authority_code
        # to get an osr.SpatialReference object. 

        # Then get the spatial reference's full name and store it in a string called srs_name

        # Build the SRS from "AUTHORITY:CODE"
        srs = osr.SpatialReference()
        err = srs.SetFromUserInput(f"{authority_str}:{authority_code}")
        if err != 0:
            QMessageBox.warning(
                self,
                "CRS Lookup Failed",
                f"Could not find spatial reference for {authority_str}:{authority_code}"
            )
            return

        # Get the human-readable name of the SRS
        srs_name = srs.GetName()

        # Now you can use srs_name however you need:
        # e.g. display it in the UI, store it in your app state, etc.
        print(f"Found CRS name: {srs_name}")
        # or, for example:
        # self._ui.label_srs_name.setText(srs_name)
        self._add_srs_to_choose_cbox(srs_name, authority_str, authority_code)


    def _on_cell_changed(self, row: int, col: int):
        table_widget = self._ui.table_gcps
        if self._suppress_cell_changed:
            return
        if col == COLUMN_ID.TARGET_X_COL:
            item = table_widget.item(row, col)
            new_val = item.text()
            new_target_x = float(new_val)
            list_entry = self._table_entry_list[row]
            target_gcp = list_entry.get_gcp_pair().get_target_gcp()
            curr_point = target_gcp.get_point()
            target_gcp.set_point([new_target_x, curr_point[1]])
            self._georeference()
        elif col == COLUMN_ID.TARGET_Y_COL:
            item = table_widget.item(row, col)
            new_val = item.text()
            new_target_y = float(new_val)
            list_entry = self._table_entry_list[row]
            target_gcp = list_entry.get_gcp_pair().get_target_gcp()
            curr_point = target_gcp.get_point()
            target_gcp.set_point([curr_point[0], new_target_y])
            self._georeference()
        elif col == COLUMN_ID.REF_X_COL:
            item = table_widget.item(row, col)
            new_val = item.text()
            new_ref_spatial_x = float(new_val)
            list_entry = self._table_entry_list[row]
            gcp_pair = list_entry.get_gcp_pair()
            ref_gcp = gcp_pair.get_reference_gcp()
            ref_gcp.set_spatial_point((new_ref_spatial_x, \
                                                  gcp_pair.get_reference_gcp_spatial_coord()[1]))
            self._georeference()
        elif col == COLUMN_ID.REF_Y_COL:
            item = table_widget.item(row, col)
            new_val = item.text()
            new_ref_spatial_y = float(new_val)
            list_entry = self._table_entry_list[row]
            gcp_pair = list_entry.get_gcp_pair()
            ref_gcp = gcp_pair.get_reference_gcp()
            ref_gcp.set_spatial_point((gcp_pair.get_reference_gcp_spatial_coord()[0], \
                                                  new_ref_spatial_y))
            self._georeference()
        else:
            return
        self._update_panes()

    def _on_choose_save_filename(self, checked=False):
        '''
        A handler for when the file-chooser for the "save-filename" is shown.
        '''

        file_dialog = QFileDialog(parent=self,
            caption=self.tr('Save raster dataset'))

        # Restrict selection to only .tif files.
        file_dialog.setNameFilter("TIFF files (*.tif)")
        # Optionally, set a default suffix to ensure the saved file gets a .tif extension.
        file_dialog.setDefaultSuffix("tif")
        file_dialog.setAcceptMode(QFileDialog.AcceptSave)

        # If there is already an initial filename, select it in the dialog.
        initial_filename = self._ui.ledit_save_path.text().strip()
        if len(initial_filename) > 0:
            base, ext = os.path.splitext(initial_filename)
            if ext.lower() != ".tif":
                initial_filename = f"{base}.tif"
            file_dialog.selectFile(initial_filename)

        result = file_dialog.exec()
        if result == QDialog.Accepted:
            filename = file_dialog.selectedFiles()[0]
            self._ui.ledit_save_path.setText(filename)
            self._georeference()

    def _on_switch_output_srs(self, index: int):
        srs = self._ui.cbox_srs.itemData(index)
        self._curr_output_srs = srs
        self._georeference()

    def _on_switch_resample_alg(self, index: int):
        resample_alg = self._ui.cbox_interpolation.itemData(index)
        self._current_resample_alg = resample_alg

    def _on_switch_transform_type(self, index: int):
        transform_type = self._ui.cbox_poly_order.itemData(index)
        self._current_transform_type = transform_type
        self._georeference()

    def _on_choose_default_color(self):
        color = QColorDialog.getColor(parent=self, initial=self._initial_default_color)
        if color.isValid():
            color_str = color.name()
            for row in range(len(self._table_entry_list)):
                # We only want to change the colors of the points that weren't explicitly
                # changed
                if self._table_entry_list[row].get_color() == self._initial_default_color:
                    self._table_entry_list[row].set_color(color_str)
                    self._set_color_icon(row, color_str)
            self._set_default_color_icon(color_str)
            self._update_panes()

    def _on_choose_color(self, table_entry: GeoRefTableEntry):
        row = table_entry.get_id()
        initial_color = QColor(self._table_entry_list[row].get_color())
        color = QColorDialog.getColor(parent=self, initial=initial_color)
        if color.isValid():
            color_str = color.name()
            self._table_entry_list[row].set_color(color_str)
            self._set_color_icon(row, color_str)
            self._update_panes()

    def _on_enabled_clicked(self, table_entry: GeoRefTableEntry, checked: bool):
        # Since the table_entry's ID can change, don't just pass in the row_to_add
        row_to_add = table_entry.get_id()
        self._set_row_enabled_state(row_to_add, checked)
        self._update_panes()
        self._georeference()

    def _on_gcp_pair_added(self, gcp_pair: GroundControlPointPair):
        # Create new table entry
        table_widget = self._ui.table_gcps
        next_row = table_widget.rowCount()
        enabled = True
        id = next_row
        residuals = 0
        color = get_random_matplotlib_color()
        color = self._initial_default_color
        table_entry = GeoRefTableEntry(gcp_pair, enabled, id, None, None, color)
        # The row that a GCP is placed on should be the same as its position in the
        # geo referencer task delegate point list

        self._add_entry_to_table(table_entry)
        self._clear_manual_ref_ledits()
        self._georeference()

    def _on_removal_button_clicked(self, table_entry: GeoRefTableEntry):
        self._remove_table_entry(table_entry)
        self._georeference()

    def _on_switch_target_dataset(self, index: int):
        ds_id = self._target_cbox.itemData(index)
        dataset = None
        try:
            if len(self._table_entry_list) > 0:
                confirm = QMessageBox.question(self, self.tr("Change reference dataset?"),
                                     self.tr("Are you sure you want to change the reference dataset?") +
                                     "\n\nThis will discard all selected GCPs")
                if confirm == QMessageBox.Yes:
                    self._reset_gcps()
            dataset = self._app_state.get_dataset(ds_id)
        except:
            pass
        self._target_rasterpane.show_dataset(dataset)

    def _on_switch_reference_dataset(self, index: int):
        ds_id = self._reference_cbox.itemData(index)
        dataset = None
        try:
            if len(self._table_entry_list) > 0:
                confirm = QMessageBox.question(self, self.tr("Change reference dataset?"),
                                     self.tr("Are you sure you want to change the reference dataset?") +
                                     "\n\nThis will discard all selected GCPs")
                if confirm == QMessageBox.Yes:
                    self._reset_gcps()
                else:
                    return
            dataset = self._app_state.get_dataset(ds_id)
            if not dataset.has_geographic_info():
                QMessageBox.warning(self, self.tr("Unreferenced Dataset"), \
                                    self.tr("You must choose a dataset with a spatial reference system"))
                return
        except:
            pass

        else:
            print(f"dataset is not none: {dataset}")
        self._reference_rasterpane.show_dataset(dataset)
        self._init_output_srs_cbox()
        self._update_manual_ref_coord_chooser(dataset)

    # region Table Entry Helpers

    def _get_save_file_path(self) -> str:
        path = self._ui.ledit_save_path.text()
        if len(path) > 0:
            abs_path = os.path.abspath(path)
            return abs_path
        return None


    def _reset_gcps(self):
        '''
        Clears all of the entries in the table widget and in the list of entries
        '''
        self._table_entry_list = []
        self._ui.table_gcps.clearContents()
        self._ui.table_gcps.setRowCount(0)

    def _set_default_color_icon(self, color: str):
        self._initial_default_color = color
        color_icon = get_color_icon(color)
        self._default_color_button.setIcon(color_icon)

    def _set_color_icon(self, row: int, color: str):
        color_icon = get_color_icon(color)
        table_widget = self._ui.table_gcps
        table_item: QPushButton = table_widget.cellWidget(row, COLUMN_ID.COLOR_COL)
        table_item.setIcon(color_icon)


    def _set_row_enabled_state(self, row: int, row_enabled_state: bool, \
                                exempt_columns: List[COLUMN_ID] = [COLUMN_ID.REMOVAL_COL, COLUMN_ID.ENABLED_COL]):
        '''
        Disable all cells in a given row except for the one columns in the exempt_columns list."
        '''
        table_widget = self._ui.table_gcps
        total_columns = table_widget.columnCount()
        for col in range(total_columns):
            if col in exempt_columns:
                continue  # Skip the removal column
            
            # Disable QTableWidgetItem if it exists
            item = table_widget.item(row, col)
            if (col == COLUMN_ID.RESIDUAL_X_COL or col == COLUMN_ID.RESIDUAL_Y_COL) and not row_enabled_state:
                item.setText("N/A")
            if item:
                if row_enabled_state:
                    item.setFlags(item.flags() | Qt.ItemIsEnabled)
                else:
                    # Remove the enabled flag from the item's flags
                    item.setFlags(item.flags() & ~Qt.ItemIsEnabled)
            
            # Also disable any cell widget if one is set (e.g., a QPushButton)
            widget = table_widget.cellWidget(row, col)
            if widget:
                widget.setEnabled(row_enabled_state)
        self._table_entry_list[row].set_enabled(row_enabled_state)
    
    def _set_all_residuals_NA(self):
        table_widget = self._ui.table_gcps
        for row in range(table_widget.rowCount()):
            item = table_widget.item(row, COLUMN_ID.RESIDUAL_X_COL)
            # A row can still be in the table but be None, so we want to skip these rows
            if item is None:
                continue
            item.setText("N/A")
            item = table_widget.item(row, COLUMN_ID.RESIDUAL_Y_COL)
            item.setText("N/A")

    def _add_entry_to_table(self, table_entry: GeoRefTableEntry):
        '''
        Adds table_entry to the table widget at the row specified by
        table_entry.get_id()
        '''
        self._table_entry_list.append(table_entry)

        table_widget = self._ui.table_gcps
        row_to_add = table_entry.get_id()
        table_widget.insertRow(row_to_add)
        gcp_pair = table_entry.get_gcp_pair()

        target_x = gcp_pair.get_target_gcp().get_point()[0]
        target_y = gcp_pair.get_target_gcp().get_point()[1]
        # ref_x = gcp_pair.get_reference_gcp().get_point()[0]
        # ref_y = gcp_pair.get_reference_gcp().get_point()[1]
        ref_x, ref_y = gcp_pair.get_reference_gcp_spatial_coord()

        residual_x = table_entry.get_residual_x()
        residual_y = table_entry.get_residual_y()

        self._suppress_cell_changed = True
        checkbox = QCheckBox()
        checkbox.setChecked(table_entry.is_enabled())
        checkbox.clicked.connect(lambda checked : self._on_enabled_clicked(table_entry, checked))

        table_widget.setCellWidget(row_to_add, COLUMN_ID.ENABLED_COL, checkbox)

        id_table_item = QTableWidgetItem(str(table_entry.get_id()))
        id_table_item.setFlags(id_table_item.flags() & ~Qt.ItemIsEditable)
        table_widget.setItem(row_to_add, COLUMN_ID.ID_COL, id_table_item)

        target_x_table_item = QTableWidgetItem(str(target_x))
        table_widget.setItem(row_to_add, COLUMN_ID.TARGET_X_COL, target_x_table_item)
        target_y_table_item = QTableWidgetItem(str(target_y))
        table_widget.setItem(row_to_add, COLUMN_ID.TARGET_Y_COL, target_y_table_item)

        ref_x_table_item = QTableWidgetItem(str(ref_x))
        # ref_x_table_item.setFlags(ref_x_table_item.flags() & ~Qt.ItemIsEditable)
        table_widget.setItem(row_to_add, COLUMN_ID.REF_X_COL, ref_x_table_item)
    
        ref_y_table_item = QTableWidgetItem(str(ref_y))
        # ref_y_table_item.setFlags(ref_y_table_item.flags() & ~Qt.ItemIsEditable)
        table_widget.setItem(row_to_add, COLUMN_ID.REF_Y_COL, ref_y_table_item)

        res_x_str = "N/A"
        if residual_x is not None:
            res_x_str = str(residual_x)

        res_y_str = "N/A"
        if residual_y is not None:
            res_y_str = str(residual_y)

        res_x_item = QTableWidgetItem(res_x_str)
        res_x_item.setFlags(res_x_item.flags() & ~Qt.ItemIsEditable)
        table_widget.setItem(row_to_add, COLUMN_ID.RESIDUAL_X_COL, res_x_item)
        res_y_item = QTableWidgetItem(res_y_str)
        res_y_item.setFlags(res_y_item.flags() & ~Qt.ItemIsEditable)
        table_widget.setItem(row_to_add, COLUMN_ID.RESIDUAL_Y_COL, res_y_item)

        color_button = QPushButton()
        color_button.clicked.connect(lambda checked : self._on_choose_color(table_entry))
        initial_color = table_entry.get_color()
        color_icon = get_color_icon(initial_color)
        color_button.setIcon(color_icon)
        table_widget.setCellWidget(row_to_add, COLUMN_ID.COLOR_COL, color_button)

        pushButton = QPushButton("Remove GCP")
        pushButton.clicked.connect(lambda checked : self._on_removal_button_clicked(table_entry))
        table_widget.setCellWidget(row_to_add, COLUMN_ID.REMOVAL_COL, pushButton)

        self._suppress_cell_changed = False
        self._update_panes()

    def _remove_table_entry(self, table_entry: GeoRefTableEntry) -> Optional[int]:
        '''
        Removes the table entry and returns the index removed. If the table entry
        is not found in the list, this errors. Table entry 
        equality is done based on memory location. 

        table_entry.get_id() should be the table entries index in both the
        TableWidget and the _table_entry_list. We uses asserts to ensure this
        '''
        table_widget = self._ui.table_gcps

        index_removed = None  # Also refers to the removed row
        for i in range(len(self._table_entry_list)):
            table_entry_in_list = self._table_entry_list[i]
            if table_entry_in_list == table_entry:
                index_removed = i
                self._table_entry_list.pop(i)
                # print(f"index removed: ", index_removed)
                # print(f"table_entry.get_id(): ", table_entry.get_id())
                assert index_removed == table_entry.get_id(), \
                        "The index that table entry was removed does not match its ID"
                break
        assert index_removed != None, "The table entry was not found in the list of entries"
        table_widget.removeRow(index_removed)
        # We must update the entry id's after we remove the rows so that
        # the table entry's are in their correct rows
        self._update_entry_ids()

        self._update_panes()

    def _update_residuals(self, table_entry: GeoRefTableEntry):
        table_widget = self._ui.table_gcps
        row_to_add = table_entry.get_id()
        assert row_to_add < table_widget.rowCount()

        residual_x = table_entry.get_residual_x()
        residual_y = table_entry.get_residual_y()
        res_x_str = "N/A"
        if residual_x is not None:
            res_x_str = str(residual_x)

        res_y_str = "N/A"
        if residual_y is not None:
            res_y_str = str(residual_y)

        res_x_item = QTableWidgetItem(res_x_str)
        res_x_item.setFlags(res_x_item.flags() & ~Qt.ItemIsEditable)
        table_widget.setItem(row_to_add, COLUMN_ID.RESIDUAL_X_COL, res_x_item)
        res_y_item = QTableWidgetItem(res_y_str)
        res_y_item.setFlags(res_y_item.flags() & ~Qt.ItemIsEditable)
        table_widget.setItem(row_to_add, COLUMN_ID.RESIDUAL_Y_COL, res_y_item)

    def _sync_gcp_table_row_with_table_entry(self, table_entry: GeoRefTableEntry):
        table_widget = self._ui.table_gcps
        row_to_add = table_entry.get_id()
        assert row_to_add < table_widget.rowCount()
        gcp_pair = table_entry.get_gcp_pair()

        target_x = gcp_pair.get_target_gcp().get_point()[0]
        target_y = gcp_pair.get_target_gcp().get_point()[1]
        ref_x = gcp_pair.get_reference_gcp_spatial_coord()[0]
        ref_y = gcp_pair.get_reference_gcp_spatial_coord()[1]

        residual_x = table_entry.get_residual_x()
        residual_y = table_entry.get_residual_y()
        table_widget.setItem(row_to_add, COLUMN_ID.TARGET_X_COL, QTableWidgetItem(str(target_x)))
        table_widget.setItem(row_to_add, COLUMN_ID.TARGET_Y_COL, QTableWidgetItem(str(target_y)))
        table_widget.setItem(row_to_add, COLUMN_ID.REF_X_COL, QTableWidgetItem(str(ref_x)))
        table_widget.setItem(row_to_add, COLUMN_ID.REF_Y_COL, QTableWidgetItem(str(ref_y)))

        res_x_str = "N/A"
        if residual_x is not None:
            res_x_str = str(residual_x)

        res_y_str = "N/A"
        if residual_y is not None:
            res_y_str = str(residual_y)

        res_x_item = QTableWidgetItem(res_x_str)
        res_x_item.setFlags(res_x_item.flags() & ~Qt.ItemIsEditable)
        table_widget.setItem(row_to_add, COLUMN_ID.RESIDUAL_X_COL, res_x_item)
        res_y_item = QTableWidgetItem(res_y_str)
        res_y_item.setFlags(res_y_item.flags() & ~Qt.ItemIsEditable)
        table_widget.setItem(row_to_add, COLUMN_ID.RESIDUAL_Y_COL, res_y_item)


    def _update_entry_ids(self):
        table_widget = self._ui.table_gcps
        for i in range(len(self._table_entry_list)):
            table_entry = self._table_entry_list[i]
            table_entry.set_id(i)
            # i also functions as the row in the table widget where
            # this entry is currently
            # print(f"updating i: {i}, for column: {COLUMN_ID.ID_COL}")
            table_widget.setItem(i, COLUMN_ID.ID_COL, QTableWidgetItem(str(i)))

    def _clear_manual_ref_ledits(self):
        self._ui.ledit_lat_north.clear()
        self._ui.ledit_lon_east.clear()


    # region Getters


    def get_table_entries(self) -> List[GeoRefTableEntry]:
        # Go through the table and extract the geo referencer dialog for each entry 
        return self._table_entry_list

    def get_gcp_table_size(self) -> int:
        # print(f"get_gcp_table_size")
        # print(f"len(self._table_entry_list): {len(self._table_entry_list)}")
        # print(f"self._ui.table_gcps.rowCount(): {self._ui.table_gcps.rowCount()}")
        assert len(self._table_entry_list) == self._ui.table_gcps.rowCount(), \
                f"Entry number mismatch. Table entry list {len(self._table_entry_list)} and QTableWidget has {self._ui.table_gcps.rowCount()} entries"
        return len(self._table_entry_list)

    def _get_target_dataset(self):
        return self._target_rasterpane.get_rasterview().get_raster_data()
    
    def _get_ref_dataset(self):
        return self._reference_rasterpane.get_rasterview().get_raster_data()

    def _get_num_active_points(self):
        count = 0
        for entry in self._table_entry_list:
            if entry.is_enabled():
                count += 1
        return count

    # region Dataset Choosers

    def _update_target_dataset_chooser(self):
        self._update_dataset_chooser(self._target_cbox)

    def _update_reference_dataset_chooser(self):
        self._update_dataset_chooser(self._reference_cbox)

    def _update_dataset_chooser(self, dataset_chooser: QComboBox):
        app_state = self._app_state

        num_datasets = app_state.num_datasets()

        current_index = dataset_chooser.currentIndex()
        current_ds_id = None
        if current_index != -1:
            current_ds_id = dataset_chooser.itemData(current_index)
        else:
            # This occurs initially, when the combobox is empty and has no
            # selection.  Make sure the "(no data)" option is selected by the
            # end of this process.
            current_index = 0
            current_ds_id = -1

        # print(f'update_toolbar_state(position={self._position}):  current_index = {current_index}, current_ds_id = {current_ds_id}')

        new_index = None
        dataset_chooser.clear()

        if num_datasets > 0:
            for (index, dataset) in enumerate(app_state.get_datasets()):
                id = dataset.get_id()
                name = dataset.get_name()

                dataset_chooser.addItem(name, id)
                if dataset.get_id() == current_ds_id:
                    new_index = index

            dataset_chooser.insertSeparator(num_datasets)
            dataset_chooser.addItem(self.tr('(no data)'), -1)
            if current_ds_id == -1:
                new_index = dataset_chooser.count() - 1
        else:
            # No datasets yet
            dataset_chooser.addItem(self.tr('(no data)'), -1)
            if current_ds_id == -1:
                new_index = 0

        # print(f'update_toolbar_state(position={self._position}):  new_index = {new_index}')

        if new_index is None:
            if num_datasets > 0:
                new_index = min(current_index, num_datasets - 1)
            else:
                new_index = 0

        dataset_chooser.setCurrentIndex(new_index)

    # def _show_dataset(self, dataset: RasterDataSet, rasterview: RasterView):
    #     '''
    #     Sets the dataset being displayed in the specified rasterview 
    #     '''
    #     # If the rasterview is already showing the specified dataset, skip!
    #     if rasterview.get_raster_data() is dataset:
    #         return

    #     bands = None
    #     stretches = None
    #     if dataset is not None:
    #         ds_id = dataset.get_id()
    #         bands = self._main_view.get_display_bands()[ds_id]
    #         stretches = self._app_state.get_stretches(ds_id, bands)

    #     rasterview.set_raster_data(dataset, bands, stretches)

    # region Misc

    def _update_panes(self):
        self._target_rasterpane.update_all_rasterviews()
        self._reference_rasterpane.update_all_rasterviews()

    def set_message_text(self, text: str):
        self._ui.lbl_message.setText(text)

    def _add_srs_to_choose_cbox(self, srs_name: str, authority_name: str, authority_code: int):
        crs_choose_cbox = self._ui.cbox_choose_crs
        # Go through crs_choose_cbox and and get the data in the cbox entry
        # The data is a 2-tuple. Compare the first element to authority_name and the second
        # element to authority_code. If both of these match print a message to the user that
        # we already have the authority code then print the name of the authority code

        # If authority_name and  authority_code don't match anything then we add this to crs_choose_cbox
        # with the display name being srs_name and the data being a 2-tuple of (authority_name, authority_code)
        # Check for existing entry
        for idx in range(crs_choose_cbox.count()):
            data = crs_choose_cbox.itemData(idx)
            # data is expected to be a (authority_name, authority_code) tuple
            if isinstance(data, tuple) and len(data) == 2:
                existing_auth, existing_code = data
                if existing_auth == authority_name and existing_code == authority_code:
                    QMessageBox.information(
                        self,
                        "CRS Already Added",
                        f"The CRS {authority_name}:{authority_code} is already in the list as “{crs_choose_cbox.itemText(idx)}.”"
                    )
                    return
            else:
                raise ValueError("CRS data is not a 2-Tuple in Choose CRS ComboBox!")
        # If not found, add as new entry
        crs_choose_cbox.addItem(srs_name, (authority_name, authority_code))
        crs_choose_cbox.setCurrentIndex(crs_choose_cbox.count()-1)

    def _get_manual_ref_chosen_crs(self) -> osr.SpatialReference:
        auth_name, auth_code = self._ui.cbox_choose_crs.currentData()

        # Build the AUTH:CODE string
        auth_code_str = f"{auth_name}:{auth_code}"

        # Create and populate the SpatialReference
        srs = osr.SpatialReference()
        err = srs.SetFromUserInput(auth_code_str)
        if err != 0:
            raise RuntimeError(f"Failed to import CRS '{auth_code_str}' (GDAL error {err})")

        return srs

    # region Geo referencing

    def _enough_points_for_transform(self):
        return False if self._get_num_active_points() < min_points_per_transform[self._current_transform_type] else True

    def _get_entry_gcp_list(self) -> List[Tuple[GeoRefTableEntry, gdal.GCP]]:
        gcps: List[Tuple[GeoRefTableEntry, gdal.GCP]] = []

        for table_entry in self._table_entry_list:
            if not table_entry.is_enabled():
                continue
            spatial_coord = table_entry.get_gcp_pair().get_reference_gcp().get_spatial_point()
            assert spatial_coord is not None, f"spatial_coord is none on reference gcp!, spatial_coord: {spatial_coord}"
            target_pixel_coord = table_entry.get_gcp_pair().get_target_gcp().get_point()
            gcps.append((table_entry, gdal.GCP(spatial_coord[0], spatial_coord[1], 0, target_pixel_coord[0], target_pixel_coord[1])))

        return gcps

    def _import_current_output_srs(self) -> osr.SpatialReference:
        # self._curr_output_srs is a 2-Tuple with the first string being an authority name.
        # The authority name is one you get back from get_authorities (from pyproj.database import get_authorities)
        # so its any of these ['EPSG', 'ESRI', 'IAU_2015', 'IGNF', 'NKG', 'NRCAN', 'OGC', 'PROJ']
        # Then the second tuple is the code. This function should return an osr.SpatialReference system object
        # using this information.
        """
        Read self._curr_output_srs (authority_name, authority_code) and
        return a corresponding OSR SpatialReference object.
        """
        authority_name, authority_code = self._curr_output_srs

        # Build the PROJ auth:code string (e.g. "EPSG:4326", "ESRI:102100")
        auth_code = f"{authority_name}:{authority_code}"

        # Create and populate the SpatialReference
        srs = osr.SpatialReference()
        err = srs.SetFromUserInput(auth_code)
        if err != 0:
            raise RuntimeError(f"Could not import CRS from '{auth_code}' (GDAL error code {err})")

        return srs

    def _get_reference_srs(self):
        # Either the reference raster pane has data selected or the manual reference
        # srs chooser has data in it
        ref_ds = self._reference_rasterpane.get_rasterview().get_raster_data()
        if ref_ds is not None:
            return ref_ds.get_spatial_ref()
        elif self._ui.widget_manual_entry.isVisible():
            return self._get_manual_ref_chosen_crs()
        else:
            raise RuntimeError("Both the dataset shown is none and the " \
                               "manual entry widget is None")

    def _georeference(self):
        save_path = self._get_save_file_path()
        if save_path is None or \
            self._target_rasterpane.get_rasterview().get_raster_data() is None:
            print(f"save path: {save_path}")
            print(f"target dataset: {self._target_rasterpane.get_rasterview().get_raster_data()}")
            return

        if not self._enough_points_for_transform():
            print(f"not enough residuals: {self._get_num_active_points()}")
            self._set_all_residuals_NA()
            return

        gdal.UseExceptions()

        gcps: List[GeoRefTableEntry, gdal.GCP] = self._get_entry_gcp_list()

        output_srs = osr.SpatialReference()
        output_srs = self._import_current_output_srs()
        output_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        print(f"output_srs: {output_srs.GetName()}")

        ref_srs = self._get_reference_srs()
        ref_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        ref_projection = ref_srs.ExportToWkt()
        # If it doesn't have a gdal dataset, instead we create one from a smaller array just to do geo referencing
        assert ref_projection is not None and ref_srs is not None, \
                f"ref_srs ({ref_srs}) or ref_project ({ref_projection}) is None!"

        temp_gdal_ds = None
        place_holder_arr = np.zeros((1, 1), np.uint8)
        temp_gdal_ds: gdal.Dataset = gdal_array.OpenNumPyArray(place_holder_arr, True)
        temp_gdal_ds.SetSpatialRef(ref_srs)
        temp_gdal_ds.SetGCPs([pair[1] for pair in gcps], ref_projection)
    
        self._warp_kwargs = {
            "copyMetadata": True,
            "resampleAlg": self._current_resample_alg,
            "dstSRS": output_srs
        }

        transformer_options = [f'DST_SRS={output_srs.ExportToWkt()}']

        if self._current_transform_type == TRANSFORM_TYPES.TPS:
            self._warp_kwargs["tps"] = True
            transformer_options += ['METHOD=GCP_TPS', 'MAX_GCP_ORDER=-1']
        elif self._current_transform_type == TRANSFORM_TYPES.POLY_1:
            self._warp_kwargs["polynomialOrder"] = 1
            transformer_options += ['METHOD=GCP_POLYNOMIAL', 'MAX_GCP_ORDER=1']
        elif self._current_transform_type == TRANSFORM_TYPES.POLY_2:
            self._warp_kwargs["polynomialOrder"] = 2
            transformer_options += ['METHOD=GCP_POLYNOMIAL', 'MAX_GCP_ORDER=2']
        elif self._current_transform_type == TRANSFORM_TYPES.POLY_3:
            self._warp_kwargs["polynomialOrder"] = 3
            transformer_options += ['METHOD=GCP_POLYNOMIAL', 'MAX_GCP_ORDER=3']
        else:
            raise RuntimeError(f"Unknown self._current_transform_type: {self._current_transform_type}")
        try:
            tr_pixel_to_output_srs: gdal.Transformer = gdal.Transformer(temp_gdal_ds, None, transformer_options)
        except BaseException as e:
            self.set_message_text(f"Error: {e}")
            return

        warp_options = gdal.WarpOptions(**self._warp_kwargs)
        warp_save_path = f'/vsimem/temp_band_{0}'
        place_holder_arr = np.zeros((1, 1), np.uint8)
        temp_gdal_ds: gdal.Dataset = gdal_array.OpenNumPyArray(place_holder_arr, True)
        temp_gdal_ds.SetSpatialRef(ref_srs)
        temp_gdal_ds.SetGCPs([pair[1] for pair in gcps], ref_projection)
        transformed_ds: gdal.Dataset = gdal.Warp(warp_save_path, temp_gdal_ds, options=warp_options)
        transformed_gt = transformed_ds.GetGeoTransform()

        tr_output_srs_to_ref_srs = osr.CoordinateTransformation(output_srs, ref_srs)

        residuals = []
        for entry, gcp in gcps:
            # These coordinates could get back to us in either lat/lon, lon/lat, or north/easting, easting/north
            ok, (output_spatial_x, output_spatial_y, z) = tr_pixel_to_output_srs.TransformPoint(False, gcp.GCPPixel, gcp.GCPLine)

            # This variable is true of the output srs is order in lat, lon or in northing,easting
            lat_lon_like_ordering_output: bool = False
            # Get the output spatial reference system is geographic or projected
            if output_srs.IsGeographic():
                lat_lon_like_ordering_output = bool(output_srs.EPSGTreatsAsLatLong())
                # Figure out the orientatino of the axis
            else:
                lat_lon_like_ordering_output = bool(output_srs.EPSGTreatsAsNorthingEasting())
            print(f"output_srs, lat_lon_like_ordering: {lat_lon_like_ordering_output}")

            # Get whether the input spatial reference system is geograhpic or projected
                # Figure out the oritentatin of the axis 
            lat_lon_like_ordering_input: bool = False
            # Get the output spatial reference system is geographic or projected
            if ref_srs.IsGeographic():
                lat_lon_like_ordering_input = bool(ref_srs.EPSGTreatsAsLatLong())
                # Figure out the orientatino of the axis
            else:
                lat_lon_like_ordering_input = bool(ref_srs.EPSGTreatsAsNorthingEasting())
            print(f"ref_srs, lat_lon_like_ordering: {lat_lon_like_ordering_input}")
            
            # # Decide whether we need to swap them
            # ref_spatial_coord = None
            # if lat_lon_like_ordering_output == lat_lon_like_ordering_input:
            #     ref_spatial_coord = tr_output_srs_to_ref_srs.TransformPoint(output_spatial_x, output_spatial_y, 0)
            # else:
            #     ref_spatial_coord = tr_output_srs_to_ref_srs.TransformPoint(output_spatial_y, output_spatial_x, 0)
            ref_spatial_coord = tr_output_srs_to_ref_srs.TransformPoint(output_spatial_x, output_spatial_y, 0)

            # print(f"output_spatial_x: {output_spatial_x}")
            # print(f"output_spatial_y: {output_spatial_y}")

            ref_spatial_x, ref_spatial_y = ref_spatial_coord[0], ref_spatial_coord[1]

            # print(f"reference_spatial_x: {ref_spatial_x}")
            # print(f"reference_spatial_y: {ref_spatial_y}")

            # print(f"gcp.GCPX: {gcp.GCPX}")
            # print(f"gcp.GCPY: {gcp.GCPY}")

            # target_pixel_coord = ref_dataset.to_geographic_coords((gcp.GCPPixel, gcp.GCPLine))
            # print(f"original x: {target_pixel_coord[0]}")
            # print(f"original y: {target_pixel_coord[1]}")

            error_spatial_x = gcp.GCPX - ref_spatial_x
            error_spatial_y = gcp.GCPY - ref_spatial_y

            # print(f"error_spatial_x: {error_spatial_x}")
            # print(f"error_spatial_y: {error_spatial_y}")

            print(f"transformed_gt[1]: {transformed_gt[1]}")
            print(f"transformed_gt[5]: {transformed_gt[5]}")

            error_raster_x = error_spatial_x / transformed_gt[1]
            error_raster_y = error_spatial_y / transformed_gt[5]

            entry.set_residual_x(round(error_raster_x, 6))
            entry.set_residual_y(round(error_raster_y, 6))

            self._update_residuals(entry)
        
            residuals.append((error_raster_x, error_raster_y))
        
        tr_pixel_to_output_srs = None
        tr_output_srs_to_ref_srs = None
        temp_gdal_ds = None


    # region Accepting


    def accept(self):
        save_path = self._get_save_file_path()
        if save_path is None or \
            self._target_rasterpane.get_rasterview().get_raster_data() is None or \
            self._reference_rasterpane.get_rasterview().get_raster_data() is None:
            super().accept()
            return

        gcps: List[GeoRefTableEntry, gdal.GCP] = []

        for table_entry in self._table_entry_list:
            spatial_coord = table_entry.get_gcp_pair().get_reference_gcp().get_spatial_point()
            assert spatial_coord is not None, f"spatial_coord is none on reference gcp!, spatial_coord: {spatial_coord}"
            target_pixel_coord = table_entry.get_gcp_pair().get_target_gcp().get_point()
            gcps.append((table_entry, gdal.GCP(spatial_coord[0], spatial_coord[1], 0, target_pixel_coord[0], target_pixel_coord[1])))

        target_dataset = self._target_rasterpane.get_rasterview().get_raster_data()
        target_dataset_impl = target_dataset.get_impl()

        ref_dataset = self._reference_rasterpane.get_rasterview().get_raster_data()
        ref_projection = ref_dataset.get_wkt_spatial_reference()
        ref_srs = ref_dataset.get_spatial_ref()
        temp_gdal_ds = None
        output_dataset = None
        print(f"self._warp_kwargs: {self._warp_kwargs}")
        if isinstance(target_dataset_impl, GDALRasterDataImpl):
            print(f"gdasl raster impl is being saved")
            target_gdal_dataset = target_dataset_impl.gdal_dataset
            temp_vrt_path = '/vsimem/ref.vrt'
            translate_opts = None
            if target_dataset.get_data_ignore_value() is not None:
                translate_opts = gdal.TranslateOptions(
                    format='VRT',
                    noData=target_dataset.get_data_ignore_value(),
                )
            else:
                translate_opts = gdal.TranslateOptions(
                    format='VRT',
                )
            temp_gdal_ds = gdal.Translate(temp_vrt_path, target_gdal_dataset, options=translate_opts)
            temp_gdal_ds.SetGCPs([pair[1] for pair in gcps], ref_projection)
            warp_options = gdal.WarpOptions(**self._warp_kwargs)
            output_dataset = gdal.Warp(save_path, temp_gdal_ds, options=warp_options)
        else:
            # I warp one band of the input dataset to a virtual memory file, 
            # so I can create the correct output data size.
            # Then I create the output dataset with width and height equal to the warp,
            # but correct number of bands
            # Then I override each band in the created array and flush cache
            output_size: Tuple[int, int] = None
            output_dataset: gdal.Dataset = None
            driver: gdal.Driver = gdal.GetDriverByName("GTiff")
            gdal_dtype = gdal_array.NumericTypeCodeToGDALTypeCode(target_dataset.get_elem_type())

            # Get the output size
            warp_options = gdal.WarpOptions(**self._warp_kwargs)
            warp_save_path = f'/vsimem/temp_band_{0}'
            band_arr = target_dataset.get_band_data(0)
            temp_gdal_ds: gdal.Dataset = gdal_array.OpenNumPyArray(band_arr, True)
            temp_gdal_ds.SetSpatialRef(ref_srs)
            temp_gdal_ds.SetGCPs([pair[1] for pair in gcps], ref_projection)
            transformed_ds: gdal.Dataset = gdal.Warp(warp_save_path, temp_gdal_ds, options=warp_options)

            width = transformed_ds.RasterXSize
            height = transformed_ds.RasterYSize
            output_size = (width, height)
            output_bytes = width * height * target_dataset.num_bands() * target_dataset.get_elem_type().itemsize

            gdal.Unlink(warp_save_path)

            ratio = MAX_RAM_BYTES / output_bytes
            if ratio > 1.0:
                print(f"numpy array saved all at once")
                warp_options = gdal.WarpOptions(**self._warp_kwargs)
                band_arr = target_dataset.get_image_data()
                temp_gdal_ds: gdal.Dataset = gdal_array.OpenNumPyArray(band_arr, True)
                temp_gdal_ds.SetSpatialRef(ref_srs)
                temp_gdal_ds.SetGCPs([pair[1] for pair in gcps], ref_projection)
                set_data_ignore_of_gdal_dataset(temp_gdal_ds, target_dataset)
                output_dataset: gdal.Dataset = gdal.Warp(save_path, temp_gdal_ds, options=warp_options)
                output_dataset.FlushCache()

            else:
                print(f"numpy array saved in chunks")
                num_bands_per = int(ratio * target_dataset.num_bands())
                for band_index in range(0, target_dataset.num_bands(), num_bands_per):
                    band_list_index = [band for band in range(band_index, band_index+num_bands_per) if band < target_dataset.num_bands()]
                    warp_options = gdal.WarpOptions(**self._warp_kwargs)
                    warp_save_path = f'/vsimem/temp_band_{min(band_list_index)}_to_{max(band_list_index)}'
                    # print(f"saving chunk: {min(band_list_index)}_to_{max(band_list_index)}")
            
                    band_arr = target_dataset.get_multiple_band_data(band_list_index)
                    temp_gdal_ds: gdal.Dataset = gdal_array.OpenNumPyArray(band_arr, True)
                    temp_gdal_ds.SetSpatialRef(ref_srs)
                    temp_gdal_ds.SetGCPs([pair[1] for pair in gcps], ref_projection)
                    set_data_ignore_of_gdal_dataset(temp_gdal_ds, target_dataset)
                    transformed_ds: gdal.Dataset = gdal.Warp(warp_save_path, temp_gdal_ds, options=warp_options)

                    width = transformed_ds.RasterXSize
                    height = transformed_ds.RasterYSize
                    assert width == output_size[0] and height == output_size[1], \
                            "Width and/or height of warped band does not equal a previous warped band"

                    if output_dataset is None:
                        output_dataset = driver.Create(save_path, width, height, target_dataset.num_bands(), gdal_dtype)
                    
                    write_raster_to_dataset(output_dataset, band_list_index, transformed_ds.ReadAsArray(), gdal_dtype)
                    # print(f"Warping bands: {min(band_list_index)} to {max(band_list_index)} out of {ref_dataset.num_bands()}")
                    gdal.Unlink(warp_save_path)
                    transformed_ds = None

        if output_dataset is None:
            raise RuntimeError("gdal.Warp failed to produce a transformed dataset.")

        copy_metadata_to_gdal_dataset(output_dataset, target_dataset)
        gt = output_dataset.GetGeoTransform()
        if gt is None:
            raise RuntimeError("Failed to retrieve geotransform from the transformed dataset.")

        output_dataset.FlushCache()
        output_dataset = None

        super().accept()

    # region Event overrides

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Return, Qt.Key_Enter, Qt.Key_Escape):
            event.accept()  # Do nothing on Enter or Escape
        else:
            super().keyPressEvent(event)
        
