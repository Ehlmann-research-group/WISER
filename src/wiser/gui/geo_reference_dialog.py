import os
from typing import List, Optional, Dict, Tuple

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

from .generated.geo_referencer_dialog_ui import Ui_GeoReferencerDialog

from wiser.gui.app_state import ApplicationState
from wiser.gui.rasterpane import RasterPane
from wiser.gui.geo_reference_pane import GeoReferencerPane
from wiser.gui.geo_reference_task_delegate import \
    (GeoReferencerTaskDelegate, GroundControlPointPair, GroundControlPoint,
     GroundControlPointCoordinate, PointSelectorType, PointSelector,
     GroundControlPointRasterPane)
from wiser.gui.util import get_random_matplotlib_color, get_color_icon, make_into_help_button

from wiser.raster.dataset import RasterDataSet
from wiser.raster.dataset_impl import GDALRasterDataImpl
from wiser.raster.utils import copy_metadata_to_gdal_dataset, set_data_ignore_of_gdal_dataset

from wiser.bandmath.utils import write_raster_to_dataset

from enum import IntEnum, Enum

from osgeo import gdal, osr, gdal_array

import numpy as np

from abc import ABC

import csv
from pathlib import Path

from pyproj import CRS
from pyproj.database import get_authorities

from wiser.bandmath.builtins.constants import MAX_RAM_BYTES

AVAILABLE_AUTHORITIES = get_authorities()

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

class GeneralCRS():
    def get_osr_crs(self) -> Optional[osr.SpatialReference]:
        '''
        Gets a osr.SpatialReference object for this class
        '''
        raise NotImplementedError("Function has not yet been implemented.")

    def __eq__(self, other: 'GeneralCRS'):
        return self.get_osr_crs().ExportToWkt() == other.get_osr_crs().ExportToWkt()

class AuthorityCodeCRS(GeneralCRS):

    def __init__(self, authority_name: str, authority_code: int):
        self.authority_name = authority_name
        self.authority_code = authority_code
    
    def get_osr_crs(self) -> Optional[osr.SpatialReference]:
        # Build the AUTH:CODE string
        auth_code_str = f"{self.authority_name}:{self.authority_code}"

        # Create and populate the SpatialReference
        srs = osr.SpatialReference()
        err = srs.SetFromUserInput(auth_code_str)
        if err != 0:
            raise RuntimeError(f"Failed to import CRS '{auth_code_str}' (GDAL error {err})")

        return srs

class UserGeneratedCRS(GeneralCRS):
    def __init__(self, name: str, crs: osr.SpatialReference):
        self._name = name
        self._crs = crs

    def get_osr_crs(self) -> Optional[osr.SpatialReference]:
        return self._crs
    
class WktGeneratedCRS(GeneralCRS):
    def __init__(self, name: str, wkt: str):
        self._name = name
        self._wkt = wkt
        self._crs = CRS.from_wkt(wkt)
        print(f"self._crs.to_wkt: {self._crs.to_wkt()}")
        crs = osr.SpatialReference()
        crs.ImportFromWkt(wkt)
        self._crs = crs
        print(f"self._crs.Export: {self._crs.ExportToWkt()}")
    
    def get_osr_crs(self) -> Optional[osr.SpatialReference]:
        return self._crs


COMMON_SRS = {
    "WGS84 EPSG:4326": AuthorityCodeCRS('EPSG', 4326),
    "Web Mercator EPSG:3857": AuthorityCodeCRS('EPSG', 3857),
    "NAD83 / UTM zone 15N EPSG:26915": AuthorityCodeCRS('EPSG', 26915)
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

        self._curr_output_srs: GeneralCRS = None
        self._curr_resample_alg = None
        self._curr_transform_type: TRANSFORM_TYPES = None

        self._default_color_button: QPushButton = None

        self._manual_entry_spacer = None

        self._first_init()

        self._warp_kwargs: Dict = None
        self._transform_options: List[str] = None
        self._suppress_cell_changed: bool = False

        self._prev_chosen_ref_crs_index: int = 0

        # These are actually always the current index, we call them previous
        # because when the current index is change on click, we need access
        # to the index before the click occured
        self._prev_ref_dataset_index: int = None
        self._prev_target_dataset_index: int = None

    def exec_(self):
        self._refresh_init()
        super().exec_()
    
    def show(self):
        self._refresh_init()
        super().show()


    # region Initialization


    def _first_init(self):
        self._init_dataset_choosers()
        self._update_dataset_choosers()
        self._init_rasterpanes()
        self._init_gcp_table()
        self._init_output_crs_finder()
        self._init_interpolation_type_cbox()
        self._init_poly_order_cbox()
        self._init_file_saver()
        self._init_default_color_chooser()
        self._init_manual_ref_crs_finder()
        self._init_manual_ref_point_enter()
        self._init_warp_button()
        self._update_manual_ref_chooser_display(None)
        self._init_help_button()
        self._init_gcp_io_buttons()

    def _refresh_init(self):
        self._update_dataset_choosers()
        self._update_ref_crs_cbox_items()
        self._update_output_srs_cbox_items()

    def _init_gcp_io_buttons(self):
        self._ui.btn_save_gcps.clicked.connect(self._on_save_gcps_clicked)
        self._ui.btn_load_gcps.clicked.connect(self._on_load_gcps_clicked)
        self._ui.btn_clear_gcps.clicked.connect(self._on_clear_gcps_clicked)
    
    def _init_help_button(self):
        btn_box = self._ui.buttonBox
        btn_box.helpRequested.connect(self._on_show_help)

    def _init_warp_button(self):
        warp_btn = self._ui.btn_run_warp
        warp_btn.clicked.connect(self._on_warp_button_clicked)

    def _init_manual_ref_point_enter(self):
        lat_north_ledit = self._ui.ledit_lat_north
        lon_east_ledit  = self._ui.ledit_lon_east

        # Validator that only allows floating-point numbers (no strict range)
        float_validator = QDoubleValidator(self)
        float_validator.setNotation(QDoubleValidator.StandardNotation)

        lat_north_ledit.setValidator(float_validator)
        lon_east_ledit.setValidator(float_validator)

        lat_north_ledit.returnPressed.connect(self._on_ref_manual_ledit_enter)
        lon_east_ledit.returnPressed.connect(self._on_ref_manual_ledit_enter)

    def _init_manual_ref_crs_finder(self):
        # Initialize the authority chooser
        authority_cbox = self._ui.cbox_authority
        authority_cbox.clear()
        for auth in AVAILABLE_AUTHORITIES:
            authority_cbox.addItem(auth, auth)

        # Initialize the code enter QLineEdit
        srs_code_ledit = self._ui.ledit_srs_code

        int_validator = QIntValidator(1, 2147483647, self)

        srs_code_ledit.setValidator(int_validator)

        # Initialize the choosable CRSs
        srs_to_choose_cbox = self._ui.cbox_choose_crs
        for name, srs in COMMON_SRS.items():
            srs_to_choose_cbox.addItem(name, srs)

        for name, (srs, _) in self._app_state.get_user_created_crs().items():
            srs_to_choose_cbox.addItem(name, UserGeneratedCRS(name, srs))

        srs_to_choose_cbox.activated.connect(self._on_switch_chosen_ref_srs)

        # initialize the find button
        find_crs_btn = self._ui.btn_find_crs
        find_crs_btn.clicked.connect(self._on_find_crs)

        # Initialize the help button
        make_into_help_button(self._ui.tbtn_help,
                              'https://ehlmann-research-group.github.io/WISER-UserManual/Georeferencer/#reference-system-information',
                              'Get help on reference systems')

    def _update_ref_crs_cbox_items(self):
        srs_to_choose_cbox = self._ui.cbox_choose_crs
        srs_to_choose_cbox.clear()
        for name, srs in COMMON_SRS.items():
            srs_to_choose_cbox.addItem(name, srs)

        for name, (srs, _) in self._app_state.get_user_created_crs().items():
            srs_to_choose_cbox.addItem(name, UserGeneratedCRS(name, srs))

    def _update_manual_ref_chooser_display(self, dataset: Optional[RasterDataSet]):
        """
        We want to enable this when no data is selected and disable this when data is selected
        """
        if self._manual_entry_spacer is None:
            self._manual_entry_spacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)
        if dataset is None:
            # We want to make sure the manual chooser is being shown
            self._ui.widget_manual_entry.show()
            self._ui.widget_ref_image.hide()
            self._add_manual_spacer_once()
        else:
            # We want to make sure it is not being shown 
            self._ui.widget_manual_entry.hide()
            self._ui.widget_ref_image.show()
            self._ui.vlayout_reference.removeItem(self._manual_entry_spacer)

    def _add_manual_spacer_once(self):
        layout = self._ui.vlayout_reference

        # scan all items in the layout…
        for idx in range(layout.count()):
            item = layout.itemAt(idx)
            # .spacerItem() returns our QSpacerItem if this layout‐item *is* a spacer
            if item.spacerItem() is self._manual_entry_spacer:
                # already in there – bail out
                return

        # if we got here, we didn’t find it yet
        layout.addItem(self._manual_entry_spacer)

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

    def _init_output_crs_finder(self):
        """
        Initialize the spatial reference combo box for the output crs
        """
        # Initialize the authority chooser
        authority_cbox = self._ui.cbox_output_authority
        authority_cbox.clear()
        for auth in AVAILABLE_AUTHORITIES:
            authority_cbox.addItem(auth, auth)

        srs_cbox = self._ui.cbox_srs
        srs_cbox.activated.connect(self._on_switch_output_srs)
        self._update_output_srs_cbox_items()

        # Initialize the code enter QLineEdit
        srs_code_ledit = self._ui.ledit_output_code

        int_validator = QIntValidator(1, 2147483647, self)

        srs_code_ledit.setValidator(int_validator)

        # initialize the find button
        find_crs_btn = self._ui.btn_find_output_crs
        find_crs_btn.clicked.connect(self._on_find_output_crs)

    def _update_output_srs_cbox_items(self):
        srs_cbox = self._ui.cbox_srs
        srs_cbox.clear()
        # Use the friendly key (e.g., "WGS84") as the display text,
        # and store the corresponding SRS string (e.g., "EPSG:4326") as userData.
        if self._reference_rasterpane is not None and self._reference_rasterpane.get_rasterview().get_raster_data() is not None:
            try:
                ref_ds = self._reference_rasterpane.get_rasterview().get_raster_data()
                reference_srs_name = "Ref CRS: " + ref_ds.get_spatial_ref().GetName()
                reference_srs_code = ref_ds.get_spatial_ref().GetAuthorityCode(None)
                if reference_srs_code is None:
                    self.set_message_text("Could not get an authority code for default dataset")
                    ref_srs = ref_ds.get_spatial_ref()
                    crs = CRS.from_wkt(ref_srs.ExportToWkt())
                    if crs is not None:
                        auth_info = crs.to_authority()
                        if auth_info is None:
                            name = crs.name if crs.name is not None else 'Uknown Name'
                            wkt_crs = WktGeneratedCRS(name, crs.to_wkt())
                            srs_cbox.addItem(name, wkt_crs)
                        else:
                            auth_name, auth_code = crs.to_authority()
                            srs_cbox.addItem(reference_srs_name, AuthorityCodeCRS(auth_name, int(auth_code)))
                else:
                    srs_cbox.addItem(reference_srs_name, AuthorityCodeCRS(ref_ds.get_spatial_ref().GetAuthorityName(None), \
                                     int(reference_srs_code)))
            except BaseException as e:
                pass

        for name, srs in COMMON_SRS.items():
            srs_cbox.addItem(name, srs)
        
        for name, (srs, _) in self._app_state.get_user_created_crs().items():
            srs_cbox.addItem(name, UserGeneratedCRS(name, srs))

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
        '''
        Initializes the columns of the table the GCPs will go into. Asigns number validators
        to each column that the user can change numbers in.
        '''
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
        '''
        Performs actions that should only be done once with dataset chooser.
        '''
        self._target_cbox.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self._target_cbox.activated.connect(self._on_switch_target_dataset)

        self._reference_cbox.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self._reference_cbox.activated.connect(self._on_switch_reference_dataset)

    def _update_dataset_choosers(self):
        '''
        Performs actions that should be done everytime geo ref dialog is reshown
        '''
        self._update_target_dataset_chooser()
        self._prev_target_dataset_index = self._target_cbox.currentIndex()

        self._update_reference_dataset_chooser()
        self._prev_ref_dataset_index = self._reference_cbox.currentIndex()

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

    def _on_find_output_crs(self):
        authority_str = self._ui.cbox_output_authority.currentText()
        authority_code = self._ui.ledit_output_code.text()
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

        self._add_srs_to_output_cbox(srs_name, AuthorityCodeCRS(authority_str, float(authority_code)))

    def _on_clear_gcps_clicked(self, checked: bool):
        """
        Asks the user for confirmation if they want to clear all the gcps.
        Clears all the gcps by removing all the rows in self._ui.table_gcps
        and then calls self._update_panes.
        """
        reply = QMessageBox.question(
            self,
            "Clear All GCPs?",
            "Are you sure you want to remove all ground control points?",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            # wipe out both the internal list and the table widget
            self._reset_gcps()
            # refresh the display panes
            self._update_panes()

    def _on_save_gcps_clicked(self, checked: bool):
        if not self._table_entry_list:
            QMessageBox.information(self, "No GCPs", "There are no ground-control points to save.")
            return

        filename, _ = QFileDialog.getSaveFileName(
            self,
            self.tr("Save ground-control points"),
            filter=self.tr("QGIS points (*.points);;ENVI ASCII (*.pts)")
        )
        if not filename:
            return

        srs = self._get_reference_srs()
        auth_name = srs.GetAuthorityName(None)
        auth_code = srs.GetAuthorityCode(None)
        wkt_str = None
        if auth_name is None or auth_code is None:
            wkt_str = srs.ExportToWkt()
            crs = CRS.from_wkt(wkt_str)
            wkt_auth = crs.to_authority()
            if wkt_auth is not None:
                auth_name, auth_code = wkt_auth
    
        ext = Path(filename).suffix.lower()
        try:
            if ext == ".points":
                self._write_qgis_points_file(filename, auth_name, auth_code, wkt_str)
            elif ext == ".pts":
                self._write_envi_pts_file(filename, auth_name, auth_code, wkt_str)
            else:
                QMessageBox.warning(self, "Extension error",
                                    "Please use either *.points or *.pts")
                return
            self.set_message_text(f"GCPs saved to {filename}")
        except Exception as e:
            QMessageBox.critical(self, "Save failed", str(e))

    def _on_load_gcps_clicked(self, checked: bool):
        filename, _ = QFileDialog.getOpenFileName(
            self,
            self.tr("Load ground-control points"),
            filter=self.tr("GCP files (*.points *.pts)")
        )
        if not filename:
            return

        try:
            points, gcp_srs = self._read_gcp_file(filename)
            if points is None or gcp_srs is None:
                raise RuntimeError(f"Passed in reference system can't be parsed. Reference system wkt:\n {gcp_srs.ExportToPrettyWkt()}")
        except Exception as e:
            QMessageBox.critical(self, "Load failed", str(e))
            return
        self.load_gcps_and_srs(gcp_points=points, gcp_srs=gcp_srs)

    def _on_show_help(self):
        QMessageBox.information(
            self,
            self.tr("How to use the Georeferencer"),
            self.tr("""
            <h3>Quick Start</h3>
            <ol>
              <li>Pick your Target and Reference images.</li>
              <li>Select or lookup the output CRS (Authority + Code)<br>
                  if you do not have a reference image.</li>
              <li>Click in the image to add ground control points.<br>
                  Enter lat/lon if adding manually. <br>
                  Hit enter after each point.</li>
              <li>Hit escape to undo your enter press. </li>
              <li>Choose your interpolation & polynomial order.</li>
              <li>Set an output path and click <b>Run Warp</b>.</li>
            </ol>
            """),
            QMessageBox.Ok
        )

    def _on_warp_button_clicked(self, checked: bool):
        self._create_warped_output()

    def _on_ref_manual_ledit_enter(self):
        lat_north_str = self._ui.ledit_lat_north.text()
        lon_east_str = self._ui.ledit_lon_east.text()

        if lat_north_str == "" or lon_east_str == "":
            self.set_message_text("Ensure both Lat/North and Lon/East have valid values")
            return

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

        self._add_srs_to_ref_choose_cbox(srs_name, AuthorityCodeCRS(authority_str, float(authority_code)))

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
            target_ds = self._get_target_dataset()
            target_ds_filepaths = []
            if target_ds is not None:
                target_ds_filepaths = target_ds.get_filepaths()
            ref_ds = self._get_ref_dataset()
            ref_ds_filepaths = []
            if ref_ds is not None:
                ref_ds_filepaths = ref_ds.get_filepaths()
            filename = file_dialog.selectedFiles()[0]
            if filename in target_ds_filepaths or filename in ref_ds_filepaths:
                QMessageBox.information(self, self.tr("Wrong Save Path"), \
                                        self.tr("The save path you chose matches either the target\n" + 
                                                "or reference dataset's save path. Please change.\n\n"
                                                f"Chosen save path:\n{filename}"))
                return
            self._ui.ledit_save_path.setText(filename)
            self._georeference()

    def _on_switch_output_srs(self, index: int):
        # We don't record the output srs because we get this
        # directly from the combo box.
        self._georeference()

    def _on_switch_chosen_ref_srs(self, index: int):
        if self._prev_chosen_ref_crs_index != index:
            if len(self._table_entry_list) > 0:
                confirm = QMessageBox.question(self, self.tr("Change reference CRS?"),
                                        self.tr("You are changing the reference CRS.") +
                                        "\n\nDo you want to discard all selected GCPs?")
                if confirm == QMessageBox.Yes:
                    self._reset_gcps()
        self._prev_chosen_ref_crs_index = self._ui.cbox_choose_crs.currentIndex()

    def _on_switch_resample_alg(self, index: int):
        resample_alg = self._ui.cbox_interpolation.itemData(index)
        self._curr_resample_alg = resample_alg

    def _on_switch_transform_type(self, index: int):
        transform_type = self._ui.cbox_poly_order.itemData(index)
        self._curr_transform_type = transform_type
        self._georeference()

    def _on_choose_default_color(self):
        color = QColorDialog.getColor(parent=self, initial=self._initial_default_color)
        if color.isValid():
            color_str = color.name()
            for row in range(len(self._table_entry_list)):
                # We only want to change the colors of the points that weren't explicitly
                # changed. We can easily disable this by removing the if statement
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
            # Check if the file save path already there matches this
            current_save_path = self._get_current_save_path()
            dataset = self._app_state.get_dataset(ds_id)
            if dataset is not None:
                if current_save_path in dataset.get_filepaths():
                    QMessageBox.information(self, self.tr("Target Dataset Path Equals Save Path"),
                                            self.tr("The target dataset path equals the save path.\n"
                                            "Change the save path before selecting this target dataset."))
                    self._target_cbox.setCurrentIndex(self._prev_target_dataset_index)
                    return
            # Check if there are already GCPs
            if len(self._table_entry_list) > 0 and self._prev_target_dataset_index != index:
                confirm = QMessageBox.question(self, self.tr("Change Target Dataset?"),
                                     self.tr("Are you sure you want to change the target dataset?") +
                                     "\n\nThis will discard all selected GCPs. Do you want\n"
                                     "to continue?")
                if confirm == QMessageBox.Yes:
                    self._reset_gcps()
                else:
                    self._target_cbox.setCurrentIndex(self._prev_target_dataset_index)
                    return
        except:
            pass
        self._target_rasterpane.show_dataset(dataset)
        self._prev_target_dataset_index = self._target_cbox.currentIndex()

    def _on_switch_reference_dataset(self, index: int):
        ds_id = self._reference_cbox.itemData(index)
        dataset = None
        try:
            # Check if the file save path already there matches this
            current_save_path = self._get_current_save_path()
            dataset = self._app_state.get_dataset(ds_id)
            if dataset is not None:
                if current_save_path in dataset.get_filepaths():
                    QMessageBox.information(self, self.tr("Reference Dataset Path Equals Save Path"),
                                            self.tr("The reference dataset path equals the save path.\n"
                                            "Change the save path before selecting this reference dataset."))
                    self._reference_cbox.setCurrentIndex(self._prev_ref_dataset_index)
                    return
            if len(self._table_entry_list) > 0 and self._prev_ref_dataset_index != index:
                confirm = QMessageBox.question(self, self.tr("Change Reference Dataset?"),
                                     self.tr("Are you sure you want to change the reference dataset?") +
                                     "\n\nThis will discard all selected GCPs. Do you want\n"
                                     "to continue?")
                if confirm == QMessageBox.Yes:
                    self._reset_gcps()
                else:
                    self._reference_cbox.setCurrentIndex(self._prev_ref_dataset_index)
                    return
            if not dataset.has_geographic_info():
                QMessageBox.warning(self, self.tr("Unreferenced Dataset"), \
                                    self.tr("You must choose a dataset with a spatial reference system"))
                self._reference_cbox.setCurrentIndex(self._prev_ref_dataset_index)
                return
        except:
            pass
        self._reference_rasterpane.show_dataset(dataset)
        self._update_output_srs_cbox_items()
        self._update_manual_ref_chooser_display(dataset)
        self._prev_ref_dataset_index = self._reference_cbox.currentIndex()


    #region Helpers

    def compare_srs_lenient(self, srs1: osr.SpatialReference, srs2: osr.SpatialReference):
        '''
        Compares srs1 and srs2, but first puts them to WKT, then reimports them as an osr.SpatialReference.
        We do this because sometimes srs's are the same but because of how they were imported, some
        less-important meta data may have been lost. We get rid of this meta data for both srs's by
        using this function
        '''
        wkt_1 = srs1.ExportToWkt()
        wkt_2 = srs2.ExportToWkt()

        srs1_clone = osr.SpatialReference()
        srs1_clone.ImportFromWkt(wkt_1)

        srs2_clone = osr.SpatialReference()
        srs2_clone.ImportFromWkt(wkt_2)

        return srs1_clone.IsSame(srs2_clone)


    def load_gcps_and_srs(self, gcp_points: List[Tuple[float, float, float, float]], gcp_srs: GeneralCRS):
        '''
        Loads the gcps in with the specified srs
        '''
        target_ds = self._get_target_dataset()
        if target_ds is None:
            return

        ref_ds = self._get_ref_dataset()

        skipped_gcps = []
        if ref_ds is not None and self.compare_srs_lenient(gcp_srs.get_osr_crs(), ref_ds.get_spatial_ref()):
            for map_x, map_y, pix_x, pix_y in gcp_points:
                # Verify pixel-within-images
                if not (0 <= pix_x < target_ds.get_width() and 0 <= pix_y < target_ds.get_height()):
                    skipped_gcps.append(((map_x, map_y, pix_x, pix_y), "Target GCP Pixel is outside of target dataset's raster bounds."))
                    # self.set_message_text("Skipped one GCP: target pixel outside image bounds.")
                    continue
                ref_px = ref_ds.geo_to_pixel_coords_exact((map_x, map_y))
                if ref_px is None or not (0 <= ref_px[0] < ref_ds.get_width() and
                                        0 <= ref_px[1] < ref_ds.get_height()):
                    skipped_gcps.append(((map_x, map_y, pix_x, pix_y), "Reference GCP coordinate is outside of reference dataset's raster bounds."))
                    continue

                tgt_gcp = GroundControlPointRasterPane((pix_x, pix_y), self._target_rasterpane)
                ref_gcp = GroundControlPointRasterPane((ref_px[0], ref_px[1]), self._reference_rasterpane)
                pair = GroundControlPointPair(tgt_gcp, ref_gcp)
                self.gcp_pair_added.emit(pair)
        else:
            # mismatch – fall back to manual entry mode
            self._reference_cbox.setCurrentIndex(self._reference_cbox.findData(-1))
            self._update_manual_ref_chooser_display(None)
            # Populate the srs in the cbox_choose_crs
            self._add_srs_to_ref_choose_cbox(gcp_srs.get_osr_crs().GetName(), gcp_srs)
            self.set_message_text("Reference CRS changed to match GCP file; select each "
                                  "target point then press Enter to pair it.")
            for map_x, map_y, pix_x, pix_y in gcp_points:
                # Verify pixel-within-images
                if not (0 <= pix_x < target_ds.get_width() and 0 <= pix_y < target_ds.get_height()):
                    skipped_gcps.append(((map_x, map_y, pix_x, pix_y), "Target GCP Pixel is outside of raster bounds."))
                    continue
                tgt_gcp = GroundControlPointRasterPane((pix_x, pix_y), self._target_rasterpane)
                ref_gcp = GroundControlPointCoordinate((map_x, map_y),
                                                    PointSelectorType.REFERENCE_POINT_SELECTOR,
                                                    gcp_srs.get_osr_crs())
                pair = GroundControlPointPair(tgt_gcp, ref_gcp)
                self.gcp_pair_added.emit(pair)

        # ────────────────────────────────────────────────────────────────
        #  Show skipped‐GCPs if any
        # ────────────────────────────────────────────────────────────────
        if skipped_gcps:
            info_lines = []
            info_lines.append(f"Skipped GCPs")
            info_lines.append("")
            for tpl, reason in skipped_gcps:
                info_lines.append(f"GCP: {tpl}")
                info_lines.append(f"Reason: {reason}")
                info_lines.append("")  # blank line between entries

            QMessageBox.information(
                self,
                "Skipped GCPs",
                "\n".join(info_lines).rstrip()
            )

    def _read_gcp_file(self, path: str):
        ext = Path(path).suffix.lower()
        if ext == ".points":
            return self._read_qgis_points_file(path)
        elif ext == ".pts":
            return self._read_envi_pts_file(path)
        raise RuntimeError("Unsupported GCP file extension")

    def _read_qgis_points_file(self, path: str) -> Tuple[List, GeneralCRS]:
        """Read a QGIS ``*.points`` file.

        If the header contains ``# CRS`` the routine returns an
        :class:`AuthorityCodeCRS`; otherwise it looks for ``# WKT`` and
        returns a :class:`WktGeneratedCRS`.

        Returns
        -------
        points : list[tuple[float, float, float, float]]
            ``(map_x, map_y, pixel_x, pixel_y)`` tuples.
        gcp_srs : GeneralCRS
            Extracted from the header
        """
        points = []
        gcp_srs = None
        pending_wkt = None

        with open(path, newline="") as f:
            rdr = csv.reader(f)
            for row in rdr:
                if not row:
                    continue
                if row[0].startswith("# CRS"):
                    _, authcode = row[:2]
                    auth, code = authcode.split(":")
                    gcp_srs = AuthorityCodeCRS(auth, int(code))
                    continue
                if row[0].startswith("# WKT"):
                    # WKT may contain commas, so rebuild the original line
                    pending_wkt = ",".join(row[1:]).strip()
                    continue
                if row[0].startswith("mapX"):
                    continue
                map_x, map_y, pix_x, pix_y, *_ = map(float, row[:5])
                points.append((map_x, map_y, pix_x, pix_y))

        if gcp_srs is None and pending_wkt:
            gcp_srs = WktGeneratedCRS("WKT", pending_wkt)
        if gcp_srs is None:
            raise RuntimeError("No CRS or WKT line found in .points file")
        return points, gcp_srs

    def _read_envi_pts_file(self, path: str) -> Tuple[List, GeneralCRS]:
        """Read an ENVI ``*.pts`` file with optional embedded WKT.

        The routine first tries the traditional ``; projection info`` comment
        to extract *(authority, code)*.  If that is missing it looks for a
        line beginning ``; wkt =`` and constructs a
        :class:`WktGeneratedCRS`.

        Returns
        -------
        points : list[tuple[float, float, float, float]]
            ``(map_x, map_y, pixel_x, pixel_y)`` tuples.
        gcp_srs : GeneralCRS
            Extracted from the header
        """
        points = []
        gcp_srs = None
        pending_wkt = None
        with open(path) as f:
            for ln in f:
                ln = ln.strip()
                if ln.lower().startswith("; projection info"):
                    inside = ln.split("{", 1)[-1].split("}", 1)[0]
                    auth, code, *_ = [x.strip().split(",")[0] for x in inside.split()]
                    gcp_srs = AuthorityCodeCRS(auth, int(code))
                elif ln.lower().startswith("; wkt ="):
                    pending_wkt = ln.split("=", 1)[1].strip()
                elif ln.startswith(";") or not ln:
                    continue
                else:
                    parts = list(map(float, ln.split()))
                    if len(parts) >= 5:
                        map_x, map_y, _elev, pix_x, pix_y = parts[:5]
                        points.append((map_x, map_y, pix_x, pix_y))
        if gcp_srs is None and pending_wkt:
            gcp_srs = WktGeneratedCRS("WKT", pending_wkt)
        if gcp_srs is None:
            raise RuntimeError("No projection info or WKT found in .pts file")
        return points, gcp_srs

    def _write_qgis_points_file(self, path: str, auth: Optional[str] = None,
                                  code: Optional[str] = None,
                                  wkt: Optional[str] = None) -> None:
        """Write ground-control points to a QGIS ``*.points`` file.

        Parameters
        ----------
        path : str
            Destination filepath (should end with ``.points``).
        auth : str or None, optional
            Authority name (e.g. ``"EPSG"``).  If *None*, the ``# CRS``
            header line is **omitted**.
        code : str or None, optional
            Authority code (e.g. ``"4326"``).  Ignored when *auth* is
            *None*.
        wkt : str or None, optional
            Well-Known Text definition of the CRS.  When provided it is
            written on a dedicated line starting with ``# WKT``.  QGIS
            will ignore this line, but *WISER* can parse it on load.

        Notes
        -----
        The file layout becomes::

            # CRS, EPSG:4326  ← optional
            # WKT,<LONG_WKT> ← optional
            mapX,mapY,pixelX,pixelY,enable
            123.4, 45.6, 100.0, 200.0, 1
            ...

        Only ASCII commas are used as delimiters so the routine is
        locale-independent.
        """
        header_rows = []
        if auth and code:
            header_rows.append(["# CRS", f"{auth}:{code}"])
        if wkt:
            header_rows.append(["# WKT", wkt])

        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(header_rows)
            writer.writerow(["mapX", "mapY", "pixelX", "pixelY", "enable"])

            for entry in self._table_entry_list:
                pair = entry.get_gcp_pair()
                map_x, map_y = pair.get_reference_gcp_spatial_coord()
                pix_x, pix_y = pair.get_target_gcp().get_point()
                writer.writerow([
                    map_x, map_y, pix_x, pix_y,
                    1 if entry.is_enabled() else 0,
                ])
    
    def _write_envi_pts_file(self, path: str, auth: Optional[str] = None,
                              code: Optional[str] = None,
                              wkt: Optional[str] = None) -> None:
        """Write ground-control points to an ENVI ``*.pts`` file. auth and code
        must be non-None or wkt must be non-None

        Parameters
        ----------
        path : str
            Destination filepath (should end with ``.pts``).
        auth, code : str or None, optional
            Authority name and code.  When either is *None*, the traditional
            ``; projection info`` comment is skipped.
        wkt : str or None, optional
            Well-Known Text to embed after a ``; wkt = `` comment.  ENVI will
            ignore this line; *WISER* uses it when the authority pair is
            missing.
        """
        with open(path, "w") as f:
            f.write("; ENVI Ground Control Points File\n")
            if auth and code:
                f.write(f"; projection info = {{{auth}, {code}, units=Degrees}}\n")
            if wkt:
                f.write(f"; wkt = {wkt}\n")
            f.write("; Map (x,y,elev), Image (x,y)\n;\n")
            for entry in self._table_entry_list:
                pair = entry.get_gcp_pair()
                map_x, map_y = pair.get_reference_gcp_spatial_coord()
                pix_x, pix_y = pair.get_target_gcp().get_point()
                f.write(f"{map_x:.10f} {map_y:.10f} 0.0 {pix_x:.3f} {pix_y:.3f}\n")

    def _get_current_save_path(self):
        return self._ui.ledit_save_path.text()


    def _get_save_file_path(self) -> str:
        path = self._ui.ledit_save_path.text()
        if len(path) > 0:
            abs_path = os.path.abspath(path)
            return abs_path
        return None

    def _clear_manual_ref_ledits(self):
        self._ui.ledit_lat_north.clear()
        self._ui.ledit_lon_east.clear()


    # region Table Entry Helpers


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
        '''
        Sets the color icon of the color at the passed in row
        '''
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
        table_widget.setItem(row_to_add, COLUMN_ID.REF_X_COL, ref_x_table_item)
    
        ref_y_table_item = QTableWidgetItem(str(ref_y))
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
        ref_x, ref_y = gcp_pair.get_reference_gcp_spatial_coord()

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
            # Index i also functions as the row in the table widget
            # where this entry is currently
            table_widget.setItem(i, COLUMN_ID.ID_COL, QTableWidgetItem(str(i)))


    #========================
    # region Getters
    #========================


    def get_table_entries(self) -> List[GeoRefTableEntry]:
        return self._table_entry_list

    def get_gcp_table_size(self) -> int:
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


    #========================
    # region Dataset Choosers
    #========================


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

        if new_index is None:
            if num_datasets > 0:
                new_index = min(current_index, num_datasets - 1)
            else:
                new_index = 0

        dataset_chooser.setCurrentIndex(new_index)


    #========================
    # region Misc
    #========================


    def _update_panes(self):
        self._target_rasterpane.update_all_rasterviews()
        self._reference_rasterpane.update_all_rasterviews()

    def set_message_text(self, text: str):
        if len(text) > 100:
            text = text[:100] + "…"
        self._ui.lbl_message.setText(text)

    def _add_srs_to_output_cbox(self, srs_name: str, crs: GeneralCRS):
        '''
        Adds the coordinate reference system that the user found to the choose combo box
        '''
        crs_choose_cbox = self._ui.cbox_srs
        osr_crs = crs.get_osr_crs()
        # Check for existing entry
        for idx in range(crs_choose_cbox.count()):
            data: GeneralCRS = crs_choose_cbox.itemData(idx)
            if data.get_osr_crs().IsSame(osr_crs):
                QMessageBox.information(
                    self,
                    "CRS Already Added",
                    f"The CRS {srs_name}: {crs} is already in the list as “{crs_choose_cbox.itemText(idx)}.”"
                )
                return

        # If not found, add as new entry
        crs_choose_cbox.addItem(srs_name, crs)
        crs_choose_cbox.setCurrentIndex(crs_choose_cbox.count()-1)
        self._on_switch_output_srs(crs_choose_cbox.count()-1)

    def _add_srs_to_ref_choose_cbox(self, srs_name: str, crs: GeneralCRS):
        '''
        Adds the coordinate reference system that the user found to the choose combo box
        '''
        crs_choose_cbox = self._ui.cbox_choose_crs
        osr_crs = crs.get_osr_crs()
        # Check for existing entry
        for idx in range(crs_choose_cbox.count()):
            data: GeneralCRS = crs_choose_cbox.itemData(idx)
            if data.get_osr_crs().IsSame(osr_crs):
                QMessageBox.information(
                    self,
                    "CRS Already Added",
                    f"The CRS {srs_name} is already in the list as “{crs_choose_cbox.itemText(idx)}.”"
                )
                return

        # If not found, add as new entry
        crs_choose_cbox.addItem(srs_name, crs)
        crs_choose_cbox.setCurrentIndex(crs_choose_cbox.count()-1)

    def _get_manual_ref_chosen_crs(self) -> osr.SpatialReference:
        '''
        From the information in the manual reference combo box, create an osr.SpatialReference
        object.
        '''
        return self._ui.cbox_choose_crs.currentData().get_osr_crs()


    #========================
    # region Geo referencing
    #========================


    def _enough_points_for_transform(self):
        return False if self._get_num_active_points() < min_points_per_transform[self._curr_transform_type] else True

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
        """
        Read self._curr_output_srs (authority_name, authority_code) and
        return a corresponding OSR SpatialReference object.
        """
        crs: GeneralCRS = self._ui.cbox_srs.currentData()
        return crs.get_osr_crs()

    def _get_reference_srs(self) -> Optional[osr.SpatialReference]:
        '''
        Get the reference coordinate reference system. It is either going to be
        in the reference raster pane or the manualy entry widget.
        '''
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
        if save_path is None:
            self.set_message_text("Must enter a save path for geo referencing to occur!")
            return
        
        if self._target_rasterpane.get_rasterview().get_raster_data() is None:
            self.set_message_text("Must select a targetr dataset for geo referencing to occur!")
            return

        if not self._enough_points_for_transform():
            self._set_all_residuals_NA()
            return

        gdal.UseExceptions()

        gcps: List[GeoRefTableEntry, gdal.GCP] = self._get_entry_gcp_list()

        output_srs = osr.SpatialReference()
        output_srs = self._import_current_output_srs()
        output_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)

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
            "resampleAlg": self._curr_resample_alg,
            "dstSRS": output_srs
        }

        self._transform_options = [f'DST_SRS={output_srs.ExportToWkt()}']

        if self._curr_transform_type == TRANSFORM_TYPES.TPS:
            self._warp_kwargs["tps"] = True
            self._transform_options += ['METHOD=GCP_TPS', 'MAX_GCP_ORDER=-1']
        elif self._curr_transform_type == TRANSFORM_TYPES.POLY_1:
            self._warp_kwargs["polynomialOrder"] = 1
            self._transform_options += ['METHOD=GCP_POLYNOMIAL', 'MAX_GCP_ORDER=1']
        elif self._curr_transform_type == TRANSFORM_TYPES.POLY_2:
            self._warp_kwargs["polynomialOrder"] = 2
            self._transform_options += ['METHOD=GCP_POLYNOMIAL', 'MAX_GCP_ORDER=2']
        elif self._curr_transform_type == TRANSFORM_TYPES.POLY_3:
            self._warp_kwargs["polynomialOrder"] = 3
            self._transform_options += ['METHOD=GCP_POLYNOMIAL', 'MAX_GCP_ORDER=3']
        else:
            raise RuntimeError(f"Unknown self._curr_transform_type: {self._curr_transform_type}")
        self._warp_kwargs['transformerOptions'] = self._transform_options

        try:
            tr_pixel_to_output_srs: gdal.Transformer = gdal.Transformer(temp_gdal_ds, None, self._transform_options)
        except BaseException as e:
            msg = str(e)
            if len(msg) > 200:
                msg = msg[:197] + "..."
            QMessageBox.critical(
                self,
                self.tr("Error!"),
                self.tr(f"Error:\n{msg}"),
                QMessageBox.Ok
            )
            return

        try:
            # Sneak peek into the transformed dataset's geo transform so we can use them later
            warp_options = gdal.WarpOptions(**self._warp_kwargs)
            warp_save_path = f'/vsimem/temp_band_{0}'
            place_holder_arr = np.zeros((1, 1), np.uint8)
            temp_gdal_ds: gdal.Dataset = gdal_array.OpenNumPyArray(place_holder_arr, True)
            temp_gdal_ds.SetGCPs([pair[1] for pair in gcps], ref_projection)
            transformed_ds: gdal.Dataset = gdal.Warp(warp_save_path, temp_gdal_ds, options=warp_options)
            transformed_gt = transformed_ds.GetGeoTransform()

            tr_output_srs_to_ref_srs = osr.CoordinateTransformation(output_srs, ref_srs)

            residuals = []
            for entry, gcp in gcps:
                # These coordinates could get back to us in either lat/lon, lon/lat, or north/easting, easting/north
                ok, (output_spatial_x, output_spatial_y, z) = tr_pixel_to_output_srs.TransformPoint(False, gcp.GCPPixel, gcp.GCPLine)

                # Since we use OAMS_TRADITIONAL_GIS_ORDER on both reference and output CRS, we don't need
                # to swap the spatial coordinates
                ref_spatial_coord = tr_output_srs_to_ref_srs.TransformPoint(output_spatial_x, output_spatial_y, 0)

                # print(f"output_spatial_x: {output_spatial_x}")
                # print(f"output_spatial_y: {output_spatial_y}")

                ref_spatial_x, ref_spatial_y = ref_spatial_coord[0], ref_spatial_coord[1]

                # print(f"reference_spatial_x: {ref_spatial_x}")
                # print(f"reference_spatial_y: {ref_spatial_y}")

                # print(f"gcp.GCPX: {gcp.GCPX}")
                # print(f"gcp.GCPY: {gcp.GCPY}")

                error_spatial_x = gcp.GCPX - ref_spatial_x
                error_spatial_y = gcp.GCPY - ref_spatial_y

                # print(f"error_spatial_x: {error_spatial_x}")
                # print(f"error_spatial_y: {error_spatial_y}")

                # print(f"transformed_gt[1]: {transformed_gt[1]}")
                # print(f"transformed_gt[5]: {transformed_gt[5]}")

                error_raster_x = error_spatial_x / transformed_gt[1]
                error_raster_y = error_spatial_y / transformed_gt[5]

                # print(f"error_raster_x: {error_raster_x}")
                # print(f"error_raster_y: {error_raster_y}")

                entry.set_residual_x(round(error_raster_x, 6))
                entry.set_residual_y(round(error_raster_y, 6))

                self._update_residuals(entry)
            
                residuals.append((error_raster_x, error_raster_y))
            
            tr_pixel_to_output_srs = None
            tr_output_srs_to_ref_srs = None
            temp_gdal_ds = None
        except BaseException as e:
            QMessageBox.warning(
                self,
                self.tr("Error"),
                self.tr(f"Error:\n{e}"),
                QMessageBox.Ok
            )
            print(f"Error:\n{e}")
        finally:
            tr_pixel_to_output_srs = None
            tr_output_srs_to_ref_srs = None
            temp_gdal_ds = None


    #========================
    # region Accepting
    #========================

    def _create_warped_output(self) -> bool:
        '''
        Returns a bool on whether this function created the warped dataset correctly.
        '''
        try:
            save_path = self._get_save_file_path()
            if save_path is None:
                QMessageBox.information(self, self.tr("No Save Path Selected"), 
                                        self.tr("In order to georeference, a save path " \
                                                "must be selected. There is no save path " \
                                                "selected, so georeferencing will not occur.\n\n" \
                                                "Please select a save path."))
                return False
            
            if not self._enough_points_for_transform() or self._warp_kwargs is None:
                QMessageBox.information(self,
                                        self.tr("Can't Run Georeferencer"),
                                        self.tr("Not enough points to run georeferencer"))
                return False

            if self._target_rasterpane.get_rasterview().get_raster_data() is None:
                QMessageBox.information(self,
                                        self.tr("No Target Dataset Selected"),
                                        self.tr("A target dataset is not selected. Please select a target dataset."))
                return False

            gcps: List[GeoRefTableEntry, gdal.GCP] = self._get_entry_gcp_list()

            target_dataset = self._target_rasterpane.get_rasterview().get_raster_data()
            target_dataset_impl = target_dataset.get_impl()

            ref_srs = self._get_reference_srs()
            ref_projection = ref_srs.ExportToWkt()
            temp_gdal_ds = None
            output_dataset = None

            self.set_message_text(self.tr("Starting warp..."))

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
            # Make sure dataset has no spatial information that could mess with warping
            temp_gdal_ds.SetGCPs([pair[1] for pair in gcps], ref_projection)
            transformed_ds: gdal.Dataset = gdal.Warp(warp_save_path, temp_gdal_ds, options=warp_options)

            width = transformed_ds.RasterXSize
            height = transformed_ds.RasterYSize
            output_size = (width, height)
            output_bytes = width * height * target_dataset.num_bands() * target_dataset.get_elem_type().itemsize

            gdal.Unlink(warp_save_path)
            output_gt = None

            ratio = MAX_RAM_BYTES / output_bytes
            if isinstance(target_dataset_impl, GDALRasterDataImpl):
                # Saving the full gdal dataste
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
                # Make sure dataset has no spatial information that could mess with warping
                temp_gdal_ds.SetGCPs([pair[1] for pair in gcps], ref_projection)
                warp_options = gdal.WarpOptions(**self._warp_kwargs)
                output_dataset = gdal.Warp(save_path, temp_gdal_ds, options=warp_options)
            elif not isinstance(target_dataset_impl, GDALRasterDataImpl) and ratio > 1.0:
                # Saving the full object array
                warp_options = gdal.WarpOptions(**self._warp_kwargs)
                dataset_arr = target_dataset.get_image_data()
                temp_gdal_ds: gdal.Dataset = gdal_array.OpenNumPyArray(dataset_arr, True)
                # Make sure dataset has no spatial information that could mess with warping
                temp_gdal_ds.SetGCPs([pair[1] for pair in gcps], ref_projection)
                set_data_ignore_of_gdal_dataset(temp_gdal_ds, target_dataset)
                output_dataset: gdal.Dataset = gdal.Warp(save_path, temp_gdal_ds, options=warp_options)
                output_dataset.FlushCache()
            else:
                # Saving incrementally using the numpy dataset
                num_bands_per = int(ratio * target_dataset.num_bands())
                for band_index in range(0, target_dataset.num_bands(), num_bands_per):
                    band_list_index = [band for band in range(band_index, band_index+num_bands_per) if band < target_dataset.num_bands()]
                    warp_options = gdal.WarpOptions(**self._warp_kwargs)
                    warp_save_path = f'/vsimem/temp_band_{min(band_list_index)}_to_{max(band_list_index)}'
                    # print(f"saving chunk: {min(band_list_index)}_to_{max(band_list_index)}")
            
                    band_arr = target_dataset.get_multiple_band_data(band_list_index)
                    temp_gdal_ds: gdal.Dataset = gdal_array.OpenNumPyArray(band_arr, True)
                    # Make sure dataset has no spatial information that could mess with warping
                    temp_gdal_ds.SetGCPs([pair[1] for pair in gcps], ref_projection)
                    set_data_ignore_of_gdal_dataset(temp_gdal_ds, target_dataset)
                    transformed_ds: gdal.Dataset = gdal.Warp(warp_save_path, temp_gdal_ds, options=warp_options)

                    width = transformed_ds.RasterXSize
                    height = transformed_ds.RasterYSize
                    assert width == output_size[0] and height == output_size[1], \
                            "Width and/or height of warped band does not equal a previous warped band"

                    if output_dataset is None:
                        output_dataset = driver.Create(save_path, width, height, target_dataset.num_bands(), gdal_dtype)
                        output_gt = transformed_ds.GetGeoTransform()
                    
                    write_raster_to_dataset(output_dataset, band_list_index, transformed_ds.ReadAsArray(), gdal_dtype)

                    # print(f"Warping bands: {min(band_list_index)} to {max(band_list_index)} out of {target_dataset.num_bands()}")

                    gdal.Unlink(warp_save_path)
                    transformed_ds = None
                output_dataset.SetGeoTransform(output_gt)
                output_dataset.SetSpatialRef(ref_srs)

            if output_dataset is None:
                raise RuntimeError("gdal.Warp failed to produce a transformed dataset.")

            copy_metadata_to_gdal_dataset(output_dataset, target_dataset)
            gt = output_dataset.GetGeoTransform()
            if gt is None:
                raise RuntimeError("Failed to retrieve geotransform from the transformed dataset.")

            output_dataset.FlushCache()
            output_dataset = None

            self.set_message_text(self.tr("Done warping!"))
        except BaseException as e:
            QMessageBox.critical(
                self,
                self.tr("Error While Creating Output"),
                self.tr(f"Error:\n{e}")
            )
            return False
        return True

    def accept(self):
        should_continue = self._create_warped_output()

        if should_continue:
            super().accept()

    # region Event overrides

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Return, Qt.Key_Enter, Qt.Key_Escape):
            event.accept()  # Do nothing on Enter or Escape
        else:
            super().keyPressEvent(event)
