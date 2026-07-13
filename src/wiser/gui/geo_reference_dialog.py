import os
from typing import List, Optional, Dict, Tuple

from PySide6.QtCore import *
from PySide6.QtGui import *
from PySide6.QtWidgets import *

from .generated.geo_referencer_dialog_ui import Ui_GeoReferencerDialog

from wiser.gui.app_state import ApplicationState
from wiser.gui.geo_reference_pane import GeoReferencerPane
from wiser.gui.geo_reference_task_delegate import (
    GeoReferencerTaskDelegate,
    GroundControlPointPair,
    GroundControlPoint,
    GroundControlPointCoordinate,
    PointSelectorType,
    PointSelector,
    GroundControlPointRasterPane,
)
from wiser.gui.util import (
    get_random_matplotlib_color,
    get_color_icon,
    make_into_help_button,
)

from wiser.raster.dataset import RasterDataSet
from wiser.raster.crs_model import (
    AVAILABLE_AUTHORITIES,
    GeneralCRS,
    AuthorityCodeCRS,
    UserGeneratedCRS,
    WktGeneratedCRS,
    COMMON_SRS,
)
from wiser.raster import gcp_io
from wiser.raster.georef_warp import (
    RESAMPLE_ALGORITHMS,
    TRANSFORM_TYPES,
    min_points_per_transform,
    build_warp_kwargs,
    compute_residuals,
    warp_dataset_to_path,
)

from wiser.gui.progress_task import run_with_progress
from wiser.gui.geo_reference_config import GeoReferencerConfig
from wiser.utils.primitives import PriorityClass

from enum import IntEnum

from osgeo import gdal, osr

from pathlib import Path

from pyproj import CRS


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


# RESAMPLE_ALGORITHMS, TRANSFORM_TYPES, and min_points_per_transform now live in
# wiser.raster.georef_warp alongside the Qt-free warp engine; they are imported at the top
# of this module and re-exported here for backwards compatibility.


# The CRS model (GeneralCRS and subclasses, COMMON_SRS, AVAILABLE_AUTHORITIES) now lives
# in wiser.raster.crs_model so it can be shared Qt-free with the mosaic CRS chooser. It is
# imported at the top of this module and re-exported here for backwards compatibility with
# existing callers that import these names from geo_reference_dialog.


class GeoRefTableEntry:
    """
    This class contains all the information needed to populate a row in the
    geo reference table.
    """

    def __init__(
        self,
        gcp_pair: GroundControlPointPair,
        enabled: bool,
        id: int,
        residual_x: float,
        residual_y: float,
        color: str,
    ):
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

    def replace_entry(self, newEntry: "GeoRefTableEntry"):
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
    """
    A simple class for validating float inputs to QLineEdits
    """

    def __init__(self, parent=None, minimum=0.0):
        super().__init__(parent)
        self._minimum = minimum

    def createEditor(self, parent, option, index):
        editor = QLineEdit(parent)
        validator = QDoubleValidator(self._minimum, 1e10, 15, editor)
        validator.setNotation(QDoubleValidator.StandardNotation)
        editor.setValidator(validator)
        return editor


class GeoReferencerDialog(QDialog):
    gcp_pair_added = Signal(GroundControlPointPair)

    gcp_add_attempt = Signal(GroundControlPoint)

    # Emitted with the written output path once a "Run Warp" finishes on its worker thread.
    warp_completed = Signal(str)

    # Internal: carries a completed residual computation (payload includes an in-flight
    # token) back to the GUI thread from the scheduler worker. Connected queued so the
    # table update always runs on the GUI thread.
    _residuals_ready = Signal(object)

    # Debounce window (ms) for coalescing a burst of GCP edits into one residual recompute.
    _RESIDUAL_DEBOUNCE_MS = 150

    def __init__(self, app_state: ApplicationState, app_services, parent=None):
        super().__init__(parent=parent)
        self._app_state = app_state
        self._app_services = app_services

        # Set up the UI state
        self._ui = Ui_GeoReferencerDialog()
        self._ui.setupUi(self)

        self._target_cbox = self._ui.cbox_target_dataset_chooser
        self._reference_cbox = self._ui.cbox_reference_dataset_chooser

        # Create the GeoReferencePanes that the user will see
        self._target_rasterpane = GeoReferencerPane(
            app_state=app_state, pane_type=PointSelectorType.TARGET_POINT_SELECTOR
        )
        self._reference_rasterpane = GeoReferencerPane(
            app_state=app_state, pane_type=PointSelectorType.REFERENCE_POINT_SELECTOR
        )
        self._georeferencer_task_delegate = GeoReferencerTaskDelegate(
            self._target_rasterpane, self._reference_rasterpane, self, app_state
        )
        # Hook up the task delegate to each pane so it can handle the logic
        # for the user adding gcps
        self._target_rasterpane.set_task_delegate(self._georeferencer_task_delegate)
        self._reference_rasterpane.set_task_delegate(self._georeferencer_task_delegate)

        self.gcp_pair_added.connect(self._on_gcp_pair_added)

        self._table_entry_list: List[GeoRefTableEntry] = []

        self._curr_output_srs: GeneralCRS = None
        self._curr_resample_alg = None
        self._curr_transform_type: TRANSFORM_TYPES = None

        self._default_color_button: QPushButton = None

        self._manual_entry_spacer = None
        self._manual_entry_shown = False

        # Chooser lock flags (set from a GeoReferencerConfig). When locked, the matching
        # chooser is caller-owned and the interactive prompts/validation are suppressed.
        self._target_locked = False
        self._reference_locked = False
        self._save_path_locked = False

        # Remembers the accept button's original label so a config-supplied
        # accept_button_text can be reverted on the next (config-less) open.
        self._default_accept_button_text: Optional[str] = None

        self._warp_kwargs: Dict = None
        self._transform_options: List[str] = None
        self._suppress_cell_changed: bool = False

        # Off-thread + debounced residual recompute. A burst of GCP edits coalesces into a
        # single background compute_residuals() once edits settle for _RESIDUAL_DEBOUNCE_MS.
        # _residual_signature is the token of the in-flight compute; a superseded result is
        # dropped so a stale computation can never clobber newer residuals. Created before
        # _first_init() because chooser wiring can trigger a residual recompute immediately.
        self._residual_signature: int = 0
        self._residual_debounce_timer = QTimer(self)
        self._residual_debounce_timer.setSingleShot(True)
        self._residual_debounce_timer.setInterval(self._RESIDUAL_DEBOUNCE_MS)
        self._residual_debounce_timer.timeout.connect(self._recompute_residuals_async)
        self._residuals_ready.connect(self._apply_residuals)

        self._first_init()

        self._prev_chosen_ref_crs_index: int = 0

        # These are actually always the current index, we call them previous
        # because when the current index is change on click, we need access
        # to the index before the click occured
        self._prev_ref_dataset_index: int = None
        self._prev_target_dataset_index: int = None

    def exec_(self, config: Optional[GeoReferencerConfig] = None):
        self._apply_config(config)
        super().exec_()

    def show(self, config: Optional[GeoReferencerConfig] = None):
        self._apply_config(config)
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
        self._show_manual_ref_chooser_display(True)
        self._init_help_button()
        self._init_gcp_io_buttons()

    def _apply_config(self, config: Optional[GeoReferencerConfig] = None):
        """
        Reset the dialog to its baseline, then apply a :class:`GeoReferencerConfig`.

        ``config=None`` reproduces the classic Tools-menu behavior exactly: repopulate the
        choosers and clear any locks / custom accept-button text left over from a prior
        (config-driven) open of this reused dialog.
        """
        # Baseline: repopulate choosers and clear prior locks / accept-button label.
        self._update_dataset_choosers()
        self._update_ref_crs_cbox_items()
        self._update_output_srs_cbox_items()
        self._set_target_locked(False)
        self._set_reference_locked(False)
        self._set_save_path_locked(False)
        self._restore_default_accept_button_text()

        if config is None:
            return

        # Presets are applied before locks so the setters (and the manual-ref chooser) can
        # still act while the reference is momentarily unlocked.
        if config.target_dataset is not None:
            self.set_target_dataset(config.target_dataset)
        if config.reference_dataset is not None:
            self.set_reference_dataset(config.reference_dataset)
        elif config.reference_crs is not None:
            self.set_reference_crs(config.reference_crs)
        if config.save_path is not None:
            self.set_save_path(config.save_path)

        if config.accept_button_text is not None:
            self._set_accept_button_text(config.accept_button_text)

        self._set_target_locked(not config.allow_change_target)
        self._set_reference_locked(not config.allow_change_reference)
        self._set_save_path_locked(not config.allow_change_save_path)

        # Now that presets are in place, refresh the residual columns.
        self._schedule_residual_recompute()

    def _set_accept_button_text(self, text: str):
        """Relabel the accept (OK) button, remembering the original text for restore."""
        btn = self._ui.buttonBox.button(QDialogButtonBox.Ok)
        if btn is None:
            return
        if self._default_accept_button_text is None:
            self._default_accept_button_text = btn.text()
        btn.setText(text)

    def _restore_default_accept_button_text(self):
        """Revert the accept button to its original label (if it was changed)."""
        if self._default_accept_button_text is None:
            return
        btn = self._ui.buttonBox.button(QDialogButtonBox.Ok)
        if btn is not None:
            btn.setText(self._default_accept_button_text)

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
        lon_east_ledit = self._ui.ledit_lon_east

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

        # Initialize the find button
        find_crs_btn = self._ui.btn_find_crs
        find_crs_btn.clicked.connect(self._on_find_crs)

        # Initialize the help button
        make_into_help_button(
            self._ui.tbtn_help,
            "https://ehlmann-research-group.github.io/WISER-UserManual/Georeferencer/#reference-system-information",
            "Learn more about reference systems",
        )

    def _update_ref_crs_cbox_items(self):
        srs_to_choose_cbox = self._ui.cbox_choose_crs
        srs_to_choose_cbox.clear()
        for name, srs in COMMON_SRS.items():
            srs_to_choose_cbox.addItem(name, srs)

        for name, (srs, _) in self._app_state.get_user_created_crs().items():
            srs_to_choose_cbox.addItem(name, UserGeneratedCRS(name, srs))

    def _show_manual_ref_chooser_display(self, show_manual_chooser: bool):
        """
        Shows the manual reference chooser UI if there is no dataset passed in. Shows
        the reference dataset if there is a dataset passed in.
        """
        # When the reference is locked (caller-owned), suppress toggling the manual chooser
        # so the reference display cannot be changed out from under the caller.
        if self._reference_locked:
            return
        if self._manual_entry_spacer is None:
            self._manual_entry_spacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)
        self._manual_entry_shown = show_manual_chooser
        if show_manual_chooser:
            # We want to make sure the manual chooser is being shown
            self._ui.widget_manual_entry.show()
            self._ui.widget_ref_image.hide()
            self._add_manual_spacer_once()
        else:
            # We want to make sure the manual reference chooser is not being shown
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
        if (
            self._reference_rasterpane is not None
            and self._reference_rasterpane.get_rasterview().get_raster_data() is not None
        ):
            try:
                ref_ds = self._reference_rasterpane.get_rasterview().get_raster_data()
                reference_srs_name = "Input Ref CRS: " + ref_ds.get_spatial_ref().GetName()
                reference_srs_code = ref_ds.get_spatial_ref().GetAuthorityCode(None)
                if reference_srs_code is None:
                    self.set_message_text("Could not get an authority code for default dataset")
                    ref_srs = ref_ds.get_spatial_ref()
                    crs = CRS.from_wkt(ref_srs.ExportToWkt())
                    if crs is not None:
                        auth_info = crs.to_authority()
                        if auth_info is None:
                            name = crs.name if crs.name is not None else "Uknown Name"
                            wkt_crs = WktGeneratedCRS(name, crs.to_wkt())
                            srs_cbox.addItem(name, wkt_crs)
                        else:
                            auth_name, auth_code = crs.to_authority()
                            srs_cbox.addItem(
                                reference_srs_name,
                                AuthorityCodeCRS(auth_name, int(auth_code)),
                            )
                else:
                    srs_cbox.addItem(
                        reference_srs_name,
                        AuthorityCodeCRS(
                            ref_ds.get_spatial_ref().GetAuthorityName(None),
                            int(reference_srs_code),
                        ),
                    )
            except BaseException:
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
        """
        Initializes the columns of the table the GCPs will go into. Asigns number validators
        to each column that the user can change numbers in.
        """
        table_widget = self._ui.table_gcps
        table_widget.setColumnCount(len(COLUMN_ID))
        headers = [
            "Enabled",
            "ID",
            "Target X",
            "Target Y",
            "Ref X",
            "Ref Y",
            "dX (Pix)",
            "dY (Pix)",
            "Color",
            "Remove",
        ]
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
        """
        Performs actions that should only be done once with dataset chooser.
        """
        self._target_cbox.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self._target_cbox.activated.connect(self._on_switch_target_dataset)

        self._reference_cbox.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self._reference_cbox.activated.connect(self._on_switch_reference_dataset)

    def _update_dataset_choosers(self):
        """
        Performs actions that should be done everytime geo ref dialog is reshown
        """
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

    # ========================
    # region Slots
    # ========================

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
                f"Could not find spatial reference for {authority_str}:{authority_code}",
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
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            # wipe out both the internal list and the table widget
            self._reset_gcps()
            # refresh the display panes
            self._update_panes()

    def _on_save_gcps_clicked(self, checked: bool):
        """
        Save GCPs either as a QGIS or ENVI gcp file format
        """
        if not self._table_entry_list:
            QMessageBox.information(self, "No GCPs", "There are no ground-control points to save.")
            return

        filename, _ = QFileDialog.getSaveFileName(
            self,
            self.tr("Save ground-control points"),
            filter=self.tr("QGIS points (*.points);;ENVI ASCII (*.pts)"),
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

        rows = self._get_gcp_rows()
        ext = Path(filename).suffix.lower()
        try:
            if ext == ".points":
                gcp_io.write_qgis_points(filename, rows, auth_name, auth_code, wkt_str)
            elif ext == ".pts":
                gcp_io.write_envi_pts(filename, rows, auth_name, auth_code, wkt_str)
            else:
                QMessageBox.warning(self, "Extension error", "Please use either *.points or *.pts")
                return
            self.set_message_text(f"GCPs saved to {filename}")
        except Exception as e:
            QMessageBox.critical(self, "Save failed", str(e))

    def _get_gcp_rows(self) -> List[Tuple[float, float, float, float, bool]]:
        """
        Flatten the current table entries into ``(map_x, map_y, pixel_x, pixel_y,
        enabled)`` rows for the Qt-free GCP writers in :mod:`wiser.raster.gcp_io`.
        """
        rows = []
        for entry in self._table_entry_list:
            pair = entry.get_gcp_pair()
            map_x, map_y = pair.get_reference_gcp_spatial_coord()
            pix_x, pix_y = pair.get_target_gcp().get_point()
            rows.append((map_x, map_y, pix_x, pix_y, entry.is_enabled()))
        return rows

    def _on_load_gcps_clicked(self, checked: bool):
        filename, _ = QFileDialog.getOpenFileName(
            self,
            self.tr("Load ground-control points"),
            filter=self.tr("GCP files (*.points *.pts)"),
        )
        if not filename:
            return

        try:
            points, gcp_srs = gcp_io.read_gcp_file(filename)
            if points is None or gcp_srs is None:
                raise RuntimeError(
                    "Passed-in reference system can't be parsed. Reference system WKT:\n"
                    f"{gcp_srs.ExportToPrettyWkt()}"
                )
        except Exception as e:
            QMessageBox.critical(self, "Load failed", str(e))
            return
        self.load_gcps_and_srs(gcp_points=points, gcp_srs=gcp_srs)

    def _on_show_help(self):
        QMessageBox.information(
            self,
            self.tr("How to use the Georeferencer"),
            self.tr(
                """
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
            """
            ),
            QMessageBox.Ok,
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
        gcp = GroundControlPointCoordinate(
            (lon_east_value, lat_north_value),
            PointSelectorType.REFERENCE_POINT_SELECTOR,
            srs=chosen_srs,
        )
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
                f"Could not find spatial reference for {authority_str}:{authority_code}",
            )
            return

        # Get the human-readable name of the SRS
        srs_name = srs.GetName()

        self._add_srs_to_ref_choose_cbox(srs_name, AuthorityCodeCRS(authority_str, float(authority_code)))

    def _on_cell_changed(self, row: int, col: int):
        """
        Correctly syncs self._table_entry_list with changes to the GUI table
        then calls the georeference function with this updated _table_entry_list
        """
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
            self._schedule_residual_recompute()
        elif col == COLUMN_ID.TARGET_Y_COL:
            item = table_widget.item(row, col)
            new_val = item.text()
            new_target_y = float(new_val)
            list_entry = self._table_entry_list[row]
            target_gcp = list_entry.get_gcp_pair().get_target_gcp()
            curr_point = target_gcp.get_point()
            target_gcp.set_point([curr_point[0], new_target_y])
            self._schedule_residual_recompute()
        elif col == COLUMN_ID.REF_X_COL:
            item = table_widget.item(row, col)
            new_val = item.text()
            new_ref_spatial_x = float(new_val)
            list_entry = self._table_entry_list[row]
            gcp_pair = list_entry.get_gcp_pair()
            ref_gcp = gcp_pair.get_reference_gcp()
            ref_gcp.set_spatial_point((new_ref_spatial_x, gcp_pair.get_reference_gcp_spatial_coord()[1]))
            self._schedule_residual_recompute()
        elif col == COLUMN_ID.REF_Y_COL:
            item = table_widget.item(row, col)
            new_val = item.text()
            new_ref_spatial_y = float(new_val)
            list_entry = self._table_entry_list[row]
            gcp_pair = list_entry.get_gcp_pair()
            ref_gcp = gcp_pair.get_reference_gcp()
            ref_gcp.set_spatial_point((gcp_pair.get_reference_gcp_spatial_coord()[0], new_ref_spatial_y))
            self._schedule_residual_recompute()
        else:
            return
        self._update_panes()

    def _on_choose_save_filename(self, checked=False):
        """
        A handler for when the file-chooser for the "save-filename" is shown.
        """
        # A locked save path is caller-owned: do not let the user re-choose it, and skip the
        # save-path-vs-dataset-path validation below (which assumes a user-chosen path).
        if self._save_path_locked:
            return
        file_dialog = QFileDialog(parent=self, caption=self.tr("Save raster dataset"))

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
                QMessageBox.information(
                    self,
                    self.tr("Wrong Save Path"),
                    self.tr(
                        "The save path you chose matches either the target\n"
                        + "or reference dataset's save path. Please change.\n\n"
                        f"Chosen save path:\n{filename}"
                    ),
                )
                return
            self._ui.ledit_save_path.setText(filename)
            self._schedule_residual_recompute()

    def _on_switch_output_srs(self, index: int):
        # We don't record the output srs because we get this
        # directly from the combo box.
        self._schedule_residual_recompute()

    def _on_switch_chosen_ref_srs(self, index: int):
        if self._prev_chosen_ref_crs_index != index:
            if len(self._table_entry_list) > 0:
                confirm = QMessageBox.question(
                    self,
                    self.tr("Change reference CRS?"),
                    self.tr("You are changing the reference CRS.")
                    + "\n\nDo you want to discard all selected GCPs?",
                )
                if confirm == QMessageBox.Yes:
                    self._reset_gcps()
        self._prev_chosen_ref_crs_index = self._ui.cbox_choose_crs.currentIndex()

    def _on_switch_resample_alg(self, index: int):
        resample_alg = self._ui.cbox_interpolation.itemData(index)
        self._curr_resample_alg = resample_alg

    def _on_switch_transform_type(self, index: int):
        transform_type = self._ui.cbox_poly_order.itemData(index)
        self._curr_transform_type = transform_type
        self._schedule_residual_recompute()

    def _on_choose_default_color(self):
        """
        Changes the default color of the already added points.
        """
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
        """
        Chooses color for a specific row in the table.

        Parameters
        ----------
        - table_entry: GeoRefTableEntry
            Our internal representation of the row whose color we
            want to change
        """
        row = table_entry.get_id()
        initial_color = QColor(self._table_entry_list[row].get_color())
        color = QColorDialog.getColor(parent=self, initial=initial_color)
        if color.isValid():
            color_str = color.name()
            self._table_entry_list[row].set_color(color_str)
            self._set_color_icon(row, color_str)
            self._update_panes()

    def _on_enabled_clicked(self, table_entry: GeoRefTableEntry, checked: bool):
        """
        This enables or disables a row in the table, making the GCPs in that row
        not used for calculation
        """
        # Since the table_entry's ID can change, don't just pass in the row_to_add
        row_to_add = table_entry.get_id()
        self._set_row_enabled_state(row_to_add, checked)
        self._update_panes()
        self._schedule_residual_recompute()

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
        self._schedule_residual_recompute()

    def _on_removal_button_clicked(self, table_entry: GeoRefTableEntry):
        """
        Removes an row from the table
        """
        self._remove_table_entry(table_entry)
        self._schedule_residual_recompute()

    def _on_switch_target_dataset(self, index: int):
        """
        User-initiated target chooser slot: run the guard + confirm-discard prompts, then
        funnel through :meth:`set_target_dataset` (the shared, prompt-free apply path).
        """
        ds_id = self._target_cbox.itemData(index)
        try:
            dataset = self._app_state.get_dataset(ds_id)
        except Exception as e:
            self.set_message_text(f"Could not load target dataset: {e}")
            self._target_cbox.setCurrentIndex(self._prev_target_dataset_index)
            return

        # The target's file must not equal the save path.
        current_save_path = self._get_current_save_path()
        if dataset is not None and current_save_path in dataset.get_filepaths():
            QMessageBox.information(
                self,
                self.tr("Target Dataset Path Equals Save Path"),
                self.tr(
                    "The target dataset path equals the save path.\n"
                    "Change the save path before selecting this target dataset."
                ),
            )
            self._target_cbox.setCurrentIndex(self._prev_target_dataset_index)
            return

        # Changing the target discards existing GCPs; confirm first.
        if len(self._table_entry_list) > 0 and self._prev_target_dataset_index != index:
            confirm = QMessageBox.question(
                self,
                self.tr("Change Target Dataset?"),
                self.tr("Are you sure you want to change the target dataset?")
                + "\n\nThis will discard all selected GCPs. Do you want\n"
                "to continue?",
            )
            if confirm == QMessageBox.Yes:
                self._reset_gcps()
            else:
                self._target_cbox.setCurrentIndex(self._prev_target_dataset_index)
                return

        self.set_target_dataset(dataset)

    def set_target_dataset(self, dataset: Optional[RasterDataSet]):
        """
        Show ``dataset`` in the target pane without any user prompts (programmatic path,
        shared by the chooser slot). Syncs the chooser to match.
        """
        self._select_dataset_in_combo(self._target_cbox, dataset)
        self._target_rasterpane.show_dataset(dataset)
        self._prev_target_dataset_index = self._target_cbox.currentIndex()

    def _select_dataset_in_combo(self, combo: QComboBox, dataset: Optional[RasterDataSet]):
        """Point ``combo`` at ``dataset`` (or the "(no data)" -1 entry when None)."""
        ds_id = dataset.get_id() if dataset is not None else -1
        idx = combo.findData(ds_id)
        if idx >= 0:
            combo.setCurrentIndex(idx)

    def _on_switch_reference_dataset(self, index: int):
        """
        User-initiated reference chooser slot: run the guard + confirm-discard prompts, then
        funnel through :meth:`set_reference_dataset` (the shared, prompt-free apply path).
        """
        ds_id = self._reference_cbox.itemData(index)
        try:
            dataset = self._app_state.get_dataset(ds_id)
        except Exception as e:
            self.set_message_text(f"Could not load reference dataset: {e}")
            self._reference_cbox.setCurrentIndex(self._prev_ref_dataset_index)
            return

        # The reference's file must not equal the save path.
        current_save_path = self._get_current_save_path()
        if dataset is not None and current_save_path in dataset.get_filepaths():
            QMessageBox.information(
                self,
                self.tr("Reference Dataset Path Equals Save Path"),
                self.tr(
                    "The reference dataset path equals the save path.\n"
                    "Change the save path before selecting this reference dataset."
                ),
            )
            self._reference_cbox.setCurrentIndex(self._prev_ref_dataset_index)
            return

        # Changing the reference discards existing GCPs; confirm first.
        if len(self._table_entry_list) > 0 and self._prev_ref_dataset_index != index:
            confirm = QMessageBox.question(
                self,
                self.tr("Change Reference Dataset?"),
                self.tr("Are you sure you want to change the reference dataset?")
                + "\n\nThis will discard all selected GCPs. Do you want\n"
                "to continue?",
            )
            if confirm == QMessageBox.Yes:
                self._reset_gcps()
            else:
                self._reference_cbox.setCurrentIndex(self._prev_ref_dataset_index)
                return

        # A real reference dataset must carry a spatial reference; guard against None so we
        # do not call has_geographic_info() on the "(no data)" selection.
        if dataset is not None and not dataset.has_geographic_info():
            QMessageBox.warning(
                self,
                self.tr("Unreferenced Dataset"),
                self.tr("You must choose a dataset with a spatial reference system"),
            )
            self._reference_cbox.setCurrentIndex(self._prev_ref_dataset_index)
            return

        self.set_reference_dataset(dataset)

    def set_reference_dataset(self, dataset: Optional[RasterDataSet]):
        """
        Show ``dataset`` in the reference pane without any user prompts (programmatic path,
        shared by the chooser slot). Syncs the chooser to match.
        """
        self._select_dataset_in_combo(self._reference_cbox, dataset)
        self._reference_rasterpane.show_dataset(dataset)
        self._update_output_srs_cbox_items()
        self._show_manual_ref_chooser_display(False)
        self._prev_ref_dataset_index = self._reference_cbox.currentIndex()

    def set_reference_crs(self, crs: Optional[GeneralCRS]):
        """
        Use a manual reference CRS (no reference dataset): show the manual chooser and
        select ``crs`` in it. Used to preset the reference when a config supplies a CRS
        instead of a reference dataset.
        """
        if crs is None:
            return
        self._select_dataset_in_combo(self._reference_cbox, None)
        self._show_manual_ref_chooser_display(True)
        srs = crs.get_osr_crs()
        name = srs.GetName() if srs is not None else "Reference CRS"
        self._add_srs_to_ref_choose_cbox(name, crs)

    def set_save_path(self, path: Optional[str]):
        """Preset the output save path (programmatic path)."""
        if path is None:
            return
        self._ui.ledit_save_path.setText(path)

    def _set_target_locked(self, locked: bool):
        """Lock the target chooser (caller owns the target dataset)."""
        self._target_locked = locked
        self._target_cbox.setEnabled(not locked)

    def _set_reference_locked(self, locked: bool):
        """Lock the reference chooser (caller owns the reference dataset/CRS)."""
        self._reference_locked = locked
        self._reference_cbox.setEnabled(not locked)

    def _set_save_path_locked(self, locked: bool):
        """Lock the save path (caller owns the output path)."""
        self._save_path_locked = locked
        self._ui.ledit_save_path.setReadOnly(locked)
        self._ui.btn_save_path.setEnabled(not locked)

    # region Helpers

    def compare_srs_lenient(
        self,
        srs1: osr.SpatialReference,
        srs2: osr.SpatialReference,
    ):
        """
        Compares srs1 and srs2, but first puts them to WKT, then reimports them as
        an osr.SpatialReference. We do this because sometimes srs's are the same
        but because of how they were imported, some less-important meta data may
        have been lost. We get rid of this meta data for both srs's by using this
        function.
        """
        wkt_1 = srs1.ExportToWkt()
        wkt_2 = srs2.ExportToWkt()

        srs1_clone = osr.SpatialReference()
        srs1_clone.ImportFromWkt(wkt_1)

        srs2_clone = osr.SpatialReference()
        srs2_clone.ImportFromWkt(wkt_2)

        return srs1_clone.IsSame(srs2_clone)

    def load_gcps_and_srs(
        self,
        gcp_points: List[Tuple[float, float, float, float]],
        gcp_srs: GeneralCRS,
    ):
        """
        Adds a list of GCPs that are each represented by (map_x, map_y, pix_x, pix_y)
        into the table with the specified spatial reference system. This ensures
        that the GCPs are correctly drawn onto the target dataset and reference
        dataset.

        Parameters
        ----------
        - gcp_points
            A list of the GCP pairs. The first two floats are the spatial
            coordinates (X, Y) of the point. The second two points are the
            raster coordinates of the point (X, Y) in the target datasets frame.

        - gcp_srs
            The spatial reference system that we give the GCPs
        """
        # Without the target dataset, we can't do anything. But we can still
        # add GCP points if we don't have the reference dataset
        target_ds = self._get_target_dataset()
        if target_ds is None:
            return
        ref_ds = self._get_ref_dataset()

        skipped_gcps = []
        if ref_ds is not None and self.compare_srs_lenient(gcp_srs.get_osr_crs(), ref_ds.get_spatial_ref()):
            for map_x, map_y, pix_x, pix_y in gcp_points:
                # Verify that the GCP is inside of the target dataset
                if not (0 <= pix_x < target_ds.get_width() and 0 <= pix_y < target_ds.get_height()):
                    skipped_gcps.append(
                        (
                            (map_x, map_y, pix_x, pix_y),
                            "Target GCP Pixel is outside of target dataset's raster bounds.",
                        )
                    )
                    continue
                # Transform the spatial coordinats to pixel coordinates in the reference dataset's
                # frame
                ref_px = ref_ds.geo_to_pixel_coords_exact((map_x, map_y))
                # Ensure the pixel is inside of the reference dataset
                if ref_px is None or not (
                    0 <= ref_px[0] < ref_ds.get_width() and 0 <= ref_px[1] < ref_ds.get_height()
                ):
                    skipped_gcps.append(
                        (
                            (map_x, map_y, pix_x, pix_y),
                            "Reference GCP coordinate is outside of reference dataset's raster bounds.",
                        )
                    )
                    continue

                tgt_gcp = GroundControlPointRasterPane((pix_x, pix_y), self._target_rasterpane)
                ref_gcp = GroundControlPointRasterPane((ref_px[0], ref_px[1]), self._reference_rasterpane)
                pair = GroundControlPointPair(tgt_gcp, ref_gcp)
                self.gcp_pair_added.emit(pair)
        else:
            # Mismatch or no reference dataset – fall back to manual entry mode

            # Set the reference dataset chosen to None and show the manual reference
            # chooser UI elements
            self._reference_cbox.setCurrentIndex(self._reference_cbox.findData(-1))
            self._show_manual_ref_chooser_display(True)
            # Populate the spatial reference system in the cbox_choose_crs
            self._add_srs_to_ref_choose_cbox(gcp_srs.get_osr_crs().GetName(), gcp_srs)
            self.set_message_text(
                "Reference CRS changed to match GCP file; select each "
                "target point then press Enter to pair it."
            )
            for map_x, map_y, pix_x, pix_y in gcp_points:
                # Verify pixel-within-images
                if not (0 <= pix_x < target_ds.get_width() and 0 <= pix_y < target_ds.get_height()):
                    skipped_gcps.append(
                        (
                            (map_x, map_y, pix_x, pix_y),
                            "Target GCP Pixel is outside of raster bounds.",
                        )
                    )
                    continue
                tgt_gcp = GroundControlPointRasterPane((pix_x, pix_y), self._target_rasterpane)
                ref_gcp = GroundControlPointCoordinate(
                    (map_x, map_y),
                    PointSelectorType.REFERENCE_POINT_SELECTOR,
                    gcp_srs.get_osr_crs(),
                )
                pair = GroundControlPointPair(tgt_gcp, ref_gcp)
                self.gcp_pair_added.emit(pair)

        # ────────────────────────────────────────────────────────────────
        #  Show skipped‐GCPs if any
        # ────────────────────────────────────────────────────────────────
        if skipped_gcps:
            info_lines = []
            info_lines.append("Skipped GCPs")
            info_lines.append("")
            for tpl, reason in skipped_gcps:
                info_lines.append(f"GCP: {tpl}")
                info_lines.append(f"Reason: {reason}")
                info_lines.append("")  # blank line between entries

            QMessageBox.information(self, "Skipped GCPs", "\n".join(info_lines).rstrip())

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
        """
        Clears all of the entries in the table widget and in the list of entries
        """
        self._table_entry_list = []
        self._ui.table_gcps.clearContents()
        self._ui.table_gcps.setRowCount(0)

    def _set_default_color_icon(self, color: str):
        self._initial_default_color = color
        color_icon = get_color_icon(color)
        self._default_color_button.setIcon(color_icon)

    def _set_color_icon(self, row: int, color: str):
        """
        Sets the color icon of the color at the passed in row
        """
        color_icon = get_color_icon(color)
        table_widget = self._ui.table_gcps
        table_item: QPushButton = table_widget.cellWidget(row, COLUMN_ID.COLOR_COL)
        table_item.setIcon(color_icon)

    def _set_row_enabled_state(
        self,
        row: int,
        row_enabled_state: bool,
        exempt_columns: List[COLUMN_ID] = [
            COLUMN_ID.REMOVAL_COL,
            COLUMN_ID.ENABLED_COL,
        ],
    ):
        """
        This is used to disable a given row so it won't be used for georeferencing.
        We visually disable the row by disabling all columns in a given row except
        for the columns in the exempt_columns list. We also set our internal
        representation of that row (in self._table_entry_list) to disabled.
        """
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
        """
        Sets all of the residual values in a row to NA. This is used when
        there aren't enough points to do georeferencing.
        """
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
        """
        Adds table_entry to the table widget at the row specified by
        table_entry.get_id()
        """
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
        checkbox.clicked.connect(lambda checked: self._on_enabled_clicked(table_entry, checked))

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
        color_button.clicked.connect(lambda checked: self._on_choose_color(table_entry))
        initial_color = table_entry.get_color()
        color_icon = get_color_icon(initial_color)
        color_button.setIcon(color_icon)
        table_widget.setCellWidget(row_to_add, COLUMN_ID.COLOR_COL, color_button)

        pushButton = QPushButton("Remove GCP")
        pushButton.clicked.connect(lambda checked: self._on_removal_button_clicked(table_entry))
        table_widget.setCellWidget(row_to_add, COLUMN_ID.REMOVAL_COL, pushButton)

        self._suppress_cell_changed = False
        self._update_panes()

    def _remove_table_entry(self, table_entry: GeoRefTableEntry) -> Optional[int]:
        """
        Removes the table entry and returns the index removed. If the table entry
        is not found in the list, this errors. Table entry equality is done based
        on reference.

        table_entry.get_id() should be the table entries index in both the
        TableWidget and the _table_entry_list. We uses asserts to ensure this
        """
        table_widget = self._ui.table_gcps

        index_removed = None  # Also refers to the removed row
        for i in range(len(self._table_entry_list)):
            table_entry_in_list = self._table_entry_list[i]
            if table_entry_in_list == table_entry:
                index_removed = i
                self._table_entry_list.pop(i)
                assert (
                    index_removed == table_entry.get_id()
                ), "The index that table entry was removed does not match its ID"
                break
        assert index_removed is not None, "The table entry was not found in the list of entries"
        table_widget.removeRow(index_removed)

        # We must update the entry id's after we remove the rows so that
        # the table entries are in their correct rows
        self._update_entry_ids()
        self._update_panes()

    def _update_residuals(self, table_entry: GeoRefTableEntry):
        """
        Visually update what the residual cell in the table widget says to accurate
        match the residual values in table_entry.
        """
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

    def _update_entry_ids(self):
        """
        Used to resync the id values that the table entries in self._table_entry_list
        have with the their position in the list. Also updates the table widget to
        properly show the new id.
        """
        table_widget = self._ui.table_gcps
        for i in range(len(self._table_entry_list)):
            table_entry = self._table_entry_list[i]
            table_entry.set_id(i)
            # Index i also functions as the row in the table widget
            # where this entry is currently
            table_widget.setItem(i, COLUMN_ID.ID_COL, QTableWidgetItem(str(i)))

    # ========================
    # region Getters
    # ========================

    def get_table_entries(self) -> List[GeoRefTableEntry]:
        return self._table_entry_list

    def get_gcp_table_size(self) -> int:
        assert len(self._table_entry_list) == self._ui.table_gcps.rowCount(), (
            f"Entry number mismatch. Table entry list "
            f"{len(self._table_entry_list)} and QTableWidget has "
            f"{self._ui.table_gcps.rowCount()} entries"
        )
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

    # ========================
    # region Dataset Choosers
    # ========================

    def _update_target_dataset_chooser(self):
        self._update_dataset_chooser(self._target_cbox)

    def _update_reference_dataset_chooser(self):
        self._update_dataset_chooser(self._reference_cbox)

    def _update_dataset_chooser(self, dataset_chooser: QComboBox):
        """
        Populates the passed in QComboBox with datasets from app_state
        """
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
            for index, dataset in enumerate(app_state.get_datasets()):
                id = dataset.get_id()
                name = dataset.get_name()

                dataset_chooser.addItem(name, id)
                if dataset.get_id() == current_ds_id:
                    new_index = index

            dataset_chooser.insertSeparator(num_datasets)
            dataset_chooser.addItem(self.tr("(no data)"), -1)
            if current_ds_id == -1:
                new_index = dataset_chooser.count() - 1
        else:
            # No datasets yet
            dataset_chooser.addItem(self.tr("(no data)"), -1)
            if current_ds_id == -1:
                new_index = 0

        if new_index is None:
            if num_datasets > 0:
                new_index = min(current_index, num_datasets - 1)
            else:
                new_index = 0

        dataset_chooser.setCurrentIndex(new_index)

    # ========================
    # region Misc
    # ========================

    def _update_panes(self):
        self._target_rasterpane.update_all_rasterviews()
        self._reference_rasterpane.update_all_rasterviews()

    def set_message_text(self, text: str):
        if len(text) > 100:
            text = text[:100] + "…"
        self._ui.lbl_message.setText(text)

    def _add_srs_to_output_cbox(self, srs_name: str, crs: GeneralCRS):
        """
        Adds the coordinate reference system that the user found to the
        manual reference combo box.
        """
        crs_choose_cbox = self._ui.cbox_srs
        osr_crs = crs.get_osr_crs()
        # Check for existing entry
        for idx in range(crs_choose_cbox.count()):
            data: GeneralCRS = crs_choose_cbox.itemData(idx)
            if data.get_osr_crs().IsSame(osr_crs):
                QMessageBox.information(
                    self,
                    "CRS Already Added",
                    f"The CRS {srs_name}: {crs} is already in the list as “{crs_choose_cbox.itemText(idx)}.”",
                )
                return

        # If not found, add as new entry
        crs_choose_cbox.addItem(srs_name, crs)
        crs_choose_cbox.setCurrentIndex(crs_choose_cbox.count() - 1)
        self._on_switch_output_srs(crs_choose_cbox.count() - 1)

    def _add_srs_to_ref_choose_cbox(self, srs_name: str, crs: GeneralCRS):
        """
        Adds the coordinate reference system that the user found to the choose combo box
        """
        crs_choose_cbox = self._ui.cbox_choose_crs
        osr_crs = crs.get_osr_crs()
        # Check for existing entry
        for idx in range(crs_choose_cbox.count()):
            data: GeneralCRS = crs_choose_cbox.itemData(idx)
            if data.get_osr_crs().IsSame(osr_crs):
                QMessageBox.information(
                    self,
                    "CRS Already Added",
                    f"The CRS {srs_name} is already in the list as “{crs_choose_cbox.itemText(idx)}.”",
                )
                return

        # If not found, add as new entry
        crs_choose_cbox.addItem(srs_name, crs)
        crs_choose_cbox.setCurrentIndex(crs_choose_cbox.count() - 1)

    def _get_manual_ref_chosen_crs(self) -> osr.SpatialReference:
        """
        From the information in the manual reference combo box, create an osr.SpatialReference
        object.
        """
        return self._ui.cbox_choose_crs.currentData().get_osr_crs()

    # ========================
    # region Geo referencing
    # ========================

    def _enough_points_for_transform(self):
        return (
            False
            if self._get_num_active_points() < min_points_per_transform[self._curr_transform_type]
            else True
        )

    def _get_entry_gcp_list(self) -> List[Tuple[GeoRefTableEntry, gdal.GCP]]:
        """
        Goes through all of the rows in the table widget and makes a gdal.GCP object
        from them so we can pass these into GDAL's georeferencer.
        """
        gcps: List[Tuple[GeoRefTableEntry, gdal.GCP]] = []
        for table_entry in self._table_entry_list:
            if not table_entry.is_enabled():
                continue
            spatial_coord = table_entry.get_gcp_pair().get_reference_gcp().get_spatial_point()
            assert (
                spatial_coord is not None
            ), f"spatial_coord is none on reference gcp!, spatial_coord: {spatial_coord}"
            target_pixel_coord = table_entry.get_gcp_pair().get_target_gcp().get_point()
            gcps.append(
                (
                    table_entry,
                    gdal.GCP(
                        spatial_coord[0],
                        spatial_coord[1],
                        0,
                        target_pixel_coord[0],
                        target_pixel_coord[1],
                    ),
                )
            )

        return gcps

    def _import_current_output_srs(self) -> osr.SpatialReference:
        """
        Read self._curr_output_srs (authority_name, authority_code) and
        return a corresponding OSR SpatialReference object.
        """
        crs: GeneralCRS = self._ui.cbox_srs.currentData()
        return crs.get_osr_crs()

    def _get_reference_srs(self) -> Optional[osr.SpatialReference]:
        """
        Get the reference coordinate reference system. It is either going to be
        in the reference raster pane or the manualy entry widget.
        """
        ref_ds = self._reference_rasterpane.get_rasterview().get_raster_data()
        if ref_ds is not None:
            return ref_ds.get_spatial_ref()
        elif self._manual_entry_shown:
            return self._get_manual_ref_chosen_crs()
        else:
            raise RuntimeError("Both the dataset shown is none and the manual entry widget is None")

    def _snapshot_residual_inputs(self) -> Optional[dict]:
        """
        Gather everything :func:`compute_residuals` needs, on the GUI thread.

        Returns a dict of the snapshotted inputs, or ``None`` if residuals cannot be
        computed yet (no save path, no target, or too few points). The GDAL/OSR objects
        captured here are plain data that can be safely handed to a worker thread.
        """
        save_path = self._get_save_file_path()
        if save_path is None:
            self.set_message_text("Must enter a save path for geo referencing to occur!")
            return None

        if self._target_rasterpane.get_rasterview().get_raster_data() is None:
            self.set_message_text("Must select a target dataset for geo referencing to occur!")
            return None

        if not self._enough_points_for_transform():
            self._set_all_residuals_NA()
            return None

        entries_and_gcps = self._get_entry_gcp_list()
        entries = [entry for entry, _ in entries_and_gcps]
        gcps = [gcp for _, gcp in entries_and_gcps]

        output_srs = self._import_current_output_srs()
        output_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)

        ref_srs = self._get_reference_srs()
        ref_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)

        warp_kwargs, transformer_options = build_warp_kwargs(
            self._curr_resample_alg, self._curr_transform_type, output_srs
        )
        # Retain the last-built kwargs/options for any callers that still read them.
        self._warp_kwargs = warp_kwargs
        self._transform_options = transformer_options

        return {
            "entries": entries,
            "gcps": gcps,
            "ref_srs": ref_srs,
            "output_srs": output_srs,
            "warp_kwargs": warp_kwargs,
            "transformer_options": transformer_options,
        }

    def _apply_residuals_to_entries(self, entries, residuals):
        """Write computed residuals back onto the table entries (GUI thread)."""
        for entry, (residual_x, residual_y) in zip(entries, residuals):
            entry.set_residual_x(round(residual_x, 6))
            entry.set_residual_y(round(residual_y, 6))
            self._update_residuals(entry)

    def _georeference(self):
        """
        Synchronously recompute residuals and update the table.

        Kept for tests and as the no-scheduler fallback; interactive edits go through
        :meth:`_schedule_residual_recompute` so the GUI never blocks on a warp.
        """
        snapshot = self._snapshot_residual_inputs()
        if snapshot is None:
            return
        try:
            residuals = compute_residuals(
                snapshot["gcps"],
                snapshot["ref_srs"],
                snapshot["output_srs"],
                snapshot["warp_kwargs"],
                snapshot["transformer_options"],
            )
        except BaseException as e:
            msg = str(e)
            if len(msg) > 200:
                msg = msg[:197] + "..."
            QMessageBox.critical(self, self.tr("Error!"), self.tr(f"Error:\n{msg}"), QMessageBox.Ok)
            return
        self._apply_residuals_to_entries(snapshot["entries"], residuals)

    def _schedule_residual_recompute(self):
        """
        Request a residual recompute after edits settle.

        Coalesces a burst of GCP edits into a single background compute. When no scheduler
        is available (e.g. some unit contexts), falls back to a synchronous recompute.
        """
        scheduler = getattr(self._app_services, "scheduler", None) if self._app_services else None
        if scheduler is None:
            self._georeference()
            return
        self._residual_debounce_timer.start()

    def _recompute_residuals_async(self):
        """
        Fired by the debounce timer: snapshot inputs on the GUI thread and run
        :func:`compute_residuals` on the work scheduler, tracking an in-flight token so a
        superseded result is dropped.
        """
        snapshot = self._snapshot_residual_inputs()
        if snapshot is None:
            return

        scheduler = getattr(self._app_services, "scheduler", None) if self._app_services else None
        if scheduler is None:
            try:
                residuals = compute_residuals(
                    snapshot["gcps"],
                    snapshot["ref_srs"],
                    snapshot["output_srs"],
                    snapshot["warp_kwargs"],
                    snapshot["transformer_options"],
                )
            except BaseException as e:
                self.set_message_text(f"Error computing residuals: {e}")
                return
            self._apply_residuals_to_entries(snapshot["entries"], residuals)
            return

        self._residual_signature += 1
        token = self._residual_signature
        entries = snapshot["entries"]

        def _done(future):
            try:
                residuals = future.result()
            except BaseException:
                residuals = None
            # Deliver back to the GUI thread; a stale token is dropped in _apply_residuals.
            self._residuals_ready.emit((token, entries, residuals))

        future = scheduler.submit_thread(
            PriorityClass.INTERACTIVE,
            compute_residuals,
            snapshot["gcps"],
            snapshot["ref_srs"],
            snapshot["output_srs"],
            snapshot["warp_kwargs"],
            snapshot["transformer_options"],
        )
        future.add_done_callback(_done)

    def _apply_residuals(self, payload):
        """Install a completed residual computation on the GUI thread (unless superseded)."""
        token, entries, residuals = payload
        if token != self._residual_signature:
            return  # superseded by a newer recompute; discard
        if residuals is None:
            self.set_message_text("Error computing residuals.")
            return
        self._apply_residuals_to_entries(entries, residuals)

    # ========================
    # region Accepting
    # ========================

    def _create_warped_output(self) -> bool:
        """
        Validate inputs and launch the multi-band warp on a background thread.

        Returns ``True`` if the warp was successfully *started* (inputs valid), ``False``
        otherwise. The GDAL work runs off the GUI thread via
        :func:`~wiser.gui.progress_task.run_with_progress`; on completion
        :meth:`_on_warp_done` emits :attr:`warp_completed` and updates the status label.
        """
        save_path = self._get_save_file_path()
        if save_path is None:
            QMessageBox.information(
                self,
                self.tr("No Save Path Selected"),
                self.tr(
                    "In order to georeference, a save path "
                    "must be selected. There is no save path "
                    "selected, so georeferencing will not occur.\n\n"
                    "Please select a save path."
                ),
            )
            return False

        if not self._enough_points_for_transform():
            QMessageBox.information(
                self,
                self.tr("Can't Run Georeferencer"),
                self.tr("Not enough points to run georeferencer"),
            )
            return False

        if self._target_rasterpane.get_rasterview().get_raster_data() is None:
            QMessageBox.information(
                self,
                self.tr("No Target Dataset Selected"),
                self.tr("A target dataset is not selected. Please select a target dataset."),
            )
            return False

        target_dataset = self._target_rasterpane.get_rasterview().get_raster_data()
        gcps = [gcp for _, gcp in self._get_entry_gcp_list()]

        output_srs = self._import_current_output_srs()
        output_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        warp_kwargs, _ = build_warp_kwargs(self._curr_resample_alg, self._curr_transform_type, output_srs)

        # The reference SRS supplies the projection attached to the GCPs. Left in its
        # native axis mapping to match the original in-dialog warp exactly.
        ref_srs = self._get_reference_srs()

        self.set_message_text(self.tr("Starting warp..."))
        run_with_progress(
            self._app_services,
            self,
            self.tr("Warping…"),
            warp_dataset_to_path,
            target_dataset,
            gcps,
            warp_kwargs,
            ref_srs,
            save_path,
            on_success=self._on_warp_done,
            on_error=self._on_warp_error,
        )
        return True

    def _on_warp_done(self, written_path: str):
        """GUI-thread callback when a warp finishes: announce it and update status."""
        self.set_message_text(self.tr("Done warping!"))
        self.warp_completed.emit(written_path)

    def _on_warp_error(self, message: str):
        """GUI-thread callback when a warp fails (or is cancelled)."""
        QMessageBox.critical(self, self.tr("Error While Creating Output"), self.tr(f"Error:\n{message}"))

    def accept(self):
        # The warp is produced by the dedicated "Run Warp" button (threaded); accept() is a
        # plain commit/close so the OK button no longer double-warps.
        super().accept()

    # region Event overrides

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Return, Qt.Key_Enter, Qt.Key_Escape):
            event.accept()  # Do nothing on Enter or Escape
        else:
            super().keyPressEvent(event)
