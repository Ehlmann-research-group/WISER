
from typing import List, Optional

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

from .generated.geo_referencer_dialog_ui import Ui_GeoReferencerDialog

from wiser.gui.app_state import ApplicationState
from wiser.gui.rasterview import RasterView
from wiser.gui.rasterpane import RasterPane
from wiser.gui.geo_reference_pane import GeoReferencerPane
from wiser.gui.georeference_task_delegate import GeoReferencerTaskDelegate, GroundControlPointPair, GroundControlPoint
from wiser.gui.util import get_random_matplotlib_color, get_color_icon

from wiser.raster.dataset import RasterDataSet

from enum import IntEnum, Enum

from osgeo import gdal

class COLUMN_ID(IntEnum):
    ENABLED_COL = 0
    ID_COL = 1
    TARGET_X_COL = 2
    TARGET_Y_COL = 3
    REF_X_COL = 4
    REF_Y_COL = 5
    COLOR_COL = 6
    REMOVAL_COL = 7

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

COMMON_SRS = {
    "WGS84 EPSG:4326": "EPSG:4326",
    "Web Mercator EPSG:3857": "EPSG:3857",
    "NAD83 / UTM zone 15N EPSG:26915": "EPSG:26915",
    # Add more as required by your application.
}

class GeoRefTableEntry:
    def __init__(self, gcp_pair: GroundControlPointPair, enabled: bool, id: int, residuals: float, color: str):
        self._gcp_pair = gcp_pair
        self._enabled = enabled
        self._id = id
        self._residuals = residuals
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
    def get_residuals(self) -> float:
        return self._residuals

    def set_residuals(self, residuals: float):
        self._residuals = residuals

    # Getter and Setter for residuals
    def get_color(self) -> str:
        return self._color

    def set_color(self, color: str):
        self._color = color

    def replace_entry(self, newEntry: 'GeoRefTableEntry'):
        self.set_gcp_pair(newEntry.get_gcp_pair())
        self.set_enabled(newEntry.is_enabled())
        self.set_id(newEntry.get_id())
        self.set_residuals(newEntry.get_residuals())

    def __str__(self):
        return (
        "=======================\n"
        f"gcp_pair: {self._gcp_pair}\n"
        f"id: {self._id}\n"
        f"enabled: {self._enabled}\n"
        f"residuals: {self._residuals}\n"
        "======================="
    )

class GeoReferencerDialog(QDialog):

    gcp_pair_added = Signal(GroundControlPointPair)

    def __init__(self, app_state: ApplicationState, main_view: RasterPane, parent=None):
        super().__init__(parent=parent)
        self._app_state = app_state
        self._main_view = main_view

        # Set up the UI state
        self._ui = Ui_GeoReferencerDialog()
        self._ui.setupUi(self)

        self._target_cbox = self._ui.cbox_target_dataset_chooser
        self._reference_cbox = self._ui.cbox_reference_dataset_chooser

        self._target_rasterpane = GeoReferencerPane(app_state=app_state)
        self._reference_rasterpane = GeoReferencerPane(app_state=app_state)
        self._georeferencer_task_delegate = GeoReferencerTaskDelegate(self._target_rasterpane,
                                                                      self._reference_rasterpane,
                                                                      self,
                                                                      app_state)
        self._target_rasterpane.set_task_delegate(self._georeferencer_task_delegate)
        self._reference_rasterpane.set_task_delegate(self._georeferencer_task_delegate)

        self.gcp_pair_added.connect(self._on_gcp_pair_added)

        self._table_entry_list: List[GeoRefTableEntry] = []

        self._current_srs: str = None
        self._current_resample_alg = None
        self._current_transform_type: TRANSFORM_TYPES = None

        # Set up dataset choosers 
        self._init_dataset_choosers()
        self._init_rasterpanes()
        self._init_gcp_table()
        self._init_srs_cbox()
        self._init_interpolation_type_cbox()
        self._init_poly_order_cbox()

    # region Initialization

    def _init_srs_cbox(self):
        """Initialize the spatial reference combo box."""
        srs_cbox = self._ui.cbox_srs
        srs_cbox.activated.connect(self._on_switch_srs)
        srs_cbox.clear()
        # Use the friendly key (e.g., "WGS84") as the display text,
        # and store the corresponding SRS string (e.g., "EPSG:4326") as userData.
        for name, srs in COMMON_SRS.items():
            srs_cbox.addItem(name, srs)
        self._on_switch_srs(0)

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
        headers = ["Enabled", "ID", "Target X", "Target Y", "Ref X", "Ref Y", "Color", "Remove"]
        table_widget.setHorizontalHeaderLabels(headers)

        table_widget.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

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

    def _on_switch_srs(self, index: int):
        srs = self._ui.cbox_srs.itemData(index)
        self._current_srs = srs

    def _on_switch_resample_alg(self, index: int):
        resample_alg = self._ui.cbox_interpolation.itemData(index)
        self._current_resample_alg = resample_alg

    def _on_switch_transform_type(self, index: int):
        transform_type = self._ui.cbox_poly_order.itemData(index)
        self._current_transform_type = transform_type

    def _on_choose_color(self, table_entry: GeoRefTableEntry):
        row = table_entry.get_id()
        initial_color = QColor(self._table_entry_list[row].get_color())
        color = QColorDialog.getColor(parent=self, initial=initial_color)
        if color.isValid():
            color_str = color.name()
            self._table_entry_list[row].set_color(color_str)
            self._set_color_icon(row, color_str)
            # TODO (Joshua G-K): Change the color icon
            self._target_rasterpane.update_all_rasterviews()
            self._reference_rasterpane.update_all_rasterviews()

    def _on_enabled_clicked(self, table_entry: GeoRefTableEntry, checked: bool):
        # Since the table_entry's ID can change, don't just pass in the row_to_add
        row_to_add = table_entry.get_id()
        self._set_row_enabled_state(row_to_add, checked)
        self._target_rasterpane.update_all_rasterviews()
        self._reference_rasterpane.update_all_rasterviews()

    def _on_gcp_pair_added(self, gcp_pair: GroundControlPointPair):
        # Create new table entry
        table_widget = self._ui.table_gcps
        next_row = table_widget.rowCount()
        enabled = True
        id = next_row
        residuals = 0
        color = get_random_matplotlib_color()
        table_entry = GeoRefTableEntry(gcp_pair, enabled, id, residuals, color)
        # The row that a GCP is placed on should be the same as its position in the
        # geo referencer task delegate point list

        self._add_entry_to_table(table_entry)

    def _on_removal_button_clicked(self, table_entry: GeoRefTableEntry):
        self._remove_table_entry(table_entry)

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
        self._reference_rasterpane.show_dataset(dataset)

    # region Table Entry Helpers

    def _reset_gcps(self):
        '''
        Clears all of the entries in the table widget and in the list of entries
        '''
        self._table_entry_list = []
        self._ui.table_gcps.clearContents()
        self._ui.table_gcps.setRowCount(0)

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
        ref_x = gcp_pair.get_reference_gcp().get_point()[0]
        ref_y = gcp_pair.get_reference_gcp().get_point()[1]

        checkbox = QCheckBox()
        checkbox.setChecked(table_entry.is_enabled())
        checkbox.clicked.connect(lambda checked : self._on_enabled_clicked(table_entry, checked))

        table_widget.setCellWidget(row_to_add, COLUMN_ID.ENABLED_COL, checkbox)
        table_widget.setItem(row_to_add, COLUMN_ID.ID_COL, QTableWidgetItem(str(table_entry.get_id())))
        table_widget.setItem(row_to_add, COLUMN_ID.TARGET_X_COL, QTableWidgetItem(str(target_x)))
        table_widget.setItem(row_to_add, COLUMN_ID.TARGET_Y_COL, QTableWidgetItem(str(target_y)))
        table_widget.setItem(row_to_add, COLUMN_ID.REF_X_COL, QTableWidgetItem(str(ref_x)))
        table_widget.setItem(row_to_add, COLUMN_ID.REF_Y_COL, QTableWidgetItem(str(ref_y)))

        color_button = QPushButton()
        color_button.clicked.connect(lambda checked : self._on_choose_color(table_entry))
        initial_color = table_entry.get_color()
        color_icon = get_color_icon(initial_color)
        color_button.setIcon(color_icon)
        table_widget.setCellWidget(row_to_add, COLUMN_ID.COLOR_COL, color_button)

        pushButton = QPushButton("Remove GCP")
        pushButton.clicked.connect(lambda checked : self._on_removal_button_clicked(table_entry))
        table_widget.setCellWidget(row_to_add, COLUMN_ID.REMOVAL_COL, pushButton)

        self._target_rasterpane.update_all_rasterviews()
        self._reference_rasterpane.update_all_rasterviews()

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

        self._target_rasterpane.update_all_rasterviews()
        self._reference_rasterpane.update_all_rasterviews()

    def _sync_entry_list_index_with_ui_row(self, row: int):
        '''
        DEPRECATED: We are suppose to update the list and ui at the
        same time so this should never be needed.
        '''
        entryInTable = self.extract_entry_from_row(row)
        print(f"entryInTable ID: {entryInTable.get_id()}")
        print(f"entry in row ID: {self._table_entry_list[row].get_id()}")
        self._table_entry_list[row].replace_entry(entryInTable)

    def _update_entry_ids(self):
        table_widget = self._ui.table_gcps
        for i in range(len(self._table_entry_list)):
            table_entry = self._table_entry_list[i]
            table_entry.set_id(i)
            # i also functions as the row in the table widget where
            # this entry is currently
            # print(f"updating i: {i}, for column: {COLUMN_ID.ID_COL}")
            table_widget.setItem(i, COLUMN_ID.ID_COL, QTableWidgetItem(str(i)))

    def extract_entry_from_row(self, row) -> GeoRefTableEntry:
        '''
        Turns an entry from a row in the QTableWidget into a GeoRefTableEntry

        DEPRECATED: We keep a list of the entries in the table so we don't
        have to do this
        '''
        table_widget = self._ui.table_gcps
        gcp_pair = GroundControlPointPair(self._target_rasterpane, self._reference_rasterpane)
        enabled = table_widget.cellWidget(row, COLUMN_ID.ENABLED_COL).isChecked()
        id = int(table_widget.item(row, COLUMN_ID.ID_COL).text())
        target_x = float(table_widget.item(row, COLUMN_ID.TARGET_X_COL).text())
        target_y = float(table_widget.item(row, COLUMN_ID.TARGET_Y_COL).text())
        target_gcp = GroundControlPoint((target_x, target_y), \
                                         self._target_rasterpane.get_rasterview().get_raster_data(), \
                                         self._target_rasterpane)

        ref_x = float(table_widget.item(row, COLUMN_ID.REF_X_COL).text())
        ref_y = float(table_widget.item(row, COLUMN_ID.REF_Y_COL).text())
        ref_gcp = GroundControlPoint((ref_x, ref_y), \
                                     self._reference_rasterpane.get_rasterview().get_raster_data(), \
                                     self._reference_rasterpane)
        gcp_pair.add_gcp(target_gcp)
        gcp_pair.add_gcp(ref_gcp)

        color_str = self._table_entry_list[row].get_color()

        table_entry = GeoRefTableEntry(gcp_pair, enabled, id, residuals=0, color=color_str)
        return table_entry
    

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

    def set_message_text(self, text: str):
        self._ui.lbl_message.setText(text)
