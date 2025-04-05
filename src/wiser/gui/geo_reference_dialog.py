
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

from wiser.raster.dataset import RasterDataSet

from enum import IntEnum

class COLUMN_ID(IntEnum):
    ENABLED_COL = 0
    ID_COL = 1
    TARGET_X_COL = 2
    TARGET_Y_COL = 3
    REF_X_COL = 4
    REF_Y_COL = 5
    REMOVAL_COL = 6

ID_PROPERTY = "ENTRY_ID"

class GeoRefTableEntry:
    def __init__(self, gcp_pair: GroundControlPointPair, enabled: bool, id: int, residuals: float):
        self._gcp_pair = gcp_pair
        self._enabled = enabled
        self._id = id
        self._residuals = residuals

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

        # Set up dataset choosers 
        self._init_dataset_choosers()
        self._init_rasterpanes()
        self._init_gcp_table()
    
    def _init_gcp_table(self):
        table_widget = self._ui.table_gcps
        table_widget.setColumnCount(len(COLUMN_ID))
        headers = ["Enabled", "ID", "Target X", "Target Y", "Ref X", "Ref Y", "Remove"]
        self._ui.table_gcps.setHorizontalHeaderLabels(headers)

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

    def _on_gcp_pair_added(self, gcp_pair: GroundControlPointPair):
        # Create new table entry
        table_widget = self._ui.table_gcps
        next_row = table_widget.rowCount()
        enabled = True
        id = next_row
        residuals = 0
        table_entry = GeoRefTableEntry(gcp_pair, enabled, id, residuals)
        # The row that a GCP is placed on should be the same as its position in the
        # geo referencer task delegate point list

        self._add_table_entry(table_entry)
        return

    def _add_table_entry(self, table_entry: GeoRefTableEntry):
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

        table_widget.setCellWidget(row_to_add, COLUMN_ID.ENABLED_COL, checkbox)
        table_widget.setItem(row_to_add, COLUMN_ID.ID_COL, QTableWidgetItem(str(table_entry.get_id())))
        table_widget.setItem(row_to_add, COLUMN_ID.TARGET_X_COL, QTableWidgetItem(str(target_x)))
        table_widget.setItem(row_to_add, COLUMN_ID.TARGET_Y_COL, QTableWidgetItem(str(target_y)))
        table_widget.setItem(row_to_add, COLUMN_ID.REF_X_COL, QTableWidgetItem(str(ref_x)))
        table_widget.setItem(row_to_add, COLUMN_ID.REF_Y_COL, QTableWidgetItem(str(ref_y)))

        pushButton = QPushButton("Remove GCP")
        pushButton.clicked.connect(lambda checked : self._on_removal_button_clicked(table_entry))
        table_widget.setCellWidget(row_to_add, COLUMN_ID.REMOVAL_COL, pushButton)

        self._target_rasterpane.update_all_rasterviews()
        self._reference_rasterpane.update_all_rasterviews()
    
    def _on_removal_button_clicked(self, table_entry: GeoRefTableEntry):
        self._remove_table_entry(table_entry)

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
                print(f"index removed: ", index_removed)
                print(f"table_entry.get_id(): ", table_entry.get_id())
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

    def _update_entry_ids(self):
        table_widget = self._ui.table_gcps
        for i in range(len(self._table_entry_list)):
            table_entry = self._table_entry_list[i]
            table_entry.set_id(i)
            # i also functions as the row in the table widget where
            # this entry is currently
            print(f"updating i: {i}, for column: {COLUMN_ID.ID_COL}")
            table_widget.setItem(i, COLUMN_ID.ID_COL, QTableWidgetItem(str(i)))
    

    def get_table_entries(self) -> List[GeoRefTableEntry]:
        # Go through the table and extract the geo referencer dialog for each entry 
        return self._table_entry_list

    def get_gcp_table_size(self) -> int:
        print(f"get_gcp_table_size")
        print(f"len(self._table_entry_list): {len(self._table_entry_list)}")
        print(f"self._ui.table_gcps.rowCount(): {self._ui.table_gcps.rowCount()}")
        assert len(self._table_entry_list) == self._ui.table_gcps.rowCount(), \
                f"Entry number mismatch. Table entry list {len(self._table_entry_list)} and QTableWidget has {self._ui.table_gcps.rowCount()} entries"
        return len(self._table_entry_list)

    def extract_entry_from_row(self, row) -> GeoRefTableEntry:
        '''
        Turns an entry from a row in the QTableWidget into a GeoRefTableEntry

        DEPRECATED: We keep a list of the entries in the table so we don't
        have to do this
        '''
        table_widget = self._ui.table_gcps
        gcp_pair = GroundControlPointPair(self._target_rasterpane, self._reference_rasterpane)
        enabled = table_widget.item(row, COLUMN_ID.ENABLED_COL)
        id = table_widget.item(row, COLUMN_ID.ID_COL)
        target_x = table_widget.item(row, COLUMN_ID.TARGET_X_COL)
        target_y = table_widget.item(row, COLUMN_ID.TARGET_Y_COL)
        target_gcp = GroundControlPoint((target_x, target_y), \
                                         self._target_rasterpane.get_rasterview().get_raster_data(), \
                                         self._target_rasterpane)

        ref_x = table_widget.item(row, COLUMN_ID.REF_X_COL)
        ref_y = table_widget.item(row, COLUMN_ID.REF_Y_COL)
        ref_gcp = GroundControlPoint((ref_x, ref_y), \
                                     self._reference_rasterpane.get_rasterview().get_raster_data(), \
                                     self._reference_rasterpane)
        gcp_pair.add_gcp(target_gcp)
        gcp_pair.add_gcp(ref_gcp)

        table_entry = GeoRefTableEntry(gcp_pair, enabled, id, residuals=0)
        return table_entry


    # Handles populating and updating the dataset choosers

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

    def _on_switch_target_dataset(self, index: int):
        ds_id = self._target_cbox.itemData(index)
        dataset = self._app_state.get_dataset(ds_id)
        self._target_rasterpane.show_dataset(dataset)

    def _on_switch_reference_dataset(self, index: int):
        ds_id = self._reference_cbox.itemData(index)
        dataset = self._app_state.get_dataset(ds_id)
        self._reference_rasterpane.show_dataset(dataset)

    def set_message_text(self, text: str):
        self._ui.lbl_message.setText(text)
