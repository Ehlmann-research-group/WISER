
from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

from .generated.geo_referencer_dialog_ui import Ui_GeoReferencerDialog

from wiser.gui.app_state import ApplicationState
from wiser.gui.rasterview import RasterView
from wiser.gui.rasterpane import RasterPane
from wiser.gui.geo_reference_pane import GeoReferencerPane

from wiser.raster.dataset import RasterDataSet

class GeoReferencerDialog(QDialog):

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

        # Set up dataset choosers 
        self._init_dataset_choosers()
        self._init_rasterpanes()
    
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
        return

    def _on_switch_reference_dataset(self, index: int):
        ds_id = self._reference_cbox.itemData(index)
        dataset = self._app_state.get_dataset(ds_id)
        self._reference_rasterpane.show_dataset(dataset)
        return

    def set_message_text(self, text: str):
        return
