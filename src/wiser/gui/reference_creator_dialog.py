import os
from typing import List, Optional

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

from .generated.reference_system_creator_ui import Ui_ReferenceSystemCreator

from wiser.gui.app_state import ApplicationState
from wiser.gui.rasterview import RasterView
from wiser.gui.rasterpane import RasterPane

from enum import Enum

class ProjectionTypes(Enum):
    EQUI_CYLINDRICAL = "Equidistance Cylindrical"
    POLAR_STEREO = "Polar Stereographic"

class ShapeTypes(Enum):
    SPHEROID = "Spheroid"
    ELLIPSOID = "Ellipsoid"

class ReferenceCreatorDialog(QDialog):

    def __init__(self, app_state: ApplicationState, parent=None):
        super().__init__(parent=parent)
        self._app_state = app_state
        
        # Set up the UI state
        self._ui = Ui_ReferenceSystemCreator()
        self._ui.setupUi(self)

        # Initialize UI
        self._init_dataset_chooser()
        self._init_projection_chooser()
        self._init_shape_chooser()
        self._init_lat_meridian_ledit()
        self._init_lon_meridian_ledit()
    
    def _init_dataset_chooser(self):
        dataset_chooser = self._ui.cbox_dataset
        dataset_chooser.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        dataset_chooser.activated.connect(self._on_switch_dataset)
        
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

    def _init_projection_chooser(self):
        proj_cbox = self._ui.cbox_proj_type
        proj_cbox.activated.connect(self._on_switch_proj_type)
        proj_cbox.clear()

        for proj in ProjectionTypes:
            proj_cbox.addItem(proj.value, proj)

        self._on_switch_proj_type(0)

    def _init_shape_chooser(self):
        shape_cbox = self._ui.cbox_shape
        shape_cbox.activated.connect(self._on_switch_shape_type)
        shape_cbox.clear()

        for shape in ShapeTypes:
            shape_cbox.addItem(shape.value, shape)

        self._on_switch_shape_type(0)

    def _init_lat_meridian_ledit(self):
        pass

    def _init_lon_meridian_ledit(self):
        pass

    # Region Slots

    def _on_switch_dataset(self, index: int):
        pass

    def _on_switch_proj_type(self, index: int):
        pass

    def _on_switch_shape_type(self, index: int):
        pass