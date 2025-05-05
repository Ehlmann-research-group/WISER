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

class Units(Enum):
    METERS = ("Meters", 1)
    DEGREES = ("Degrees", 0.0174532925199433)

class Ellipsoid_Axis_Type(Enum):
    SEMI_MINOR = "Semi Minor"
    INVERSE_FLATTENING = "Inverse Flattening"

class ProjectionTypes(Enum):
    EQUI_CYLINDRICAL = "Equidistance Cylindrical"
    POLAR_STEREO = "Polar Stereographic"
    NO_PROJECTION = "No Projection"

class ShapeTypes(Enum):
    ELLIPSOID = "Ellipsoid"
    SPHEROID = "Spheroid"

class ReferenceCreatorDialog(QDialog):

    def __init__(self, app_state: ApplicationState, parent=None):
        super().__init__(parent=parent)
        self._app_state = app_state
        
        # Set up the UI state
        self._ui = Ui_ReferenceSystemCreator()
        self._ui.setupUi(self)

        # Init variables
        self._lat_equator: Optional[float] = None
        self._lon_meridian: Optional[float] = None
        self._proj_type : Optional[ProjectionTypes]
        self._axis_ingest_type: Optional[Ellipsoid_Axis_Type] = None
        self._axis_ingestion_value: Optional[float] = None
        self._semi_major_value: Optional[float] = None

        # Initialize UI
        self._init_dataset_chooser()
        self._init_projection_chooser()
        self._init_shape_chooser()
        self._init_units()
        self._init_ellipsoid_params()
        self._init_lat_equator_ledit()
        self._init_lon_meridian_ledit()
        self._init_crs_name()
    
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

    def _init_units(self):
        # Populate units combo box
        self._ui.cbox_units.clear()
        for unit in Units:
            display_text, _factor = unit.value
            self._ui.cbox_units.addItem(display_text, unit)

        # Connect combo box signal
        self._ui.cbox_units.currentIndexChanged.connect(
            self._on_cbox_units_changed
        )

    def _init_ellipsoid_params(self):
        # Populate the axis type combo box
        self._ui.cbox_flat_minor.clear()
        for axis_type in Ellipsoid_Axis_Type:
            # display text is the enum value, store enum itself as user data
            self._ui.cbox_flat_minor.addItem(axis_type.value, axis_type)

        # Connect combo box signal to slot
        self._ui.cbox_flat_minor.currentIndexChanged.connect(
            self._on_axis_ingest_type_changed
        )

        # Configure flat minor value entry with float validator
        flat_validator = QDoubleValidator(self._ui.ledit_flat_minor)
        flat_validator.setNotation(QDoubleValidator.StandardNotation)
        flat_validator.setDecimals(10)
        self._ui.ledit_flat_minor.setValidator(flat_validator)
        self._ui.ledit_flat_minor.textChanged.connect(
            self._on_axis_ingestion_value_changed
        )

        # Configure semi-major entry with float validator
        semi_validator = QDoubleValidator(self._ui.ledit_semi_major)
        semi_validator.setNotation(QDoubleValidator.StandardNotation)
        semi_validator.setDecimals(10)
        self._ui.ledit_semi_major.setValidator(semi_validator)
        self._ui.ledit_semi_major.textChanged.connect(
            self._on_semi_major_changed
        )

    def _init_lat_equator_ledit(self):
        validator = QDoubleValidator(self._ui.ledit_lat_equator)
        validator.setNotation(QDoubleValidator.StandardNotation)
        validator.setRange(-90.0, 90.0, 10)
        self._ui.ledit_lat_equator.setValidator(validator)
        self._ui.ledit_lat_equator.textChanged.connect(
            self._on_lat_equator_changed
        )

    def _init_lon_meridian_ledit(self):
        validator = QDoubleValidator(self._ui.ledit_lon_meridian)
        validator.setNotation(QDoubleValidator.StandardNotation)
        validator.setRange(-180.0, 180.0, 10)
        self._ui.ledit_lon_meridian.setValidator(validator)
        self._ui.ledit_lon_meridian.textChanged.connect(
            self._on_lon_meridian_changed
        )

    def _init_crs_name(self):
        regex = QRegExp(r"^[A-Za-z0-9_]+$")
        validator = QRegExpValidator(regex, self._ui.ledit_crs_name)
        self._ui.ledit_crs_name.setValidator(validator)
        self._ui.ledit_crs_name.textEdited.connect(
            self._on_crs_name_changed
        )


    # Region Slots


    def _on_cbox_units_changed(self, index: int):
        self._cbox_units = self._ui.cbox_units.itemData(index)

    def _on_axis_ingest_type_changed(self, index: int):
        # retrieve the enum stored as user data
        self._axis_ingest_type = self._ui.cbox_flat_minor.itemData(index)

    def _on_axis_ingestion_value_changed(self, text: str):
        # convert to float if possible
        try:
            self._axis_ingestion_value = float(text)
        except ValueError:
            self._axis_ingestion_value = None

    def _on_semi_major_changed(self, text: str):
        try:
            self._semi_major_value = float(text)
        except ValueError:
            self._semi_major_value = None
    
    def _on_switch_dataset(self, index: int):
        pass

    def _on_lat_equator_changed(self, text: str):
        try:
            self._lat_equator = float(text)
        except ValueError:
            self._lat_equator = None

    def _on_lon_meridian_changed(self, text: str):
        try:
            self._lon_meridian = float(text)
        except ValueError:
            self._lon_meridian = None

    def _on_switch_proj_type(self, index: int):
        self._proj_type = self._ui.cbox_proj_type.itemData(index)

    def _on_switch_shape_type(self, index: int):
        self._shape_type = self._ui.cbox_shape.itemData(index)
        if self._shape_type == ShapeTypes.SPHEROID:
            self._ui.cbox_flat_minor.setEnabled(False)
            self._ui.ledit_flat_minor.setEnabled(False)
            self._ui.lbl_semi_major.setText("Radius")
        elif self._shape_type == ShapeTypes.ELLIPSOID:
            self._ui.cbox_flat_minor.setEnabled(True)
            self._ui.ledit_flat_minor.setEnabled(True)
            self._ui.lbl_semi_major.setText("Semi-Major Axis")

    def _on_crs_name_changed(self):
        self._crs_name = self._ui.ledit_crs_name.text()
