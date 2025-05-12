import os
import sys

from typing import List, Optional

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

from .generated.reference_system_creator_ui import Ui_ReferenceSystemCreator

from wiser.gui.app_state import ApplicationState

from osgeo import osr

import pyproj

from enum import Enum

ALLOWED_DECIMALS = 15
MAX_SCALE_FACTOR = 65535
NO_CRS_NAME = "(None)"

class Units(Enum):
    METERS = ("Meters", 1)
    DEGREES = ("Degrees", 0.0174532925199433)

class EllipsoidAxisType(Enum):
    SEMI_MINOR = "Semi Minor"
    INVERSE_FLATTENING = "Inverse Flattening"

class LatitudeTypes(Enum):
    CENTRAL_LATITUDE = "Central Latitude"
    TRUE_SCALE_LATITUDE = "True Scale Lat"

class ProjectionTypes(Enum):
    EQUI_CYLINDRICAL = "Equidistance Cylindrical"
    POLAR_STEREO = "Polar Stereographic"
    NO_PROJECTION = "No Projection"

class ShapeTypes(Enum):
    ELLIPSOID = "Ellipsoid"
    SPHEROID = "Spheroid"

class Sign(Enum):
    POSITIVE = "+"
    NEGATIVE = "-"

class CrsCreatorState:
    def __init__(
        self,
        lon_meridian: Optional[float] = None,
        proj_type: Optional[ProjectionTypes] = None,
        axis_ingest_type: Optional[EllipsoidAxisType] = EllipsoidAxisType.SEMI_MINOR,
        axis_ingestion_value: Optional[float] = None,
        semi_major_value: Optional[float] = None,
        latitude_choice: Optional[LatitudeTypes] = None,
        latitude: Optional[float] = None,
        center_lon: Optional[float] = None,
        polar_stereo_scale: Optional[float] = None,
        polar_stereo_latitude_sign: Optional[str] = None,
    ):
        self._lon_meridian = lon_meridian
        self._proj_type = proj_type
        self._axis_ingest_type = axis_ingest_type
        self._axis_ingestion_value = axis_ingestion_value
        self._semi_major_value = semi_major_value
        self._latitude_choice = latitude_choice
        self._latitude = latitude
        self._center_lon = center_lon
        self._polar_stereo_scale = polar_stereo_scale
        self._polar_stereo_latitude_sign = polar_stereo_latitude_sign

    @property
    def lon_meridian(self) -> Optional[float]:
        return self._lon_meridian

    @property
    def proj_type(self) -> Optional[ProjectionTypes]:
        return self._proj_type

    @property
    def axis_ingest_type(self) -> Optional[EllipsoidAxisType]:
        return self._axis_ingest_type

    @property
    def axis_ingestion_value(self) -> Optional[float]:
        return self._axis_ingestion_value

    @property
    def semi_major_value(self) -> Optional[float]:
        return self._semi_major_value

    @property
    def latitude_choice(self) -> Optional[LatitudeTypes]:
        return self._latitude_choice

    @property
    def latitude(self) -> Optional[float]:
        return self._latitude

    @property
    def center_lon(self) -> Optional[float]:
        return self._center_lon

    @property
    def polar_stereo_scale(self) -> Optional[float]:
        return self._polar_stereo_scale

    @property
    def polar_stereo_latitude_sign(self) -> Optional[str]:
        return self._polar_stereo_latitude_sign

class ReferenceCreatorDialog(QDialog):

    def __init__(self, app_state: ApplicationState, parent=None):
        super().__init__(parent=parent)
        self._app_state = app_state
        
        # Set up the UI state
        self._ui = Ui_ReferenceSystemCreator()
        self._ui.setupUi(self)

        # Init variables
        self._lon_meridian: Optional[float] = None
        self._proj_type : Optional[ProjectionTypes]
        self._axis_ingest_type: Optional[EllipsoidAxisType] = EllipsoidAxisType.SEMI_MINOR
        self._axis_ingestion_value: Optional[float] = None
        self._semi_major_value: Optional[float] = None
        self._latitude_choice: Optional[LatitudeTypes] = None
        self._latitude: Optional[float] = None
        self._center_lon: Optional[float] = None
        self._polar_stereo_scale: Optional[float] = None
        self._polar_stereo_latitude_sign: Optional[str] = None

        # save current name so we can tell if the user picks something new later
        self._current_starting_crs_name: Optional[str] = None

        # Initialize UI
        self._init_user_created_crs()
        self._init_projection_chooser()
        self._init_shape_chooser()
        self._init_ellipsoid_params()
        self._init_lon_meridian_ledit()
        self._init_center_longitude_ledit()
        self._init_crs_name()
        self._init_cbox_lat_chooser()
        self._init_ledit_lat_value()
        self._init_reset_button()
        self._init_create_crs_button()
        self._init_extra_polar_stereo_params()

    def _init_extra_polar_stereo_params(self):
        # Initialize the central lat cbox
        cbox = self._ui.cbox_pstereo_sign
        cbox.clear()

        # Add each enum member
        for sign in Sign:
            cbox.addItem(sign.value, sign.value)

        # When the user picks a new item, update self._latitude_choice
        cbox.currentIndexChanged.connect(self._on_stereo_pos_neg_changed)
        cbox.setCurrentIndex(0)
        cbox.currentIndexChanged.emit(0)
        # self._on_stereo_pos_neg_changed(cbox.currentIndex())

        # Initialize the Scale Factor Line Edit 
        validator = QDoubleValidator(self._ui.ledit_pstereo_scale_factor)
        validator.setNotation(QDoubleValidator.StandardNotation)
        validator.setRange(0, MAX_SCALE_FACTOR, ALLOWED_DECIMALS)
        self._ui.ledit_pstereo_scale_factor.setValidator(validator)
        self._ui.ledit_pstereo_scale_factor.textChanged.connect(
            self._on_stereo_scale_factor_changed
        )
        self._on_stereo_scale_factor_changed(self._ui.ledit_pstereo_scale_factor.text())

    def _on_stereo_scale_factor_changed(self, text: str):
        try:
            self._polar_stereo_scale = float(text)
            print(f"self._polar: {self._polar_stereo_scale}")
        except ValueError:
            self._polar_stereo_scale = None

    def _on_stereo_pos_neg_changed(self, index: int) -> None:
        self._polar_stereo_latitude_sign = self._ui.cbox_pstereo_sign.itemData(index)
        print(f"self._polar_stereo_latitude_sign: {self._polar_stereo_latitude_sign};;;;;;;;;;")

    def _update_extra_polar_stereo_params_display(self):
        if self._proj_type == ProjectionTypes.POLAR_STEREO:
            if self._latitude_choice == LatitudeTypes.CENTRAL_LATITUDE:
                self._ui.wdgt_ts_central_lat.hide()
                self._ui.wdgt_scale_factor.show()
            elif self._latitude_choice == LatitudeTypes.TRUE_SCALE_LATITUDE:
                self._ui.wdgt_ts_central_lat.show()
                self._ui.wdgt_scale_factor.hide()
            else:
                raise ValueError(f"Latitude choice is incorrect. It is: {self._latitude_choice}")
        else:
            self._ui.wdgt_ts_central_lat.hide()
            self._ui.wdgt_scale_factor.hide()
            

    def _init_reset_button(self):
        '''
        Should reset the fields _axis_ingestion_value, _semi_major_value, _latitude, _center_lon, and _lon_meridian.
        Shoudl reset their class values and their line edits. 
        Should reset _current_starting_crs_name
        Should reset the value of self._ui.cbox_user_crs to be the value of (None)
        '''
        # Resolve the reset‑button name used in the .ui file
        reset_btn = self._ui.btn_reset_fields
        if reset_btn is None:
            raise AttributeError("Reset button not found in UI")

        reset_btn.clicked.connect(self._on_reset_clicked)

        # Do one reset immediately so the dialog starts in a clean state
        self._on_reset_clicked()

    def _on_reset_clicked(self):
        """Slot that really performs the reset."""
        # ---- 1.  Clear internal values ---------------------------------
        self._axis_ingestion_value = None
        self._semi_major_value     = None
        self._latitude             = None
        self._center_lon           = None
        self._lon_meridian         = None
        self._current_starting_crs_name = NO_CRS_NAME

        # ---- 2.  Clear the editor widgets ------------------------------
        for le in (
            self._ui.ledit_flat_minor,
            self._ui.ledit_semi_major,
            self._ui.ledit_lat_value,
            self._ui.ledit_center_lon,
            self._ui.ledit_prime_meridian,
            self._ui.ledit_crs_name
        ):
            le.clear()

        # ---- 3.  Put the “Starting CRS” combo back to “(None)” ----------
        cbox = self._ui.cbox_user_crs
        none_idx = cbox.findText(self.tr(NO_CRS_NAME))
        if none_idx == -1:                        # fallback: last entry
            none_idx = cbox.count() - 1
        if none_idx >= 0:
            cbox.blockSignals(True)              # suppress _on_starting_crs_changed
            cbox.setCurrentIndex(none_idx)
            cbox.blockSignals(False)


    def _init_create_crs_button(self):
        '''
        Should just run self._create_crs()
        '''
        create_btn = self._ui.btn_create_crs
        if create_btn is None:
            raise AttributeError("Create-CRS button not found in UI")

        create_btn.clicked.connect(self._create_crs)

    def _init_user_created_crs(self):
        """
        Gets the dictionary Dict[str, osr.SpatialReference] from self._app_state
        by doing self._app_state.get_user_created_crs

        If the dictionary is empty, we disable self._ui.cbox_user_crs. If the dictionary is not empty, we enable it.

        We populate the cbox with the text as the key (str) in the dictionary and the data as the value (osr.SpatialReference)
        
        When the user clicks on a cbox entry, it should call the function _on_starting_crs_changed. It should use a lambda to call this
        so that we can also pas the key (str) of the dictionary by the name parameter
        """
        """
        Populate the “Starting CRS” combo box with user-defined CRS objects that
        were persisted in ApplicationState.  Each entry's *text* is the dict key
        and the *userData* is the osr.SpatialReference itself.
        """
        cbox = self._ui.cbox_user_crs
        self._update_user_created_crs_cbox()

        # When the user chooses an entry we call _on_starting_crs_changed and pass
        # the *name* (key) via a small lambda wrapper.
        cbox.activated.connect(
            lambda idx: self._on_starting_crs_changed(cbox.itemText(idx))
        )

    def _switch_user_crs_cbox_selection(self, name: str):
        cbox = self._ui.cbox_user_crs
        idx = cbox.findText(name)
        if idx != -1:
            cbox.blockSignals(True)
            cbox.setCurrentIndex(idx)
            cbox.blockSignals(False)


    def _update_user_created_crs_cbox(self):
        app_state = self._app_state
        cbox = self._ui.cbox_user_crs

        num_crs = len(app_state.get_user_created_crs())

        current_index = cbox.currentIndex()
        current_crs_name = None
        if current_index != -1:
            current_crs_name= cbox.itemText(current_index)
        else:
            # This occurs initially, when the combobox is empty and has no
            # selection.  Make sure the "(no data)" option is selected by the
            # end of this process.
            current_index = 0
            current_crs_name = ""

        new_index = None
        cbox.clear()

        if num_crs > 0:
            for (index, name) in enumerate(sorted(list(app_state.get_user_created_crs().keys()))):
                crs = self._app_state.get_user_created_crs()[name][0]
                cbox.addItem(name, crs)
    
                if name == current_crs_name:
                    new_index = index

            cbox.insertSeparator(num_crs)
            cbox.addItem(self.tr('(None)'), -1)
            if current_crs_name == "":
                new_index = cbox.count() - 1
        else:
            # No datasets yet
            cbox.addItem(self.tr('(None)'), -1)
            if current_crs_name == "":
                new_index = 0

        if new_index is None:
            if num_crs > 0:
                new_index = min(current_index, num_crs - 1)
            else:
                new_index = 0

        cbox.setCurrentIndex(new_index)
        # cbox = self._ui.cbox_user_crs
        # cbox.clear()

        # self._user_created_crs = self._app_state.get_user_created_crs() or {}

        # if not self._user_created_crs:            # nothing stored yet
        #     cbox.setEnabled(False)
        #     return

        # cbox.setEnabled(True)
        # for name, srs in self._user_created_crs.items():
        #     cbox.addItem(name, srs)


    def _on_starting_crs_changed(self, name: str):
        """
        Should somehow check to see if a different starting CRS was clicked. If it was then we ask the user i they want 
        to replace all the fields and warn them that this will overwrite their data. If they do then we bring in the osr.SpatialReference
        and get its prime meridian value, equator value, projection type (it will either have no projection, be Equirectangular, or be stereo graphic)
        We figure out if it is an ellipsoid or a spheroid and its semi-major axis value and (if applicable) semi-minoir axis value.
        We make sure to change the cbox_flat_minor to either be the inverse flattening or the semi-minor depending on what the ellipsoid
        is represented as. 

        We figure out its center latitude and center longitude and true scale latitude (if it is a projection)

        We replace the old name with the new name
        """
        print(f"_on_starting_crs_changed CALEDDDDDD!!!!")
        print(f"self._currentstarting_name: {self._current_starting_crs_name}")
        if not name \
           or name == self._current_starting_crs_name:
            return                        # same choice – nothing to do
    
        if name == NO_CRS_NAME:
            self._current_starting_crs_name= NO_CRS_NAME
            return

        print(f"name: {name}")
        print(f" self._app_state.get_user_created_crs(): { self._app_state.get_user_created_crs()}")
        srs: osr.SpatialReference = self._app_state.get_user_created_crs().get(name)[0]
        creator_state: CrsCreatorState = self._app_state.get_user_created_crs().get(name)[1]
        if srs is None:                   # defensive – shouldn’t happen
            return

        # Ask before clobbering whatever the user already entered
        if QMessageBox.question(
            self,
            "Replace current parameters?",
            "Loading “{0}” will overwrite all fields you have entered so far.\n"
            "Continue?".format(name),
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        ) == QMessageBox.No:
            # Re‑select the previous item (if any) so the user sees no change
            cbox = self._ui.cbox_user_crs
            if self._current_starting_crs_name:
                old_idx = cbox.findText(self._current_starting_crs_name)
                if old_idx != -1:
                    cbox.blockSignals(True)
                    cbox.setCurrentIndex(old_idx)
                    cbox.blockSignals(False)
            return

        # -----------------------------------------------------------------
        # Convert to pyproj for convenient interrogation
        # -----------------------------------------------------------------
        pycrs = pyproj.CRS.from_wkt(srs.ExportToWkt())

        # ---------- Prime meridian & “equator” (lat_0) --------------------
        pm_lon = creator_state.lon_meridian
        self._ui.ledit_prime_meridian.setText(str(pm_lon))
        self._lon_meridian = pm_lon

        # ---------- Shape / ellipsoid parameters -------------------------
        a = creator_state.semi_major_value
        if creator_state.axis_ingest_type == EllipsoidAxisType.SEMI_MINOR:
            inv_f = creator_state.semi_major_value / (creator_state.semi_major_value - creator_state.axis_ingestion_value)
        else:
            inv_f = creator_state.axis_ingestion_value
        spheroid = (inv_f == 0.0)

        # Shape type
        shape_cbox = self._ui.cbox_shape
        if spheroid:
            shape_idx = shape_cbox.findData(ShapeTypes.SPHEROID)
            self._shape_type = ShapeTypes.SPHEROID
        else:
            shape_idx = shape_cbox.findData(ShapeTypes.ELLIPSOID)
            self._shape_type = ShapeTypes.ELLIPSOID
        shape_cbox.setCurrentIndex(shape_idx)

        # Semi‑major
        self._ui.ledit_semi_major.setText(f"{a}")
        self._semi_major_value = a

        # Semi‑minor / inverse‑flattening
        if spheroid:
            # Sphere – disable the flat/minor widgets
            self._ui.ledit_flat_minor.clear()
            self._ui.cbox_flat_minor.setEnabled(False)
            self._ui.ledit_flat_minor.setEnabled(False)
            self._axis_ingest_type = None
            self._axis_ingestion_value = None
        else:
            # Choose to display inverse‑flattening by default
            axis_cbox = self._ui.cbox_flat_minor
            axis_idx = axis_cbox.findData(EllipsoidAxisType.INVERSE_FLATTENING)
            axis_cbox.setCurrentIndex(axis_idx)
            self._axis_ingest_type = EllipsoidAxisType.INVERSE_FLATTENING
            self._axis_ingestion_value = inv_f
            self._ui.ledit_flat_minor.setEnabled(True)
            self._ui.cbox_flat_minor.setEnabled(True)
            self._ui.ledit_flat_minor.setText(f"{inv_f}")

        # ---------- Projection (or none) ---------------------------------
        # if pycrs.is_geographic:
        #     proj_type = ProjectionTypes.NO_PROJECTION
        # else:
        #     m = pycrs.coordinate_operation.method_name.lower()
        #     if "equidistant cylindrical" in m or "equirectangular" in m:
        #         proj_type = ProjectionTypes.EQUI_CYLINDRICAL
        #     elif "polar stereographic" in m or "stereographic" in m:
        #         proj_type = ProjectionTypes.POLAR_STEREO
        #     else:
        #         proj_type = ProjectionTypes.NO_PROJECTION  # fallback
        proj_type = creator_state.proj_type

        self._update_extra_polar_stereo_params_display()

        proj_cbox = self._ui.cbox_proj_type
        proj_idx = proj_cbox.findData(proj_type)
        proj_cbox.setCurrentIndex(proj_idx)
        self._proj_type = proj_type
        self._update_units()                             # refresh “Units” display

        # Projection‑specific parameters
        if not pycrs.is_geographic:
            op = pycrs.coordinate_operation
            params = op.params
            print(f"op: {params}")
            # self._center_lon = self._find_param(op,
            #                             "Longitude of natural origin",
            #                             "Longitude of origin",
            #                             "Central meridian",
            #                             "Longitude of projection centre",
            #                             "central_meridian")
            # center_lat = self._find_param(op,
            #                             "Latitude of natural origin",
            #                             "Latitude of projection centre",
            #                             "latitude_of_origin")
            # true_scale_lat = self._find_param(op,
            #                                 "Latitude of true scale",
            #                                 "Latitude of 1st standard parallel",
            #                                 "Latitude of standard parallel",
            #                                 "standard_parallel_1")
            self._center_lon = creator_state.center_lon
            latitude_value = creator_state.latitude
            latitude_choice = creator_state.latitude_choice
            # print(f"center_lat: {center_lat}")
            # print(f"true_scale_lat: {true_scale_lat}")
            # assert not (center_lat == None and true_scale_lat == None), \
            #     "Center Latitude and True Scale Latitude should not both be None."

            # if center_lat is None:
            #     chosen_enum = LatitudeTypes.TRUE_SCALE_LATITUDE
            #     chosen_value = true_scale_lat
            # elif true_scale_lat is None or (center_lat == 0.0 and true_scale_lat == 0.0):
            #     print(f"true_scale_lat is None or 0.0: {true_scale_lat}")
            #     print(f"center_lat is 0.0: {center_lat}")
            #     chosen_enum = LatitudeTypes.CENTRAL_LATITUDE
            #     chosen_value = center_lat
            # elif center_lat > true_scale_lat:
            #     print(f"center_lat > true_scale_lat: { center_lat } > {true_scale_lat}")
            #     chosen_enum = LatitudeTypes.CENTRAL_LATITUDE
            #     chosen_value = center_lat
            # else:
            #     chosen_enum = LatitudeTypes.TRUE_SCALE_LATITUDE
            #     chosen_value = true_scale_lat

            self._ui.ledit_center_lon.setText("" if self._center_lon is None
                                            else str(self._center_lon))
            
            # Save the choice
            self._latitude_choice = latitude_choice

            # Update the combo‑box without re‑entering the slot
            cbox = self._ui.cbox_lat_chooser
            idx  = cbox.findData(latitude_choice)
            if idx != -1:
                cbox.blockSignals(True)
                cbox.setCurrentIndex(idx)
                cbox.blockSignals(False)

            self._ui.ledit_lat_value.setText("" if latitude_value is None
                                            else str(latitude_value))

            self._ui.lbl_center_lon.setEnabled(True)
            self._ui.ledit_center_lon.setEnabled(True)
            self._ui.cbox_lat_chooser.setEnabled(True)
            self._ui.ledit_lat_value.setEnabled(True)
        else:
            # Clear projection fields
            for w in (self._ui.ledit_center_lon,
                    self._ui.ledit_lat_value):
                w.clear()
            self._center_lon = self._latitude = None
            self._ui.lbl_center_lon.setEnabled(False)
            self._ui.ledit_center_lon.setEnabled(False)
            self._ui.cbox_lat_chooser.setEnabled(False)
            self._ui.ledit_lat_value.setEnabled(False)

        # ---------- CRS name ---------------------------------------------
        self._ui.ledit_crs_name.setText(name)
        self._ui.ledit_crs_name.editingFinished.emit()
        self._crs_name = name

        # Remember selection so we can detect future changes
        print(f"changing starting crs_name to: {name}")
        self._current_starting_crs_name = name


        if self._proj_type == ProjectionTypes.POLAR_STEREO:
            if self._latitude_choice == LatitudeTypes.CENTRAL_LATITUDE:
                assert creator_state.polar_stereo_scale is not None
                ledit = self._ui.ledit_pstereo_scale_factor
                ledit.setText(creator_state.polar_stereo_scale)
                ledit.textChanged.emit(creator_state.polar_stereo_scale)
            elif self._latitude_choice == LatitudeTypes.TRUE_SCALE_LATITUDE:
                assert creator_state._polar_stereo_latitude_sign is not None
                cbox = self._ui.cbox_pstereo_sign
                idx = cbox.findData(creator_state._polar_stereo_latitude_sign)
                if idx != -1 :
                    cbox.setCurrentIndex(idx)
                    cbox.currentIndexChanged.emit(idx)


    def _find_param(self, op, *names):
        """
        Return the first matching operation-parameter value (or None) from a
        pyproj CoordinateOperation.  Usage:
            lat_0 = _find_param(op, "Latitude of natural origin",
                                    "Latitude of projection centre")
        """
        lname = {n.lower() for n in names}
        for p in getattr(op, "params", []):
            if p.name.lower() in lname:
                return p.value
        return None

    def _init_cbox_lat_chooser(self):
        """
        Initializes self._ui.cbox_lat_chooser to have all the values in LatitudeTypes. The text shown should be
        the value of the enum and the value of the cbox should be the enum
        
        class LatitudeTypes(Enum):
            CENTRAL_LATITUDE = "Central Latitude"
            TRUE_SCALE_LATITUDE = "True Scale Lat"

        WHen a new cbox item is clicked the function _on_change_lat_choice should be called which sets an instance varialbe
        called self._latitude_choice
        """
        cbox = self._ui.cbox_lat_chooser
        cbox.clear()

        # Add each enum member
        for lat_type in LatitudeTypes:
            cbox.addItem(lat_type.value, lat_type)

        # When the user picks a new item, update self._latitude_choice
        cbox.currentIndexChanged.connect(self._on_change_lat_choice)

        # Initialize to the first entry (if any)
        if cbox.count() > 0:
            # This will call _on_change_lat_choice and set self._latitude_choice
            self._on_change_lat_choice(cbox.currentIndex())
    
    def _init_ledit_lat_value(self):
        """
        Adds a double validator to ledit lat value that is in the 
        """
        validator = QDoubleValidator(self._ui.ledit_lat_value)
        validator.setNotation(QDoubleValidator.StandardNotation)
        validator.setRange(-90.0, 90.0, ALLOWED_DECIMALS)
        self._ui.ledit_lat_value.setValidator(validator)
        self._ui.ledit_lat_value.textChanged.connect(
            self._on_latitude_changed
        )

    def _init_projection_chooser(self):
        proj_cbox = self._ui.cbox_proj_type
        proj_cbox.activated.connect(self._on_switch_proj_type)
        proj_cbox.activated.connect(self._update_units)
        proj_cbox.clear()

        for proj in ProjectionTypes:
            proj_cbox.addItem(proj.value, proj)

        self._update_units()
        self._on_switch_proj_type(0)

    def _update_units(self):
        self._ui.ledit_units.setReadOnly(True)

        proj_type = self._ui.cbox_proj_type.currentData()
        if proj_type in (ProjectionTypes.EQUI_CYLINDRICAL, ProjectionTypes.POLAR_STEREO):
            text = "Meters"
        elif proj_type == ProjectionTypes.NO_PROJECTION:
            text = "Degrees"
        else:
            text = ""
        self._ui.ledit_units.setText(text)

    def _init_shape_chooser(self):
        shape_cbox = self._ui.cbox_shape
        shape_cbox.activated.connect(self._on_switch_shape_type)
        shape_cbox.clear()

        for shape in ShapeTypes:
            shape_cbox.addItem(shape.value, shape)

        self._on_switch_shape_type(0)


    def _init_ellipsoid_params(self):
        # Populate the axis type combo box
        self._ui.cbox_flat_minor.clear()
        for axis_type in EllipsoidAxisType:
            # display text is the enum value, store enum itself as user data
            self._ui.cbox_flat_minor.addItem(axis_type.value, axis_type)

        # Connect combo box signal to slot
        self._ui.cbox_flat_minor.currentIndexChanged.connect(
            self._on_axis_ingest_type_changed
        )

        self._axis_ingest_type = self._ui.cbox_flat_minor.itemData(self._ui.cbox_flat_minor.currentIndex())

        # Configure flat minor value entry with float validator
        flat_validator = QDoubleValidator(self._ui.ledit_flat_minor)
        flat_validator.setNotation(QDoubleValidator.StandardNotation)
        flat_validator.setDecimals(ALLOWED_DECIMALS)
        flat_validator.setBottom(0.0)
        self._ui.ledit_flat_minor.setValidator(flat_validator)
        self._ui.ledit_flat_minor.textChanged.connect(
            self._on_axis_ingestion_value_changed
        )

        # Configure semi-major entry with float validator
        semi_validator = QDoubleValidator(self._ui.ledit_semi_major)
        semi_validator.setNotation(QDoubleValidator.StandardNotation)
        semi_validator.setDecimals(ALLOWED_DECIMALS)
        semi_validator.setBottom(0.1)
        self._ui.ledit_semi_major.setValidator(semi_validator)
        self._ui.ledit_semi_major.textChanged.connect(
            self._on_semi_major_changed
        )

    # def _init_true_scale_lat_ledit(self):
    #     validator = QDoubleValidator(self._ui.ledit_true_scale_lat)
    #     validator.setNotation(QDoubleValidator.StandardNotation)
    #     validator.setRange(-90.0, 90.0, ALLOWED_DECIMALS)
    #     self._ui.ledit_true_scale_lat.setValidator(validator)
    #     self._ui.ledit_true_scale_lat.textChanged.connect(
    #         self._on_true_scale_lat_changed
    #     )

    # def _init_center_latitude_ledit(self):
    #     validator = QDoubleValidator(self._ui.ledit_center_lat)
    #     validator.setNotation(QDoubleValidator.StandardNotation)
    #     validator.setRange(-90.0, 90.0, ALLOWED_DECIMALS)
    #     self._ui.ledit_center_lat.setValidator(validator)
    #     self._ui.ledit_center_lat.textChanged.connect(
    #         self._on_center_lat_changed
    #     )

    def _init_center_longitude_ledit(self):
        validator = QDoubleValidator(self._ui.ledit_center_lon)
        validator.setNotation(QDoubleValidator.StandardNotation)
        validator.setRange(-180.0, 180.0, ALLOWED_DECIMALS)
        self._ui.ledit_center_lon.setValidator(validator)
        self._ui.ledit_center_lon.textChanged.connect(
            self._on_center_lon_changed
        )

    def _init_lon_meridian_ledit(self):
        validator = QDoubleValidator(self._ui.ledit_prime_meridian)
        validator.setNotation(QDoubleValidator.StandardNotation)
        validator.setRange(-180.0, 180.0, ALLOWED_DECIMALS)
        self._ui.ledit_prime_meridian.setValidator(validator)
        self._ui.ledit_prime_meridian.textChanged.connect(
            self._on_lon_meridian_changed
        )

    # def _init_lat_equator_ledit(self):
    #     validator = QDoubleValidator(self._ui.ledit_equator)
    #     validator.setNotation(QDoubleValidator.StandardNotation)
    #     validator.setRange(-90.0, 90.0, ALLOWED_DECIMALS)
    #     self._ui.ledit_equator.setValidator(validator)
    #     self._ui.ledit_equator.textChanged.connect(
    #         self._on_lat_equator_changed
    #     )

    def _init_crs_name(self):
        regex = QRegExp(r"^[A-Za-z0-9_]+$")
        validator = QRegExpValidator(regex, self._ui.ledit_crs_name)
        self._ui.ledit_crs_name.setValidator(validator)
        self._ui.ledit_crs_name.textEdited.connect(
            self._on_crs_name_changed
        )


    # region Slots


    def _on_change_lat_choice(self, index: int) -> None:
        """
        Slot called when the latitude-type combo box changes.
        Stores the chosen LatitudeTypes enum in self._latitude_choice.
        """
        self._latitude_choice = self._ui.cbox_lat_chooser.itemData(index)
        self._update_extra_polar_stereo_params_display()

    def _on_true_scale_lat_changed(self, text: str) -> None:
        """Slot for when the true scale latitude QLineEdit text changes."""
        try:
            self._true_scale_lat = float(text)
        except ValueError:
            # empty or invalid text → clear or leave as None
            self._true_scale_lat = None

    def _on_latitude_changed(self, text: str) -> None:
        """Slot for when the center latitude QLineEdit text changes."""
        try:
            self._latitude = float(text)
        except ValueError:
            self._latitude = None

    def _on_center_lon_changed(self, text: str) -> None:
        """Slot for when the center longitude QLineEdit text changes."""
        try:
            self._center_lon = float(text)
        except ValueError:
            self._center_lon = None

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

        needs_params = (self._proj_type != ProjectionTypes.NO_PROJECTION)

        for widget in (
            self._ui.ledit_center_lon,
            self._ui.cbox_lat_chooser,
            self._ui.ledit_lat_value,
        ):
            widget.setEnabled(needs_params)
        self._update_extra_polar_stereo_params_display()

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

    def _create_crs(self):
        # --- 1. Basic validation -------------------------------------------------
        if self._crs_name is None:
            QMessageBox.warning(self, self.tr("Missing value"),
                                self.tr("Please supply a name for the CRS."))
            return

        if self._semi_major_value is None or self._axis_ingestion_value is None:
            QMessageBox.warning(self, self.tr("Missing value"),
                                self.tr("Please supply the semi-major axis or the "
                                        "semi-minor axis / inverse flattening."))
            return
    
        if (self._shape_type == ShapeTypes.ELLIPSOID and
                (self._axis_ingestion_value is None or self._axis_ingest_type is None)):
            QMessageBox.warning(self, self.tr("Missing value"),
                                self.tr("For an ellipsoid you must fill the second axis\n"
                                "value and choose whether it is the semi-minor axis\n"
                                "or the inverse flattening."))
            return

        # Safe defaults if the user left them blank
        if self._proj_type != ProjectionTypes.NO_PROJECTION and \
            (self._lon_meridian is None or self._latitude is None or \
            self._center_lon) is None:
            QMessageBox.warning(self, self.tr("Missing value"),
                                self.tr("When doing a projection, the prime meridian, center latitude,\n"
                                        "center longitude, and latitude of true scale must be set. One\n"
                                        "of them is not set."))
            return

        lon_0 = self._lon_meridian if self._lon_meridian is not None else 0.0

        a = self._semi_major_value
        if self._shape_type == ShapeTypes.SPHEROID:
            inv_f = 0.0  # sphere
        else:
            if self._axis_ingest_type == EllipsoidAxisType.SEMI_MINOR:
                b = self._axis_ingestion_value
                inv_f = a / (a - b) if a != b else 0.0
            else:  # inverse flattening entered directly
                inv_f = self._axis_ingestion_value

        _internal_use_proj = True

        if _internal_use_proj:
            # Ellipsoid description for proj

            if inv_f == 0.0:
                ellps_part = f"+R={a}"
            else:
                ellps_part = f"+a={a} +rf={inv_f}"

            base = f"{ellps_part} +pm={self._lon_meridian} +no_defs"

            if self._proj_type == ProjectionTypes.NO_PROJECTION:
                proj_str = f"+proj=longlat {base} +units=deg"
            elif self._proj_type == ProjectionTypes.EQUI_CYLINDRICAL:
                if self._latitude_choice == LatitudeTypes.CENTRAL_LATITUDE:
                    proj_str = (f"+proj=eqc +lon_0={self._center_lon} +lat_0={self._latitude} "
                                f"{base}")
                else:
                    proj_str = (f"+proj=eqc +lon_0={self._center_lon} +lat_ts={self._latitude} "
                                f"{base}")
                    
            elif self._proj_type == ProjectionTypes.POLAR_STEREO:
                if self._latitude_choice == LatitudeTypes.CENTRAL_LATITUDE:
                    if self._polar_stereo_scale is None:
                        QMessageBox.warning(self, self.tr("Missing value"),
                                            self.tr("The scale factor value is None. Please enter\n"
                                            "a scale factor value."))
                        return

                    proj_str = (f"+proj=stere +lat_0={self._latitude} +lon_0={self._center_lon} "
                                f"+k={self._polar_stereo_scale} +x_0=0 +y_0=0 {base}")
                else:
                    if self._polar_stereo_latitude_sign is None:
                        QMessageBox.warning(self, self.tr("Missing value"),
                                            self.tr("The central latitude sign is None. Please select\n"
                                            "a central latitude sign."))
                        return
                    proj_str = (f"+proj=stere +lon_0={self._center_lon} +lat_0={self._polar_stereo_latitude_sign}90 "
                                f"+lat_ts={self._latitude} +x_0=0 +y_0=0 {base}")
                    print(f"proj str: {proj_str}")
                    
            else:
                QMessageBox.critical(self, "Error",
                                    f"Unknown projection type: {self._proj_type}")
                return

            pyproj_crs = pyproj.CRS.from_proj4(proj_str)
            self._new_crs = osr.SpatialReference()
            wkt = pyproj_crs.to_wkt()
            print(f"wkt: {wkt}")
            self._new_crs.ImportFromWkt(pyproj_crs.to_wkt())
        # else:
        #     gcs = osr.SpatialReference()
        #     gcs.SetGeogCS(self._crs_name or "USER_GCS",
        #                 "USER_DATUM",
        #                 "USER_ELLIPSOID",
        #                 a, inv_f,
        #                 "Prime_Meridian", lon_0,
        #                 "degree", 0.0174532925199433)  

        #     # Axis order: lon/lat as in “traditional GIS” gives us consistent
        #     # lon/lat ordering
        #     gcs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)

        #     if self._proj_type == ProjectionTypes.NO_PROJECTION:
        #         srs = gcs      # geographic only
        #     else:
        #         srs: osr.SpatialReference = gcs.Clone()
        #         srs.SetProjCS(self._crs_name or "USER_PCS")

        #         if self._proj_type == ProjectionTypes.EQUI_CYLINDRICAL:
        #             if self._latitude_choice == LatitudeTypes.CENTRAL_LATITUDE:
        #                 # SetEquirectangular2​(double clat, double clong, double pseudostdparallellat, double fe, double fn)
        #                 srs.SetEquirectangular2(self._latitude, self._center_lon, 0, 0, 0)
        #             else:
        #                 srs.SetEquirectangular2(0.0, self._center_lon, self._latitude, 0, 0)
        #         elif self._proj_type == ProjectionTypes.POLAR_STEREO:
        #             # lat_ts: true‑scale latitude (use pole if user left blank)
        #             # lat_ts = 90.0 if lat_0 >= 0 else -90.0
        #             # SetPS​(double clat, double clong, double scale, double fe, double fn)
        #             if self._latitude_choice == LatitudeTypes.CENTRAL_LATITUDE:
        #                 srs.SetPS(self._latitude, self._center_lon, 1.0, 0, 0)
        #             else:
        #                 srs.SetPS(self._latitude, self._center_lon, 1.0, 0, 0)
        #         else:
        #             raise RuntimeError("Unsupported projection choice")
        #     self._new_crs = srs
            
        self._new_crs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)

        self._app_state.add_user_created_crs(self._crs_name, self._new_crs, self._export_creator_state())

        self._update_user_created_crs_cbox()

        self._switch_user_crs_cbox_selection(self._crs_name)

    def _export_creator_state(self) -> CrsCreatorState:
        crs_creator_state = CrsCreatorState(
            lon_meridian=self._lon_meridian,
            proj_type=self._proj_type,
            axis_ingest_type=self._axis_ingest_type,
            axis_ingestion_value=self._axis_ingestion_value,
            semi_major_value=self._semi_major_value,
            latitude_choice=self._latitude_choice,
            latitude=self._latitude,
            center_lon=self._center_lon,
            polar_stereo_scale=self._polar_stereo_scale,
            polar_stereo_latitude_sign=self._polar_stereo_latitude_sign,
        )
        return crs_creator_state

    def accept(self):
        self._create_crs()

        super().accept()
