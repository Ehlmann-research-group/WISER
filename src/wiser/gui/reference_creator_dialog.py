import os
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
        self._lon_meridian: Optional[float] = None
        self._lat_equator: Optional[float] = None
        self._proj_type : Optional[ProjectionTypes]
        self._axis_ingest_type: Optional[Ellipsoid_Axis_Type] = Ellipsoid_Axis_Type.SEMI_MINOR
        self._axis_ingestion_value: Optional[float] = None
        self._semi_major_value: Optional[float] = None
        self._true_scale_lat: Optional[float] = None
        self._center_lat: Optional[float] = None
        self._center_lon: Optional[float] = None

        # save current name so we can tell if the user picks something new later
        self._current_starting_crs_name: Optional[str] = None

        # Initialize UI
        self._init_user_created_crs()
        self._init_projection_chooser()
        self._init_shape_chooser()
        self._init_ellipsoid_params()
        self._init_lon_meridian_ledit()
        self._init_crs_name()
        self._init_true_scale_lat_ledit()
        self._init_center_latitude_ledit()
        self._init_center_longitude_ledit()

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
        Populate the “Starting CRS” combo box with user‑defined CRS objects that
        were persisted in ApplicationState.  Each entry’s *text* is the dict key
        and the *userData* is the osr.SpatialReference itself.
        """
        cbox = self._ui.cbox_user_crs
        cbox.clear()

        self._user_created_crs = self._app_state.get_user_created_crs() or {}

        if not self._user_created_crs:            # nothing stored yet
            cbox.setEnabled(False)
            return

        cbox.setEnabled(True)

        for name, srs in self._user_created_crs.items():
            cbox.addItem(name, srs)

        # When the user chooses an entry we call _on_starting_crs_changed and pass
        # the *name* (key) via a small lambda wrapper.
        cbox.activated.connect(
            lambda idx: self._on_starting_crs_changed(cbox.itemText(idx))
        )

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
        if not name                       \
        or name == self._current_starting_crs_name:
            return                        # same choice – nothing to do

        srs: osr.SpatialReference = self._user_created_crs.get(name)
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
        pm_lon = pycrs.prime_meridian.longitude or 0.0
        self._ui.ledit_prime_meridian.setText(str(pm_lon))
        self._lon_meridian = pm_lon

        eq_lat = 0.0   # latitude of the “equator/origin”; most CRSs use 0
        self._ui.ledit_equator.setText(str(eq_lat))
        self._lat_equator = eq_lat

        # ---------- Shape / ellipsoid parameters -------------------------
        a = pycrs.ellipsoid.semi_major_metre
        inv_f = pycrs.ellipsoid.inverse_flattening
        spheroid = (inv_f == 0.0 or not pycrs.ellipsoid.is_ellipsoidal)

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
        self._ui.ledit_semi_major.setText(f"{a:.{ALLOWED_DECIMALS}f}")
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
            axis_idx = axis_cbox.findData(Ellipsoid_Axis_Type.INVERSE_FLATTENING)
            axis_cbox.setCurrentIndex(axis_idx)
            self._axis_ingest_type = Ellipsoid_Axis_Type.INVERSE_FLATTENING
            self._axis_ingestion_value = inv_f
            self._ui.ledit_flat_minor.setEnabled(True)
            self._ui.cbox_flat_minor.setEnabled(True)
            self._ui.ledit_flat_minor.setText(f"{inv_f:.{ALLOWED_DECIMALS}f}")

        # ---------- Projection (or none) ---------------------------------
        if pycrs.is_geographic:
            proj_type = ProjectionTypes.NO_PROJECTION
        else:
            m = pycrs.coordinate_operation.method_name.lower()
            if "equidistant cylindrical" in m or "equirectangular" in m:
                proj_type = ProjectionTypes.EQUI_CYLINDRICAL
            elif "polar stereographic" in m or "stereographic" in m:
                proj_type = ProjectionTypes.POLAR_STEREO
            else:
                proj_type = ProjectionTypes.NO_PROJECTION  # fallback

        proj_cbox = self._ui.cbox_proj_type
        proj_idx = proj_cbox.findData(proj_type)
        proj_cbox.setCurrentIndex(proj_idx)
        self._proj_type = proj_type
        self._update_units()                             # refresh “Units” display

        # Projection‑specific parameters
        if not pycrs.is_geographic:
            op = pycrs.coordinate_operation
            self._center_lon = self._find_param(op,
                                        "Longitude of natural origin",
                                        "Central meridian",
                                        "Longitude of projection centre")
            self._center_lat = self._find_param(op,
                                        "Latitude of natural origin",
                                        "Latitude of projection centre")
            self._true_scale_lat = self._find_param(op,
                                            "Latitude of true scale",
                                            "Latitude of 1st standard parallel")

            self._ui.ledit_center_long.setText("" if self._center_lon is None
                                            else str(self._center_lon))
            self._ui.ledit_center_lat.setText("" if self._center_lat is None
                                            else str(self._center_lat))
            self._ui.ledit_true_scale_lat.setText("" if self._true_scale_lat is None
                                                else str(self._true_scale_lat))
        else:
            # Clear projection fields
            for w in (self._ui.ledit_center_long,
                    self._ui.ledit_center_lat,
                    self._ui.ledit_true_scale_lat):
                w.clear()
            self._center_lon = self._center_lat = self._true_scale_lat = None

        # ---------- CRS name ---------------------------------------------
        self._ui.ledit_crs_name.setText(name)
        self._crs_name = name

        # Remember selection so we can detect future changes
        self._current_starting_crs_name = name

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
        for axis_type in Ellipsoid_Axis_Type:
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

    def _init_true_scale_lat_ledit(self):
        validator = QDoubleValidator(self._ui.ledit_true_scale_lat)
        validator.setNotation(QDoubleValidator.StandardNotation)
        validator.setRange(-90.0, 90.0, ALLOWED_DECIMALS)
        self._ui.ledit_true_scale_lat.setValidator(validator)
        self._ui.ledit_true_scale_lat.textChanged.connect(
            self._on_true_scale_lat_changed
        )

    def _init_center_latitude_ledit(self):
        validator = QDoubleValidator(self._ui.ledit_center_lat)
        validator.setNotation(QDoubleValidator.StandardNotation)
        validator.setRange(-180.0, 180.0, ALLOWED_DECIMALS)
        self._ui.ledit_center_lat.setValidator(validator)
        self._ui.ledit_center_lat.textChanged.connect(
            self._on_center_lat_changed
        )

    def _init_center_longitude_ledit(self):
        validator = QDoubleValidator(self._ui.ledit_center_long)
        validator.setNotation(QDoubleValidator.StandardNotation)
        validator.setRange(-180.0, 180.0, ALLOWED_DECIMALS)
        self._ui.ledit_center_long.setValidator(validator)
        self._ui.ledit_center_long.textChanged.connect(
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

    def _init_lat_equator_ledit(self):
        validator = QDoubleValidator(self._ui.ledit_equator)
        validator.setNotation(QDoubleValidator.StandardNotation)
        validator.setRange(-90.0, 90.0, ALLOWED_DECIMALS)
        self._ui.ledit_equator.setValidator(validator)
        self._ui.ledit_equator.textChanged.connect(
            self._on_lat_equator_changed
        )

    def _init_crs_name(self):
        regex = QRegExp(r"^[A-Za-z0-9_]+$")
        validator = QRegExpValidator(regex, self._ui.ledit_crs_name)
        self._ui.ledit_crs_name.setValidator(validator)
        self._ui.ledit_crs_name.textEdited.connect(
            self._on_crs_name_changed
        )


    # Region Slots


    def _on_true_scale_lat_changed(self, text: str) -> None:
        """Slot for when the true scale latitude QLineEdit text changes."""
        try:
            self._true_scale_lat = float(text)
        except ValueError:
            # empty or invalid text → clear or leave as None
            self._true_scale_lat = None

    def _on_center_lat_changed(self, text: str) -> None:
        """Slot for when the center latitude QLineEdit text changes."""
        try:
            self._center_lat = float(text)
        except ValueError:
            self._center_lat = None

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
            self._ui.ledit_center_long,
            self._ui.ledit_center_lat,
            self._ui.ledit_true_scale_lat,
        ):
            widget.setEnabled(needs_params)


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

    def accept(self):
        # --- 1. Basic validation -------------------------------------------------
        if self._semi_major_value is None:
            QMessageBox.warning(self, self.tr("Missing value"),
                                self.tr("Please supply the semi-major axis / radius."))
            return
    
        if (self._shape_type == ShapeTypes.ELLIPSOID and
                (self._axis_ingestion_value is None or self._axis_ingest_type is None)):
            QMessageBox.warning(self, self.tr("Missing value"),
                                self.tr("For an ellipsoid you must fill the second axis\n"
                                "value and choose whether it is the semi-minor axis\n"
                                "or the inverse flattening."))
            return

        # # Safe defaults if the user left them blank
        # if self._proj_type != ProjectionTypes.NO_PROJECTION and \
        #     (self._lon_meridian is None or self._center_lat is None or \
        #     self._center_lon is None or self._true_scale_lat is None):
        #     QMessageBox.warning(self, self.tr("Missing value"),
        #                         self.tr("When doing a projection, the prime meridian, center latitude,\n"
        #                                 "center longitude, and latitude of true scale must be set. One\n"
        #                                 "of them is not set."))

        lon_0 = self._lon_meridian if self._lon_meridian is not None else 0.0
        lat_0 = self._lat_equator if self._lat_equator is not None else 0.0
        center_lat = self._center_lat if self._center_lat is not None else 0.0
        center_lon = self._center_lon if self._center_lon is not None else 0.0
        true_scale_lat = self._true_scale_lat if self._true_scale_lat is not None else 0.0


        a = self._semi_major_value                                           # semi-major
        if self._shape_type == ShapeTypes.SPHEROID:
            inv_f = 0.0                                                      # sphere
            print(f"its a spheroid!!")
        else:
            if self._axis_ingest_type == Ellipsoid_Axis_Type.SEMI_MINOR:
                b = self._axis_ingestion_value                               # semi‑minor
                inv_f = a / (a - b) if a != b else 0.0
                print(f"in else if inv: {inv_f}")
            else:  # inverse flattening entered directly
                inv_f = self._axis_ingestion_value
                print(f"in else if inv: {inv_f}")

        gcs = osr.SpatialReference()
        gcs.SetGeogCS(self._crs_name or "USER_GCS",
                      "USER_DATUM",
                      "USER_ELLIPSOID",
                      a, inv_f,
                      "Prime_Meridian", lon_0,
                      "degree", 0.0174532925199433)  

        # Axis order: lon/lat as in “traditional GIS” gives us consistent
        # lon/lat ordering
        gcs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)

        # if self._proj_type == ProjectionTypes.NO_PROJECTION:
        #     srs = gcs      # geographic only
        # else:
        #     srs = gcs.Clone()
        #     srs.SetProjCS(self._crs_name or "USER_PCS")

        #     if self._proj_type == ProjectionTypes.EQUI_CYLINDRICAL:
        #         # SetEquirectangular2​(double clat, double clong, double pseudostdparallellat, double fe, double fn)
        #         srs.SetEquirectangular2(center_lat, center_lon, true_scale_lat, 0, 0)
        #     elif self._proj_type == ProjectionTypes.POLAR_STEREO:
        #         # lat_ts: true‑scale latitude (use pole if user left blank)
        #         # lat_ts = 90.0 if lat_0 >= 0 else -90.0
        #         # SetPS​(double clat, double clong, double scale, double fe, double fn)
        #         srs.SetPS(center_lat, center_lat, true_scale_lat, 0, 0)
        #     else:
        #         raise RuntimeError("Unsupported projection choice")

        # Ellipsoid description for proj
        print(f"inv_f: {inv_f}")

        if inv_f == 0.0:
            print(f"Making a circle!")
            ellps_part = f"+R={a}"
        else:
            ellps_part = f"+a={a} +rf={inv_f}"
            print(f"ellipse made: {ellps_part}")

        base = f"{ellps_part} +pm={lon_0} +no_defs"

        if self._proj_type == ProjectionTypes.NO_PROJECTION:
            proj_str = f"+proj=longlat {base} +units=deg"
        elif self._proj_type == ProjectionTypes.EQUI_CYLINDRICAL:
            print(f"self._true_scale_lat: {self._true_scale_lat}")
            if self._true_scale_lat is None:
                proj_str = (f"+proj=eqc +lon_0={self._center_lon} +lat_0={self._center_lat} "
                            f"{base}")
            else:
                proj_str = (f"+proj=eqc +lon_0={self._center_lon} +lat_ts={self._true_scale_lat} "
                            f"{base}")
                
        elif self._proj_type == ProjectionTypes.POLAR_STEREO:
            # proj_str = (f"+proj=stere +lat_0={lat_0} +lon_0={lon_0} +lat_ts={self._true_scale_lat}"
            #             f"+k=1 +x_0=0 +y_0=0 {ellps_part} "
            #             f"+no_defs")
            print(f"self._true_scale_lat: {self._true_scale_lat}")
            if self._true_scale_lat is None:
                proj_str = (f"+proj=stere +lat_0={self._center_lat} +lon_0={self._center_lon} "
                            f"+k=1 +x_0=0 +y_0=0 {base}")
            else:
                proj_str = (f"+proj=stere +lon_0={self._center_lon} +lat_ts={self._true_scale_lat} "
                            f"+k=1 +x_0=0 +y_0=0 {base}")
                
        else:
            QMessageBox.critical(self, "Error",
                                 f"Unknown projection type: {self._proj_type}")
            return
        print(f"proj_str: {proj_str}")
        pyproj_crs = pyproj.CRS.from_proj4(proj_str)
        self._new_crs = osr.SpatialReference()
        wkt = pyproj_crs.to_wkt()
        print(f"wkt: {wkt}")
        print(f"type: {type(wkt)}")
        self._new_crs.ImportFromWkt(pyproj_crs.to_wkt())

        # If everything worked we keep the OSR object
        # self._new_crs = srs
        print(f"self._new_crs wkt: {self._new_crs.ExportToWkt()}")
        print(f"self._new_crs: {self._new_crs}")
        print(f"type self._new_crs: {type(self._new_crs)}")
        super().accept()
