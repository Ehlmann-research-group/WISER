import os
import sys

from typing import List, Optional

from PySide6.QtCore import *
from PySide6.QtGui import *
from PySide6.QtWidgets import *

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
        shape_type: Optional[ShapeTypes] = None,
        source_wkt: Optional[str] = None,
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
        self._shape_type = shape_type
        # When set, this CRS was built by pasting a raw CRS string on the "From
        # CRS String" tab rather than from the parameter fields.  The stored WKT
        # lets the dialog reload it into the string tab instead of trying (and
        # failing) to drive the parameter widgets.
        self._source_wkt = source_wkt

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

    @property
    def shape_type(self) -> Optional[str]:
        return self._shape_type

    @property
    def source_wkt(self) -> Optional[str]:
        return self._source_wkt

    @property
    def is_string_origin(self) -> bool:
        return self._source_wkt is not None


class ReferenceCreatorDialog(QDialog):
    def __init__(self, app_state: ApplicationState, parent=None):
        super().__init__(parent=parent)
        self._app_state = app_state

        # Set up the UI state
        self._ui = Ui_ReferenceSystemCreator()
        self._ui.setupUi(self)

        # Init variables
        self._lon_meridian: Optional[float] = None
        self._proj_type: Optional[ProjectionTypes]
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
        self._crs_name: Optional[str] = None

        # The last CRS successfully parsed on the "From CRS String" tab, if any.
        self._validated_string_srs: Optional[osr.SpatialReference] = None

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
        self._init_crs_string_tab()

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

        # Initialize the Scale Factor Line Edit
        validator = QDoubleValidator(self._ui.ledit_pstereo_scale_factor)
        validator.setNotation(QDoubleValidator.StandardNotation)
        validator.setRange(0, MAX_SCALE_FACTOR, ALLOWED_DECIMALS)
        self._ui.ledit_pstereo_scale_factor.setValidator(validator)
        self._ui.ledit_pstereo_scale_factor.textChanged.connect(self._on_stereo_scale_factor_changed)
        self._on_stereo_scale_factor_changed(self._ui.ledit_pstereo_scale_factor.text())

    def _on_stereo_scale_factor_changed(self, text: str):
        try:
            self._polar_stereo_scale = float(text)
        except ValueError:
            self._polar_stereo_scale = None

    def _on_stereo_pos_neg_changed(self, index: int) -> None:
        self._polar_stereo_latitude_sign = self._ui.cbox_pstereo_sign.itemData(index)

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
        # Resolve the reset‑button name used in the .ui file
        reset_btn = self._ui.btn_reset_fields
        if reset_btn is None:
            raise AttributeError("Reset button not found in UI")

        reset_btn.clicked.connect(self._on_reset_clicked)

        # Do one reset immediately so the dialog starts in a clean state
        self._on_reset_clicked()

    def _on_reset_clicked(self):
        """Slot that really performs the reset."""
        # Clear internal values
        self._axis_ingestion_value = None
        self._semi_major_value = None
        self._latitude = None
        self._center_lon = None
        self._lon_meridian = None
        self._current_starting_crs_name = NO_CRS_NAME

        # Clear the editor widgets
        for le in (
            self._ui.ledit_flat_minor,
            self._ui.ledit_semi_major,
            self._ui.ledit_lat_value,
            self._ui.ledit_center_lon,
            self._ui.ledit_prime_meridian,
            self._ui.ledit_crs_name,
        ):
            le.clear()

        # Clear the "From CRS String" tab too, if it has been built yet.
        if hasattr(self._ui, "pedit_crs_string"):
            self._validated_string_srs = None
            self._ui.pedit_crs_string.clear()
            self._ui.ledit_string_crs_name.clear()
            self._ui.pedit_crs_string_result.clear()
            self._ui.btn_add_crs_string.setEnabled(False)

        # Put the "Starting CRS" combo back to "(None)"
        cbox = self._ui.cbox_user_crs
        none_idx = cbox.findText(self.tr(NO_CRS_NAME))
        if none_idx == -1:  # fallback: last entry
            none_idx = cbox.count() - 1
        if none_idx >= 0:
            cbox.blockSignals(True)
            cbox.setCurrentIndex(none_idx)
            cbox.blockSignals(False)

    def _init_create_crs_button(self):
        """
        Initialize the button that creates the CRS from all the information
        in this dialog.
        """
        create_btn = self._ui.btn_create_crs
        if create_btn is None:
            raise AttributeError("Create-CRS button not found in UI")

        create_btn.clicked.connect(self._create_crs)

    # region CRS-string tab

    def _init_crs_string_tab(self):
        """
        Wire up the "From CRS String" tab, where a user pastes a raw CRS
        definition (WKT, PROJ, or an authority code such as ``EPSG:4326``) and
        checks whether WISER can load it.
        """
        # Restrict the name to the same character set as the parameter tab.
        regex = QRegularExpression(r"^[A-Za-z0-9_]+$")
        validator = QRegularExpressionValidator(regex, self._ui.ledit_string_crs_name)
        self._ui.ledit_string_crs_name.setValidator(validator)

        self._ui.btn_validate_crs_string.clicked.connect(self._on_validate_crs_string)
        self._ui.btn_add_crs_string.clicked.connect(self._on_add_crs_string)

        # Any edit to the pasted text invalidates the previous validation.
        self._ui.pedit_crs_string.textChanged.connect(self._on_crs_string_text_changed)

        self._validated_string_srs = None
        self._ui.btn_add_crs_string.setEnabled(False)

    def _build_srs_from_string(self, text: str):
        """
        Try to build an ``osr.SpatialReference`` from a user-supplied CRS
        string.  Returns ``(srs, None)`` on success or ``(None, error)`` on
        failure, where ``error`` is a human-readable message.
        """
        srs = osr.SpatialReference()
        try:
            # SetFromUserInput accepts WKT (1 & 2), PROJ strings, and authority
            # codes like "EPSG:4326" - the broadest entry point GDAL offers.
            err = srs.SetFromUserInput(text)
        except Exception as exc:  # GDAL may raise when exceptions are enabled
            return None, str(exc)

        if err != 0:
            return None, "GDAL could not interpret the given CRS string."

        # Match the axis convention the parameter path (and the georeferencer)
        # uses, so the CRS behaves the same everywhere in WISER.
        srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        return srs, None

    def _describe_srs(self, srs: osr.SpatialReference) -> str:
        """Build a short success summary for a validated CRS."""
        name = srs.GetName() or "(unnamed)"
        if srs.IsProjected():
            kind = "Projected CRS"
        elif srs.IsGeographic():
            kind = "Geographic CRS"
        else:
            kind = "CRS"

        lines = [
            "Valid — WISER can load this CRS.",
            "",
            f"Name: {name}",
            f"Type: {kind}",
        ]
        authority = srs.GetAuthorityName(None)
        code = srs.GetAuthorityCode(None)
        if authority and code:
            lines.append(f"Authority: {authority}:{code}")
        return "\n".join(lines)

    def _set_string_result(self, text: str, ok: bool) -> None:
        """Show ``text`` in the result box, coloured green (ok) or red (error)."""
        self._ui.pedit_crs_string_result.setPlainText(text)
        color = "#137333" if ok else "#a50e0e"
        self._ui.pedit_crs_string_result.setStyleSheet(f"color: {color};")

    def _on_crs_string_text_changed(self) -> None:
        """Invalidate any prior validation whenever the pasted text changes."""
        self._validated_string_srs = None
        self._ui.btn_add_crs_string.setEnabled(False)

    def _on_validate_crs_string(self) -> None:
        """Validate the pasted CRS string and report success or the error."""
        self._validated_string_srs = None
        self._ui.btn_add_crs_string.setEnabled(False)

        text = self._ui.pedit_crs_string.toPlainText().strip()
        if not text:
            self._set_string_result("Please paste a CRS string to validate.", ok=False)
            return

        srs, error = self._build_srs_from_string(text)
        if srs is None:
            self._set_string_result(f"Invalid — WISER cannot load this CRS.\n\n{error}", ok=False)
            return

        self._validated_string_srs = srs
        self._ui.btn_add_crs_string.setEnabled(True)
        self._set_string_result(self._describe_srs(srs), ok=True)

    def _on_add_crs_string(self) -> bool:
        """
        Persist the validated string CRS into ``ApplicationState`` under the
        name in the name field.  Returns ``True`` on success.
        """
        name = self._ui.ledit_string_crs_name.text().strip()
        if not name:
            QMessageBox.warning(
                self,
                self.tr("Missing value"),
                self.tr("Please supply a name for the CRS."),
            )
            return False

        srs = self._validated_string_srs
        if srs is None:
            # The user may click Add without validating first - try now.
            text = self._ui.pedit_crs_string.toPlainText().strip()
            srs, error = self._build_srs_from_string(text)
            if srs is None:
                self._set_string_result(f"Invalid — WISER cannot load this CRS.\n\n{error}", ok=False)
                QMessageBox.warning(
                    self,
                    self.tr("Invalid CRS"),
                    self.tr("The CRS string could not be validated. Fix it and try again."),
                )
                return False
            self._validated_string_srs = srs
            self._set_string_result(self._describe_srs(srs), ok=True)

        state = self._export_creator_state(source_wkt=srs.ExportToWkt())
        self._app_state.add_user_created_crs(name, srs, state)
        self._update_user_created_crs_cbox()
        self._switch_user_crs_cbox_selection(name)
        return True

    # endregion

    def _init_user_created_crs(self):
        """
        Populate the "Starting CRS" combo box with user-defined CRS objects that
        were persisted in ApplicationState.  Each entry's *text* is the dict key
        and the *userData* is the osr.SpatialReference itself.
        """
        cbox = self._ui.cbox_user_crs
        self._update_user_created_crs_cbox()
        cbox.activated.connect(lambda idx: self._on_starting_crs_changed(cbox.itemText(idx)))

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
            current_crs_name = cbox.itemText(current_index)
        else:
            # This occurs initially, when the combobox is empty and has no
            # selection.  Make sure the "(no data)" option is selected by the
            # end of this process.
            current_index = 0
            current_crs_name = ""

        new_index = None
        cbox.clear()

        if num_crs > 0:
            for index, name in enumerate(sorted(list(app_state.get_user_created_crs().keys()))):
                crs = self._app_state.get_user_created_crs()[name][0]
                cbox.addItem(name, crs)

                if name == current_crs_name:
                    new_index = index

            cbox.insertSeparator(num_crs)
            cbox.addItem(self.tr("(None)"), -1)
            if current_crs_name == "":
                new_index = cbox.count() - 1
        else:
            # No datasets yet
            cbox.addItem(self.tr("(None)"), -1)
            if current_crs_name == "":
                new_index = 0

        if new_index is None:
            if num_crs > 0:
                new_index = min(current_index, num_crs - 1)
            else:
                new_index = 0

        cbox.setCurrentIndex(new_index)

    def _on_starting_crs_changed(self, name: str):
        if not name or name == self._current_starting_crs_name:
            return

        if name == NO_CRS_NAME:
            self._current_starting_crs_name = NO_CRS_NAME
            return

        srs: osr.SpatialReference = self._app_state.get_user_created_crs().get(name)[0]
        creator_state: CrsCreatorState = self._app_state.get_user_created_crs().get(name)[1]
        if srs is None:  # shouldn't happen
            return

        # Ask before removing whatever the user already entered
        if (
            QMessageBox.question(
                self,
                "Replace current parameters?",
                "Loading '{0}' will overwrite all fields you have entered so far.\n Continue?".format(name),
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            == QMessageBox.No
        ):
            # Re‑select the previous item (if any) so the user sees no change
            cbox = self._ui.cbox_user_crs
            if self._current_starting_crs_name:
                old_idx = cbox.findText(self._current_starting_crs_name)
                if old_idx != -1:
                    cbox.blockSignals(True)
                    cbox.setCurrentIndex(old_idx)
                    cbox.blockSignals(False)
            return

        # A CRS that was created by pasting a raw string has no parameter state
        # to reload into the form.  Show it on the "From CRS String" tab
        # instead of driving the parameter widgets (which would fail).
        if creator_state is not None and creator_state.is_string_origin:
            self._ui.tabWidget.setCurrentWidget(self._ui.page_string)
            self._ui.ledit_string_crs_name.setText(name)
            self._ui.pedit_crs_string.setPlainText(creator_state.source_wkt)
            self._on_validate_crs_string()
            self._crs_name = name
            self._current_starting_crs_name = name
            return

        # Convert to pyproj for convenient interrogation
        pycrs = pyproj.CRS.from_wkt(srs.ExportToWkt())

        # Prime meridian & "equator" (lat_0)
        pm_lon = creator_state.lon_meridian
        self._ui.ledit_prime_meridian.setText(str(pm_lon))
        self._lon_meridian = pm_lon

        # Shape / ellipsoid parameters
        a = creator_state.semi_major_value
        spheroid = creator_state.shape_type == ShapeTypes.SPHEROID

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
            # Sphere - disable the flat/minor widgets
            self._ui.ledit_flat_minor.clear()
            self._ui.cbox_flat_minor.setEnabled(False)
            self._ui.ledit_flat_minor.setEnabled(False)
            self._axis_ingest_type = None
            self._axis_ingestion_value = None
        else:
            if creator_state.axis_ingest_type == EllipsoidAxisType.SEMI_MINOR:
                semi_minor = creator_state.axis_ingestion_value
                # Choose to display inverse‑flattening by default
                axis_cbox = self._ui.cbox_flat_minor
                axis_idx = axis_cbox.findData(EllipsoidAxisType.SEMI_MINOR)
                axis_cbox.setCurrentIndex(axis_idx)
                self._axis_ingest_type = EllipsoidAxisType.SEMI_MINOR
                self._ui.lbl_flat_minor_units.setText("Meters")
                self._axis_ingestion_value = semi_minor
                self._ui.ledit_flat_minor.setEnabled(True)
                self._ui.cbox_flat_minor.setEnabled(True)
                self._ui.ledit_flat_minor.setText(f"{semi_minor}")
            else:
                inv_f = creator_state.axis_ingestion_value
                # Choose to display inverse‑flattening by default
                axis_cbox = self._ui.cbox_flat_minor
                axis_idx = axis_cbox.findData(EllipsoidAxisType.INVERSE_FLATTENING)
                axis_cbox.setCurrentIndex(axis_idx)
                self._axis_ingest_type = EllipsoidAxisType.INVERSE_FLATTENING
                self._ui.lbl_flat_minor_units.setText("No Units")
                self._axis_ingestion_value = inv_f
                self._ui.ledit_flat_minor.setEnabled(True)
                self._ui.cbox_flat_minor.setEnabled(True)
                self._ui.ledit_flat_minor.setText(f"{inv_f}")

        # Projection (or none)
        proj_type = creator_state.proj_type

        self._update_extra_polar_stereo_params_display()

        proj_cbox = self._ui.cbox_proj_type
        proj_idx = proj_cbox.findData(proj_type)
        proj_cbox.setCurrentIndex(proj_idx)
        self._proj_type = proj_type
        self._update_units()

        # Projection‑specific parameters
        if not pycrs.is_geographic:
            self._center_lon = creator_state.center_lon
            latitude_value = creator_state.latitude
            latitude_choice = creator_state.latitude_choice

            self._ui.ledit_center_lon.setText("" if self._center_lon is None else str(self._center_lon))

            # Save the choice
            self._latitude_choice = latitude_choice

            # Update the combo‑box without re‑entering the slot
            cbox = self._ui.cbox_lat_chooser
            idx = cbox.findData(latitude_choice)
            if idx != -1:
                cbox.blockSignals(True)
                cbox.setCurrentIndex(idx)
                cbox.blockSignals(False)

            self._ui.ledit_lat_value.setText("" if latitude_value is None else str(latitude_value))

            self._ui.lbl_center_lon.setEnabled(True)
            self._ui.ledit_center_lon.setEnabled(True)
            self._ui.cbox_lat_chooser.setEnabled(True)
            self._ui.ledit_lat_value.setEnabled(True)
        else:
            # Clear projection fields
            for w in (self._ui.ledit_center_lon, self._ui.ledit_lat_value):
                w.clear()
            self._center_lon = self._latitude = None
            self._ui.lbl_center_lon.setEnabled(False)
            self._ui.ledit_center_lon.setEnabled(False)
            self._ui.cbox_lat_chooser.setEnabled(False)
            self._ui.ledit_lat_value.setEnabled(False)

        # CRS name
        self._ui.ledit_crs_name.setText(name)
        self._ui.ledit_crs_name.editingFinished.emit()
        self._crs_name = name

        # Remember selection so we can detect future changes
        self._current_starting_crs_name = name

        if self._proj_type == ProjectionTypes.POLAR_STEREO:
            if self._latitude_choice == LatitudeTypes.CENTRAL_LATITUDE:
                assert creator_state.polar_stereo_scale is not None
                ledit = self._ui.ledit_pstereo_scale_factor
                ledit.setText(str(creator_state.polar_stereo_scale))
                ledit.textChanged.emit(creator_state.polar_stereo_scale)
            elif self._latitude_choice == LatitudeTypes.TRUE_SCALE_LATITUDE:
                assert creator_state._polar_stereo_latitude_sign is not None
                cbox = self._ui.cbox_pstereo_sign
                idx = cbox.findData(creator_state._polar_stereo_latitude_sign)
                if idx != -1:
                    cbox.setCurrentIndex(idx)
                    cbox.currentIndexChanged.emit(idx)

    def _init_cbox_lat_chooser(self):
        """
        Initializes self._ui.cbox_lat_chooser to have all the values in
        LatitudeTypes. The text shown should be the value of the enum and
        the value of the cbox should be the enum.

        class LatitudeTypes(Enum):
            CENTRAL_LATITUDE = "Central Latitude"
            TRUE_SCALE_LATITUDE = "True Scale Lat"

        When a new cbox item is clicked the function _on_change_lat_choice
        should be called which sets an instance variable called
        self._latitude_choice.
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
        self._ui.ledit_lat_value.textChanged.connect(self._on_latitude_changed)

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
        if proj_type in (
            ProjectionTypes.EQUI_CYLINDRICAL,
            ProjectionTypes.POLAR_STEREO,
        ):
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
            # Display text is the enum value, store enum itself as user data
            self._ui.cbox_flat_minor.addItem(axis_type.value, axis_type)

        # Connect combo box signal to slot
        self._ui.cbox_flat_minor.currentIndexChanged.connect(self._on_axis_ingest_type_changed)

        self._axis_ingest_type = self._ui.cbox_flat_minor.itemData(self._ui.cbox_flat_minor.currentIndex())

        # Configure flat minor value entry with float validator
        flat_validator = QDoubleValidator(self._ui.ledit_flat_minor)
        flat_validator.setNotation(QDoubleValidator.StandardNotation)
        flat_validator.setDecimals(ALLOWED_DECIMALS)
        flat_validator.setBottom(0.0)
        self._ui.ledit_flat_minor.setValidator(flat_validator)
        self._ui.ledit_flat_minor.textChanged.connect(self._on_axis_ingestion_value_changed)

        # Configure semi-major entry with float validator
        semi_validator = QDoubleValidator(self._ui.ledit_semi_major)
        semi_validator.setNotation(QDoubleValidator.StandardNotation)
        semi_validator.setDecimals(ALLOWED_DECIMALS)
        semi_validator.setBottom(0.1)
        self._ui.ledit_semi_major.setValidator(semi_validator)
        self._ui.ledit_semi_major.textChanged.connect(self._on_semi_major_changed)

    def _init_center_longitude_ledit(self):
        validator = QDoubleValidator(self._ui.ledit_center_lon)
        validator.setNotation(QDoubleValidator.StandardNotation)
        validator.setRange(-180.0, 180.0, ALLOWED_DECIMALS)
        self._ui.ledit_center_lon.setValidator(validator)
        self._ui.ledit_center_lon.textChanged.connect(self._on_center_lon_changed)

    def _init_lon_meridian_ledit(self):
        validator = QDoubleValidator(self._ui.ledit_prime_meridian)
        validator.setNotation(QDoubleValidator.StandardNotation)
        validator.setRange(-180.0, 180.0, ALLOWED_DECIMALS)
        self._ui.ledit_prime_meridian.setValidator(validator)
        self._ui.ledit_prime_meridian.textChanged.connect(self._on_lon_meridian_changed)

    def _init_crs_name(self):
        regex = QRegularExpression(r"^[A-Za-z0-9_]+$")
        validator = QRegularExpressionValidator(regex, self._ui.ledit_crs_name)
        self._ui.ledit_crs_name.setValidator(validator)
        self._ui.ledit_crs_name.textEdited.connect(self._on_crs_name_changed)

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
        self._axis_ingest_type = self._ui.cbox_flat_minor.itemData(index)
        if self._axis_ingest_type == EllipsoidAxisType.INVERSE_FLATTENING:
            self._ui.lbl_flat_minor_units.setText("No Units")
        elif self._axis_ingest_type == EllipsoidAxisType.SEMI_MINOR:
            self._ui.lbl_flat_minor_units.setText("Meters")
        else:
            raise TypeError(
                f"Axis ingestion type is neither inverse flatting nor semi minor. "
                f"Instead, it is {self._axis_ingest_type}"
            )

    def _on_axis_ingestion_value_changed(self, text: str):
        try:
            self._axis_ingestion_value = float(text)
        except ValueError:
            self._axis_ingestion_value = None

    def _on_semi_major_changed(self, text: str):
        try:
            self._semi_major_value = float(text)
        except ValueError:
            self._semi_major_value = None

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

        needs_params = self._proj_type != ProjectionTypes.NO_PROJECTION

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
        # Basic validation
        if self._crs_name is None:
            QMessageBox.warning(
                self,
                self.tr("Missing value"),
                self.tr("Please supply a name for the CRS."),
            )
            return

        if self._shape_type == ShapeTypes.SPHEROID and self._semi_major_value is None:
            QMessageBox.warning(
                self,
                self.tr("Missing value"),
                self.tr("Please supply the radius value."),
            )
            return

        if self._shape_type == ShapeTypes.ELLIPSOID and (
            self._axis_ingestion_value is None
            or self._axis_ingest_type is None
            or self._semi_major_value is None
        ):
            QMessageBox.warning(
                self,
                self.tr("Missing value"),
                self.tr(
                    "For an ellipsoid you must fill the second axis\n"
                    "value and choose whether it is the semi-minor axis\n"
                    "or the inverse flattening."
                ),
            )
            return

        # Safe defaults if the user left them blank
        if self._proj_type != ProjectionTypes.NO_PROJECTION and (
            self._lon_meridian is None or self._latitude is None or self._center_lon is None
        ):
            QMessageBox.warning(
                self,
                self.tr("Missing value"),
                self.tr(
                    "When doing a projection, the prime meridian, center latitude,\n"
                    "center longitude, and latitude of true scale must be set. One\n"
                    "of them is not set."
                ),
            )
            return
        elif self._proj_type == ProjectionTypes.NO_PROJECTION and self._lon_meridian is None:
            QMessageBox.warning(
                self,
                self.tr("Missing value"),
                self.tr("When doing 'No Projection', the Prime Meridian field must be set."),
            )
            return

        a = self._semi_major_value
        if self._shape_type == ShapeTypes.SPHEROID:
            inv_f = 0.0  # sphere
        else:
            if self._axis_ingest_type == EllipsoidAxisType.SEMI_MINOR:
                b = self._axis_ingestion_value
                inv_f = a / (a - b) if a != b else 0.0
            else:  # inverse flattening entered directly
                inv_f = self._axis_ingestion_value

        # Ellipsoid description for proj

        if inv_f == 0.0:
            ellps_part = f"+R={a}"
        else:
            ellps_part = f"+a={a} +rf={inv_f}"

        base = f"{ellps_part} +pm={self._lon_meridian} +no_defs"

        if self._proj_type == ProjectionTypes.NO_PROJECTION:
            proj_str = f"+proj=longlat {base}"
        elif self._proj_type == ProjectionTypes.EQUI_CYLINDRICAL:
            if self._latitude_choice == LatitudeTypes.CENTRAL_LATITUDE:
                proj_str = f"+proj=eqc +lon_0={self._center_lon} +lat_0={self._latitude} " f"{base}"
            else:
                proj_str = f"+proj=eqc +lon_0={self._center_lon} +lat_ts={self._latitude} " f"{base}"

        elif self._proj_type == ProjectionTypes.POLAR_STEREO:
            if self._latitude_choice == LatitudeTypes.CENTRAL_LATITUDE:
                if self._polar_stereo_scale is None:
                    QMessageBox.warning(
                        self,
                        self.tr("Missing value"),
                        self.tr("The scale factor value is None. Please enter\n a scale factor value."),
                    )
                    return

                proj_str = (
                    f"+proj=stere +lat_0={self._latitude} +lon_0={self._center_lon} "
                    f"+k={self._polar_stereo_scale} +x_0=0 +y_0=0 {base}"
                )
            else:
                if self._polar_stereo_latitude_sign is None:
                    QMessageBox.warning(
                        self,
                        self.tr("Missing value"),
                        self.tr(
                            "The central latitude sign is None. Please select\n a central latitude sign."
                        ),
                    )
                    return
                proj_str = (
                    f"+proj=stere +lon_0={self._center_lon} +lat_0={self._polar_stereo_latitude_sign}90 "
                    f"+lat_ts={self._latitude} +x_0=0 +y_0=0 {base}"
                )

        else:
            QMessageBox.critical(self, "Error", f"Unknown projection type: {self._proj_type}")
            return

        pyproj_crs = pyproj.CRS.from_proj4(proj_str)
        self._new_crs = osr.SpatialReference()
        self._new_crs.ImportFromWkt(pyproj_crs.to_wkt())

        self._new_crs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)

        self._app_state.add_user_created_crs(self._crs_name, self._new_crs, self._export_creator_state())

        self._update_user_created_crs_cbox()

        self._switch_user_crs_cbox_selection(self._crs_name)

    def _export_creator_state(self, source_wkt: Optional[str] = None) -> CrsCreatorState:
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
            shape_type=self._shape_type,
            source_wkt=source_wkt,
        )
        return crs_creator_state

    def accept(self):
        # Which builder runs depends on the active tab: the parameter form or
        # the pasted-string tab.
        if self._ui.tabWidget.currentWidget() is self._ui.page_string:
            if not self._on_add_crs_string():
                # Validation/name problem - keep the dialog open so the user
                # can fix it rather than silently discarding their input.
                return
        else:
            self._create_crs()

        super().accept()
