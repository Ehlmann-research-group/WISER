import os
from typing import List, Optional, Dict, Tuple, TYPE_CHECKING

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *


from wiser.gui.generated.subdataset_chooser_dialog_ui import Ui_SubdatasetChooser
from .util import populate_combo_box_with_units
from osgeo import gdal, osr

import numpy as np

from astropy import units as u

if TYPE_CHECKING:
    from wiser.raster.dataset import RasterDataSet
    from wiser.raster.dataset_impl import RasterDataImpl

import netCDF4 as nc

class SubdatasetFileOpenerDialog(QDialog):
    '''
    Dialog that allows the user to pick a sub-dataset from a NetCDF file and
    configure optional geo-referencing information before the file is opened
    inside WISER.
    '''

    # Metadata keys we know GDAL uses to store geotransform / SRS for NetCDF
    _geotransform_search_keys = {
        "NC_GLOBAL#geotransform",
        "geotransform",  # Fallback – rare but seen in some files
    }
    _spatial_ref_search_keys = {
        "NC_GLOBAL#spatial_ref",
        "spatial_ref",  # Fallback – rare but seen in some files
    }

    # ---------------------------------------------------------------------
    # Construction / UI initialisation
    # ---------------------------------------------------------------------
    def __init__(self, gdal_dataset: gdal.Dataset, netcdf_dataset: nc.Dataset, parent: QWidget = None):
        """Create the dialog.

        Parameters
        ----------
        gdal_dataset
            **Already opened** GDAL dataset that *must* expose sub-datasets.
        netcdf_dataset
            **Already opened** ``netCDF4.Dataset`` that corresponds to the same
            physical file as *gdal_dataset* - used to extract wavelength
            metadata for band display.
        parent
            Usual Qt parent widget.
        """
        super().__init__(parent=parent)

        if not gdal_dataset.GetSubDatasets():
            raise AssertionError(
                "A GDAL dataset without sub-datasets was passed to SubdatasetFileOpenerDialog"
            )

        # --- Build the Qt UI generated from Qt-Designer
        self._ui = Ui_SubdatasetChooser()
        self._ui.setupUi(self)

        self._gdal_dataset = gdal_dataset
        self._netcdf_dataset = netcdf_dataset

        # Keep internal copies of potentially optional metadata so getters can
        # return quickly later on.
        self._geo_transform: Optional[Tuple[float, float, float, float, float, float]] = None
        self._spatial_ref_wkt: Optional[str] = None  # Keep raw string until asked for osr object
        self._wavelengths: Optional[np.ndarray] = None
        self._use_wavelengths: bool = None
        self.netcdf_impl = None  # Of type NetCDF_GDALRasterDataImpl
        self._band_count: int = None
        # Update dataset name label so the user sees which file they are working with
        description = self._gdal_dataset.GetDescription() or "<unknown>"
        self._ui.lbl_dataset_name.setText(os.path.basename(description))

        # Populate every part of the dialog
        self._init_subdataset_cbox()
        self._init_geo_transform()
        self._init_spatial_ref()
        # Must be before _init_bands_table_widget
        self._init_bands_table_widget()

        # When the geo-transform / SRS check-boxes are toggled we dim / un-dim
        # the text so the user has quick visual feedback.
        self._ui.chk_box_geo_transform.toggled.connect(
            lambda checked: self._set_checkbox_enabled_state(self._ui.chk_box_geo_transform, checked)
        )
        self._ui.chk_box_srs.toggled.connect(self.set_srs_checkbox_enabled_state)

    def set_srs_checkbox_enabled_state(self, checked: bool):
        self._set_checkbox_enabled_state(self._ui.chk_box_srs, checked)
        if not checked:
            self._set_checkbox_enabled_state(self._ui.chk_box_geo_transform, checked)
            self._ui.chk_box_geo_transform.setCheckState(Qt.Unchecked)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_geotransform_string(gtr: str) -> Optional[Tuple[float, float, float, float, float, float]]:
        """Convert the NC_GLOBAL#geotransform style string into a 6-tuple of
        floats in the order GDAL expects.
        Example input::
            "{47.6381040971226,0.000542232520256367,-0,-20.9576106620263,-0,-0.000542232520256367}"
        """
        cleaned = gtr.strip().lstrip("{").rstrip("}")
        parts = [p.strip() for p in cleaned.split(",") if p.strip()]
        if len(parts) != 6:
            return None
        try:
            return tuple(float(p) for p in parts)  # type: ignore[return-value]
        except ValueError:
            return None

    @staticmethod
    def _set_checkbox_enabled_state(chk: QCheckBox, enabled: bool) -> None:
        """Grey-out the text on *chk* when *enabled* is *False* without disabling
        the widget itself (the user can still re-enable it).
        """
        pal: QPalette = chk.palette()
        role = QPalette.WindowText if chk.isEnabled() else QPalette.Disabled
        default_colour = chk.style().standardPalette().color(QPalette.WindowText)
        dim_colour = chk.style().standardPalette().color(QPalette.Disabled, QPalette.Text)
        pal.setColor(QPalette.WindowText, default_colour if enabled else dim_colour)
        chk.setPalette(pal)

    # ------------------------------------------------------------------
    # UI section initialisers
    # ------------------------------------------------------------------
    def _init_subdataset_cbox(self) -> None:
        """Populate *cbox_subdataset_choice* with the available sub-datasets."""
        self._ui.cbox_subdataset_choice.clear()

        self._ui.cbox_subdataset_choice.activated.connect(self._init_bands_table_widget)

        subdatasets = self._gdal_dataset.GetSubDatasets()
        for subdataset_name, description in subdatasets:
            subdataset_key = subdataset_name.split(":")[-1]
            display_text = description or subdataset_key  # Human-readable if available.
            # Store *subdataset_key* as the user data; display the friendly text.
            self._ui.cbox_subdataset_choice.addItem(display_text, (subdataset_key, subdataset_name))

        # When the list is not empty select the first item by default.
        if self._ui.cbox_subdataset_choice.count():
            self._ui.cbox_subdataset_choice.setCurrentIndex(0)

    def _init_geo_transform(self) -> None:
        """Initialise the geo-transform check-box."""
        metadata = self._gdal_dataset.GetMetadata() or {}
        geo_string: Optional[str] = next(
            (metadata[k] for k in self._geotransform_search_keys if k in metadata),
            None,
        )

        if geo_string is not None:
            self._geo_transform = self._parse_geotransform_string(geo_string)

        if self._geo_transform:
            self._ui.chk_box_geo_transform.setText(str(self._geo_transform))
            self._ui.chk_box_geo_transform.setChecked(True)
        else:
            self._ui.chk_box_geo_transform.setText("No geotransform available")
            self._ui.chk_box_geo_transform.setChecked(False)

        # Ensure the grey-out is correct on first show.
        self._set_checkbox_enabled_state(self._ui.chk_box_geo_transform, self._ui.chk_box_geo_transform.isChecked())

    def _init_spatial_ref(self) -> None:
        """Initialise the spatial reference system check-box."""
        metadata = self._gdal_dataset.GetMetadata() or {}
        srs_string: Optional[str] = next(
            (metadata[k] for k in self._spatial_ref_search_keys if k in metadata),
            None,
        )

        def _truncate_pretty_wkt(wkt_str: str, max_lines: int, max_len: int) -> str:
            """
            1) Split the pretty-WKT into its lines.
            2) For each line, if it exceeds max_len, truncate it to (max_len - 3) + '...'.
            3) If there are more than max_lines lines, keep only the first max_lines,
                and append '...' to the last visible line (again respecting max_len).
            """
            lines = wkt_str.splitlines()
            truncated: list[str] = []

            for idx, ln in enumerate(lines):
                if idx >= max_lines:
                    # We have already reached max_lines, so append '...' to the last stored line:
                    last = truncated[-1]
                    if len(last) > max_len:
                        last = last[: max_len - 3] + "..."
                    else:
                        # If the last item is shorter than max_len, just tack on '...'
                        last = last + "..."
                    truncated[-1] = last
                    break

                if len(ln) > max_len:
                    truncated.append(ln[: max_len - 3] + "...")
                else:
                    truncated.append(ln)

            return "\n".join(truncated)

        # These two constants control how much of the WKT we display:
        MAX_LINES = 12
        MAX_LINE_LENGTH = 60

        if srs_string:
            self._spatial_ref_wkt = srs_string
            try:
                srs = osr.SpatialReference()
                srs.ImportFromWkt(srs_string)
                pretty_wkt = srs.ExportToPrettyWkt()
                display_text = _truncate_pretty_wkt(pretty_wkt, MAX_LINES, MAX_LINE_LENGTH)
            except Exception:
                # Fallback: just truncate the raw srs_string to a single line:
                raw = srs_string.replace("\n", " ")  # collapse any existing newlines
                if len(raw) > MAX_LINE_LENGTH:
                    display_text = raw[: MAX_LINE_LENGTH - 3] + "..."
                else:
                    display_text = raw
            finally:
                self._ui.chk_box_srs.setText(display_text)
                self._ui.chk_box_srs.setChecked(True)
        else:
            self._ui.chk_box_srs.setText("No spatial reference system available")
            self._ui.chk_box_srs.setChecked(False)

        self._set_checkbox_enabled_state(self._ui.chk_box_srs, self._ui.chk_box_srs.isChecked())

    def _init_bands_table_widget(self) -> None:
        """Populate the *Bands List* QTableWidget with band indices and optional
        wavelength values obtained from the NetCDF *sensor_band_parameters*
        group.
        """
        tbl = self._ui.table_wdgt_bands
        tbl.clear()
        tbl.setColumnCount(1)
        tbl.setHorizontalHeaderLabels(["Bands List"])

        # Try to extract wavelength array from the NetCDF side first.
        wavelengths: Optional[List[float]] = None
        try:
            sbp_group = self._netcdf_dataset.groups.get("sensor_band_parameters")
            if sbp_group and "wavelengths" in sbp_group.variables:
                wl_var = sbp_group["wavelengths"]
                wavelengths = wl_var[:]
            else:
                wl_var = self._netcdf_dataset.variables["wavelengths"]
                wavelengths = wl_var[:]

        except Exception:
            # Any failure means we fallback to just the indices
            wavelengths = None

        self._wavelengths = wavelengths

        subdataset = self._get_selected_subdataset()

        # Init the combo box for band unit
        cmb = self._ui.cbox_wavelength_units
        cmb.clear()

        populate_combo_box_with_units(cmb)

        cmb.activated.connect(self.on_unit_cbox_changed)

        # If we have no wavelength information we leave the combo disabled.
        if self._wavelengths is None:
            cmb.setEnabled(False)

        self._band_count = subdataset.RasterCount
        if wavelengths is not None and self._band_count == len(wavelengths) and self._get_wavelength_units() is not None:
            self._use_wavelengths = True
        else:
            self._use_wavelengths = False
    
        # Disable wavelength-unit selection if we did not manage to extract any.
        self._ui.cbox_wavelength_units.setEnabled(self._use_wavelengths)

        tbl.setRowCount(self._band_count)
        for i in range(self._band_count):
            if self._use_wavelengths:
                text = f"Band {i}: {wavelengths[i]:.2f}"
            else:
                text = f"Band {i}"
            tbl.setItem(i, 0, QTableWidgetItem(text))

        # Hide the vertical header (row numbers)
        self._ui.table_wdgt_bands.verticalHeader().setVisible(False)

        # A little polish: resize to fit contents.
        tbl.resizeColumnsToContents()
        tbl.resizeRowsToContents()

    def on_unit_cbox_changed(self, index):
        data = self._ui.cbox_wavelength_units.currentData()
        if data is None:
            self._use_wavelengths = False
        elif self._wavelengths is not None and self._band_count == len(self._wavelengths):
            self._use_wavelengths = True
        # The else case is if we are on a dataset that doesn't have wavelengths like glt_x,
        # so we let the logic in init_bands_table_widget handle this.

    # ------------------------------------------------------------------
    # Getters – public helpers that callers can use once the dialog returns
    # ------------------------------------------------------------------

    def _get_selected_subdataset(self) -> gdal.Dataset:
        '''
        Gets the selected subdataset from the subdataset chooser
        '''
        current_data = self._ui.cbox_subdataset_choice.currentData()
        subdataset_name = current_data[1]
        gdal.PushErrorHandler('CPLQuietErrorHandler')  # :contentReference[oaicite:0]{index=0}
        try:
            subdataset: gdal.Dataset = gdal.Open(subdataset_name)
        finally:
            gdal.PopErrorHandler()
        assert subdataset is not None, "Selected subdataset can not be opened!"
        return subdataset

    def  _get_wavelength_units(self) -> Optional[u.Unit]:
        """Return the currently selected *astropy.units.Unit* or *None*."""
        return self._ui.cbox_wavelength_units.currentData()

    def _get_subdataset_choice(self) -> Tuple[str, str]:
        """Return the key for the currently selected sub-dataset."""
        return self._ui.cbox_subdataset_choice.currentData()

    def _get_geo_transform(self) -> Optional[Tuple[float, float, float, float, float, float]]:
        """Return the geo-transform tuple if the user left the check-box enabled
        **and** the dataset actually has one. Otherwise *None*.
        """
        if not self._ui.chk_box_geo_transform.isChecked():
            return None
        return self._geo_transform

    def _get_spatial_reference(self) -> Optional[osr.SpatialReference]:
        """Return an *osgeo.osr.SpatialReference* built from the WKT in the GDAL
        metadata **if** the user left the SRS check-box enabled. Otherwise
        *None*.
        """
        if not self._ui.chk_box_srs.isChecked() or not self._spatial_ref_wkt:
            return None

        srs = osr.SpatialReference()
        # ImportFromWkt expects *str* in Python 3
        srs.ImportFromWkt(self._spatial_ref_wkt)
        return srs

    def accept(self):
        from wiser.raster.dataset_impl import NetCDF_GDALRasterDataImpl

        wl_unit = self._get_wavelength_units()

        srs = self._get_spatial_reference()
        subdataset_choice_data = self._get_subdataset_choice()
        subdataset_name = subdataset_choice_data[1]
        subdataset: gdal.Dataset = gdal.Open(subdataset_name)
        if self._use_wavelengths:
            wavelengths = self._wavelengths
        else:
            wavelengths = None
        geotransform = self._get_geo_transform()
        self.netcdf_impl = NetCDF_GDALRasterDataImpl(subdataset, self._netcdf_dataset, subdataset_name,
                                                srs, wl_unit, wavelengths, geotransform)

        super().accept()
