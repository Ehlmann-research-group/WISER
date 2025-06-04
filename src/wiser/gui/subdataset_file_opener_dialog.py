import os
from typing import List, Optional, Dict, Tuple, TYPE_CHECKING

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *


from wiser.gui.generated.subdataset_chooser_dialog_ui import Ui_SubdatasetChooser

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
        # Update dataset name label so the user sees which file they are working with
        description = self._gdal_dataset.GetDescription() or "<unknown>"
        self._ui.lbl_dataset_name.setText(os.path.basename(description))

        # Populate every part of the dialog
        self._init_subdataset_cbox()
        self._init_geo_transform()
        self._init_spatial_ref()
        self._init_bands_table_widget()
        self._init_band_units()

        # When the geo-transform / SRS check-boxes are toggled we dim / un-dim
        # the text so the user has quick visual feedback.
        self._ui.chk_box_geo_transform.toggled.connect(
            lambda checked: self._set_checkbox_enabled_state(self._ui.chk_box_geo_transform, checked)
        )
        self._ui.chk_box_srs.toggled.connect(self.set_srs_checkbox_enabled_state)

    def set_srs_checkbox_enabled_state(self, checked: bool):
        self._set_checkbox_enabled_state(self._ui.chk_box_srs, checked)
        if checked == False:
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
                print(f"srs pretty: {srs.ExportToPrettyWkt()}")
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
        except Exception:
            # Any failure means we fallback to just the indices
            wavelengths = None

        self._wavelengths = wavelengths

        subdataset = self._get_selected_subdataset()

        band_count = subdataset.RasterCount
        if wavelengths is not None and band_count == len(wavelengths):
            self._use_wavelengths = True
        else:
            self._use_wavelengths = False

        tbl.setRowCount(band_count)
        print(f'BAND COUNT: {band_count}')
        for i in range(band_count):
            if self._use_wavelengths:
                text = f"Band {i}: {wavelengths[i]:.2f}"
            else:
                text = f"Band {i}"
            tbl.setItem(i, 0, QTableWidgetItem(text))

        # Disable wavelength-unit selection if we did not manage to extract any.
        self._ui.cbox_wavelength_units.setEnabled(self._use_wavelengths)

        # Hide the vertical header (row numbers)
        self._ui.table_wdgt_bands.verticalHeader().setVisible(False)

        # A little polish: resize to fit contents.
        tbl.resizeColumnsToContents()
        tbl.resizeRowsToContents()

    def _init_band_units(self) -> None:
        """Populate the wavelength-units combo-box with common choices."""
        cmb = self._ui.cbox_wavelength_units
        cmb.clear()

        # If we have no wavelength information we leave the combo disabled.
        if self._wavelengths is None:
            cmb.setEnabled(False)
            return

        # Helpful mapping of units (some duplicates removed for clarity)
        unit_options: list[tuple[str, Optional[u.Unit]]] = [
            ("None", None),  # Allow the caller to opt-out of unit conversion
            ("nm (nanometer)", u.nanometer),
            ("µm (micrometer)", u.micrometer),
            ("mm (millimeter)", u.millimeter),
            ("cm (centimeter)", u.centimeter),
            ("m (meter)", u.meter),
            ("Å (angstrom)", u.angstrom),
            ("cm⁻¹ (wavenumber)", u.cm ** -1),
            ("GHz", u.GHz),
            ("MHz", u.MHz),
        ]

        for text, unit_obj in unit_options:
            cmb.addItem(text, userData=unit_obj)

        # Default to nanometers if present.
        default_index = next((i for i, (_, uobj) in enumerate(unit_options) if uobj == u.nanometer), 0)
        cmb.setCurrentIndex(default_index)

    # ------------------------------------------------------------------
    # Getters – public helpers that callers can use once the dialog returns
    # ------------------------------------------------------------------

    def _get_selected_subdataset(self) -> gdal.Dataset:
        '''
        Gets the selected subdataset from the subdataset chooser
        '''
        current_data = self._ui.cbox_subdataset_choice.currentData()
        subdataset_name = current_data[1]
        subdataset: gdal.Dataset = gdal.Open(subdataset_name)
        assert subdataset is not None, "Selected subdataset can not be opened!"
        return subdataset

    def _get_wavelength_units(self) -> Optional[u.Unit]:
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
        # Figure out which dataset to open (we open it as an EMIT/netCDF type)
            # Things to override in NetCDF_GDALRasterDataImpl
                # geo transform getting, read_geo_transform
                # srs getting, read_spatial_ref(), get_wkt_spatial_reference()

                # band unit getting, read_band_unit
                # band info, read_band_info
                # data ignore value, read_data_ignore_value

        wl_unit = self._get_wavelength_units()

        srs = self._get_spatial_reference()
        subdataset_choice_data = self._get_subdataset_choice()
        subdataset_name = subdataset_choice_data[1]
        print(f"!!## subdataset name: {subdataset_name}")
        subdataset: gdal.Dataset = gdal.Open(subdataset_name)
        if self._use_wavelengths:
            wavelengths = self._wavelengths
        else:
            wavelengths = None
        geotransform = self._get_geo_transform()
        print(f"wl_unit: {srs}")
        print(f"wl_unit: {wl_unit}")
        print(f"wavelengths: {wavelengths}")
        print(f"geotransform: {geotransform}")
        print(f"type(subdataset): {type(subdataset)}")
        print(f"type(self._netcdf_dataset): {type(self._netcdf_dataset)}")
        print(f"!!@@subdataset raster band: {subdataset.GetRasterBand(1).ReadAsArray()}")
        self.netcdf_impl = NetCDF_GDALRasterDataImpl(subdataset, self._netcdf_dataset, subdataset_name,
                                                srs, wl_unit, wavelengths, geotransform)

        print(f"netcdf_impl.read_band_info(): {self.netcdf_impl.read_band_info()}")
        print(f"netcdf_impl.get_wkt_spatial_reference(): {self.netcdf_impl.get_wkt_spatial_reference()}")
        print(f"netcdf_impl.read_spatial_ref(): {self.netcdf_impl.read_spatial_ref()}")
        print(f"netcdf_impl.read_geo_transform(): {self.netcdf_impl.read_geo_transform()}")
        print(f"netcdf_impl.read_band_unit(): {self.netcdf_impl.read_band_unit()}")

        # Figure out if we should put SRS info on
            # FIgure out if we should put geo transform information on there
        
        # Figure out if we should put wavelength info on. 

        super().accept()

#     def __init__(self, gdal_dataset: gdal.Dataset, netcdf_dataset: nc.Dataset, parent=None):
#         '''
#         gdal_dataset should have subdatasets. Whatever function calls this should check that it does
#         '''
#         super().__init__(parent=parent)
#         print(f"About to check for subdatasetses")
#         assert gdal_dataset.GetSubDatasets(), (f"A gdal_dataset without subdatasets was passed into \
#                                                 subdataset file opener dialog.")
#         print(f"About to initialize UI")
#         self._ui = Ui_SubdatasetChooser()
#         self._ui.setupUi(self)
#         print(f"Set up UI")
#         self._gdal_dataset = gdal_dataset
#         self._netcdf_dataset = netcdf_dataset

#     def _init_subdataset_cbox(self):
#         '''
#         Should get the subdatasets from gdal_dataset and populate the cbox with the 
#         subdataset options. (the cbox is self._ui.cbox_subdataset_choice). The data for the subdataset should be the subdataset key.
#         The text shown should be the name. Refer to the below code for subdataset key
#         and name:
#             for subdataset_name, description in subdatasets:
#                 print(f"going through each subdaataset")
#                 # Extract the actual subdataset name from the path (e.g., "reflectance", "Calcite", etc.)
#                 subdataset_key = subdataset_name.split(':')[-1]
#                 # Check if the subdataset name is in the emit_data_names set
#                 # if subdataset_key in emit_data_names:
#                 # Open the subdataset
#                 gdal_subdataset = gdal.Open(subdataset_name)
#                 if gdal_subdataset is None:
#                     raise ValueError(f"Unable to open subdataset: {subdataset_name}")

#                 # Create an instance of the class for each matching subdataset
#                 instance = cls(gdal_subdataset)
#                 instance.subdataset_name = subdataset_name
#                 instance.subdataset_key = subdataset_key
#                 # Add the instance to the list
#                 instances_list.append(instance)
#         '''
#         pass

#     def _init_geo_transform(self):
#         '''
#         Should initialize the geo transform checkbox (called self._ui.chk_box_geotransform).
#         First we must see if there's a geotransform in the gdal_dataset and if so we get the
#         geo transform. If there is no geo transform, we make sure the QCheckBox is unchecked
#         and the text says 'No geotransform available'. 
#         TO get the geotransform we will have to go into the gdal_dataset.GetMetadata and find 
#         a key that looks like NC_GLOBAL#geotransform. However, this will return to us a 
#         geo transform of the form {47.6381040971226,0.000542232520256367,-0,-20.9576106620263,-0,-0.000542232520256367}
#         BUT its a string. So we will have to parse this string to get the numbers out in the 
#         order that they appeared in the string and put it in a 6 tuple. We will then set the
#         text in the checkbox to be the 6 tuple and make sure the checkbox is checked. Also,
#         whenever the checkbox is unchecked, the text in it should be a bit greyed out.  The key
#         NC_GLOBAL#geotransform should be in a set called self._geotransform_search_keys
#         '''

#     def _init_spatial_ref(self):
#         '''
#         This should initialize the spatial reference system checkbox (called self._ui.chk_box_srs).
#         This should get the spatial reference system in the gdal dataset. If there is no
#         spatial reference system, we make sure the QCheckbox is unchecked and the checkbox's
#         text should say 'No spatial reference system available'.

#         To get the spatial reference system we will have to go into the gdal_dataset.GetMetadata
#         and find a key that looks like NC_GLOBAL#spatial_ref. This will give us a string that looks
#         like this: GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AXIS["Latitude",NORTH],AXIS["Longitude",EAST],AUTHORITY["EPSG","4326"]]
#         For now we just want to display this full string in the text of the check box. Also, the keys that
#         we search for should be in a set called self._spatial_ref_search_keys. So NC_GLOBAL#spatial_ref should be 
#         in the set self._spatial_ref_search_keys. For now this set will only have one thing. Also, if
#         there is a spatial reference system the check box should be checked. If there is not one, the
#         checkbox should be unchecked and all the text greyed out.
#         '''

#     def _init_bands_table_widget(self):
#         '''
#         This should initialize the bands table widget (called self._ui.tbl_widgt_bands). It should have
#         there just be one header called 'Bands List' and it should be populated with all the bands, but the
#         format should be like this: 'Band {index}: {wavelength_value}'. For example
#         it should look like 'Band 0: 398.27' for an entry

#         If there are no wavelength values, it should just say 'Band {index}' like
#         'Band 0'. First try to get the bandmath values. THese must be found using the netcdf
#         dataset that is passed in (self._netcdf_dataset).  If there are no wavelength values then
#         self._ui.cbox_wavelength_units should be greyed out. 

#         TO get the wavelength values you will first make sure the self._netcdf_dataset as the sensor_band_parameters 
#         group. Then you will make sure that group has the wavelengths variable. Then you will extract all wavelength values 
#         by doing ds_nc_1b_rad.groups['sensor_band_parameters']['wavelengths'][:] which returns a numpy array of wavelengths.
#         You will then make Band 0 have the wavelength_value of the first index of the array and so on. If any of the steps to
#         get the wavelengths failed, we will just show the band index so like 'Band 0'
#         '''

#     def _init_band_units(self):
#         '''
#         This should initialize the band units QComboBox (called self._ui.cbox_wavelength_units)
#         It should default to displaying nanometers, but then also have these other wavelength units:
#         UNIT_NAME_MAPPING = {
#     u.cm: "Wavelength",
#     u.m: "Wavelength",
#     u.micrometer: "Wavelength",
#     u.millimeter: "Wavelength",
#     u.micron: "Wavelength",
#     u.nanometer: "Wavelength",
#     u.centimeter: "Wavelength",
#     u.meter: "Wavelength",
#     u.millimeter: "Wavelength",
#     u.nanometer: "Wavelength",
#     u.micrometer: "Wavelength",
#     u.cm ** -1: "Wavenumber",
#     u.angstrom: "Wavelength",
#     u.GHz: "Wavelength",
#     u.MHz: "Wavelength"
# }
#     And it should also have a none option
#         These units should be in astropy units when they are entered in to the combo box.

#         If there are no wavelength values, then this should just be disabled.
#         '''

#     def _get_wavelength_units(self):
#         '''
#         This function should get the current wavelength units in self._ui.cbox_wavelength_units.
#         And return it. If the none option is selected in the wavelength units, the units should be none.
#         '''

#     def _get_subdataset_choice(self):
#         '''
#         This should return the combobox choice currently selected in self._ui.cbox_subdataset_choice
#         '''

#     def _get_geo_transform(self):
#         '''
#         This should return none of the geo transform is disabled (the checkbox is unchecked) or if there
#         is no geo transform. This should return the geo transform if it is enabled and there is a geo transform.
#         '''

#     def _get_spatial_reference(self):
#         '''
#         If there is no spatial reference system or the checkbox for this is disabled it should return NOne.
#         If that is not the case then it should get the spatial reference string and make it into an osr.SpatialReference object
#         using from wkt.
#         '''
