import logging

from wiser.plugins import ToolsMenuPlugin

from PySide2.QtWidgets import QMenu, QMessageBox

from wiser.raster.dataset import RasterDataSet

from wiser.gui.app_state import ApplicationState

import numpy as np

from astropy import units as u

from typing import Any, Dict, List, Optional, Union

from wiser.plugins import ContextMenuPlugin, ContextMenuType

logger = logging.getLogger(__name__)

quick_indices = {
    "NDVI": [860 * u.nm, 660 * u.nm],
    "NDWI": [550 * u.nm, 860 * u.nm],
    "NDBI": [1600 * u.nm, 860 * u.nm],
}


class ApplyNDIndex(ContextMenuPlugin):
    """
    Use a user selected normalized difference index quickly on a dataset
    """

    def __init__(self):
        super().__init__()

    def add_context_menu_items(
        self,
        context_type: ContextMenuType,
        context_menu: QMenu,
        context: Dict[str, Any],
    ) -> None:
        """
        Use QMenu.addAction() to add individual actions, or QMenu.addMenu() to
        add sub-menus to the Tools menu.
        """
        if context_type == ContextMenuType.RASTER_VIEW:
            """
            Context-menu display in a raster-view, which probably is showing a
            dataset.  The current dataset is passed to the plugin.

            Example code to get all necessary pieces of data for this context_type:
            ```python
            # A RasterDataSet object
            dataset = context["dataset"]
            # A 3 or 1 tuple of integers
            display_bands = context["display_bands"]
            # Every context_type has the app_state in the "wiser" key
            app_state = context["wiser"]
            ```
            """
            act = context_menu.addAction("Use NDI")
            act.triggered.connect(lambda checked=False: self.select_and_use_ndi(context))

        else:
            raise ValueError(f"Unrecognized context_type value {context_type}")

    def select_and_use_ndi(self, context):
        app_state: ApplicationState = context["wiser"]
        form_inputs = [
            ("Select the NDI method to use", "ndi_method", 0, list(quick_indices.keys())),
        ]
        return_dict: Dict[str, Any] = app_state.create_form(
            form_inputs,
            title="Analysis Params",
            description="The equation for the NDI is (a-b)/(a+b)",
        )
        ndi_method: str = return_dict["ndi_method"]
        wvl_a, wvl_b = quick_indices[ndi_method]

        dataset: RasterDataSet = context["dataset"]
        wavelengths = dataset.get_spectral_metadata().get_wavelengths()
        if wavelengths is None:
            QMessageBox.warning(
                None,
                "Couldn't Apply NDI",
                "The selected dataset has no wavelengths, cannot apply NDI.",
            )
            return
        wvl_a_index_closest = find_closest_wavelength(
            wavelengths=wavelengths,
            input_wavelength=wvl_a,
            max_distance=5 * u.nm,
        )
        wvl_b_index_closest = find_closest_wavelength(
            wavelengths=wavelengths,
            input_wavelength=wvl_b,
            max_distance=5 * u.nm,
        )

        assert wvl_a_index_closest is not None, f"Couldn't find a matching wavelength for {wvl_a} within 5 nm"
        assert wvl_b_index_closest is not None, f"Couldn't find a matching wavelength for {wvl_b} within 5 nm"

        band_a = dataset.get_band_data(wvl_a_index_closest)
        band_b = dataset.get_band_data(wvl_b_index_closest)

        ndi = (band_a - band_b) / (band_a + band_b)

        # Get object that helps us load datasets from numpy arrays or file paths
        data_loader = app_state.get_loader()
        # We must add a dimension to the beginning of the array because WISER takes
        # 3D arrays in the form [band][y][x]
        ndi = ndi[np.newaxis, :, :]
        # Create the dataset object, you must set its name
        ndi_dataset = data_loader.dataset_from_numpy_array(ndi, app_state.get_cache())
        ndi_dataset.set_name(f"{ndi_method} for {dataset.get_name()}")
        # Now add it back to our app
        app_state.add_dataset(ndi_dataset)


class NormalizedDifferenceIndexMaker(ToolsMenuPlugin):
    """
    A plugin to make a new normalized difference index
    """

    def __init__(self):
        super().__init__()

    def add_tool_menu_items(self, tool_menu: QMenu, wiser) -> None:
        """
        Use QMenu.addAction() to add individual actions, or QMenu.addMenu() to
        add sub-menus to the Tools menu.
        """
        logger.info("NormalizedDifferenceIndexMaker is adding tool-menu items")
        act = tool_menu.addAction("Created new NDI")
        act.triggered.connect(self.create_nd_index)
        self._app_state = wiser

    def create_nd_index(self) -> None:
        # To create the ND index, we need a name and the two wavelengths to use
        # for the index, so lets make a form to get that

        form_inputs = [
            ("Enter Wavelength for A", "wvl_a", 2),
            ("Enter Wavelength for B", "wvl_b", 2),
            ("Enter Normalized Difference Index Name", "ndi_name", 5),
        ]
        return_dict: Dict[str, Any] = self._app_state.create_form(
            form_inputs,
            title="Analysis Params",
            description="The equation for the ND is (a-b)/(a+b)",
        )
        wvl_a: u.Quantity = return_dict["wvl_a"]
        wvl_b: u.Quantity = return_dict["wvl_b"]
        ndi_name: str = return_dict["ndi_name"]

        quick_indices[ndi_name] = [wvl_a, wvl_b]


def find_closest_wavelength(
    wavelengths: List[u.Quantity],
    input_wavelength: u.Quantity,
    max_distance: u.Quantity = None,
) -> Optional[int]:
    """
    Given a list of wavelengths and an input wavelength, this function returns
    the index of the wavelength closest to the input wavelength.  If no
    wavelength is within max_distance of the input then None is returned.
    """

    # Do the whole calculation in nm to keep things simple.
    if max_distance is None:
        max_distance = 20 * input_wavelength.unit.si
    input_value = convert_spectral(input_wavelength, u.nm).value
    max_dist_value = None
    if max_distance is not None:
        max_dist_value = convert_spectral(max_distance, u.nm).value

    values = [convert_spectral(v, u.nm).value for v in wavelengths]

    return find_closest_value(values, input_value, max_dist_value)


Number = Union[float, int]


def find_closest_value(
    values: List[Number], input_value: Number, max_distance: Optional[Number] = None
) -> Optional[int]:
    """
    Given a list of numbers (ints and/or floats) and an input number, this
    function returns the index of the number closest to the input number.
    If no number is within max_distance of the input then None is returned.
    """
    best_index = None
    best_distance = None

    for index, value in enumerate(values):
        distance = abs(value - input_value)

        if max_distance is not None and distance > max_distance:
            continue

        if best_index is None or distance < best_distance:
            best_index = index
            best_distance = distance

    return best_index


def convert_spectral(value: u.Quantity, to_unit: u.Unit) -> u.Quantity:
    """
    Convert a spectral value with units (e.g. a frequency or wavelength),
    to the specified units.
    """
    return value.to(to_unit, equivalencies=u.spectral())
