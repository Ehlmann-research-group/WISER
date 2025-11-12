Plugin Utilities
================

WISER exposes 4 ways to create your own UI items that get wiser state for your plugins. These ways let you 
prompt the user to choose datasets, spectra, ROIs, and bands. You access them
through :class:`wiser.gui.app_state.ApplicationState`. The methods are:

.. code-block:: python

    dataset: RasterDataSet = app_state.choose_dataset_ui()
    spectrum: Spectrm = app_state.choose_spectrum_ui()
    roi: RegionOfInterest = app_state.choose_roi_ui()
    band: RasterDataBand = app_state.choose_band_ui()

The :class:`wiser.gui.ui_library.DynamicInputDialog` provides a simple way to collect user input that isn't tied to WISER's internal state.
By passing in a list of input specifications, you can dynamically generate a dialog that displays text fields and
combo boxes. This is useful for plugins or tools that need to ask users for parameters,
options, or configuration values before running. Here is an example of its use:

.. code-block:: python

    form_inputs = [
            ("Enter Wavelength", "wvl", 2),
            ("Enter Divisor", "divisor", 1),
            ("Enter Dimensionality Reduction", "dim_reduction", 0, ["PCA", "SVD"]),
        ]
    return_dict: Dict[str, Any] = self._app_state.create_form(
        form_inputs,
        title="Analysis Params",
        description="In the above text enter the parameters needed to perform the algorithm.",
    )

This will construct a GUI element that looks like this:

.. image:: images/dynamic_plugin_input.png

Below is an example that asks the user to first select a dataset and then a region
of interest in a tools menu plugin:

.. code-block:: python

    from wiser.raster.spectrum import SpectrumAverageMode, calc_roi_spectrum

    class ExampleToolPlugin(ToolsMenuPlugin):
        def __init__(self):
            super().__init__()

        def add_tool_menu_items(self, tool_menu: QMenu, wiser) -> None:
            """
            Use QMenu.addAction() to add individual actions, or QMenu.addMenu() to
            add sub-menus to the Tools menu.
            """
            act = tool_menu.addAction("ROI Angle Mapper")
            act.triggered.connect(self.roi_angle_mapper)

            self._app_state = wiser

        def roi_angle_mapper(self):
            dataset = self._app_state.choose_dataset_ui(description="Description: choose dataset to perform analysis on")
            roi = self._app_state.choose_roi_ui(description="Description: choose ROI to get Average Spectrum")

            mean_roi_spec_arr = calc_roi_spectrum(dataset=dataset, roi=roi, mode=SpectrumAverageMode.MEAN)
            image_arr = dataset.get_image_data()

            # Do some math...
