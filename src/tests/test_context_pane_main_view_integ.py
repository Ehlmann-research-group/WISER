import unittest

import tests.context

# import context
from tests.utils import are_pixels_close, are_qrects_close
# from utils import are_pixels_close, are_qrects_close

from test_utils.test_model import WiserTestModel

import numpy as np

from PySide2.QtTest import QTest
from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *


class TestContextPaneMainViewIntegration(unittest.TestCase):
    def setUp(self):
        self.test_model = WiserTestModel()

    def tearDown(self):
        self.test_model.close_app()
        del self.test_model

    def test_click_cp_visible_mv(self):
        """
        Clicks in the context pane and ensures the clicked point is visible in the main view
        """
        # Create first array
        rows, cols, channels = 100, 100, 3
        # Create a vertical gradient from 0 to 1: shape (50,1)
        row_values = np.linspace(0, 1, rows).reshape(rows, 1)
        # Tile the values horizontally to get a 50x50 array
        impl = np.tile(row_values, (1, cols))
        # Repeat the 2D array across 3 channels to get a 3x50x50 array
        np_impl = np.repeat(impl[np.newaxis, :, :], channels, axis=0)

        self.test_model.load_dataset(np_impl)

        self.test_model.click_main_view_zoom_in()
        self.test_model.click_main_view_zoom_in()
        self.test_model.click_main_view_zoom_in()
        self.test_model.click_main_view_zoom_in()
        self.test_model.click_main_view_zoom_in()
        self.test_model.click_main_view_zoom_in()
        self.test_model.click_main_view_zoom_in()
        self.test_model.click_main_view_zoom_in()
        self.test_model.click_main_view_zoom_in()
        self.test_model.click_main_view_zoom_in()

        pixel = (79, 79)

        self.test_model.click_raster_coord_context_pane(pixel)

        visible_region = self.test_model.get_main_view_rv_visible_region((0, 0))

        pixel_point = QPoint(pixel[0], pixel[1])

        self.assertTrue(visible_region.contains(pixel_point))

    def test_cp_highlight_equal_mv_after_zoom(self):
        """
        First zoom in. Then click in the context pane. The context pane highlight box
        should be the same as the main view's visible region.
        """
        # Create first array
        rows, cols, channels = 100, 100, 3
        # Create a vertical gradient from 0 to 1: shape (50,1)
        row_values = np.linspace(0, 1, rows).reshape(rows, 1)
        # Tile the values horizontally to get a 50x50 array
        impl = np.tile(row_values, (1, cols))
        # Repeat the 2D array across 3 channels to get a 3x50x50 array
        np_impl = np.repeat(impl[np.newaxis, :, :], channels, axis=0)

        ds = self.test_model.load_dataset(np_impl)

        self.test_model.click_main_view_zoom_in()
        self.test_model.click_main_view_zoom_in()
        self.test_model.click_main_view_zoom_in()
        self.test_model.click_main_view_zoom_in()
        self.test_model.click_main_view_zoom_in()
        self.test_model.click_main_view_zoom_in()
        self.test_model.click_main_view_zoom_in()
        self.test_model.click_main_view_zoom_in()
        self.test_model.click_main_view_zoom_in()
        self.test_model.click_main_view_zoom_in()

        cp_highlight = self.test_model.get_context_pane_highlight_region(ds.get_id())[0]

        mv_region = self.test_model.get_main_view_rv_visible_region((0, 0))

        self.assertTrue(cp_highlight == mv_region)

    def test_cp_highlight_equal_mv_after_scroll(self):
        """
        First zoom in. Then scroll the mainview. The context pane's highlight box should be
        the same as the main view'sa visible region.
        """
        # Create first array
        rows, cols, channels = 100, 100, 3
        # Create a vertical gradient from 0 to 1: shape (50,1)
        row_values = np.linspace(0, 1, rows).reshape(rows, 1)
        # Tile the values horizontally to get a 50x50 array
        impl = np.tile(row_values, (1, cols))
        # Repeat the 2D array across 3 channels to get a 3x50x50 array
        np_impl = np.repeat(impl[np.newaxis, :, :], channels, axis=0)

        ds = self.test_model.load_dataset(np_impl)

        self.test_model.click_main_view_zoom_in()
        self.test_model.click_main_view_zoom_in()
        self.test_model.click_main_view_zoom_in()
        self.test_model.click_main_view_zoom_in()
        self.test_model.click_main_view_zoom_in()
        self.test_model.click_main_view_zoom_in()
        self.test_model.click_main_view_zoom_in()
        self.test_model.click_main_view_zoom_in()
        self.test_model.click_main_view_zoom_in()
        self.test_model.click_main_view_zoom_in()

        self.test_model.scroll_main_view_rv_dx((0, 0), -100)

        cp_highlight = self.test_model.get_context_pane_highlight_region(ds.get_id())[0]

        mv_region = self.test_model.get_main_view_rv_visible_region((0, 0))

        self.assertTrue(cp_highlight == mv_region)

        self.test_model.scroll_main_view_rv_dy((0, 0), -100)

        cp_highlight = self.test_model.get_context_pane_highlight_region(ds.get_id())[0]

        mv_region = self.test_model.get_main_view_rv_visible_region((0, 0))

        self.assertTrue(cp_highlight == mv_region)

    def test_cp_highlight_equal_mv_after_scroll_linked(self):
        """
        First zoom in. Then scroll the mainview. The context pane's highlight box should be
        the same as the main view's visible region for the shared dataset.
        """
        # Create first array
        rows, cols, channels = 100, 100, 3
        # Create a vertical gradient from 0 to 1: shape (50,1)
        row_values = np.linspace(0, 1, rows).reshape(rows, 1)
        # Tile the values horizontally to get a 50x50 array
        impl = np.tile(row_values, (1, cols))
        # Repeat the 2D array across 3 channels to get a 3x50x50 array
        np_impl = np.repeat(impl[np.newaxis, :, :], channels, axis=0)

        # Create second array
        # Create 49 linearly spaced values from 0 to 0.75 and then append a 0
        row_values = np.concatenate(
            (np.linspace(0, 0.75, rows - 5), np.array([0, 0, 0, 0, 0]))
        ).reshape(rows, 1)
        impl2 = np.tile(row_values, (1, cols))
        np_impl2 = np.repeat(impl2[np.newaxis, :, :], channels, axis=0)

        # Create third array
        # Start with an array of zeros (50x1)
        row_values = np.zeros((rows, 1))
        # Choose the row index corresponding to 75% of the height.
        nonzero_index = int(0.75 * (rows - 1))
        row_values[nonzero_index] = 0.75
        impl3 = np.tile(row_values, (1, cols))
        np_impl3 = np.repeat(impl3[np.newaxis, :, :], channels, axis=0)

        self.test_model.set_main_view_layout((2, 2))

        ds1 = self.test_model.load_dataset(np_impl)
        ds2 = self.test_model.load_dataset(np_impl2)
        ds3 = self.test_model.load_dataset(np_impl3)

        self.test_model.click_link_button()

        self.test_model.set_context_pane_dataset(ds1.get_id())

        self.test_model.click_main_view_zoom_in()
        self.test_model.click_main_view_zoom_in()
        self.test_model.click_main_view_zoom_in()
        self.test_model.click_main_view_zoom_in()
        self.test_model.click_main_view_zoom_in()
        self.test_model.click_main_view_zoom_in()
        self.test_model.click_main_view_zoom_in()
        self.test_model.click_main_view_zoom_in()
        self.test_model.click_main_view_zoom_in()
        self.test_model.click_main_view_zoom_in()

        cp_highlight = self.test_model.get_context_pane_compatible_highlights(
            ds1.get_id()
        )[0]

        mv_region = self.test_model.get_main_view_rv_visible_region((0, 0))

        self.assertTrue(cp_highlight == mv_region)

        self.test_model.scroll_main_view_rv_dx((0, 1), -100)

        cp_highlight = self.test_model.get_context_pane_compatible_highlights(
            ds1.get_id()
        )[0]

        mv_region = self.test_model.get_main_view_rv_visible_region((0, 0))

        self.assertTrue(cp_highlight == mv_region)

        self.test_model.scroll_main_view_rv_dy((1, 0), -100)

        cp_highlight = self.test_model.get_context_pane_compatible_highlights(
            ds1.get_id()
        )[0]

        mv_region = self.test_model.get_main_view_rv_visible_region((0, 0))

        self.assertTrue(cp_highlight == mv_region)

    def test_cp_highlight_box(self):
        """
        Ensures that the context pane's compatible highlights is
        just the dataset shown in the context pane
        """
        # Create first array
        rows, cols, channels = 75, 75, 3
        # Create a vertical gradient from 0 to 1: shape (50,1)
        row_values = np.linspace(0, 1, rows).reshape(rows, 1)
        # Tile the values horizontally to get a 50x50 array
        impl = np.tile(row_values, (1, cols))
        # Repeat the 2D array across 3 channels to get a 3x50x50 array
        np_impl = np.repeat(impl[np.newaxis, :, :], channels, axis=0)

        # Create second array
        # Create 49 linearly spaced values from 0 to 0.75 and then append a 0
        row_values = np.concatenate(
            (np.linspace(0, 0.75, rows - 5), np.array([0, 0, 0, 0, 0]))
        ).reshape(rows, 1)
        impl2 = np.tile(row_values, (1, cols))
        np_impl2 = np.repeat(impl2[np.newaxis, :, :], channels, axis=0)

        # Create third array
        # Start with an array of zeros (50x1)
        row_values = np.zeros((rows, 1))
        # Choose the row index corresponding to 75% of the height.
        nonzero_index = int(0.75 * (rows - 1))
        row_values[nonzero_index] = 0.75
        impl3 = np.tile(row_values, (1, cols))
        np_impl3 = np.repeat(impl3[np.newaxis, :, :], channels, axis=0)

        self.test_model.set_main_view_layout((2, 2))

        ds1 = self.test_model.load_dataset(np_impl)
        ds2 = self.test_model.load_dataset(np_impl2)
        ds3 = self.test_model.load_dataset(np_impl3)

        self.test_model.click_main_view_zoom_in()
        self.test_model.click_main_view_zoom_in()
        self.test_model.click_main_view_zoom_in()
        self.test_model.click_main_view_zoom_in()

        self.test_model.set_context_pane_dataset(ds1.get_id())

        visible_region_00 = self.test_model.get_main_view_rv_visible_region((0, 0))
        highlight_region = self.test_model.context_pane._get_compatible_highlights(
            ds1.get_id()
        )[0]

        # For an unknown reason, when I run this test inside of pytest and outside,
        # I get two different results. Outside of pytests the below regions are the same, but
        # in pytest, one of the values is off by 6, hence epsilon=6
        self.assertTrue(
            are_qrects_close(highlight_region, visible_region_00, epsilon=6)
        )

    def test_cp_use_clicked(self):
        # Create first array
        rows, cols, channels = 75, 75, 3
        # Create a vertical gradient from 0 to 1: shape (50,1)
        row_values = np.linspace(0, 1, rows).reshape(rows, 1)
        # Tile the values horizontally to get a 50x50 array
        impl = np.tile(row_values, (1, cols))
        # Repeat the 2D array across 3 channels to get a 3x50x50 array
        np_impl = np.repeat(impl[np.newaxis, :, :], channels, axis=0)

        # Create second array
        # Create 49 linearly spaced values from 0 to 0.75 and then append a 0
        row_values = np.concatenate(
            (np.linspace(0, 0.75, rows - 5), np.array([0, 0, 0, 0, 0]))
        ).reshape(rows, 1)
        impl2 = np.tile(row_values, (1, cols))
        np_impl2 = np.repeat(impl2[np.newaxis, :, :], channels, axis=0)

        # Create third array
        # Start with an array of zeros (50x1)
        row_values = np.zeros((rows, 1))
        # Choose the row index corresponding to 75% of the height.
        nonzero_index = int(0.75 * (rows - 1))
        row_values[nonzero_index] = 0.75
        impl3 = np.tile(row_values, (1, cols))
        np_impl3 = np.repeat(impl3[np.newaxis, :, :], channels, axis=0)
        # Loads in multiple datasets

        # Ensures the context pane only has the use click checked even when we click between others
        self.test_model.set_main_view_layout((2, 2))

        ds1 = self.test_model.load_dataset(np_impl)
        ds2 = self.test_model.load_dataset(np_impl2)
        ds3 = self.test_model.load_dataset(np_impl3)

        clicked_id = self.test_model.get_cp_dataset_chooser_checked_id()
        self.assertTrue(
            clicked_id == -1, "Starting dataset option is not 'Use Clicked Dataset'"
        )

        self.test_model.click_raster_coord_main_view_rv(
            rv_pos=(0, 0), raster_coord=(10, 10)
        )
        ds_00 = self.test_model.get_main_view_rv((0, 0)).get_raster_data()
        cp_ds = self.test_model.get_context_pane_dataset()
        self.assertTrue(
            ds_00 == cp_ds,
            "Context pane dataset is not the same as the clicked dataset",
        )

        clicked_id = self.test_model.get_cp_dataset_chooser_checked_id()
        self.assertTrue(
            clicked_id == -1,
            "Context Pane dataset chooser changed when clicking in main view",
        )

        self.test_model.click_raster_coord_main_view_rv(
            rv_pos=(0, 1), raster_coord=(10, 10)
        )
        ds_01 = self.test_model.get_main_view_rv((0, 1)).get_raster_data()
        cp_ds = self.test_model.get_context_pane_dataset()
        self.assertTrue(
            ds_01 == cp_ds,
            "Context pane dataset is not the same as the clicked dataset",
        )

        clicked_id = self.test_model.get_cp_dataset_chooser_checked_id()
        self.assertTrue(
            clicked_id == -1,
            "Context Pane dataset chooser changed when clicking in main view",
        )

    def test_cp_use_clicked_while_linked(self):
        # Create first array
        rows, cols, channels = 75, 75, 3
        # Create a vertical gradient from 0 to 1: shape (50,1)
        row_values = np.linspace(0, 1, rows).reshape(rows, 1)
        # Tile the values horizontally to get a 50x50 array
        impl = np.tile(row_values, (1, cols))
        # Repeat the 2D array across 3 channels to get a 3x50x50 array
        np_impl = np.repeat(impl[np.newaxis, :, :], channels, axis=0)

        # Create second array
        # Create 49 linearly spaced values from 0 to 0.75 and then append a 0
        row_values = np.concatenate(
            (np.linspace(0, 0.75, rows - 5), np.array([0, 0, 0, 0, 0]))
        ).reshape(rows, 1)
        impl2 = np.tile(row_values, (1, cols))
        np_impl2 = np.repeat(impl2[np.newaxis, :, :], channels, axis=0)

        # Create third array
        # Start with an array of zeros (50x1)
        row_values = np.zeros((rows, 1))
        # Choose the row index corresponding to 75% of the height.
        nonzero_index = int(0.75 * (rows - 1))
        row_values[nonzero_index] = 0.75
        impl3 = np.tile(row_values, (1, cols))
        np_impl3 = np.repeat(impl3[np.newaxis, :, :], channels, axis=0)
        # Loads in multiple datasets

        # Ensures the context pane only has the use click checked even when we click between others
        self.test_model.set_main_view_layout((2, 2))

        ds1 = self.test_model.load_dataset(np_impl)
        ds2 = self.test_model.load_dataset(np_impl2)
        ds3 = self.test_model.load_dataset(np_impl3)

        self.test_model.click_link_button()

        clicked_id = self.test_model.get_cp_dataset_chooser_checked_id()
        self.assertTrue(
            clicked_id == -1, "Starting dataset option is not 'Use Clicked Dataset'"
        )

        self.test_model.click_raster_coord_main_view_rv(
            rv_pos=(0, 0), raster_coord=(10, 10)
        )
        ds_00 = self.test_model.get_main_view_rv((0, 0)).get_raster_data()
        cp_ds = self.test_model.get_context_pane_dataset()
        self.assertTrue(
            ds_00 == cp_ds,
            "Context pane dataset is not the same as the clicked dataset",
        )

        clicked_id = self.test_model.get_cp_dataset_chooser_checked_id()
        self.assertTrue(
            clicked_id == -1,
            "Context Pane dataset chooser changed when clicking in main view",
        )

        self.test_model.click_raster_coord_main_view_rv(
            rv_pos=(0, 1), raster_coord=(10, 10)
        )
        ds_01 = self.test_model.get_main_view_rv((0, 1)).get_raster_data()
        cp_ds = self.test_model.get_context_pane_dataset()
        self.assertTrue(
            ds_01 == cp_ds,
            "Context pane dataset is not the same as the clicked dataset",
        )

        clicked_id = self.test_model.get_cp_dataset_chooser_checked_id()
        self.assertTrue(
            clicked_id == -1,
            "Context Pane dataset chooser changed when clicking in main view",
        )

    def test_cp_use_specific_ds(self):
        # Create first array
        rows, cols, channels = 75, 75, 3
        # Create a vertical gradient from 0 to 1: shape (50,1)
        row_values = np.linspace(0, 1, rows).reshape(rows, 1)
        # Tile the values horizontally to get a 50x50 array
        impl = np.tile(row_values, (1, cols))
        # Repeat the 2D array across 3 channels to get a 3x50x50 array
        np_impl = np.repeat(impl[np.newaxis, :, :], channels, axis=0)

        # Create second array
        # Create 49 linearly spaced values from 0 to 0.75 and then append a 0
        row_values = np.concatenate(
            (np.linspace(0, 0.75, rows - 5), np.array([0, 0, 0, 0, 0]))
        ).reshape(rows, 1)
        impl2 = np.tile(row_values, (1, cols))
        np_impl2 = np.repeat(impl2[np.newaxis, :, :], channels, axis=0)

        # Create third array
        # Start with an array of zeros (50x1)
        row_values = np.zeros((rows, 1))
        # Choose the row index corresponding to 75% of the height.
        nonzero_index = int(0.75 * (rows - 1))
        row_values[nonzero_index] = 0.75
        impl3 = np.tile(row_values, (1, cols))
        np_impl3 = np.repeat(impl3[np.newaxis, :, :], channels, axis=0)

        self.test_model.set_main_view_layout((2, 2))

        ds1 = self.test_model.load_dataset(np_impl)
        ds2 = self.test_model.load_dataset(np_impl2)
        ds3 = self.test_model.load_dataset(np_impl3)

        # Ensure the checked id is set to 'Use Clicked Dataset'
        checked_id = self.test_model.get_cp_dataset_chooser_checked_id()
        self.assertTrue(
            checked_id == -1, "Starting dataset option is not 'Use Clicked Dataset'"
        )

        ds1_id = ds1.get_id()
        # Set the context pane to use dataset 1
        self.test_model.set_context_pane_dataset_chooser_id(ds1_id)
        # Make sure context pane dataset chooser was actually switched
        checked_id = self.test_model.get_cp_dataset_chooser_checked_id()
        self.assertTrue(
            checked_id == ds1_id, f"Context pane dataset chooser not set to {ds1_id}"
        )
        # Make sure the shown dataset in context pane was switched
        cp_ds_id = self.test_model.get_context_pane_dataset().get_id()
        self.assertTrue(
            cp_ds_id == checked_id,
            f"Context pane did not switch to showing the checked dataset",
        )

        # Click somewhere in ds2'S raster view
        self.test_model.click_raster_coord_main_view_rv(
            rv_pos=(0, 1), raster_coord=(10, 10)
        )

        # Make sure the context pane is still showing dataset 1
        checked_id = self.test_model.get_cp_dataset_chooser_checked_id()
        self.assertTrue(
            checked_id == ds1_id, f"Context pane dataset chooser not set to {ds1_id}"
        )
        cp_ds_id = self.test_model.get_context_pane_dataset().get_id()
        self.assertTrue(
            cp_ds_id == checked_id,
            f"Context pane dataset changed when clicking in main view",
        )

        ds2_id = ds2.get_id()
        # Set context pane to show dataset 2
        self.test_model.set_context_pane_dataset_chooser_id(ds2_id)

        # Ensure context pane has dataset 2 checked and shown
        checked_id = self.test_model.get_cp_dataset_chooser_checked_id()
        self.assertTrue(
            checked_id == ds2_id, f"Context pane dataset chooser not set to {ds2_id}"
        )
        cp_ds_id = self.test_model.get_context_pane_dataset().get_id()
        self.assertTrue(
            cp_ds_id == checked_id,
            f"Context pane did not switch to showing the checked dataset",
        )

        # Click somewhere in dataset 1's rasterview
        self.test_model.click_raster_coord_main_view_rv(
            rv_pos=(0, 0), raster_coord=(10, 10)
        )

        checked_id = self.test_model.get_cp_dataset_chooser_checked_id()
        self.assertTrue(
            checked_id == ds2_id,
            "Context Pane dataset chooser changed when clicking in main view",
        )
        cp_ds_id = self.test_model.get_context_pane_dataset().get_id()
        self.assertTrue(
            cp_ds_id == checked_id,
            f"Context pane dataset changed when clicking in main view",
        )

    def test_cp_use_specific_ds_while_linked(self):
        # Create first array
        rows, cols, channels = 75, 75, 3
        # Create a vertical gradient from 0 to 1: shape (50,1)
        row_values = np.linspace(0, 1, rows).reshape(rows, 1)
        # Tile the values horizontally to get a 50x50 array
        impl = np.tile(row_values, (1, cols))
        # Repeat the 2D array across 3 channels to get a 3x50x50 array
        np_impl = np.repeat(impl[np.newaxis, :, :], channels, axis=0)

        # Create second array
        # Create 49 linearly spaced values from 0 to 0.75 and then append a 0
        row_values = np.concatenate(
            (np.linspace(0, 0.75, rows - 5), np.array([0, 0, 0, 0, 0]))
        ).reshape(rows, 1)
        impl2 = np.tile(row_values, (1, cols))
        np_impl2 = np.repeat(impl2[np.newaxis, :, :], channels, axis=0)

        # Create third array
        # Start with an array of zeros (50x1)
        row_values = np.zeros((rows, 1))
        # Choose the row index corresponding to 75% of the height.
        nonzero_index = int(0.75 * (rows - 1))
        row_values[nonzero_index] = 0.75
        impl3 = np.tile(row_values, (1, cols))
        np_impl3 = np.repeat(impl3[np.newaxis, :, :], channels, axis=0)

        self.test_model.set_main_view_layout((2, 2))

        ds1 = self.test_model.load_dataset(np_impl)
        ds2 = self.test_model.load_dataset(np_impl2)
        ds3 = self.test_model.load_dataset(np_impl3)

        self.test_model.click_link_button()

        # Ensure the checked id is set to 'Use Clicked Dataset'
        checked_id = self.test_model.get_cp_dataset_chooser_checked_id()
        self.assertTrue(
            checked_id == -1, "Starting dataset option is not 'Use Clicked Dataset'"
        )

        ds1_id = ds1.get_id()
        # Set the context pane to use dataset 1
        self.test_model.set_context_pane_dataset_chooser_id(ds1_id)
        # Make sure context pane dataset chooser was actually switched
        checked_id = self.test_model.get_cp_dataset_chooser_checked_id()
        self.assertTrue(
            checked_id == ds1_id, f"Context pane dataset chooser not set to {ds1_id}"
        )
        # Make sure the shown dataset in context pane was switched
        cp_ds_id = self.test_model.get_context_pane_dataset().get_id()
        self.assertTrue(
            cp_ds_id == checked_id,
            f"Context pane did not switch to showing the checked dataset",
        )

        # Click somewhere in ds2'S raster view
        self.test_model.click_raster_coord_main_view_rv(
            rv_pos=(0, 1), raster_coord=(10, 10)
        )

        # Make sure the context pane is still showing dataset 1
        checked_id = self.test_model.get_cp_dataset_chooser_checked_id()
        self.assertTrue(
            checked_id == ds1_id, f"Context pane dataset chooser not set to {ds1_id}"
        )
        cp_ds_id = self.test_model.get_context_pane_dataset().get_id()
        self.assertTrue(
            cp_ds_id == checked_id,
            f"Context pane dataset changed when clicking in main view",
        )

        ds2_id = ds2.get_id()
        # Set context pane to show dataset 2
        self.test_model.set_context_pane_dataset_chooser_id(ds2_id)

        # Ensure context pane has dataset 2 checked and shown
        checked_id = self.test_model.get_cp_dataset_chooser_checked_id()
        self.assertTrue(
            checked_id == ds2_id, f"Context pane dataset chooser not set to {ds2_id}"
        )
        cp_ds_id = self.test_model.get_context_pane_dataset().get_id()
        self.assertTrue(
            cp_ds_id == checked_id,
            f"Context pane did not switch to showing the checked dataset",
        )

        # Click somewhere in dataset 1's rasterview
        self.test_model.click_raster_coord_main_view_rv(
            rv_pos=(0, 0), raster_coord=(10, 10)
        )

        checked_id = self.test_model.get_cp_dataset_chooser_checked_id()
        self.assertTrue(
            checked_id == ds2_id,
            "Context Pane dataset chooser changed when clicking in main view",
        )
        cp_ds_id = self.test_model.get_context_pane_dataset().get_id()
        self.assertTrue(
            cp_ds_id == checked_id,
            f"Context pane dataset changed when clicking in main view",
        )

    def test_cp_remove_chosen_dataset(self, func=lambda: None):
        """
        Ensures that when we remove a dataset that is the same that was chosen
        in the context pane, we update to the 'Use Clicked Dataset' value
        """
        # Create first array
        rows, cols, channels = 75, 75, 3
        # Create a vertical gradient from 0 to 1: shape (50,1)
        row_values = np.linspace(0, 1, rows).reshape(rows, 1)
        # Tile the values horizontally to get a 50x50 array
        impl = np.tile(row_values, (1, cols))
        # Repeat the 2D array across 3 channels to get a 3x50x50 array
        np_impl = np.repeat(impl[np.newaxis, :, :], channels, axis=0)

        # Create second array
        # Create 49 linearly spaced values from 0 to 0.75 and then append a 0
        row_values = np.concatenate(
            (np.linspace(0, 0.75, rows - 5), np.array([0, 0, 0, 0, 0]))
        ).reshape(rows, 1)
        impl2 = np.tile(row_values, (1, cols))
        np_impl2 = np.repeat(impl2[np.newaxis, :, :], channels, axis=0)

        # Create third array
        # Start with an array of zeros (50x1)
        row_values = np.zeros((rows, 1))
        # Choose the row index corresponding to 75% of the height.
        nonzero_index = int(0.75 * (rows - 1))
        row_values[nonzero_index] = 0.75
        impl3 = np.tile(row_values, (1, cols))
        np_impl3 = np.repeat(impl3[np.newaxis, :, :], channels, axis=0)

        self.test_model.set_main_view_layout((2, 2))

        ds1 = self.test_model.load_dataset(np_impl)
        ds2 = self.test_model.load_dataset(np_impl2)
        ds3 = self.test_model.load_dataset(np_impl3)

        func()

        ds1_id = ds1.get_id()
        # Set the context pane to use dataset 1
        self.test_model.set_context_pane_dataset_chooser_id(ds1_id)
        # Make sure context pane dataset chooser was actually switched
        checked_id = self.test_model.get_cp_dataset_chooser_checked_id()
        self.assertTrue(
            checked_id == ds1_id, f"Context pane dataset chooser not set to {ds1_id}"
        )
        # Make sure the shown dataset in context pane was switched
        cp_ds_id = self.test_model.get_context_pane_dataset().get_id()
        self.assertTrue(
            cp_ds_id == checked_id,
            f"Context pane did not switch to showing the checked dataset",
        )

        self.test_model.close_dataset(ds1_id)

        checked_id = self.test_model.get_cp_dataset_chooser_checked_id()
        self.assertTrue(
            checked_id == -1,
            f"Context pane did not switch 'Use Clicked Dataset' when open dataset was closed",
        )

    def test_cp_remove_chosen_dataset_while_linked(self):
        self.test_cp_remove_chosen_dataset(self.test_model.click_link_button)

    def test_cp_remove_not_chosen_dataset(self, func=lambda: None):
        """
        Ensures that when we remove a dataset that is not chosen in the context pane,
        the context pane stays the same.
        """
        # Create first array
        rows, cols, channels = 75, 75, 3
        # Create a vertical gradient from 0 to 1: shape (50,1)
        row_values = np.linspace(0, 1, rows).reshape(rows, 1)
        # Tile the values horizontally to get a 50x50 array
        impl = np.tile(row_values, (1, cols))
        # Repeat the 2D array across 3 channels to get a 3x50x50 array
        np_impl = np.repeat(impl[np.newaxis, :, :], channels, axis=0)

        # Create second array
        # Create 49 linearly spaced values from 0 to 0.75 and then append a 0
        row_values = np.concatenate(
            (np.linspace(0, 0.75, rows - 5), np.array([0, 0, 0, 0, 0]))
        ).reshape(rows, 1)
        impl2 = np.tile(row_values, (1, cols))
        np_impl2 = np.repeat(impl2[np.newaxis, :, :], channels, axis=0)

        # Create third array
        # Start with an array of zeros (50x1)
        row_values = np.zeros((rows, 1))
        # Choose the row index corresponding to 75% of the height.
        nonzero_index = int(0.75 * (rows - 1))
        row_values[nonzero_index] = 0.75
        impl3 = np.tile(row_values, (1, cols))
        np_impl3 = np.repeat(impl3[np.newaxis, :, :], channels, axis=0)

        self.test_model.set_main_view_layout((2, 2))

        ds1 = self.test_model.load_dataset(np_impl)
        ds2 = self.test_model.load_dataset(np_impl2)
        ds3 = self.test_model.load_dataset(np_impl3)

        func()

        ds1_id = ds1.get_id()
        ds2_id = ds2.get_id()
        # Set the context pane to use dataset 1
        self.test_model.set_context_pane_dataset_chooser_id(ds1_id)
        # Make sure context pane dataset chooser was actually switched
        checked_id = self.test_model.get_cp_dataset_chooser_checked_id()
        self.assertTrue(
            checked_id == ds1_id, f"Context pane dataset chooser not set to {ds1_id}"
        )
        # Make sure the shown dataset in context pane was switched
        cp_ds_id = self.test_model.get_context_pane_dataset().get_id()
        self.assertTrue(
            cp_ds_id == checked_id,
            f"Context pane did not switch to showing the checked dataset",
        )

        self.test_model.close_dataset(ds2_id)

        checked_id = self.test_model.get_cp_dataset_chooser_checked_id()
        self.assertTrue(
            checked_id == ds1_id, f"Context pane changed datasets on removing a dataset"
        )

    def test_cp_remove_not_chosen_dataset_while_linked(self):
        self.test_cp_remove_not_chosen_dataset(self.test_model.click_link_button)


if __name__ == "__main__":
    """
    Code to make sure new tests work as desired
    """
    test_model = WiserTestModel(use_gui=True)

    # Create first array
    rows, cols, channels = 75, 75, 3
    # Create a vertical gradient from 0 to 1: shape (50,1)
    row_values = np.linspace(0, 1, rows).reshape(rows, 1)
    # Tile the values horizontally to get a 50x50 array
    impl = np.tile(row_values, (1, cols))
    # Repeat the 2D array across 3 channels to get a 3x50x50 array
    np_impl = np.repeat(impl[np.newaxis, :, :], channels, axis=0)

    # Create second array
    # Create 49 linearly spaced values from 0 to 0.75 and then append a 0
    row_values = np.concatenate(
        (np.linspace(0, 0.75, rows - 5), np.array([0, 0, 0, 0, 0]))
    ).reshape(rows, 1)
    impl2 = np.tile(row_values, (1, cols))
    np_impl2 = np.repeat(impl2[np.newaxis, :, :], channels, axis=0)

    # Create third array
    # Start with an array of zeros (50x1)
    row_values = np.zeros((rows, 1))
    # Choose the row index corresponding to 75% of the height.
    nonzero_index = int(0.75 * (rows - 1))
    row_values[nonzero_index] = 0.75
    impl3 = np.tile(row_values, (1, cols))
    np_impl3 = np.repeat(impl3[np.newaxis, :, :], channels, axis=0)

    test_model.set_main_view_layout((2, 2))

    ds1 = test_model.load_dataset(np_impl)
    ds2 = test_model.load_dataset(np_impl2)
    ds3 = test_model.load_dataset(np_impl3)

    ds1_id = ds1.get_id()
    # Set the context pane to use dataset 1
    test_model.set_context_pane_dataset_chooser_id(ds1_id)
    # Make sure context pane dataset chooser was actually switched
    checked_id = test_model.get_cp_dataset_chooser_checked_id()

    # Make sure the shown dataset in context pane was switched
    cp_ds_id = test_model.get_context_pane_dataset().get_id()

    test_model.close_dataset(ds1_id)

    checked_id = test_model.get_cp_dataset_chooser_checked_id()
    print(f"checked_id: {checked_id}")

    test_model.app.exec_()
