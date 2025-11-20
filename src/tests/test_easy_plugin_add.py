import os
import unittest

import tests.context

from pathlib import Path

from test_utils.test_model import WiserTestModel

import pytest

pytestmarker = [
    pytest.mark.functional,
]


class TestAddingPlugins(unittest.TestCase):
    """
    Tests to make sure that we can extract the plugins from plugin files
    """

    def setUp(self):
        """Sets up a fresh WISER test model before each test."""
        self.test_model = WiserTestModel()

    def tearDown(self):
        """Cleans up the WISER application and test model after each test."""
        self.test_model.close_app()
        del self.test_model

    def test_extract_bandmath_plugin(self):
        folder_path = os.path.dirname(__file__)
        bandmath_plugin_file_path = os.path.join(
            folder_path,
            "..",
            "example_plugins",
            "bandmath_plugin.py",
        )

        plugins, base_dir_path = self.test_model.load_plugin_by_file(bandmath_plugin_file_path)
        plugin_fqcn = plugins[0]["fqcn"]
        plugin_name = plugins[0]["name"]

        truth_base_dir_path = os.path.join(os.path.dirname(__file__), "..")
        truth_plugin_fqcn = "example_plugins.bandmath_plugin.SpectralAnglePlugin"
        truth_plugin_name = "SpectralAnglePlugin"

        assert Path(base_dir_path).resolve(strict=False) == Path(truth_base_dir_path).resolve(strict=False)
        assert plugin_fqcn == truth_plugin_fqcn
        assert plugin_name == truth_plugin_name

    def test_extract_ctx_menu_plugin(self):
        folder_path = os.path.dirname(__file__)
        plugin_file_name = "ctxmenu_plugin"
        bandmath_plugin_file_path = os.path.join(
            folder_path,
            "..",
            "example_plugins",
            f"{plugin_file_name}.py",
        )

        plugins, base_dir_path = self.test_model.load_plugin_by_file(bandmath_plugin_file_path)
        plugin_fqcn = plugins[0]["fqcn"]
        plugin_name = plugins[0]["name"]

        truth_base_dir_path = os.path.join(os.path.dirname(__file__), "..")
        truth_plugin_name = "HelloContextPlugin"
        truth_plugin_fqcn = f"example_plugins.{plugin_file_name}.{truth_plugin_name}"

        assert Path(base_dir_path).resolve(strict=False) == Path(truth_base_dir_path).resolve(strict=False)
        assert plugin_fqcn == truth_plugin_fqcn
        assert plugin_name == truth_plugin_name

    def test_extract_tool_plugin(self):
        folder_path = os.path.dirname(__file__)
        plugin_file_name = "tool_plugin"
        bandmath_plugin_file_path = os.path.join(
            folder_path,
            "..",
            "example_plugins",
            f"{plugin_file_name}.py",
        )

        plugins, base_dir_path = self.test_model.load_plugin_by_file(bandmath_plugin_file_path)
        plugin_fqcn = plugins[0]["fqcn"]
        plugin_name = plugins[0]["name"]

        truth_base_dir_path = os.path.join(os.path.dirname(__file__), "..")
        truth_plugin_name = "HelloToolPlugin"
        truth_plugin_fqcn = f"example_plugins.{plugin_file_name}.{truth_plugin_name}"

        assert Path(base_dir_path).resolve(strict=False) == Path(truth_base_dir_path).resolve(strict=False)
        assert plugin_fqcn == truth_plugin_fqcn
        assert plugin_name == truth_plugin_name
