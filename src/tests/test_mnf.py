import unittest
from pathlib import Path
from unittest.mock import patch

import pytest
import tests.context

from wiser.gui.app_services import AppServices
from wiser.gui.mnf import MinimumNoiseFractionDialog
from wiser.raster.loader import RasterDataLoader
from wiser.utils.storage_client import StorageClient
from wiser.utils.storage_layer import ExternalRasterHandle

pytestmark = [
    pytest.mark.integration,
]


class TestMnf(unittest.TestCase):
    def test_perform_mnf_runs_with_app_services_and_waits_for_future(self) -> None:
        dataset_path = (
            Path(__file__).resolve().parent / ".." / "test_utils" / "test_datasets" / "caltech_425_7_7_nm"
        ).resolve()
        dataset = RasterDataLoader().load_from_file(str(dataset_path), interactive=False)[0]

        app_services = AppServices()

        try:
            dataset_ref = app_services.storage_service.register_external(
                ExternalRasterHandle(dataset_obj=dataset)
            )
            dialog = MinimumNoiseFractionDialog(app_services=app_services)

            future = dialog.perform_mnf(dataset_ref)

            future.result(timeout=120)
        finally:
            app_services.scheduler.shutdown(wait=True)
            app_services.storage_service.close()
