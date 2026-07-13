"""
Top-level dialog shell for the Seamless Mosaic feature (EPIC #629).

Scaffolding only (issue #633): a non-modal :class:`QDialog` that hosts a
:class:`MosaicPane`. It follows the dependency-injection convention of the other tool
dialogs — ``(app_state, app_services, parent)`` — see
:class:`~wiser.gui.kmeans.KMeansDialog` /
:class:`~wiser.gui.linear_unmixing.LinearUnmixingDialog`.

All real workflow (add scenes, reorder, choose grid/CRS/resampling, export) lands in
later issues via the :class:`MosaicPane` and :class:`MosaicController`.
"""

from typing import Callable, Optional, Tuple, TYPE_CHECKING

from PySide6.QtCore import *
from PySide6.QtGui import *
from PySide6.QtWidgets import *

from .app_state import ApplicationState
from .app_services import AppServices
from .mosaic_pane import MosaicPane

from wiser.raster.mosaic_materialize import SceneMaterializer

if TYPE_CHECKING:
    from wiser.raster.dataset import RasterDataSet

# Resolves a dataset to the display bands currently shown for it in the main view
# (or None if not shown). Injected by the app so the mosaic can honor the user's live
# band-chooser choice when baking display-only preview artifacts (#677).
DisplayBandsResolver = Callable[["RasterDataSet"], Optional[Tuple[int, ...]]]


class SeamlessMosaicDialog(QDialog):
    """
    Non-modal shell window for building a seamless mosaic.

    Owns a session-scoped :class:`SceneMaterializer` (#634) that turns each added
    :class:`RasterDataSet` into a warpable temp GeoTIFF, and passes it to the
    :class:`MosaicPane` that drives ingestion.
    """

    def __init__(
        self,
        app_state: ApplicationState,
        app_services: AppServices,
        display_bands_resolver: Optional[DisplayBandsResolver] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent=parent)
        self._app_state = app_state
        self._app_services = app_services
        self._display_bands_resolver = display_bands_resolver

        self.setModal(False)
        self.setWindowTitle(self.tr("Seamless Mosaic"))
        self.resize(1000, 700)

        # One materializer per dialog instance. The main window caches and *reuses*
        # this dialog across open/close (a mosaic is a long-lived, resumable
        # workflow), so the materialized temp files must survive `close()` — cleaning
        # up here on `closeEvent` would orphan every added scene's `gdal_path` on the
        # next open. Instead we tear the temp dir down when the dialog is actually
        # destroyed (app teardown); `TemporaryDirectory`'s own finalizer is the
        # backstop on GC / interpreter exit.
        self._materializer = SceneMaterializer()
        materializer = self._materializer
        self.destroyed.connect(lambda *_: materializer.close())

        self._mosaic_pane = MosaicPane(
            app_state=app_state,
            app_services=app_services,
            materializer=self._materializer,
            display_bands_resolver=display_bands_resolver,
            parent=self,
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._mosaic_pane)

    def get_mosaic_pane(self) -> MosaicPane:
        return self._mosaic_pane
