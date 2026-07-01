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

from typing import Optional

from PySide6.QtCore import *
from PySide6.QtGui import *
from PySide6.QtWidgets import *

from .app_state import ApplicationState
from .app_services import AppServices
from .mosaic_pane import MosaicPane


class SeamlessMosaicDialog(QDialog):
    """
    Non-modal shell window for building a seamless mosaic.

    Behavior-free in this issue: it constructs the widget tree (dialog -> pane ->
    view) and nothing more.
    """

    def __init__(
        self,
        app_state: ApplicationState,
        app_services: AppServices,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent=parent)
        self._app_state = app_state
        self._app_services = app_services

        self.setModal(False)
        self.setWindowTitle(self.tr("Seamless Mosaic"))
        self.resize(1000, 700)

        self._mosaic_pane = MosaicPane(
            app_state=app_state,
            app_services=app_services,
            parent=self,
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._mosaic_pane)

    def get_mosaic_pane(self) -> MosaicPane:
        return self._mosaic_pane
