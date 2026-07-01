"""
Control panel for the Seamless Mosaic feature (EPIC #629).

Scaffolding only (issue #633): the pane lays out the :class:`MosaicView` alongside a
placeholder controls area, and owns the non-GUI :class:`MosaicController` that both
share. It is modeled on :class:`~wiser.gui.rasterpane.RasterPane` /
:class:`~wiser.gui.similarity_transform_pane.SimilarityTransformPane` for how a
feature-specific pane hosts its view.

The controls area is intentionally empty here; #638 fills it in with the scene list
(drag-to-reorder for z-order), per-scene visibility toggles, resolution / CRS /
resampling selectors, the band-metadata chooser, and the Export/Finish action.
"""

from typing import Optional

from PySide6.QtCore import *
from PySide6.QtGui import *
from PySide6.QtWidgets import *

from .app_state import ApplicationState
from .app_services import AppServices
from .mosaic_view import MosaicView

from wiser.raster.mosaic_controller import MosaicController


class MosaicPane(QWidget):
    """
    Hosts a :class:`MosaicView` plus a (placeholder) controls area.

    Behavior-free in this issue. Creates the shared :class:`MosaicController` and
    wires it into the view so that the model is the single source of truth for both
    the rendering (#636/#637) and the control panel (#638).
    """

    def __init__(
        self,
        app_state: ApplicationState,
        app_services: Optional[AppServices] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent=parent)
        self._app_state = app_state
        self._app_services = app_services
        self._controller = MosaicController()

        self._init_ui()

    def _init_ui(self) -> None:
        # View on the left, controls on the right, split so the panel can grow later.
        self._splitter = QSplitter(Qt.Horizontal, self)

        self._mosaic_view = MosaicView(
            parent=self._splitter,
            app_state=self._app_state,
            controller=self._controller,
            mosaicpane=self,
        )
        self._splitter.addWidget(self._mosaic_view)

        # Placeholder controls area (#638 fills this in).
        self._controls = QWidget(self._splitter)
        self._controls_layout = QVBoxLayout(self._controls)
        self._controls_layout.addStretch(1)
        self._splitter.addWidget(self._controls)

        # Give the view the bulk of the width; controls stay a slim side panel.
        self._splitter.setStretchFactor(0, 1)
        self._splitter.setStretchFactor(1, 0)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._splitter)

    def get_controller(self) -> MosaicController:
        return self._controller

    def get_mosaic_view(self) -> MosaicView:
        return self._mosaic_view
