import os
from typing import List, Optional, Dict, Tuple

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

from wiser.gui.rasterpane import RasterPane
from wiser.gui.similarity_transform_pane import SimilarityTransformPane
from wiser.gui.app_state import ApplicationState

from wiser.raster.dataset import RasterDataSet, pixel_coord_to_geo_coord

from .generated.similarity_transform_dialog_ui import Ui_SimilarityTransform


class SimilarityTransformDialog(QDialog):

    def __init__(self, app_state: ApplicationState, parent=None):
        super().__init__(parent=parent)

        self._ui = Ui_SimilarityTransform()
        self._ui.setupUi(self)

        # --- Image renderings ----------------------------------------------------
        self._rotate_scale_pane = SimilarityTransformPane(app_state)
        self._translate_pane = SimilarityTransformPane(app_state, translation=True)
        self._translate_pane.pixel_selected_for_translation.connect(self._on_translation_pixel_selected)
        self._init_rasterpanes()

        # --- Internal state ----------------------------------------------------
        self._image_rotation: float = 0.0  # degrees CCW
        self._image_scale: float = 1.0     # isotropic scale factor
        self._lat_north_translate: float = 0.0
        self._lon_east_translate: float = 0.0

        # --- Validators --------------------------------------------------------
        rot_validator = QDoubleValidator(0.0, 360.0, 4, self)
        rot_validator.setNotation(QDoubleValidator.StandardNotation)
        self._ui.ledit_rotation.setValidator(rot_validator)

        scale_validator = QDoubleValidator(0.0, 100.0, 6, self)
        scale_validator.setNotation(QDoubleValidator.StandardNotation)
        self._ui.ledit_scale.setValidator(scale_validator)

        # --- Defaults ----------------------------------------------------------
        self._ui.ledit_rotation.setText("0.0")
        self._ui.ledit_scale.setText("1.0")

        self._ui.slider_rotation.setRange(0, 360)
        self._ui.slider_rotation.setSingleStep(1)
        self._ui.slider_rotation.setValue(0)

        # --- Signal ↔ slot wiring ----------------------------------------------
        # 1. Rotation – keep QLineEdit and QSlider in sync, but avoid infinite loops.
        self._ui.ledit_rotation.textEdited.connect(self._on_rotation_edited)
        self._ui.slider_rotation.valueChanged.connect(self._on_slider_rotation_changed)

        # 2. Scale.
        self._ui.ledit_scale.textEdited.connect(self._on_scale_edited)

        # 3. Buttons.
        self._ui.btn_rotate_scale.clicked.connect(self._on_run_rotate_scale)
        self._ui.btn_create_translation.clicked.connect(self._on_create_translation)

        # 4. Translation edits.
        self._ui.ledit_lat_north.editingFinished.connect(self._on_lat_north_changed)
        self._ui.ledit_lon_east.editingFinished.connect(self._on_lon_east_changed)


    # -------------------------------------------------------------------------
    # Initializers
    # -------------------------------------------------------------------------

    def _init_rasterpanes(self):
        rotate_scale_layout = QVBoxLayout(self._ui.widget_rotate_scale)
        self._ui.widget_rotate_scale.setLayout(rotate_scale_layout)
        rotate_scale_layout.addWidget(self._rotate_scale_pane)

        translate_layout = QVBoxLayout(self._ui.widget_translate)
        self._ui.widget_translate.setLayout(translate_layout)
        translate_layout.addWidget(self._translate_pane)


    # -------------------------------------------------------------------------
    # Rotation handlers
    # -------------------------------------------------------------------------

    @Slot(str)
    def _on_rotation_edited(self, text: str) -> None:
        """User typed a rotation value – sync slider and store state."""
        if not text:
            return  # ignore empty edits
        try:
            value = float(text)
        except ValueError:
            return  # validator should prevent this

        # Clamp to [0, 360]
        value = max(0.0, min(360.0, value))
        self._image_rotation = value

        # Update slider without triggering its signal.
        self._ui.slider_rotation.blockSignals(True)
        self._ui.slider_rotation.setValue(int(round(value)))
        self._ui.slider_rotation.blockSignals(False)
        print(f"self._image_rotation: {self._image_rotation}")
        self._rotate_scale_pane.rotate_and_scale_rasterview(self._image_rotation, self._image_scale)

    @Slot(int)
    def _on_slider_rotation_changed(self, value: int) -> None:
        """Slider moved by user – sync line‑edit and store integer rotation."""
        # Update line‑edit without re‑entering _on_rotation_edited.
        self._ui.ledit_rotation.blockSignals(True)
        self._ui.ledit_rotation.setText(f"{value}")
        self._ui.ledit_rotation.blockSignals(False)

        self._image_rotation = float(value)
        print(f"self._image_rotation: {self._image_rotation}")

        self._rotate_scale_pane.rotate_and_scale_rasterview(self._image_rotation, self._image_scale)

    # -------------------------------------------------------------------------
    # Scale handlers
    # -------------------------------------------------------------------------

    @Slot(str)
    def _on_scale_edited(self, text: str) -> None:
        """Scale edited – store value between 0 and 100 (float)."""
        if not text:
            return
        try:
            value = float(text)
        except ValueError:
            return

        # Clamp to [0, 100]
        value = max(0.0, min(100.0, value))
        self._image_scale = value
        print(f"self._image_scale: {self._image_scale}")
        self._rotate_scale_pane.rotate_and_scale_rasterview(self._image_rotation, self._image_scale)

    # -------------------------------------------------------------------------
    # Translation handlers
    # -------------------------------------------------------------------------

    @Slot()
    def _on_lat_north_changed(self) -> None:
        text = self._ui.ledit_lat_north.text()
        try:
            self._lat_north_translate = float(text)
        except ValueError:
            pass
        print(f"self._lat_north_translate: {self._lat_north_translate}")

    @Slot()
    def _on_lon_east_changed(self) -> None:
        text = self._ui.ledit_lon_east.text()
        try:
            self._lon_east_translate = float(text)
        except ValueError:
            pass
        print(f"self._lon_east_translate: {self._lon_east_translate}")

    def _make_point_to_text(self, point):
        return f"({point[0]}, {point[1]})"

    @Slot()
    def _on_translation_pixel_selected(self, dataset: RasterDataSet, point: QPoint) -> None:
        print(f"_on_translation_pixel_selected")
        print(f"dataset: {dataset}")
        print(f"point: {point}")
        point = (point.x(), point.y())
        # Get current point's dataset
        orig_geo_coords = dataset.to_geographic_coords(point)
        if orig_geo_coords is None:
            print(f"Dataset has no geo transform!")
            return
        origin_lon_east, pixel_w, rot_x, origin_lat_north, rot_y, pixel_h = dataset.get_geo_transform()
        new_lon_east = origin_lon_east + self._lon_east_translate
        new_lat_north = origin_lat_north + self._lat_north_translate

        new_gt = (new_lon_east, pixel_w, rot_x, new_lat_north, rot_y, pixel_h)
        new_geo_coord = pixel_coord_to_geo_coord(point, new_gt)

        print(f"orig_geo_coords: {orig_geo_coords}")
        print(f"new_geo_coord: {new_geo_coord}")

        self.set_orig_coord_text(self._make_point_to_text(orig_geo_coords))
        self.set_new_coord_text(self._make_point_to_text(new_geo_coord))


    # -------------------------------------------------------------------------
    # Button handlers – currently stubs
    # -------------------------------------------------------------------------

    @Slot()
    def _on_run_rotate_scale(self) -> None:
        print("Running rotate and scale")
        # Placeholder – real implementation will apply transform.

    @Slot()
    def _on_create_translation(self) -> None:
        print("Creating translation")
        # Placeholder – real implementation will apply translation.

    # -------------------------------------------------------------------------
    # Public helper methods for external callers
    # -------------------------------------------------------------------------

    def set_orig_coord_text(self, text: str) -> None:
        """Update the previous coordinate label."""
        self._ui.lbl_orig_coord_input.setText(text)

    def set_new_coord_text(self, text: str) -> None:
        """Update the current coordinate label."""
        self._ui.lbl_new_coord_input.setText(text)

    def set_crs_text(self, text: str) -> None:
        """Update (read‑only) CRS line‑edit."""
        self._ui.ledit_crs.setText(text)

    # -------------------------------------------------------------------------
    # Convenience getters – optional but handy
    # -------------------------------------------------------------------------

    def image_rotation(self) -> float:
        return self._image_rotation

    def image_scale(self) -> float:
        return self._image_scale

    def translation(self) -> tuple[float, float]:
        return self._lat_north_translate, self._lon_east_translate
