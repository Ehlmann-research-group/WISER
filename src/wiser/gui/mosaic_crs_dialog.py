"""
Reproject prompt dialog for the Seamless Mosaic feature (EPIC #629, issue #635).

When the scenes added to a mosaic have differing coordinate reference systems,
:meth:`MosaicController.build_common_grid` raises
:class:`~wiser.raster.mosaic_controller.TargetCrsRequired`. The GUI catches that and
shows this modal dialog so the user can see the mismatch (dataset -> CRS table) and
choose the single target CRS the whole mosaic is placed onto.

The dialog is deliberately data-driven: it takes plain lists (not the controller),
so it can be constructed and exercised without GDAL/OSR objects in hand. The caller
passes ``controller.scene_crs_summary()`` for the table and
``controller.scene_crs_choices()`` for the target chooser.
"""

from typing import List, Optional, Tuple

from osgeo import osr

from PySide6.QtGui import QIntValidator
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from wiser.gui.app_state import ApplicationState

# Reuse the CRS building blocks from the georeferencer so the target chooser behaves
# identically to the one users already know (same common CRSs, same authority lookup).
from wiser.gui.geo_reference_dialog import (
    AVAILABLE_AUTHORITIES,
    COMMON_SRS,
    AuthorityCodeCRS,
    UserGeneratedCRS,
    WktGeneratedCRS,
)


class ReprojectPromptDialog(QDialog):
    """
    Modal dialog that asks the user to pick a single target CRS for a mosaic whose
    scenes have differing CRSs.

    Parameters
    ----------
    scene_summary
        ``(dataset_name, crs_display_name)`` per scene (from
        :meth:`MosaicController.scene_crs_summary`); shown read-only so the mismatch
        is visible.
    scene_crs_choices
        ``(crs_display_name, crs_wkt)`` for each distinct visible-scene CRS (from
        :meth:`MosaicController.scene_crs_choices`); seeds the target chooser. The
        **last** entry is the top scene's CRS and becomes the default selection.
    app_state
        Used to offer any user-created CRSs as target options.
    """

    def __init__(
        self,
        scene_summary: List[Tuple[str, str]],
        scene_crs_choices: List[Tuple[str, str]],
        app_state: ApplicationState,
        parent=None,
    ):
        super().__init__(parent=parent)
        self._scene_summary = scene_summary
        self._scene_crs_choices = scene_crs_choices
        self._app_state = app_state

        self.setWindowTitle(self.tr("Choose a target CRS"))
        self.setModal(True)

        self._cbox_target: QComboBox = None
        self._cbox_authority: QComboBox = None
        self._ledit_code: QLineEdit = None

        self._build_ui()
        self._init_scene_table()
        self._init_target_chooser()

    # region UI construction

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        intro = QLabel(
            self.tr(
                "These scenes may use different coordinate reference systems. Choose a "
                "single target CRS to place the whole mosaic onto."
            )
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        self._table = QTableWidget(0, 2, self)
        self._table.setHorizontalHeaderLabels([self.tr("Dataset"), self.tr("CRS")])
        self._table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._table.setSelectionMode(QTableWidget.NoSelection)
        self._table.verticalHeader().setVisible(False)
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        layout.addWidget(self._table)

        layout.addWidget(QLabel(self.tr("Target CRS:")))
        self._cbox_target = QComboBox(self)
        layout.addWidget(self._cbox_target)

        # Authority + code lookup for a fully custom target CRS.
        lookup_row = QHBoxLayout()
        lookup_row.addWidget(QLabel(self.tr("Add by authority + code:")))
        self._cbox_authority = QComboBox(self)
        for auth in AVAILABLE_AUTHORITIES:
            self._cbox_authority.addItem(auth, auth)
        lookup_row.addWidget(self._cbox_authority)

        self._ledit_code = QLineEdit(self)
        self._ledit_code.setValidator(QIntValidator(1, 2147483647, self))
        self._ledit_code.setPlaceholderText(self.tr("code"))
        lookup_row.addWidget(self._ledit_code)

        add_btn = QPushButton(self.tr("Add"), self)
        add_btn.clicked.connect(self._on_add_authority_crs)
        lookup_row.addWidget(add_btn)
        layout.addLayout(lookup_row)

        self._button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        self._button_box.accepted.connect(self.accept)
        self._button_box.rejected.connect(self.reject)
        layout.addWidget(self._button_box)

    def _init_scene_table(self) -> None:
        self._table.setRowCount(len(self._scene_summary))
        for row, (name, crs_display) in enumerate(self._scene_summary):
            self._table.setItem(row, 0, QTableWidgetItem(name))
            self._table.setItem(row, 1, QTableWidgetItem(crs_display))

    def _init_target_chooser(self) -> None:
        """
        Populate the target chooser: the distinct scene CRSs first (default = the top
        scene's CRS, i.e. the last one), then the common CRSs, then any user-created
        CRSs. Every entry stores a ``GeneralCRS`` as ``userData`` so
        :meth:`selected_target_wkt` reads uniformly.
        """
        cbox = self._cbox_target

        for display, wkt in self._scene_crs_choices:
            cbox.addItem(display, WktGeneratedCRS(display, wkt))
        # Default to the top scene's CRS (the last scene choice).
        default_index = len(self._scene_crs_choices) - 1

        for name, srs in COMMON_SRS.items():
            cbox.addItem(name, srs)

        for name, (srs, _) in self._app_state.get_user_created_crs().items():
            cbox.addItem(name, UserGeneratedCRS(name, srs))

        if default_index >= 0:
            cbox.setCurrentIndex(default_index)

    # region Slots

    def _on_add_authority_crs(self) -> None:
        authority = self._cbox_authority.currentText()
        code = self._ledit_code.text()
        if not code:
            QMessageBox.warning(
                self,
                self.tr("Missing code"),
                self.tr("Enter an authority code to look up."),
            )
            return

        srs = osr.SpatialReference()
        try:
            err = srs.SetFromUserInput(f"{authority}:{code}")
        except RuntimeError:
            QMessageBox.warning(
                self,
                self.tr("CRS Lookup Failed"),
                self.tr(f"Could not find spatial reference for {authority}:{code}"),
            )
            return

        if err != 0:
            QMessageBox.warning(
                self,
                self.tr("CRS Lookup Failed"),
                self.tr(f"Could not find spatial reference for {authority}:{code}"),
            )
            return

        name = srs.GetName() or f"{authority}:{code}"
        self._cbox_target.addItem(f"{name} ({authority}:{code})", AuthorityCodeCRS(authority, int(code)))
        self._cbox_target.setCurrentIndex(self._cbox_target.count() - 1)

    # region Public API

    def selected_target_wkt(self) -> Optional[str]:
        """Return the chosen target CRS as WKT, or ``None`` if nothing is selected."""
        data = self._cbox_target.currentData()
        if data is None:
            return None
        srs = data.get_osr_crs()
        if srs is None:
            return None
        return srs.ExportToWkt()

    def accept(self) -> None:
        if self.selected_target_wkt() is None:
            QMessageBox.warning(
                self,
                self.tr("No target CRS"),
                self.tr("Choose a target CRS before continuing."),
            )
            return
        super().accept()
