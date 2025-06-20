from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

import matplotlib

from .generated.roi_info_editor_ui import Ui_ROIInfoEditor


class ROIInfoEditor(QDialog):
    """
    A dialog for editing information about collected spectra and library
    spectra.  Depending on the type of spectrum, different fields will be
    made available to the user.
    """

    def __init__(self, app_state, parent=None):
        super().__init__(parent=parent)
        self._ui = Ui_ROIInfoEditor()
        self._ui.setupUi(self)

        self._app_state = app_state
        self._roi = None

        # TODO(donnie):  Pull the default color from config.
        self._ui.lineedit_color.setText("yellow")
        self._ui.btn_color_chooser.clicked.connect(self._on_choose_color)

    def configure_ui(self, roi):
        self._roi = roi

        self._ui.lineedit_name.setText(self._roi.get_name())
        self._ui.textedit_desc.setPlainText(self._roi.get_description())
        self._ui.lineedit_color.setText(self._roi.get_color())

    def _on_choose_color(self, checked):
        initial_color = QColor(self._ui.lineedit_color.text())
        color = QColorDialog.getColor(parent=self, initial=initial_color)
        if color.isValid():
            self._ui.lineedit_color.setText(color.name())

    def accept(self):
        # Region of Interest name

        name = self._ui.lineedit_name.text().strip()

        if len(name) == 0:
            QMessageBox.critical(
                self,
                self.tr("Missing or invalid values"),
                self.tr("Region of Interest name must be specified."),
                QMessageBox.Ok,
            )
            return

        existing_roi = self._app_state.get_roi(name=name)
        if existing_roi is not None and existing_roi is not self._roi:
            QMessageBox.critical(
                self,
                self.tr("Missing or invalid values"),
                self.tr("There is already a Region of Interest with that name."),
                QMessageBox.Ok,
            )
            return

        # Region of Interest color

        color_name = self._ui.lineedit_color.text().strip()

        try:
            matplotlib.colors.to_rgb(color_name)
        except:
            QMessageBox.critical(
                self,
                self.tr("Missing or invalid values"),
                self.tr("Region of Interest color name is unrecognized."),
                QMessageBox.Ok,
            )
            return

        # =======================================================================
        # All done!

        super().accept()

    def store_values(self, roi=None):
        """Store UI values into the ROI object"""

        if roi is None:
            if self._roi is None:
                raise ValueError(
                    "ROI must be specified either to configure_ui() or store_values()"
                )

            roi = self._roi

        roi.set_name(self._ui.lineedit_name.text().strip())

        desc = self._ui.textedit_desc.toPlainText().strip()
        if len(desc) == 0:
            desc = None

        roi.set_description(desc)

        roi.set_color(self._ui.lineedit_color.text().strip())
