import os
import traceback
from typing import Dict, List, Optional, Tuple

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

from .generated.export_plot_image_ui import Ui_ExportPlotImageDialog

from .util import generate_unused_filename



class ExportPlotImageDialog(QDialog):
    '''
    This class implements the functionality of the export plot-image dialog.
    '''

    def __init__(self, figure, parent=None):
        super().__init__(parent=parent)

        self._figure = figure

        # Set up the UI state
        self._ui = Ui_ExportPlotImageDialog()
        self._ui.setupUi(self)

        # Populate the supported formats.  We only want to support the formats
        # specified here, but we also ask the matplotlib canvas what it can do,
        # and we don't show anything that matplotlib can't do.  If we end up
        # having no supported formats, raise an exception.

        check_formats = ['eps', 'pdf', 'png', 'svg']
        self._supported_formats = []
        canvas_formats = self._figure.canvas.get_supported_filetypes()
        for name in check_formats:
            if name in canvas_formats:
                # TODO(donnie):  Delete?  No one cares what "EPS" stands for.
                # desc = f'{name.upper()} - {canvas_formats[name]}'
                desc = name.upper()
                self._supported_formats.append((name, desc))
                self._ui.cbox_image_format.addItem(desc, userData=name)

        if len(self._supported_formats) == 0:
            raise ValueError(
                'matplotlib does not recognize any of these formats:  ' +
                ' '.join(check_formats))

        # DPI combobox

        self._ui.cbox_image_dpi.addItems(['72', '100', '300'])
        self._ui.cbox_image_dpi.setCurrentIndex(1)
        self._ui.cbox_image_dpi.lineEdit().setValidator(QIntValidator(1, 1000))

        # Events:

        self._ui.btn_filename.clicked.connect(self._on_btn_filename_clicked)


    def _on_btn_filename_clicked(self, checked):
        '''
        This helper function shows the file-chooser dialog when the user clicks
        the corresponding button in the UI.
        '''
        extensions = ' '.join([f'*.{f[0]}' for f in self._supported_formats])
        file_dialog = QFileDialog(parent=self, caption=self.tr('Image Filename'),
            filter=self.tr('Image files ({extensions})').format(extensions=extensions))
        file_dialog.setFileMode(QFileDialog.AnyFile)
        file_dialog.setAcceptMode(QFileDialog.AcceptSave)

        # If there is already an initial filename, select it in the dialog.
        initial_filename = self._ui.ledit_filename.text().strip()
        if len(initial_filename) > 0:
            print(f'initial = "{initial_filename}"')
            file_dialog.selectFile(initial_filename)

        result = file_dialog.exec()
        if result == QDialog.Accepted:
            filename = file_dialog.selectedFiles()[0]
            self._ui.ledit_filename.setText(filename)
            # TODO(donnie):  self._set_image_format_from_filename()


    def accept(self):
        filename = self._ui.ledit_filename.text().strip()
        if len(filename) == 0:
            QMessage.critical(self, self.tr('Invalid filename'),
                self.tr('Filename must be specified'))
            return

        format = self._ui.cbox_image_format.currentData()
        dpi = int(self._ui.cbox_image_dpi.currentText())

        # Finally, try to save the image data to the file.
        try:
            self._figure.savefig(filename, format=format, dpi=dpi)

        except Exception as e:
            mbox = QMessageBox(QMessageBox.Critical,
                self.tr('Could not export plot image'),
                self.tr('Could not export plot image to file\n{0}').format(filename),
                QMessageBox.Ok, parent=self)

            mbox.setInformativeText(str(e))
            mbox.setDetailedText(traceback.format_exc())

            mbox.exec()

            return

        self._figure = None
        super().accept()
