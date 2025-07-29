import os
import traceback
from typing import Dict, List, Optional, Tuple

from PySide6.QtCore import *
from PySide6.QtGui import *
from PySide6.QtWidgets import *

from .generated.export_plot_image_ui import Ui_ExportPlotImageDialog

from .util import generate_unused_filename

# Do not get rid of these imports! They're needed for pyinstaller.
import matplotlib.backends.backend_pdf
import matplotlib.backends.backend_svg
import matplotlib.backends.backend_ps


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
                self._supported_formats.append(name)
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

        self._ui.ledit_filename.editingFinished.connect(self._on_ledit_filename_edited)
        self._ui.btn_filename.clicked.connect(self._on_btn_filename_clicked)
        self._ui.cbox_image_format.activated.connect(self._on_cbox_image_format_changed)


    def _on_ledit_filename_edited(self):
        self._set_image_format_from_filename()


    def _on_btn_filename_clicked(self, checked):
        '''
        This helper function shows the file-chooser dialog when the user clicks
        the corresponding button in the UI.
        '''
        extensions = ' '.join([f'*.{fmt}' for fmt in self._supported_formats])
        file_dialog = QFileDialog(parent=self, caption=self.tr('Image Filename'),
            filter=self.tr('Image files ({extensions})').format(extensions=extensions))
        file_dialog.setFileMode(QFileDialog.AnyFile)
        file_dialog.setAcceptMode(QFileDialog.AcceptSave)

        # If there is already an initial filename, select it in the dialog.
        initial_filename = self._ui.ledit_filename.text().strip()
        if len(initial_filename) > 0:
            file_dialog.selectFile(initial_filename)

        result = file_dialog.exec()
        if result == QDialog.Accepted:
            filename = file_dialog.selectedFiles()[0]
            self._ui.ledit_filename.setText(filename)
            self._set_image_format_from_filename()


    def _on_cbox_image_format_changed(self, index):
        '''
        If the image-format combobox changes, this function updates the filename
        to match the new image format.
        '''
        idx = self._ui.cbox_image_format.currentIndex()
        self._update_filename_from_image_format()


    def _set_image_format_from_filename(self) -> None:
        '''
        This helper function tries to update the image-format attributes pane
        to reflect the image format indicated by the file extension.  If the
        extension is unrecognized or unspecified, the UI state is not changed.
        '''

        filename = self._ui.ledit_filename.text().strip()
        ext = os.path.splitext(filename)[1].lower()
        if len(ext) > 0:
            # Extension will have a leading "."
            ext = ext[1:]

        try:
            # Try to find the image extension in the recognized image types.
            # If we find it, update the UI to show that format's options.
            idx = self._supported_formats.index(ext)
            self._ui.cbox_image_format.setCurrentIndex(idx)

        except ValueError:
            # Unrecognized image file extension.  Don't try to update anything
            # in the UI.
            pass


    def _update_filename_from_image_format(self) -> None:
        '''
        This helper function tries to update the filename extension to reflect
        the currently chosen image format.  If the filename's current extension
        is a recognized image format that is also different from the selected
        format then the filename's extension will be updated.  If the filename's
        current extension is not a recognized image format then it will not be
        modified.
        '''

        filename = self._ui.ledit_filename.text().strip()
        (base, ext) = os.path.splitext(filename)
        if len(ext) > 0:
            # Chop off the '.' at the start of the extension, and convert to
            # lowercase
            ext = ext[1:].lower()

        try:
            # Try to find the image extension in the recognized image types.
            # If we find it, update the filename with the new extension.
            idx = self._supported_formats.index(ext)
            new_idx = self._ui.cbox_image_format.currentIndex()
            if idx != new_idx:
                # Filename's extension and selected image format don't match.
                filename = f'{base}.{self._supported_formats[new_idx]}'
                self._ui.ledit_filename.setText(filename)

        except ValueError:
            # Unrecognized image file extension.  Don't update anything.
            pass


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
