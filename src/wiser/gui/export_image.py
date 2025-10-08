import os
import traceback
from typing import Dict, List, Optional, Tuple

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

import numpy as np
import PIL

from .generated.export_image_ui import Ui_ExportImageDialog
from .rasterview import RasterView
from .util import generate_unused_filename


def export_raster_image(
    image_data: np.ndarray,
    filename: str,
    output_format: str,
    output_attrs: Dict = {},
    data_format: str = "ARGB",
    crop: Optional[Tuple[int, int, int, int]] = None,
    scale: float = 1.0,
) -> None:
    """
    Export an image generated from raster data, as an image file of the
    specified format and details.  The image data may optionally be cropped
    and/or scaled in the generated image.

    image_data is a NumPy array containing the data to output.  The array should
    have at least 2 dimensions, with a shape (height, width).

    filename is the output file to write to.  Note that if the file already
    exists, this function will overwrite the file's contents, with no warning or
    exception.

    The output_format value is a value recognized by PIL (or pillow, the PIL
    fork).  Similarly, output_attrs is an optional dictionary of attributes
    recognized by the PIL image-writer being used.

    data_format specifies the format of the input data.  Currently, only a value
    of 'ARGB' is supported, since this is what WISER uses internally for its
    image representations.  'ARGB' means that each pixel is a 32-bit unsigned
    integer, with each color channel taking 8 bits, and the alpha channel taking
    8 bits.  The alpha value is the top byte, and blue is the bottom byte, in
    each 32-bit value.

    crop is an optional 4-tuple specifying the part of the data to output, of
    the form (x, y, width, height).  All values must be integers.  The default
    value is None, which produces no cropping.

    scale is a floating-point value used to scale the image.  The default value
    is 1.0, which will not scale the image up or down.  All scaling is done with
    a nearest-neighbor approach; no blending of pixel values will be performed.
    """

    # TODO(donnie):  Currently we only support 'ARGB' format.
    if data_format != "ARGB":
        raise ValueError('Currently only support data_format of "ARGB"')

    # Apply cropping on the NumPy array, so we can avoid generating a larger
    # image than we must have.
    if crop is not None:
        (crop_x, crop_y, crop_width, crop_height) = crop
        image_data = image_data[crop_y : crop_y + crop_height, crop_x : crop_x + crop_width]

    # PIL expects data to be in one of a few formats.  The 32-bit pixel format
    # is expected to be ABGR (A = top byte, R = bottom byte).
    image_data = image_data.copy()
    with np.nditer(image_data, op_flags=["readwrite"]) as it:
        for value in it:
            a = (value >> 24) & 0xFF
            r = (value >> 16) & 0xFF
            g = (value >> 8) & 0xFF
            b = (value) & 0xFF
            value[...] = (a << 24) | (b << 16) | (g << 8) | r

    (height, width) = image_data.shape

    im = PIL.Image.fromarray(image_data, mode="RGBA")

    if scale != 1.0:
        im = im.resize(
            size=(int(width * scale + 0.5), int(height * scale + 0.5)),
            resample=PIL.Image.NEAREST,
        )

    im.save(filename, output_format, **output_attrs)


class ExportImageDialog(QDialog):
    """
    This class implements the functionality of the export-image dialog.
    """

    def __init__(self, parent=None):
        super().__init__(parent=parent)

        # TODO(donnie):  Make this a global in this module?
        self._img_extensions: List[str] = ["png", "tiff", "jpg"]

        self._rasterview: Optional[RasterView] = None

        # Set up the UI state
        self._ui = Ui_ExportImageDialog()
        self._ui.setupUi(self)

        # NOTE:  These need to be in the same order as the stacked pane, or else
        # the index of entries in the combobox won't match up with the indexes
        # of UIs in the stacked pane.
        self._ui.cbox_image_format.addItems(
            [
                self.tr("PNG"),
                self.tr("TIFF"),
                self.tr("JPEG"),
            ]
        )
        self._ui.cbox_image_format.setCurrentIndex(0)
        self._ui.stack_image_config.setCurrentIndex(0)

        # DPI combobox
        self._ui.cbox_image_dpi.addItems(["72", "100", "300"])
        self._ui.cbox_image_dpi.setCurrentIndex(1)
        self._ui.cbox_image_dpi.lineEdit().setValidator(QIntValidator(1, 1000))

        # Events:

        self._ui.ledit_filename.editingFinished.connect(self._on_ledit_filename_edited)
        self._ui.btn_filename.clicked.connect(self._on_btn_filename_clicked)
        self._ui.cbox_image_format.activated.connect(self._on_cbox_image_format_changed)

    def _on_ledit_filename_edited(self):
        self._set_image_format_from_filename()

    def _on_btn_filename_clicked(self, checked):
        """
        This helper function shows the file-chooser dialog when the user clicks
        the corresponding button in the UI.
        """
        file_dialog = QFileDialog(
            parent=self,
            caption=self.tr("Image Filename"),
            filter=self.tr("Image files (*.png *.tiff *.jpg)"),
        )
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
        """
        If the image-format combobox changes, this function updates the stacked
        widget to show the image format's configuration.  it also updates the
        filename to match the new image format.
        """
        idx = self._ui.cbox_image_format.currentIndex()
        self._ui.stack_image_config.setCurrentIndex(idx)
        self._update_filename_from_image_format()

    def _set_image_format_from_filename(self) -> None:
        """
        This helper function tries to update the image-format attributes pane
        to reflect the image format indicated by the file extension.  If the
        extension is unrecognized or unspecified, the UI state is not changed.
        """

        filename = self._ui.ledit_filename.text().strip()
        ext = os.path.splitext(filename)[1].lower()
        if len(ext) > 0:
            # Extension will have a leading "."
            ext = ext[1:]

        try:
            # Try to find the image extension in the recognized image types.
            # If we find it, update the UI to show that format's options.
            idx = self._img_extensions.index(ext)
            self._ui.cbox_image_format.setCurrentIndex(idx)
            self._ui.stack_image_config.setCurrentIndex(idx)

        except ValueError:
            # Unrecognized image file extension.  Don't try to update anything
            # in the UI.
            pass

    def _update_filename_from_image_format(self) -> None:
        """
        This helper function tries to update the filename extension to reflect
        the currently chosen image format.  If the filename's current extension
        is a recognized image format that is also different from the selected
        format then the filename's extension will be updated.  If the filename's
        current extension is not a recognized image format then it will not be
        modified.
        """

        filename = self._ui.ledit_filename.text().strip()
        (base, ext) = os.path.splitext(filename)
        if len(ext) > 0:
            # Chop off the '.' at the start of the extension, and convert to
            # lowercase
            ext = ext[1:].lower()

        try:
            # Try to find the image extension in the recognized image types.
            # If we find it, update the filename with the new extension.
            idx = self._img_extensions.index(ext)
            new_idx = self._ui.cbox_image_format.currentIndex()
            if idx != new_idx:
                # Filename's extension and selected image format don't match.
                filename = f"{base}.{self._img_extensions[new_idx]}"
                self._ui.ledit_filename.setText(filename)

        except ValueError:
            # Unrecognized image file extension.  Don't update anything.
            pass

    def configure(
        self,
        rasterview: RasterView,
        x: int,
        y: int,
        width: int,
        height: int,
        scale: float,
    ):
        self._rasterview = rasterview
        dataset = rasterview.get_raster_data()
        raster_width = dataset.get_width()
        raster_height = dataset.get_height()

        basename = "image"
        paths = dataset.get_filepaths()
        if paths:
            # Use the first filename for the image filename, chopping off the
            # extension, if any.
            basename = os.path.splitext(paths[0])[0]

        self._ui.ledit_x.setText(f"{x}")
        self._ui.ledit_height.setValidator(QIntValidator(0, raster_width - 1))

        self._ui.ledit_y.setText(f"{y}")
        self._ui.ledit_height.setValidator(QIntValidator(0, raster_height - 1))

        self._ui.ledit_width.setText(f"{width}")
        self._ui.ledit_height.setValidator(QIntValidator(1, raster_width))

        self._ui.ledit_height.setText(f"{height}")
        self._ui.ledit_height.setValidator(QIntValidator(1, raster_height))

        # The UI scale value is in percentage, so we allow 1%-1600%
        self._ui.ledit_scale.setText(f"{int(scale * 100)}")
        self._ui.ledit_scale.setValidator(QIntValidator(1, 1600))

        # Generate a suggested filename for the user that is presently unused.
        idx = self._ui.cbox_image_format.currentIndex()
        filename = generate_unused_filename(basename, self._img_extensions[idx])
        self._ui.ledit_filename.setText(filename)
        self._set_image_format_from_filename()

    def accept(self):
        """
        When the user clicks the "OK" button, verify the dialog's contents, and
        then attempt to save the image file.  If an error is encountered, leave
        the dialog open so the user can fix any issues.
        """

        dataset = self._rasterview.get_raster_data()

        # These fields all have an int-validator on them, so we can trust that
        # the results are integers, at least.  They may still be out of proper
        # ranges, though.

        x = int(self._ui.ledit_x.text())
        y = int(self._ui.ledit_y.text())
        width = int(self._ui.ledit_width.text())
        height = int(self._ui.ledit_height.text())
        scale = int(self._ui.ledit_scale.text()) / 100.0

        # TODO(donnie):  Maybe figure out how to satisfy some of these things
        #     with validators on the fields.

        if x + width > dataset.get_width():
            QMessageBox.critical(
                self,
                self.tr("Invalid X coordinate and width"),
                self.tr("X coordinate + width must fall within the image"),
            )
            return

        if y + height > dataset.get_height():
            QMessageBox.critical(
                self,
                self.tr("Invalid Y coordinate and height"),
                self.tr("Y coordinate + height must fall within the image"),
            )
            return

        filename = self._ui.ledit_filename.text().strip()
        if len(filename) == 0:
            QMessage.critical(self, self.tr("Invalid filename"), self.tr("Filename must be specified"))
            return

        idx_format = self._ui.cbox_image_format.currentIndex()
        format = self._img_extensions[idx_format]

        dpi = int(self._ui.cbox_image_dpi.currentText())

        format_attrs = {
            "dpi": (dpi, dpi),  # Must specify DPI on each dimension
        }

        # TODO(donnie):  Format-specific attributes!

        # Finally, try to save the image data to the file.
        try:
            image_data = self._rasterview.get_image_data()
            export_raster_image(
                image_data,
                filename,
                format,
                format_attrs,
                crop=(x, y, width, height),
                scale=scale,
            )

        except Exception as e:
            mbox = QMessageBox(
                QMessageBox.Critical,
                self.tr("Could not export image"),
                self.tr("Could not export image to file\n{0}").format(filename),
                QMessageBox.Ok,
                parent=self,
            )

            mbox.setInformativeText(str(e))
            mbox.setDetailedText(traceback.format_exc())

            mbox.exec()

            return

        self._rasterview = None
        super().accept()
