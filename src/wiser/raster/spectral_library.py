import abc
import os

from typing import List, Optional


from .spectrum import Spectrum


class SpectralLibrary(abc.ABC):
    """
    A spectral library, comprised of one or more spectra of use in analyzing
    raster data files.

    If specific steps must be taken when a data-set is closed, the
    implementation should implement the __del__ function.
    """

    def __init__(self):
        self._id = None

    def get_id(self) -> Optional[int]:
        return self._id

    def set_id(self, id: int) -> None:
        self._id = id

    def get_name(self):
        # TODO(donnie):  Temporary hack for spectral libraries that are unnamed.
        #     Definitely want to support spectral libraries being constructed in
        #     memory and then saving them to disk.
        paths = self.get_filepaths()
        if len(paths) == 0:
            return "unnamed"

        name = os.path.basename(paths[0])
        return name

    def get_description(self):
        """
        Returns a description of the spectral library that might be specified
        in the library's metadata.  A missing description is indicated by the
        empty string "".
        """
        pass

    def get_filetype(self):
        """
        Returns a string describing the type of file that backs this spectral
        library.  The file-type string will be specific to the kind of loader
        used to load the library.
        """
        pass

    def get_filepaths(self):
        """
        Returns the paths and filenames of all files associated with this
        spectral library.  This may be None if the data is in-memory only.
        """
        pass

    def num_spectra(self):
        """
        Returns the number of spectra in the spectral library.
        """
        pass

    def get_spectrum_name(self, index):
        """
        Returns the name of the specified spectrum in the spectral library.
        """
        pass

    def get_spectrum(self, index) -> Spectrum:
        """
        Returns a Spectrum object corresponding to the specified spectrum in the
        spectral library.
        """
        pass


class ListSpectralLibrary(SpectralLibrary):
    """
    A simple implementation of the ``SpectralLibrary`` base-type that simply
    holds a list of spectra.  Individual spectra in the library may have
    different numbers of bands, band details, band units, etc.
    """

    def __init__(self, spectra: List[Spectrum], **kwargs):
        self._spectra = spectra

        self._name = kwargs.pop("name", None)
        self._path = kwargs.pop("path", None)
        self._description = kwargs.pop("description", None)
        if kwargs:
            raise ValueError(f"Unrecognized arguments:  {kwargs.keys()}")

    def set_id(self, id: int) -> None:
        # Go through and set the ID of every spectrum in the library as well.
        super().set_id(id)
        for index, s in enumerate(self._spectra):
            s.set_id((self._id, index))

    def get_description(self):
        """
        Returns a description of the spectral library that might be specified
        in the library's metadata.  A missing description is indicated by the
        empty string "".
        """
        return self._description

    def get_filetype(self):
        """
        Returns a string describing the type of file that backs this spectral
        library.  The file-type string will be specific to the kind of loader
        used to load the library.
        """
        return ""

    def get_filepaths(self):
        """
        Returns the paths and filenames of all files associated with this
        spectral library.  This may be None if the data is in-memory only.
        """
        return [self._path]

    def num_spectra(self):
        """
        Returns the number of spectra in the spectral library.
        """
        return len(self._spectra)

    def get_spectrum_name(self, index):
        """
        Returns the name of the specified spectrum in the spectral library.
        """
        return self._spectra[index].get_name()

    def get_spectrum(self, index) -> Spectrum:
        """
        Returns a Spectrum object corresponding to the specified spectrum in the
        spectral library.
        """
        return self._spectra[index]
