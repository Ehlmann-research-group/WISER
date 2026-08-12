import logging
import os

from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

from osgeo import gdal

from wiser.utils.progress import ProgressReporter

from .dataset import RasterDataSet
from .dataset_impl import (
    Confidence,
    RasterDataImpl,
    ENVI_GDALRasterDataImpl,
    NumPyRasterDataImpl,
)
from .format_registry import FormatSpec, candidates_for, format_names, get_format

if TYPE_CHECKING:
    from wiser.raster.dataset import DataCache

logger = logging.getLogger(__name__)


class RasterDataLoader:
    """
    A loader for loading 2D raster data-sets from the local filesystem, using
    GDAL (Geospatial Data Abstraction Library) for reading the data.

    Which format is used for a given file is decided by the registry in
    :mod:`wiser.raster.format_registry`; see that module for the dispatch rules.
    """

    def __init__(self):
        # This is a counter so we can generate names for unnamed datasets.
        self._unnamed_datasets: int = 0

    # ------------------------------------------------------------------
    # Format selection
    # ------------------------------------------------------------------

    def _candidate_formats(self, path: str, format: Optional[str]) -> List[FormatSpec]:
        """
        The formats to consider for ``path``, best first.

        An explicit ``format`` collapses this to exactly one spec:  the caller
        has stated the format, so guessing past it would only hide their error.
        """
        if format:
            spec = get_format(format)
            if spec is None:
                raise ValueError(
                    f'Unknown raster format "{format}".  Known formats:  ' + ", ".join(format_names())
                )
            return [spec]

        return candidates_for(path)

    def _rank_candidates(self, path: str, candidates: List[FormatSpec]) -> List[FormatSpec]:
        """
        Narrow ``candidates`` to those worth opening, in the order to try them.

        Each candidate is asked to :meth:`~.RasterDataImpl.identify` the file --
        a cheap check that opens nothing.  The first that answers
        :attr:`Confidence.YES` wins outright and the walk stops there, so a
        confident format is never overtaken by one that merely could have
        worked.  Anything answering :attr:`Confidence.NO` is dropped.

        Formats that are merely plausible follow the winner, in priority order,
        as fallbacks in case the winner turns out to be unreadable.
        """
        winner: Optional[FormatSpec] = None
        maybes: List[FormatSpec] = []

        for spec in candidates:
            try:
                confidence = spec.impl.identify(path)
            except Exception as e:
                # identify() is documented not to raise, but a broken one must
                # not take down the whole load.
                logger.debug("identify() raised for format %s on %s:  %s", spec.name, path, e)
                continue

            if confidence == Confidence.YES:
                winner = spec
                break
            if confidence == Confidence.MAYBE:
                maybes.append(spec)

        if winner is None:
            return maybes
        return [winner] + maybes

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load_from_file(
        self,
        path,
        data_cache=None,
        interactive=True,
        subdataset_name="",
        format: Optional[str] = None,
    ) -> List[RasterDataSet]:
        """
        Load a raster data-set from the specified path.  Returns a list of
        :class:`RasterDataSet` objects -- a list, because one file may contain
        several sub-datasets.

        :param path: the file to load.
        :param data_cache: cache the resulting datasets should read through.
        :param interactive: when False, never prompt; make a sensible default
            choice instead.  Required for project restore and headless use.
        :param subdataset_name: open this specific sub-dataset rather than
            asking which one is wanted.
        :param format: force a specific registered format by name (for example
            ``"ENVI"``).  When given, no other format is tried and a failure is
            raised rather than silently falling back to guessing.

        Raises :class:`FileNotFoundError` if the path does not exist, or
        :class:`ValueError` if no registered format can read it.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"File path {path} does not exist!")

        candidates = self._candidate_formats(path, format)
        attempts = self._rank_candidates(path, candidates)

        if not attempts:
            raise ValueError(f"Couldn't load file {path}:  unsupported format")

        spec, impl_list = self._open_first_that_works(path, attempts, interactive, subdataset_name)

        # An empty (rather than None) result means the format opened the file but
        # deliberately produced nothing -- the user cancelled a sub-dataset
        # prompt.  That is a decision, not a failure, so don't try other formats.
        if not impl_list:
            return []

        logger.debug("Loaded %s as format %s", path, spec.name)
        return self._build_datasets(path, spec, impl_list, data_cache)

    def _open_first_that_works(
        self,
        path: str,
        attempts: List[FormatSpec],
        interactive: bool,
        subdataset_name: str,
    ) -> Tuple[FormatSpec, List[RasterDataImpl]]:
        """
        Open ``path`` with the first spec in ``attempts`` that succeeds.

        In the normal case the first attempt is a format that positively
        identified the file, so exactly one file handle is ever opened.  The
        remaining entries only matter when a file identifies as a format but
        then turns out to be unreadable.
        """
        errors: List[str] = []

        for spec in attempts:
            try:
                if subdataset_name:
                    impl_list = spec.impl.try_load_file(
                        path, subdataset_name=subdataset_name, interactive=interactive
                    )
                else:
                    impl_list = spec.impl.try_load_file(path, interactive=interactive)
            except Exception as e:
                logger.debug("Couldn't load %s as format %s:  %s", path, spec.name, e)
                errors.append(f"{spec.name}: {e}")
                continue

            if impl_list is not None:
                return spec, impl_list

            errors.append(f"{spec.name}: returned no implementation")

        detail = "; ".join(errors) if errors else "no format claimed the file"
        raise ValueError(f"Couldn't load file {path}:  unsupported format ({detail})")

    def _build_datasets(
        self,
        path: str,
        spec: FormatSpec,
        impl_list: List[RasterDataImpl],
        data_cache: "DataCache",
    ) -> List[RasterDataSet]:
        """Turn opened implementations into named datasets."""
        datasets: List[RasterDataSet] = []

        for impl in impl_list:
            for ds in spec.loader(impl, data_cache):
                ds.set_name(self._dataset_name(path, ds))
                datasets.append(ds)

        return datasets

    @staticmethod
    def _dataset_name(path: str, ds: RasterDataSet) -> str:
        """
        Name a dataset after its own file, falling back to the requested path,
        and qualify it with the sub-dataset name when there is one.
        """
        files = ds.get_filepaths()
        name = os.path.basename(files[0]) if files else os.path.basename(path)

        subdataset_name = ds.get_subdataset_name()
        if subdataset_name is not None:
            name += ":" + subdataset_name.split(":")[-1]

        return name

    # ------------------------------------------------------------------
    # Saving and in-memory construction
    # ------------------------------------------------------------------

    def get_save_filenames(self, path: str, format: str = "ENVI") -> List[str]:
        if format == "ENVI":
            return ENVI_GDALRasterDataImpl.get_save_filenames(path)
        else:
            raise ValueError(f'Unsupported format "{format}"')

    def save_dataset_as(
        self,
        dataset: RasterDataSet,
        path: str,
        format: str,
        config: Dict[str, Any],
        progress: Optional[ProgressReporter] = None,
    ) -> ENVI_GDALRasterDataImpl:
        if format == "ENVI":
            return ENVI_GDALRasterDataImpl.save_dataset_as(dataset, path, config, progress=progress)
        else:
            raise ValueError(f'Unsupported format "{format}"')

    def dataset_from_numpy_array(self, arr: np.ndarray, cache: "DataCache" = None) -> RasterDataSet:
        """
        Given a NumPy ndarray, this function returns a RasterDataSet object
        that uses the array for its raster data.  The input ndarray must have
        three dimensions; they are interpreted as
        [spectral][spatial_y][spatial_x].

        Raises a ValueError if the input array doesn't have 3 dimensions.
        """

        if len(arr.shape) != 3:
            raise ValueError("NumPy array must have 3 dimensions")

        impl = NumPyRasterDataImpl(arr)
        return RasterDataSet(impl, cache)

    def dataset_from_gdal_dataset(self, dataset: gdal.Dataset, cache: "DataCache") -> RasterDataSet:
        impl = ENVI_GDALRasterDataImpl(dataset)
        return RasterDataSet(impl, cache)

    # TODO(donnie):  Not presently needed - can instantiate a NumPyArraySpectrum
    #     object from a NumPy array...
    # def spectrum_from_numpy_array(self, arr: np.ndarray) -> Spectrum:
    #     return None
