"""Regression tests for importing/exporting spectra as delimited text.

These tests capture three defects reported by WISER users around the
"Import Spectra from Text" dialog and the spectrum-export feature:

1.  Trailing delimiters on *data rows only* (a common artifact of Excel's
    "Save As -> Tab delimited (.txt)") make every data row have one more
    column than the header, so parsing aborts with a "Line N has X columns,
    but first line has Y columns" error even though the data is fine.

2.  Trailing delimiters on *every* row (header included) are consistent, so
    parsing "succeeds" -- but the extra empty trailing columns are imported as
    phantom, empty spectra.  A file with 3 real spectra plus a run of trailing
    tabs imports as many more (the user's report: 3 spectra read as 22).

3.  Spectra exported by WISER (``export_spectrum_list``, which writes
    wavelength/value column pairs) cannot be re-imported with the matching
    "odd columns specify wavelength values" option: the importer crashes with
    an ``IndexError`` because it mis-indexes the header names.

Each test asserts the *desired* behavior, so it currently fails (red) and
should pass once the corresponding bug is fixed.
"""

import os
import tempfile
import unittest

import numpy as np
import pytest
from astropy import units as u

import tests.context  # noqa: F401 -- adds src/ to sys.path

from wiser.raster.spectra_export import (
    WavelengthCols,
    import_spectra_text,
    import_spectra_textfile,
    export_spectrum_list,
)
from wiser.raster.spectrum import NumPyArraySpectrum

pytestmark = [
    pytest.mark.unit,
    pytest.mark.regression,
]


class TestTrailingDelimiterImport(unittest.TestCase):
    """Bugs 1 & 2: trailing delimiters left behind by Excel text exports."""

    def test_trailing_delimiter_on_data_rows_is_tolerated(self):
        """Bug 1: header has no trailing tab but every data row does (the
        ``Muscovite_test.txt`` case).  The trailing empty column should be
        ignored, not treated as a fatal column-count mismatch."""
        # Header: 3 columns, no trailing tab.  Data rows: 3 values + trailing
        # tab -> would split into 4 columns.
        lines = [
            "WVS\tMUSC_SC\tMUSC\n",
            "0.3\t6.4616\t0.32308\t\n",
            "0.305\t6.646\t0.3323\t\n",
            "0.31\t6.794\t0.3397\t\n",
        ]

        spectra = import_spectra_text(
            lines,
            delim="\t",
            has_header=True,
            wavelength_cols=WavelengthCols.FIRST_COL,
        )

        # First column is the wavelength, so we expect the two named spectra.
        self.assertEqual([s.get_name() for s in spectra], ["MUSC_SC", "MUSC"])
        self.assertEqual([s.num_bands() for s in spectra], [3, 3])
        np.testing.assert_allclose(spectra[0].get_spectrum(), [6.4616, 6.646, 6.794])
        np.testing.assert_allclose(spectra[1].get_spectrum(), [0.32308, 0.3323, 0.3397])

    def test_trailing_delimiters_on_all_rows_do_not_create_phantom_spectra(self):
        """Bug 2: every row (header included) ends in a run of trailing tabs
        (the ``C1 Samples`` case where 3 spectra were read as 22).  The empty
        trailing columns must not become empty spectra."""
        # Four real columns (wavelength + 3 spectra), then four trailing tabs.
        trailing = "\t\t\t\t"
        lines = [
            "Wavelength_um\tARKSAW_18_G\tARKSAW_15_TS\tARKSAW_17_M" + trailing + "\n",
            "0.3466\t26.5672\t19.726\t27.2727" + trailing + "\n",
            "0.3482\t26.9417\t19.3805\t27.3134" + trailing + "\n",
        ]

        spectra = import_spectra_text(
            lines,
            delim="\t",
            has_header=True,
            wavelength_cols=WavelengthCols.FIRST_COL,
        )

        # Exactly the three real spectra -- no empty-named phantoms.
        self.assertEqual(
            [s.get_name() for s in spectra],
            ["ARKSAW_18_G", "ARKSAW_15_TS", "ARKSAW_17_M"],
        )
        self.assertTrue(all(s.get_name() for s in spectra), "no spectrum should be unnamed")
        self.assertTrue(all(s.num_bands() == 2 for s in spectra))

    def test_ragged_trailing_delimiters_are_tolerated(self):
        """Bug 1/2 variant: Excel does not always emit the *same* number of
        trailing tabs on every line -- the header and each data row can carry a
        different-length run (here 4, 1, 3, 4, 2).  After stripping trailing
        empties every row should normalize to the four real columns, yielding
        the three real spectra with no phantom or truncated ones."""
        lines = [
            "Wavelength_um\tARKSAW_18_G\tARKSAW_15_TS\tARKSAW_17_M\t\t\t\t\n",  # 4 trailing
            "0.3466\t26.5672\t19.726\t27.2727\t\n",  # 1 trailing
            "0.3482\t26.9417\t19.3805\t27.3134\t\t\t\n",  # 3 trailing
            "0.3498\t26.8571\t19.6957\t27.2464\t\t\t\t\n",  # 4 trailing
            "0.3514\t26.682\t19.7034\t27.0283\t\t\n",  # 2 trailing
        ]

        spectra = import_spectra_text(
            lines,
            delim="\t",
            has_header=True,
            wavelength_cols=WavelengthCols.FIRST_COL,
        )

        self.assertEqual(
            [s.get_name() for s in spectra],
            ["ARKSAW_18_G", "ARKSAW_15_TS", "ARKSAW_17_M"],
        )
        self.assertTrue(all(s.get_name() for s in spectra), "no spectrum should be unnamed")
        # All four data rows survive for every real spectrum.
        self.assertTrue(all(s.num_bands() == 4 for s in spectra))
        np.testing.assert_allclose(spectra[0].get_spectrum(), [26.5672, 26.9417, 26.8571, 26.682])
        np.testing.assert_allclose(spectra[2].get_spectrum(), [27.2727, 27.3134, 27.2464, 27.0283])


class TestExportImportRoundTrip(unittest.TestCase):
    """Bug 3: spectra exported by WISER should be re-importable."""

    def test_exported_spectra_can_be_reimported(self):
        """``export_spectrum_list`` writes (wavelength, value) column pairs.
        Re-importing with ``ODD_COLS`` (what the dialog auto-selects for these
        files) should recover the original spectra."""
        wavelengths = [0.5 * u.um, 0.6 * u.um, 0.7 * u.um]
        src = [
            NumPyArraySpectrum(np.array([0.1, 0.2, 0.3]), name="specA", wavelengths=wavelengths),
            NumPyArraySpectrum(np.array([0.4, 0.5, 0.6]), name="specB", wavelengths=wavelengths),
        ]

        fd, path = tempfile.mkstemp(suffix=".txt")
        os.close(fd)
        try:
            export_spectrum_list(path, src)

            imported = import_spectra_textfile(
                path,
                delim="\t",
                has_header=True,
                wavelength_cols=WavelengthCols.ODD_COLS,
            )
        finally:
            os.remove(path)

        self.assertEqual([s.get_name() for s in imported], ["specA", "specB"])
        np.testing.assert_allclose(imported[0].get_spectrum(), [0.1, 0.2, 0.3])
        np.testing.assert_allclose(imported[1].get_spectrum(), [0.4, 0.5, 0.6])

        # Wavelengths should survive the round trip too.
        for spec in imported:
            values = [w.to(u.um).value for w in spec.get_wavelengths()]
            np.testing.assert_allclose(values, [0.5, 0.6, 0.7])


if __name__ == "__main__":
    unittest.main()
