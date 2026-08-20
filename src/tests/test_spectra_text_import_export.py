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

    def test_variable_length_spectra_with_trailing_padding(self):
        """Variable-length spectra AND ragtag trailing padding at once.

        The file holds three spectra of different lengths (A: 5 values, B: 3,
        C: 2) sharing one wavelength column.  A shorter spectrum signals its end
        with an empty value *within* the real columns, so those in-width blanks
        must be preserved.  On top of that, every row carries a different number
        of trailing tabs that correspond to no spectrum at all -- pure padding
        that must be stripped.  Both behaviors have to hold simultaneously: strip
        the blanks beyond the four real columns, keep the blanks inside them.
        """
        # Four real columns: WVS + A + B + C.  Empty A/B/C cells mark a spectrum
        # as finished; the tabs appended after column C are padding only.
        lines = [
            "WVS\tA\tB\tC\t\t\n",  # header + 2 padding tabs
            "0.3\t1\t2\t3\t\t\t\n",  # all present + 3 padding tabs
            "0.4\t4\t5\t6\n",  # all present, no padding
            "0.5\t7\t8\t\t\t\t\t\n",  # C ends here + 4 padding tabs
            "0.6\t9\t\t\t\n",  # B ends here (C already ended) + 1 padding tab
            "0.7\t11\t\t\t\t\t\t\n",  # only A remains + 5 padding tabs
        ]

        spectra = import_spectra_text(
            lines,
            delim="\t",
            has_header=True,
            wavelength_cols=WavelengthCols.FIRST_COL,
        )

        # The padding never becomes a spectrum: exactly the three real ones.
        self.assertEqual([s.get_name() for s in spectra], ["A", "B", "C"])
        # Each spectrum keeps only the values it actually had; the in-width
        # blanks correctly truncated the shorter ones.
        self.assertEqual([s.num_bands() for s in spectra], [5, 3, 2])
        np.testing.assert_allclose(spectra[0].get_spectrum(), [1, 4, 7, 9, 11])
        np.testing.assert_allclose(spectra[1].get_spectrum(), [2, 5, 8])
        np.testing.assert_allclose(spectra[2].get_spectrum(), [3, 6])

    def test_comma_delimited_trailing_padding_is_stripped(self):
        """The comma-delimited flavor of the Excel export (the exact example
        from issue #725): every row carries a long run of trailing commas that
        must not become phantom spectra."""
        trailing = "," * 19
        lines = [
            "Wavelength_um,ARKSAW_18_G,ARKSAW_15_TS,ARKSAW_17_M" + trailing + "\n",
            "0.3466,26.5672,19.726,27.2727" + trailing + "\n",
            "0.3482,26.9417,19.3805,27.3134" + trailing + "\n",
            "0.3498,26.8571,19.6957,27.2464" + trailing + "\n",
            "0.3514,26.682,19.7034,27.0283" + trailing + "\n",
            "0.353,27.1622,19.7531,27.6606" + trailing + "\n",
        ]

        spectra = import_spectra_text(
            lines,
            delim=",",
            has_header=True,
            wavelength_cols=WavelengthCols.FIRST_COL,
        )

        self.assertEqual(
            [s.get_name() for s in spectra],
            ["ARKSAW_18_G", "ARKSAW_15_TS", "ARKSAW_17_M"],
        )
        self.assertTrue(all(s.num_bands() == 5 for s in spectra))
        np.testing.assert_allclose(spectra[0].get_spectrum(), [26.5672, 26.9417, 26.8571, 26.682, 27.1622])

    def test_stray_value_in_padding_region_is_an_error(self):
        """A value sitting in the padding region (beyond the real columns) is
        not padding -- the row genuinely has more columns than the header, and
        the import must reject it rather than silently dropping the value.
        The error names the stray value and its column, since post-stripping
        column counts would not match what the user sees in their file."""
        lines = [
            "Wavelength_um,ARKSAW_18_G,ARKSAW_15_TS,ARKSAW_17_M,,,\n",
            "0.3466,26.5672,19.726,27.2727,,,\n",
            "0.3482,26.9417,19.3805,27.3134,,3.14,\n",
        ]

        with self.assertRaisesRegex(ValueError, r"'3\.14'.*column 6"):
            import_spectra_text(
                lines,
                delim=",",
                has_header=True,
                wavelength_cols=WavelengthCols.FIRST_COL,
            )

    def test_header_only_column_is_kept_as_empty_spectrum(self):
        """A column that has a header name but no values is a real (if empty)
        spectrum, not padding: the name is within the real column count, so the
        spectrum is kept with zero bands."""
        lines = [
            "Wavelength_um,ARKSAW_18_G,EMPTY_SAMPLE,,\n",
            "0.3466,26.5672,,,\n",
            "0.3482,26.9417,,,\n",
        ]

        spectra = import_spectra_text(
            lines,
            delim=",",
            has_header=True,
            wavelength_cols=WavelengthCols.FIRST_COL,
        )

        self.assertEqual([s.get_name() for s in spectra], ["ARKSAW_18_G", "EMPTY_SAMPLE"])
        self.assertEqual([s.num_bands() for s in spectra], [2, 0])


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
