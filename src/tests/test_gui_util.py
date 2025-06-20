import unittest

from wiser.gui.util import make_filename


class TestGuiUtil(unittest.TestCase):
    """
    Exercise code in the gui.util module.
    """

    # ======================================================
    # gui.util.make_filename()

    def test_make_filename_throws_on_empty_string(self):
        with self.assertRaises(ValueError):
            make_filename("")

    def test_make_filename_throws_on_whitespace_string(self):
        with self.assertRaises(ValueError):
            make_filename("    ")

    def test_make_filename_valid_chars(self):
        self.assertEqual(make_filename("foo-bar_abc def.txt"), "foo-bar_abc def.txt")

    def test_make_filename_collapse_spaces(self):
        self.assertEqual(make_filename("a    b.txt"), "a b.txt")

    def test_make_filename_collapse_spaces_tabs(self):
        self.assertEqual(make_filename("a \t   \tb.txt"), "a b.txt")

    def test_make_filename_remove_punctuation(self):
        self.assertEqual(
            make_filename("/a\\bc!@d#ef$g%hi^&j*klm(n)o_pq+-r=st.uv~\"w`x~y'z.txt"),
            "abcdefghijklmno_pq-rst.uvwxyz.txt",
        )
