import unittest

import tests.context

from wiser import plugins
from wiser.plugins import utils


class TestPluginsUtils(unittest.TestCase):
    '''
    Exercise code in the plugins.utils module.
    '''

    #======================================================
    # gui.util.make_filename()

    def test_none_is_not_plugin(self):
        ''' is_plugin(None) should be False '''
        self.assertFalse(utils.is_plugin(None))

    def test_str_is_not_plugin(self):
        ''' is_plugin('hello') should be False '''
        self.assertFalse(utils.is_plugin('hello'))

    def test_int_is_not_plugin(self):
        ''' is_plugin(345) should be False '''
        self.assertFalse(utils.is_plugin(345))

    def test_basetype_is_not_plugin(self):
        '''
        The Plugin base-type is abstract, and should not be reported as a
        plugin.
        '''
        plugin_base = plugins.Plugin()
        self.assertFalse(utils.is_plugin(plugin_base))

    def test_tool_is_plugin(self):
        ''' The ToolsMenuPlugin type should be reported as a plugin. '''
        p = plugins.ToolsMenuPlugin()
        self.assertTrue(utils.is_plugin(p))

    def test_ctxmenu_is_plugin(self):
        ''' The ContextMenuPlugin type should be reported as a plugin. '''
        p = plugins.ContextMenuPlugin()
        self.assertTrue(utils.is_plugin(p))

    def test_bandmath_is_plugin(self):
        ''' The BandMathPlugin type should be reported as a plugin. '''
        p = plugins.BandMathPlugin()
        self.assertTrue(utils.is_plugin(p))
