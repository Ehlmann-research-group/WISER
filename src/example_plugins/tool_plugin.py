import logging

from wiser.plugins import ToolsMenuPlugin, WISERControl

from PySide2.QtWidgets import QMenu, QMessageBox


logger = logging.getLogger(__name__)


class HelloToolPlugin(ToolsMenuPlugin):
    '''
    A simple "Hello world!" example of a Tools plugin.
    '''

    def __init__(self):
        super().__init__()

    def add_tool_menu_items(self, tool_menu: QMenu, wiser: WISERControl) -> None:
        '''
        Use QMenu.addAction() to add individual actions, or QMenu.addMenu() to
        add sub-menus to the Tools menu.
        '''
        logger.info('HelloToolPlugin is adding tool-menu items')
        act = tool_menu.addAction('Say hello...')
        act.triggered.connect(self.say_hello)

    def say_hello(self, checked: bool = False):
        logger.info('HelloToolPlugin.say_hello() was called!')
        QMessageBox.information(None, 'Hello-Tool Plugin',
            'Hello from the toolbar!')
