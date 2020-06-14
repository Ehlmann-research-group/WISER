from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

from typing import Any, List, Tuple


class ToolbarMenu(QToolButton):
    '''
    This toolbar button shows a menu of items, one of which is selected at any
    given time.  Each item has a display string, and a data value that can be
    used by the application.
    '''

    def __init__(self, icon: QIcon, items: List[Tuple[str, Any]]):
        super().__init__()

        self._item_menu = QMenu()
        self._items = items
        self._item_actions = []

        self.setIcon(icon)
        self.setPopupMode(QToolButton.InstantPopup)
        self.setMenu(self._item_menu)

        self.triggered.connect(self._on_selection)

        self._populate_menu()


    def _populate_menu(self):
        # Remove all existing actions
        self._item_actions = []
        self._item_menu.clear()

        # Add an action for each dataset
        for (item_text, item_data) in self._items:
            act = QAction(item_text)
            act.setCheckable(True)
            act.setData(item_data)

            self._item_menu.addAction(act)
            self._item_actions.append(act)

        if len(self._item_actions) > 0:
            self._item_actions[0].setChecked(True)


    def _on_selection(self, act: QAction):
        '''
        This helper function ensures that only the currently selected dataset
        has a check-mark by it; all other datasets will be (or become)
        deselected.
        '''
        for oact in self._item_actions:
            if oact is not act and oact.isChecked():
                # This is some other action, and it is checked.  Uncheck it.
                oact.setChecked(False)

            elif oact is act and not oact.isChecked():
                # This is the selected action, but the user just unchecked it.
                # Re-check it.
                oact.setChecked(True)
