from PySide6.QtCore import *
from PySide6.QtGui import *
from PySide6.QtWidgets import *


class DockablePane(QDockWidget):
    def __init__(self, widget, name, title, app_state, icon=None, tooltip=None,
                 parent=None):
        super().__init__(title, parent=parent)

        self._name = name
        self._widget = widget
        self._app_state = app_state

        self.setObjectName(name)
        self.setWidget(widget)

        if isinstance(icon, str):
            self._icon = QIcon(icon)
        elif isinstance(icon, QIcon):
            self._icon = icon
        elif icon is not None:
            raise TypeError(f'Unrecognized type for icon argument:  {type(icon)}')

        self._tooltip = tooltip

        # Make sure we get visibility-changed notifications
        self.visibilityChanged.connect(self._on_visibility_changed)


    def get_icon(self):
        return self._icon


    def get_tooltip(self):
        return self._tooltip


    def _on_visibility_changed(self, visible):
        # Work around a known Qt bug:  if a dockable window is floating, and is
        # closed while floating, it can't be redocked unless we toggle its
        # floating state.
        if self.isFloating() and not visible:
            self.setFloating(False)
            self.setFloating(True)
