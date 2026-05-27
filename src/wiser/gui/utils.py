from typing import Callable

from PySide2.QtWidgets import QPushButton, QStyle, QWidget


def build_trash_button(
    parent: QWidget,
    callback: Callable,
    tooltip: str = "Remove",
    fallback_text: str = "Remove",
) -> QPushButton:
    """Build a QPushButton showing the standard trash icon, falling back to text."""
    button = QPushButton()
    button.setToolTip(tooltip)
    trash_icon_enum = getattr(QStyle, "SP_TrashIcon", None)
    icon = parent.style().standardIcon(trash_icon_enum) if trash_icon_enum is not None else None
    if icon is None or icon.isNull():
        button.setText(fallback_text)
    else:
        button.setIcon(icon)
    button.clicked.connect(callback)
    return button
