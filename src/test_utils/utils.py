from PySide2.QtTest import QTest
from PySide2.QtCore import Qt
from PySide2.QtWidgets import QComboBox, QApplication

import time

def click_combo_index(combo: QComboBox, index: int, delay: int = 50):
    """
    Open `combo`’s popup and click on the item at `index`.

    :param combo: the QComboBox to drive
    :param index: 0-based row to select
    :param delay: milliseconds to wait after opening before clicking
    """
    # 1) click the combobox to open its popup
    QTest.mouseClick(combo, Qt.LeftButton)
    # 2) give Qt a moment to show the popup
    # QTest.qWaitForWindowActive(combo.view().window(), timeout=500)
    time.sleep(0.1)
    popup = combo.view()
    # 3) locate the item’s rectangle and click it
    model_idx = combo.model().index(index, 0)
    rect = popup.visualRect(model_idx)
    QTest.mousePress(popup.viewport(), Qt.LeftButton, Qt.NoModifier, rect.center())
    time.sleep(0.01)
    QApplication.processEvents()
    time.sleep(0.01)
    QTest.mouseRelease(popup.viewport(), Qt.LeftButton, Qt.NoModifier, rect.center())
    QApplication.processEvents()
    time.sleep(0.01)
