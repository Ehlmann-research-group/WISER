
from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

from PySide2 import QtCore, QtWidgets, QtTest, QtGui

from test_utils.test_function_decorator import run_in_wiser_decorator

import time

def _action_center(menu: QtWidgets.QMenu, action: QtWidgets.QAction) -> QtCore.QPoint:
    rect = menu.actionGeometry(action)
    return rect.center()

def _find_action_by(menu: QtWidgets.QMenu, *, text=None, object_name=None):
    def norm(s): return s.replace("&", "").strip()
    for a in menu.actions():
        if object_name and a.objectName() == object_name:
            return a
        if text and norm(a.text()) == norm(text):
            return a
    return None

@run_in_wiser_decorator
def right_click_widget(test_model, widget: QtWidgets.QWidget, pos: QtCore.QPoint):
    QtTest.QTest.mouseClick(widget, QtCore.Qt.RightButton, QtCore.Qt.NoModifier, pos)
    time.sleep(0.01)

def click_active_context_menu_path(test_model, menu: QtWidgets.QMenu, path, right_click_pos=None, delay_ms=80):
    """
    Right-click on `widget`, then left-click submenu path.
    path: list like ["Edit", "Advanced", "Rename"] or list of dicts with {"text": "..."} / {"object_name": "..."}.
    """
    # if right_click_pos is None:
    #     # default to center of the widget
    #     right_click_pos = widget.rect().center()

    # right_click_widget(test_model, widget, right_click_pos)
    # time.sleep(1)
    # print(f"right_click_pos: {right_click_pos}")
    # # 1) Right click to show the menu
    # QtTest.QTest.mouseClick(widget, QtCore.Qt.RightButton, QtCore.Qt.NoModifier, right_click_pos)

    # # 2) Grab the topmost popup (the QMenu)
    # menu = QtWidgets.QApplication.activePopupWidget()
    print(f"type of menu: {type(menu)}")
    if not isinstance(test_model.main_view._menu, QtWidgets.QMenu):
        raise RuntimeError("No context menu appeared.")

    # Used to keep a reference to objects so they don't get garbage collected
    menus = []
    current_menu = test_model.main_view._menu
    menus.append(current_menu)
    # 3) Walk through the path
    for i, key in enumerate(path):
        print(f"new key: {key}")
        kw = key if isinstance(key, dict) else {"text": key}
        print(f"new actions!")
        print(f"current menu: {current_menu}")
        print(f"type of current menu: {type(current_menu)}")
        for a in current_menu.actions():
            name = a.text()
            print(f"name: {name}")
        act = _find_action_by(current_menu, **kw)
        print(f"act 111: {act}")
        if not act:
            raise RuntimeError(f"Menu item not found at level {i}: {key}")

        # Hover over the action so its submenu (if any) opens
        # center = _action_center(current_menu, act)
        # global_pt = current_menu.mapToGlobal(center)
        # test_model.move_mouse(current_menu, global_pt)

        if i < len(path) - 1:
            # Some styles open submenu on hover; some need a click—support both.
            sub = act.menu()
            print(f"type of sub: {type(sub)}")
            if sub is None:
                raise RuntimeError(f"Could not open submenu at level {i}: {key}")
            else:
                # give the hover-open time
                time.sleep(delay_ms/1000)

            if not isinstance(sub, QtWidgets.QMenu):
                raise RuntimeError(f"Could not open submenu at level {i}: {key}")
            print(f"changing current menu")
            menus.append(sub)
            current_menu = sub
            test_model.popup_menu(sub, QPoint(0, 0))
        else:
            print(f"i: {i}")
            print(f"kw: {kw}")
            print(f"act: {act}")
            print(f"type of act: {type(act)}")
            act.trigger()

def are_pixels_close(pixel1, pixel2) -> bool:
        '''
        Helper functions to determine if two pixels are close. Used for when scrolling
        in zoom pane and center's don't exactly align.
        '''
        if isinstance(pixel1, (QPoint, QPointF)):
            pixel1 = (pixel1.x(), pixel1.y())

        if isinstance(pixel2, (QPoint, QPointF)):
            pixel2 = (pixel2.x(), pixel2.y())

        pixel1_diff = abs(pixel1[0]-pixel1[1])
        pixel2_diff = abs(pixel2[0]-pixel2[1])

        diff_similar = abs(pixel1_diff - pixel2_diff) <= 2 

        epsilon = 2
        return abs(pixel1[0]-pixel2[0]) <= epsilon and diff_similar

def are_qrects_close(qrect1: QRect, qrect2: QRect, epsilon=3) -> bool:
    '''
    Helper functions to determine if two qrects are close. Used for when scrolling
    in zoom pane and center's don't exactly align. Not exact alignment seems to
    occur when we enter the event loop.
    '''
    top_left_1 = qrect1.topLeft()
    top_left_1 = (top_left_1.x(), top_left_1.y())

    width_1 = qrect1.width()
    height_1 = qrect1.height()

    top_left_2 = qrect2.topLeft()
    top_left_2 = (top_left_2.x(), top_left_2.y())

    width_2 = qrect2.width()
    height_2 = qrect2.height()

    width_diff = abs(width_1-width_2)
    height_diff = abs(height_1-height_2)

    diff_similar = abs(width_diff-height_diff) <= epsilon

    return top_left_1 == top_left_2 \
        and diff_similar \
        and width_diff <= epsilon \
        and height_diff <= epsilon