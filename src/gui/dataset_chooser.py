import os

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *


class DatasetChooser(QToolButton):
    '''
    This toolbar button allows the user to quickly select a dataset to display,
    by popping up a menu that shows the list of datasets.  The dataset-list menu
    stays in sync with the current state of the model, and what datasets have
    been loaded.
    '''

    def __init__(self, app_state):
        super().__init__()

        self._app_state = app_state

        self._dataset_menu = QMenu()
        self._action_list = []

        self.setIcon(QIcon('resources/stack.svg'))
        self.setToolTip(self.tr('Select dataset to view'))
        self.setPopupMode(QToolButton.InstantPopup)
        self.setMenu(self._dataset_menu)

        self.triggered.connect(self._on_dataset_changed)

        self._app_state.dataset_added.connect(self._on_dataset_added)
        self._app_state.dataset_removed.connect(self._on_dataset_removed)

        self._populate_dataset_menu()


    def _make_dataset_action(self, index, dataset):
        file_path = dataset.get_filepaths()[0]
        file_name = os.path.basename(file_path)

        act = QAction(file_name)
        act.setCheckable(True)
        act.setData(index)

        return act


    def _populate_dataset_menu(self):
        # Remove all existing actions
        self._action_list = []
        self._dataset_menu.clear()

        # Add an action for each dataset
        model = self._app_state.get_model()
        for (index, dataset) in enumerate(model.get_datasets()):
            act = self._make_dataset_action(index, dataset)
            self._dataset_menu.addAction(act)
            self._action_list.append(act)

        if len(self._action_list) > 0:
            self._action_list[0].setChecked(True)


    def _on_dataset_changed(self, act):
        for oact in self._action_list:
            if oact != act and oact.isChecked():
                oact.setChecked(False)


    def _on_dataset_added(self, index):
        model = self._app_state.get_model()
        dataset = model.get_dataset(index)
        act = self._make_dataset_action(index, dataset)

        if index < len(self._action_list):
            # Inserting the action into the middle of the menu
            self._dataset_menu.insertAction(self._action_list[index], act)
            self._action_list.insert(index, act)

            # Need to update the indexes of all subsequent actions
            for i in range(index, len(self._action_list)):
                self._action_list[index].setData(i)

        else:
            # Appending the action to the end of the menu
            self._dataset_menu.addAction(act)
            self._action_list.append(act)

        if len(self._action_list) == 1:
            # First item being added.  Make sure to check it.
            self._action_list[0].setChecked(True)


    def _on_dataset_removed(self, index):
        act = self._action_list[index]

        self._dataset_menu.removeAction(act)
        del self._action_list[index]

        for i in range(index, len(self._action_list)):
            self._action_list[index].setData(i)

        if act.checked():
            # The action was checked.  Switch the check to a different entry.
            self._action_list[0].setChecked(True)
