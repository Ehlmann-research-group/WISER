import os

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

from .app_state import ApplicationState
from raster.dataset import RasterDataSet


class DatasetChooser(QToolButton):
    '''
    This toolbar button allows the user to quickly select a dataset to display,
    by popping up a menu that shows the list of datasets.  The dataset-list menu
    stays in sync with the current application state, and what datasets have
    been loaded.
    '''

    def __init__(self, app_state: ApplicationState):
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


    def _make_dataset_action(self, dataset: RasterDataSet) -> QAction:
        '''
        This helper function generates a QAction for selecting the specified
        dataset, suitable for adding to the Dataset Chooser's pop-up menu.
        '''


        file_path = dataset.get_filepaths()[0]
        file_name = os.path.basename(file_path)

        act = QAction(file_name)
        act.setCheckable(True)
        act.setData(dataset.get_id())

        return act


    def _populate_dataset_menu(self):
        '''
        This helper function clears and repopulates the dataset menu from the
        current list of datasets in the application state.
        '''

        # Remove all existing actions
        self._action_list = []
        self._dataset_menu.clear()

        # Add an action for each dataset
        for dataset in enumerate(self._app_state.get_datasets()):
            act = self._make_dataset_action(dataset)
            self._dataset_menu.addAction(act)
            self._action_list.append(act)

        if len(self._action_list) > 0:
            self._action_list[0].setChecked(True)


    def _on_dataset_changed(self, act: QAction):
        '''
        This helper function ensures that only the currently selected dataset
        has a check-mark by it; all other datasets will be (or become)
        deselected.
        '''

        for oact in self._action_list:
            if oact != act and oact.isChecked():
                oact.setChecked(False)


    def _on_dataset_added(self, ds_id: int):
        '''
        This helper function updates the dataset-chooser when a new dataset is
        added to the application state.
        '''

        dataset = self._app_state.get_dataset(ds_id)
        act = self._make_dataset_action(dataset)

        # Always append the new action to the end of the menu and list.
        self._dataset_menu.addAction(act)
        self._action_list.append(act)

        if len(self._action_list) == 1:
            # We added an item to an empty list, so make sure to check it.
            self._action_list[0].setChecked(True)


    def _on_dataset_removed(self, ds_id: int):
        '''
        This helper function updates the dataset-chooser when a dataset is
        removed from the application state.
        '''

        # Find the QAction corresponding to the specified dataset ID.
        act = None
        for i in range(len(self._action_list)):
            act = self._action_list[i]
            if act.data() == ds_id:
                break

        if act is None:
            print(f'WARNING:  Dataset Chooser encountered unrecognized dataset ID {ds_id}')
            return

        self._dataset_menu.removeAction(act)
        del self._action_list[i]

        if act.checked():
            # The action was checked.  Switch the check to a different entry.
            self._action_list[0].setChecked(True)
