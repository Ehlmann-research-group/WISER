import os

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

import gui.resources

from .app_state import ApplicationState
from raster.dataset import RasterDataSet

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .rasterpane import RasterPane


class DatasetChooser(QToolButton):
    '''
    This toolbar button allows the user to quickly select a dataset to display,
    by popping up a menu that shows the list of datasets.  The dataset-list menu
    stays in sync with the current application state, and what datasets have
    been loaded.
    '''

    def __init__(self, rasterpane: 'RasterPane', app_state: ApplicationState):
        '''
        NOTE:  The RasterPane argument allows the dataset chooser to be informed
               when the raster-pane switches to multiple views.  It may be set
               to None, in which case the chooser will act like it is only
               managing one view.
        '''
        super().__init__()

        self._rasterpane = rasterpane
        self._app_state = app_state

        self._dataset_menu = QMenu()
        self._action_list = []

        self.setIcon(QIcon(':/icons/stack.svg'))
        self.setToolTip(self.tr('Select dataset to view'))
        self.setPopupMode(QToolButton.InstantPopup)
        self.setMenu(self._dataset_menu)

        self.triggered.connect(self._on_dataset_changed)

        self._app_state.dataset_added.connect(self._on_dataset_added)
        self._app_state.dataset_removed.connect(self._on_dataset_removed)

        if self._rasterpane is not None:
            self._rasterpane.views_changed.connect(self._on_views_changed)

        self._populate_dataset_menu()


    def _populate_dataset_menu(self):
        '''
        This helper function clears and repopulates the dataset menu from the
        current list of datasets in the application state.
        '''

        # Remove all existing actions
        self._action_list = []
        self._dataset_menu.clear()

        if self._rasterpane is not None:
            num_views = self._rasterpane.get_num_views()
        else:
            num_views = (1, 1)

        if num_views != (1, 1):
            # We have multiple raster-views in the pane.  Generate a menu for
            # each view.

            (rows, cols) = num_views
            for r in range(rows):
                for c in range(cols):
                    rv_menu = QMenu(self.tr('Row {r} column {c}').format(r=r, c=c))
                    self._dataset_menu.addMenu(rv_menu)

                    self._add_dataset_menu_items(rv_menu, (r, c))

        else:
            # We only have one raster-view in the pane.  Use the top level menu
            # for the dataset list.
            self._add_dataset_menu_items(self._dataset_menu)


    def _add_dataset_menu_items(self, menu, rasterview_pos=(0, 0)):
        # Add an action for each dataset
        for dataset in self._app_state.get_datasets():
            file_path = dataset.get_filepaths()[0]
            file_name = os.path.basename(file_path)

            act = QAction(file_name, parent=menu)
            act.setCheckable(True)
            act.setData( (rasterview_pos, dataset.get_id()) )

            menu.addAction(act)
            # self._action_list.append(act)

        # if len(self._action_list) > 0:
        #     self._action_list[0].setChecked(True)


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
        self._populate_dataset_menu()

    def _on_dataset_removed(self, ds_id: int):
        self._populate_dataset_menu()

    def _on_views_changed(self, shape):
        # print(f'TODO:  _on_views_changed(shape={shape})')
        self._populate_dataset_menu()
