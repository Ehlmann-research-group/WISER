"""Unified Save-Project dialog (issue #626).

One checkable tree is the whole decision.  Three top-level groups -- Datasets,
ROIs, and Analysis outputs -- list every item in the session, checked by default;
unchecking an item leaves it out of the project.  Unchecking a dataset or ROI
cascades to the products under it -- a dependent spectrum freezes to a snapshot, a
stretch is dropped -- shown live as a ``(snapshot)`` / ``(dropped)`` annotation on
the child.  A top checkbox chooses a self-contained (portable) save that copies
file-backed data into the bundle instead of referencing it.

The dialog is a thin view over :mod:`wiser.project.save_plan`: :func:`save_tree`
builds the model and :func:`resolver_for_selection` turns the user's exclusions
into the :class:`~wiser.project.resolver.DependencyResolver` handed to
:func:`~wiser.project.orchestrate.save_project`.  The tree is built once and its
annotations and group tri-states are updated in place as the user toggles, so the
item whose signal is being handled is never deleted mid-update.
"""

from typing import TYPE_CHECKING, Any, Dict, List, Set, Tuple

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
)

from wiser.project.save_plan import resolver_for_selection, save_tree

if TYPE_CHECKING:
    from wiser.gui.app_state import ApplicationState
    from wiser.project.resolver import DependencyResolver

Handle = Tuple[str, object]  # (kind, id) -- the resolver's exclusion key for an item


class SaveProjectDialog(QDialog):
    def __init__(self, app_state: "ApplicationState", parent=None):
        super().__init__(parent)
        self._app_state = app_state
        self._excluded: Set[Handle] = set()
        self._updating = False
        self._item_widgets: Dict[Handle, QTreeWidgetItem] = {}
        self._group_handles: Dict[str, List[Handle]] = {}
        self.setWindowTitle(self.tr("Save Project"))

        layout = QVBoxLayout(self)
        intro = QLabel(
            self.tr(
                "Choose what to include in the project. Everything is included by default; "
                "uncheck an item to leave it out. A product under a dataset or ROI follows it "
                "-- a spectrum freezes to a snapshot and a stretch is dropped when its dataset "
                "is left out."
            )
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        self._self_contained = QCheckBox(
            self.tr("Save self-contained (copy data into the project so it can be shared or moved)")
        )
        layout.addWidget(self._self_contained)

        self._tree = QTreeWidget(self)
        self._tree.setColumnCount(1)
        self._tree.setHeaderHidden(True)
        self._tree.setEditTriggers(QAbstractItemView.NoEditTriggers)
        layout.addWidget(self._tree)

        row = QHBoxLayout()
        include_all = QPushButton(self.tr("Include All"))
        exclude_all = QPushButton(self.tr("Exclude All"))
        include_all.clicked.connect(lambda: self._set_all(excluded=False))
        exclude_all.clicked.connect(lambda: self._set_all(excluded=True))
        row.addWidget(include_all)
        row.addWidget(exclude_all)
        row.addStretch()
        layout.addLayout(row)

        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self._build_tree()
        self._tree.itemChanged.connect(self._on_item_changed)

    # -- resolver / mode ---------------------------------------------------

    def _current_resolver(self) -> "DependencyResolver":
        excluded_datasets = [i for (k, i) in self._excluded if k == "dataset"]
        excluded_rois = [i for (k, i) in self._excluded if k == "roi"]
        excluded_items = {(k, i) for (k, i) in self._excluded if k not in ("dataset", "roi")}
        return resolver_for_selection(self._app_state, excluded_datasets, excluded_rois, excluded_items)

    def get_resolver(self) -> "DependencyResolver":
        """The resolver for the user's current selection (call after ``exec()``)."""
        return self._current_resolver()

    def get_self_contained(self) -> bool:
        """Whether the user asked for a portable, self-contained save."""
        return self._self_contained.isChecked()

    # -- build (once) ------------------------------------------------------

    def _build_tree(self) -> None:
        self._tree.clear()
        self._item_widgets.clear()
        self._group_handles.clear()
        for group in save_tree(self._app_state, self._current_resolver()):
            group_item = QTreeWidgetItem(self._tree)
            group_item.setText(0, group["label"])
            group_item.setData(0, Qt.UserRole, ("group", group["group"]))
            group_item.setFlags(group_item.flags() | Qt.ItemIsUserCheckable)
            group_item.setCheckState(0, Qt.Checked)
            group_item.setExpanded(True)
            handles: List[Handle] = []
            for node in group["children"]:
                handle: Handle = (node["kind"], node["id"])
                handles.append(handle)
                item = QTreeWidgetItem(group_item)
                item.setText(0, self._item_label(node))
                item.setData(0, Qt.UserRole, ("item", node["kind"], node["id"]))
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                item.setCheckState(0, Qt.Checked)
                item.setExpanded(True)
                self._item_widgets[handle] = item
                for child in node.get("children", []):
                    leaf = QTreeWidgetItem(item)
                    leaf.setText(0, self._child_label(child))
                    leaf.setFlags(leaf.flags() & ~Qt.ItemIsUserCheckable)
            self._group_handles[group["group"]] = handles

    # -- toggles -----------------------------------------------------------

    def _on_item_changed(self, item: QTreeWidgetItem, column: int) -> None:
        if self._updating:
            return
        data = item.data(0, Qt.UserRole)
        if not data:
            return
        self._updating = True
        try:
            excluded = item.checkState(0) != Qt.Checked
            if data[0] == "group":
                self._apply_group(data[1], excluded)
            elif data[0] == "item":
                self._set_excluded((data[1], data[2]), excluded)
            self._sync()
        finally:
            self._updating = False

    def _set_all(self, excluded: bool) -> None:
        self._updating = True
        try:
            for handles in self._group_handles.values():
                for handle in handles:
                    self._set_excluded(handle, excluded)
                    self._reflect_checkbox(handle, excluded)
            self._sync()
        finally:
            self._updating = False

    def _apply_group(self, group_key: str, excluded: bool) -> None:
        for handle in self._group_handles.get(group_key, []):
            self._set_excluded(handle, excluded)
            self._reflect_checkbox(handle, excluded)

    def _set_excluded(self, handle: Handle, excluded: bool) -> None:
        if excluded:
            self._excluded.add(handle)
        else:
            self._excluded.discard(handle)

    def _reflect_checkbox(self, handle: Handle, excluded: bool) -> None:
        widget = self._item_widgets.get(handle)
        if widget is not None:
            widget.setCheckState(0, Qt.Unchecked if excluded else Qt.Checked)

    def _sync(self) -> None:
        # Re-annotate each cascade child with the policy it now takes, and recompute
        # every group's tri-state -- in place, so the tree is never rebuilt.
        for group in save_tree(self._app_state, self._current_resolver()):
            for node in group["children"]:
                widget = self._item_widgets.get((node["kind"], node["id"]))
                if widget is None:
                    continue
                for index, child in enumerate(node.get("children", [])):
                    if index < widget.childCount():
                        widget.child(index).setText(0, self._child_label(child))
        self._update_group_states()

    def _update_group_states(self) -> None:
        for i in range(self._tree.topLevelItemCount()):
            group_item = self._tree.topLevelItem(i)
            items = [
                group_item.child(j)
                for j in range(group_item.childCount())
                if (group_item.child(j).data(0, Qt.UserRole) or (None,))[0] == "item"
            ]
            checked = sum(1 for it in items if it.checkState(0) == Qt.Checked)
            if not items or checked == len(items):
                state = Qt.Checked
            elif checked == 0:
                state = Qt.Unchecked
            else:
                state = Qt.PartiallyChecked
            group_item.setCheckState(0, state)

    # -- labels ------------------------------------------------------------

    def _item_label(self, node: Dict[str, Any]) -> str:
        kind = node["kind"]
        if kind == "dataset":
            backing = self.tr("file") if node.get("backing") == "file" else self.tr("in-memory")
            return self.tr("{0} ({1})").format(node["label"], backing)
        if kind == "library":
            return self.tr("Library: {0}").format(node["label"])
        if kind == "crs":
            return self.tr("CRS: {0}").format(node["label"])
        if kind == "bandmath":
            return self.tr("Band math: {0}").format(node["label"])
        return node["label"]

    def _child_label(self, child: Dict[str, Any]) -> str:
        label = child["label"]
        if child["policy"] == "snapshot":
            return self.tr("{0} (snapshot)").format(label)
        if child["policy"] == "drop":
            return self.tr("{0} (dropped)").format(label)
        return label
