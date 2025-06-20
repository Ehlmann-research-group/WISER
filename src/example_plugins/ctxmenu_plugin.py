import pprint
import textwrap

from typing import Any, Dict

from wiser.plugins import ContextMenuPlugin, ContextMenuType

from PySide2.QtWidgets import QMenu, QMessageBox


class HelloContextPlugin(ContextMenuPlugin):
    """
    A simple "Hello world!" example of a context-menu plugin.
    """

    def __init__(self):
        super().__init__()

    def add_context_menu_items(
        self,
        context_type: ContextMenuType,
        context_menu: QMenu,
        context: Dict[str, Any],
    ) -> None:
        """
        Use QMenu.addAction() to add individual actions, or QMenu.addMenu() to
        add sub-menus to the Tools menu.
        """
        if context_type == ContextMenuType.RASTER_VIEW:
            # Context-menu display in a raster-view, which probably is showing a
            # dataset.  The current dataset is passed to the plugin.
            pass

        elif context_type == ContextMenuType.SPECTRUM_PLOT:
            # Context-menu display in the spectrum-plot window.
            pass

        elif context_type == ContextMenuType.DATASET_PICK:
            # A specific dataset was picked.  This may not be in the context of
            # a raster-view window, e.g. if the user right-clicks on a dataset
            # in the info viewer.
            pass

        elif context_type == ContextMenuType.SPECTRUM_PICK:
            # A specific spectrum was picked.  The spectrum is passed to the
            # plugin.
            pass

        elif context_type == ContextMenuType.ROI_PICK:
            # A specific ROI was picked.  The ROI is passed, along with the
            # current dataset (if available).
            pass

        else:
            raise ValueError(f"Unrecognized context_type value {context_type}")

        act = context_menu.addAction(f"Say hello {context_type}...")
        act.triggered.connect(
            lambda checked=False: self.say_hello(context_type, context)
        )

    def say_hello(self, context_type: ContextMenuType, context: Dict[str, Any]):
        context_str = pprint.pformat(context)

        print("HelloContextPlugin.say_hello() was called!")
        print(f" * context_type = {context_type}")
        print(f" * context =\n{textwrap.indent(context_str, ' ' * 8)}")

        QMessageBox.information(
            None,
            "Hello-Context Plugin",
            f"Hello from a {context_type} context menu!\n\n{context_str}",
        )
