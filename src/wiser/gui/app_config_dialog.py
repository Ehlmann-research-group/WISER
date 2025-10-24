import abc
import sys
import json
import os
import subprocess

from typing import List, Optional, Tuple, Dict
from pathlib import Path
import inspect
import importlib

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

from astropy import units as u

from .generated.app_config_ui import Ui_AppConfigDialog

from .app_state import ApplicationState

# We have a lot of variables named "plugins" so make the package name distinct.
from wiser.plugins import utils as plugutils
from wiser.plugins.types import ContextMenuPlugin, ToolsMenuPlugin, BandMathPlugin


def qlistwidget_to_list(list_widget: QListWidget) -> List[str]:
    result: List[str] = []
    for i in range(list_widget.count()):
        result.append(list_widget.item(i).text())

    return result


def qlistwidget_selections(list_widget: QListWidget) -> List[Tuple[int, str]]:
    result = []
    for i in range(list_widget.count()):
        item = list_widget.item(i)
        if item.isSelected():
            result.append((i, item.text()))

    return result


class EnvironmentManager(abc.ABC):
    """
    This class is the base class for all environment managers. It is used
    to get the packages for a specific environment manager given an
    environment name.
    """

    def get_env_manager_name(self) -> str:
        raise NotImplementedError("Must be implemented by subclass")

    def get_packages_directory(self, env_name: str) -> Path:
        raise NotImplementedError("Must be implemented by subclass")


class CondaEnvironmentManager(EnvironmentManager):
    """
    This class is used to get the packages for a conda environment.
    """

    @staticmethod
    def get_env_manager_name() -> str:
        return "Conda"

    @staticmethod
    def get_packages_directory(env_name: str) -> Path:
        """
        Return the absolute path to the Python package directory ("site-packages")
        for a given Conda environment.

        Resolution strategy (in order):
          1) If `env_name` looks like a filesystem path, treat it as an explicit
             prefix (env created with `-p/--prefix`).
          2) If `env_name` is "base" (or "root"), use Conda's root prefix.
          3) Search configured environment directories (envs_dirs), including any
             provided via CONDA_ENVS_PATH, for `<dir>/<env_name>`.
          4) As a final heuristic, if an active env matches the name, use CONDA_PREFIX.

        Cross-platform site-packages layout:
          - Windows:   <prefix>/Lib/site-packages
          - POSIX:     <prefix>/lib/pythonX.Y/site-packages (chosen by glob)

        Raises:
            FileNotFoundError: If the environment or its site-packages cannot be found.
        """
        # Query `conda info --json` if available (programmatic/portable).
        info = {}
        try:
            res = subprocess.run(["conda", "info", "--json"], check=True, text=True, capture_output=True)
            info = json.loads(res.stdout or "{}")
        except Exception:
            # Non-fatal: fall back to env vars/heuristics below.
            info = {}
        # 1) Explicit prefix path support (env created by --prefix).
        env_path = Path(os.path.expanduser(env_name))
        candidate_prefixes = []
        if env_path.is_absolute() or env_path.exists():
            candidate_prefixes.append(env_path)

        # 2) Handle base/root by using Conda's root prefix.
        if env_name.lower() in {"base", "root"}:
            root_prefix = info.get("root_prefix") or os.environ.get("CONDA_PREFIX")
            if root_prefix:
                candidate_prefixes.append(Path(root_prefix))

        # 3) Resolve named env via envs_dirs (respect CONDA_ENVS_PATH override).
        envs_dirs: List[Path] = []

        # Environment variable takes precedence (os.pathsep handles ; on Win, : on POSIX).
        envs_path_env = os.environ.get("CONDA_ENVS_PATH")
        if envs_path_env:
            envs_dirs.extend(Path(p) for p in envs_path_env.split(os.pathsep) if p)

        # Add conda-configured envs_dirs and the conventional <root_prefix>/envs.
        envs_dirs.extend(Path(p) for p in info.get("envs_dirs", []))
        envs_dirs.extend(Path(p) for p in info.get("envs", []))
        if info.get("root_prefix"):
            envs_dirs.append(Path(info["root_prefix"]) / "envs")

        # Deduplicate while preserving order.
        seen = set()
        envs_dirs = [p for p in envs_dirs if not (str(p) in seen or seen.add(str(p)))]

        # See what paths match our environment
        for d in envs_dirs:
            cand = d / env_name
            if cand.exists():
                candidate_prefixes.append(cand)
            if str(d.resolve()).endswith(env_name):
                candidate_prefixes.append(d)

        # 4) If still nothing, but active env matches by name, use it.
        active_prefix = info.get("active_prefix") or os.environ.get("CONDA_PREFIX")
        if active_prefix and Path(active_prefix).name == env_name:
            candidate_prefixes.append(Path(active_prefix))

        # Pick the first existing prefix we found.
        prefix = next((p for p in candidate_prefixes if p.exists()), None)
        if prefix is None:
            raise FileNotFoundError(
                f"Conda environment '{env_name}' not found. "
                "Checked explicit path, CONDA_ENVS_PATH, configured"
                "envs_dirs, root_prefix/envs, and active env."
            )

        # Compute site-packages per platform layout.
        if os.name == "nt":
            sp = prefix / "Lib" / "site-packages"
        else:
            # Typical POSIX layout: <prefix>/lib/pythonX.Y/site-packages
            candidates = sorted((prefix / "lib").glob("python*/site-packages"))
            sp = candidates[-1] if candidates else None

        if not sp or not sp.exists():
            raise FileNotFoundError(
                f"site-packages not found under '{prefix}'. "
                "Expected 'Lib/site-packages' (Windows) or 'lib/pythonX.Y/site-packages' (POSIX)."
            )

        return sp


PluginBases = (ContextMenuPlugin, ToolsMenuPlugin, BandMathPlugin)

ENV_MANAGERS: List[EnvironmentManager] = [CondaEnvironmentManager]


class AppConfigDialog(QDialog):
    """
    This dialog provides configuration options for the spectrum plot component,
    and for spectrum collection.
    """

    def __init__(self, app_state, parent=None):
        super().__init__(parent=parent)
        self._ui = Ui_AppConfigDialog()
        self._ui.setupUi(self)

        self._app_state = app_state

        # Duplicate the initial plugin-paths and plugin lists, so that we can
        # act based on the changes made by the user.  Also, duplicate the
        # Python system path, so that we can restore it when needed.
        self._initial_sys_path: List[str] = list(sys.path)
        self._initial_plugin_paths: List[str] = list(self._app_state.get_config("plugin_paths"))
        self._initial_plugins: List[str] = list(self._app_state.get_config("plugin_paths"))

        self._init_general_tab()
        self._init_plugins_tab()

    def _init_general_tab(self):
        # ==============================
        # Error-Reporting group-box

        self._ui.ckbox_online_bug_reporting.setChecked(
            self._app_state.get_config("general.online_bug_reporting")
        )

        # ==============================
        # Visible-Light group-box

        self._ui.ledit_red_wavelength.setValidator(QIntValidator())
        self._ui.ledit_green_wavelength.setValidator(QIntValidator())
        self._ui.ledit_blue_wavelength.setValidator(QIntValidator())

        red = self._app_state.get_config("general.red_wavelength_nm")
        green = self._app_state.get_config("general.green_wavelength_nm")
        blue = self._app_state.get_config("general.blue_wavelength_nm")

        self._ui.ledit_red_wavelength.setText(f"{red}")
        self._ui.ledit_green_wavelength.setText(f"{green}")
        self._ui.ledit_blue_wavelength.setText(f"{blue}")

        # ==============================
        # Raster Display group-box

        self._ui.ledit_viewport_highlight_color.setText(
            self._app_state.get_config("raster.viewport_highlight_color")
        )

        self._ui.cbox_pixel_cursor_type.addItem(self.tr("Crosshair"), "SMALL_CROSS")
        self._ui.cbox_pixel_cursor_type.addItem(self.tr("Large crosshair"), "LARGE_CROSS")
        self._ui.cbox_pixel_cursor_type.addItem(self.tr("Crosshair with box"), "SMALL_CROSS_BOX")

        self._ui.ledit_pixel_cursor_color.setText(self._app_state.get_config("raster.pixel_cursor_color"))

        self._ui.btn_viewport_highlight_color.clicked.connect(self._on_choose_viewport_highlight_color)
        self._ui.btn_pixel_cursor_color.clicked.connect(self._on_choose_pixel_cursor_color)

        # Fetch the cursor type as a string
        cursor = self._app_state.get_config("raster.pixel_cursor_type")
        index = self._ui.cbox_pixel_cursor_type.findData(cursor)
        if index == -1:
            index = 0
        self._ui.cbox_pixel_cursor_type.setCurrentIndex(index)

        # ==============================
        # New Spectra group-box

        self._ui.ledit_aavg_x.setValidator(QIntValidator(1, 99))
        self._ui.ledit_aavg_y.setValidator(QIntValidator(1, 99))

        self._ui.ledit_aavg_x.setText(str(self._app_state.get_config("spectra.default_area_avg_x")))
        self._ui.ledit_aavg_y.setText(str(self._app_state.get_config("spectra.default_area_avg_y")))

        self._ui.cbox_default_avg_mode.addItem(self.tr("Mean"), "MEAN")
        self._ui.cbox_default_avg_mode.addItem(self.tr("Median"), "MEDIAN")

        # Fetch the mode as a string
        mode = self._app_state.get_config("spectra.default_area_avg_mode")
        index = self._ui.cbox_default_avg_mode.findData(mode)
        if index == -1:
            index = 0
        self._ui.cbox_default_avg_mode.setCurrentIndex(index)

    def _init_plugins_tab(self):
        # Plugin paths

        plugin_paths = self._app_state.get_config("plugin_paths")
        for p in plugin_paths:
            item = QListWidgetItem(p)
            item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            self._ui.list_plugin_paths.addItem(item)

        self._ui.btn_edit_plugin_path.setEnabled(False)
        self._ui.btn_del_plugin_path.setEnabled(False)

        self._ui.list_plugin_paths.itemSelectionChanged.connect(self._on_plugin_path_selection_changed)
        self._ui.btn_add_plugin_path.clicked.connect(self._on_add_plugin_path)
        self._ui.btn_edit_plugin_path.clicked.connect(self._on_edit_plugin_path)
        self._ui.btn_del_plugin_path.clicked.connect(self._on_del_plugin_path)

        # Plugins

        plugins = self._app_state.get_config("plugins")
        for p in plugins:
            item = QListWidgetItem(p)
            item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            self._ui.list_plugins.addItem(item)

        self._ui.btn_edit_plugin.setEnabled(False)
        self._ui.btn_del_plugin.setEnabled(False)

        self._ui.list_plugins.itemSelectionChanged.connect(self._on_plugin_selection_changed)
        self._ui.btn_add_plugin.clicked.connect(self._on_add_plugin)
        self._ui.btn_edit_plugin.clicked.connect(self._on_edit_plugin)
        self._ui.btn_del_plugin.clicked.connect(self._on_del_plugin)
        self._ui.btn_verify_plugins.clicked.connect(self._on_verify_plugins)

        # Set easy add plugin section
        self._ui.btn_add_plugin_file.clicked.connect(self._on_add_plugin_by_file)
        for env_manager in ENV_MANAGERS:
            self._ui.cbox_env_manager.addItem(env_manager.get_env_manager_name(), env_manager)
        self._ui.btn_add_env.clicked.connect(self._on_add_env)

        # Set advanced button functionality
        self._ui.btn_show_advanced.clicked.connect(self._on_show_advanced)
        self._ui.gbox_plugins.setVisible(False)
        self._ui.gbox_plugin_paths.setVisible(False)

    def _on_choose_viewport_highlight_color(self, checked):
        initial_color = QColor(self._ui.ledit_viewport_highlight_color.text())
        color = QColorDialog.getColor(parent=self, initial=initial_color)
        if color.isValid():
            self._ui.ledit_viewport_highlight_color.setText(color.name())

    def _on_choose_pixel_cursor_color(self, checked):
        initial_color = QColor(self._ui.ledit_pixel_cursor_color.text())
        color = QColorDialog.getColor(parent=self, initial=initial_color)
        if color.isValid():
            self._ui.ledit_pixel_cursor_color.setText(color.name())

    # ========================================================================
    # EASY ADD PLUGIN BY FILE UI
    # ========================================================================

    def _on_show_advanced(self, checked=False):
        # Determine current visibility (just use one of them as reference)
        is_visible = self._ui.gbox_plugins.isVisible()
        should_show = not is_visible

        # Toggle visibility
        self._ui.gbox_plugins.setVisible(should_show)
        self._ui.gbox_plugin_paths.setVisible(should_show)

    def _on_add_plugin_by_file(self, checked=False):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Plugin File",
            "",
            "Python Files (*.py);;All Files (*)",
        )

        if not file_path:
            return

        self._load_plugin_from_file(file_path)

    def _load_plugin_from_file(self, file_path: str) -> Tuple[List[Dict], str]:
        """
        Discover plugin classes in the given file and add them to the UI lists.

        Plugins' base directory (if new) is added to `list_plugin_paths`, and each
        plugin's fully qualified class name (if new) is added to `list_plugins`.
        If duplicates are detected, a warning dialog is shown explaining what wasn't added.

        Args:
            file_path (str): Path to a Python file which may define plugin classes.

        Raises:
            FileNotFoundError: If the file is missing or cannot be discovered.
            ValueError: If `_discover_plugin_classes` returns no valid plugins.
        """
        plugins_dict = self._discover_plugin_classes(file_path)

        base_dir_abs: str = plugins_dict["base_dir_abs"]
        base_dir_duplicate = False
        if base_dir_abs not in qlistwidget_to_list(self._ui.list_plugin_paths):
            # Add the path to the list widget.
            item = QListWidgetItem(base_dir_abs)
            item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            self._ui.list_plugin_paths.addItem(item)
        else:
            base_dir_duplicate = True

        plugins: List[Dict] = plugins_dict["plugins"]

        plugin_duplicates: List[str] = []
        for plugin in plugins:
            fqcn = plugin["fqcn"]
            if fqcn not in qlistwidget_to_list(self._ui.list_plugins):
                # Add the plugin to the list widget.
                item = QListWidgetItem(fqcn)
                item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                self._ui.list_plugins.addItem(item)
            else:
                plugin_duplicates.append(fqcn)
        if plugin_duplicates or base_dir_duplicate:
            msg = QMessageBox(self)
            msg.setIcon(QMessageBox.Warning)
            msg.setWindowTitle("Some items were not added")
            msg.setText("Couldn't add some items.")

            details = []
            if base_dir_duplicate:
                details.append(f"• The base directory is already in the list:\n    {base_dir_abs}")
            if plugin_duplicates:
                details.append(
                    "• The following plugins were already present and were not added:\n    - "
                    + "\n    - ".join(plugin_duplicates)
                )

            msg.setInformativeText("\n\n".join(details))
            msg.setStandardButtons(QMessageBox.Ok)
            # Qt6 uses exec(), Qt5 uses exec_(); this handles both:
            try:
                msg.exec()
            except AttributeError:
                msg.exec_()

        return plugins, base_dir_abs

    def _derive_paths_and_module(self, file_path: str) -> Tuple[str, Path, Path]:
        """
        Given a path to a .py file, find:
        - the nearest parent directory that contains __init__.py (package_root)
        - the fully-qualified module path using that package root
        - the directory above the package root (base_dir, to put on sys.path)

        Returns:
            module_path: e.g. 'example_plugins.context_plugin'
            package_root_abs: absolute Path to the nearest package root
            base_dir_abs: absolute Path to the directory above package_root
        """
        p = Path(file_path).resolve()
        if not p.is_file() or p.suffix != ".py":
            raise ValueError(f"Expected a .py file, got: {p}")

        current = p.parent
        package_root = None
        while True:
            if (current / "__init__.py").exists():
                package_root = current
                break
            if current == current.parent:
                break  # hit filesystem root
            current = current.parent

        if package_root is None:
            # No package root found; treat the folder containing the file as base
            module_path = p.stem
            package_root_abs = p.parent.resolve()
            base_dir_abs = package_root_abs
        else:
            module_path = f"{package_root.name}.{p.stem}"
            package_root_abs = package_root.resolve()
            base_dir_abs = package_root_abs.parent.resolve()

        return module_path, package_root_abs, base_dir_abs

    def _discover_plugin_classes(self, file_path: str) -> Dict:
        """
        Imports the target file as a module using its derived fully-qualified
        name and returns all classes that subclass the known Plugin base classes.
        """
        module_path, package_root_abs, base_dir_abs = self._derive_paths_and_module(file_path)

        # Ensure base_dir is importable so 'import example_plugins.context_plugin' works
        added = False
        if str(base_dir_abs) not in sys.path:
            sys.path.insert(0, str(base_dir_abs))
            added = True

        # Import the module via its fully-qualified name
        mod = importlib.import_module(module_path)

        found = []
        for name, obj in vars(mod).items():
            if inspect.isclass(obj):
                matches = [b.__name__ for b in PluginBases if issubclass(obj, b) and obj is not b]
                if matches:
                    found.append(
                        {
                            "name": name,
                            "fqcn": f"{module_path}.{name}",
                            "base_matches": matches,
                            "cls": obj,
                        }
                    )
        if added:
            sys.path.remove(base_dir_abs)

        return {
            "module_path": module_path,
            "package_root_abs": str(package_root_abs),
            "base_dir_abs": str(base_dir_abs),
            "plugins": found,
        }

    def _on_add_env(self):
        env_manager: EnvironmentManager = self._ui.cbox_env_manager.currentData()
        env_name: str = self._ui.ledit_env_name.text()
        try:
            packages_path: Path = env_manager.get_packages_directory(env_name)
            packages_path_str = str(packages_path.resolve())
            if packages_path_str not in qlistwidget_to_list(self._ui.list_plugin_paths):
                # Add the path to the list widget.
                item = QListWidgetItem(packages_path_str)
                item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                self._ui.list_plugin_paths.addItem(item)
                QMessageBox.information(
                    self,
                    self.tr("Environment Found"),
                    self.tr(f"The environment {env_name} was successfully found and added!"),
                )
            else:
                QMessageBox.warning(
                    self,
                    self.tr("Path already in plugin paths"),
                    self.tr(f"Path {packages_path_str} already in plugin paths"),
                )
        except FileNotFoundError as e:
            QMessageBox.warning(
                self,
                self.tr(f"Could Not Find {env_manager.get_env_manager_name()} Environment"),
                self.tr(f"Received Error:\n\n{e}"),
            )

    # ========================================================================
    # PLUGIN PATH UI
    # ========================================================================

    def _on_plugin_path_selection_changed(self):
        # Enable or disable the "Remove path" button based on whether something
        # is actually selected.
        item_selected = len(self._ui.list_plugin_paths.selectedItems()) > 0
        self._ui.btn_edit_plugin_path.setEnabled(item_selected)
        self._ui.btn_del_plugin_path.setEnabled(item_selected)

    def _on_add_plugin_path(self, checked=False):
        self._edit_plugin_path_helper()

    def _on_edit_plugin_path(self, checked=False):
        item = self._ui.list_plugin_paths.currentItem()
        self._edit_plugin_path_helper(item)

    def _edit_plugin_path_helper(self, existing_item=None):
        initial_path = None
        if existing_item is not None:
            initial_path = existing_item.text()

        path = QFileDialog.getExistingDirectory(
            parent=self, dir=initial_path, caption=self.tr("Choose plugin path")
        )

        if path:
            # Make sure the path isn't already in the list of paths.
            if path in qlistwidget_to_list(self._ui.list_plugin_paths):
                QMessageBox.information(
                    self,
                    self.tr("Path already in plugin paths"),
                    self.tr('Path "{0}" already in plugin-paths list').format(path),
                )
                return

            # Add the path to the list widget.
            if existing_item is not None:
                # Update the list-widget item with the updated path.
                existing_item.setText(path)

            else:
                # Add the path to the list widget.
                item = QListWidgetItem(path)
                item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                self._ui.list_plugin_paths.addItem(item)

    def _on_del_plugin_path(self, checked=False):
        # Find the path or paths to be removed
        paths: List[Tuple[int, str]] = qlistwidget_selections(self._ui.list_plugin_paths)

        # Get user confirmation
        if len(paths) == 1:
            msg = self.tr("Remove this plugin path?") + f"\n\n{paths[0][1]}"
        else:
            msg = self.tr("Remove these plugin paths?") + "\n" + "\n".join([p[1] for p in paths])

        result = QMessageBox.question(self, self.tr("Remove plugin paths?"), msg)
        if result != QMessageBox.Yes:
            return  # User decided not to remove the path(s).

        # Delete all affected plugin paths.  Delete in decreasing index order,
        # so that indexes aren't shifted/invalidated by deleting lower-index
        # entries.
        for i in sorted([p[0] for p in paths], reverse=True):
            self._ui.list_plugin_paths.takeItem(i)

        # Now, no paths should be selected
        self._ui.btn_del_plugin_path.setEnabled(False)

    # ========================================================================
    # PLUGIN UI
    # ========================================================================

    def _on_plugin_selection_changed(self):
        # Enable or disable the "Remove plugin" button based on whether
        # something is actually selected.
        item_selected = len(self._ui.list_plugins.selectedItems()) > 0
        self._ui.btn_edit_plugin.setEnabled(item_selected)
        self._ui.btn_del_plugin.setEnabled(item_selected)

    def _on_add_plugin(self, checked=False):
        self._edit_plugin_helper()

    def _on_edit_plugin(self, checked=False):
        item = self._ui.list_plugins.currentItem()
        self._edit_plugin_helper(item)

    def _edit_plugin_helper(self, existing_item=None):
        initial_plugin = None
        if existing_item is not None:
            initial_plugin = existing_item.text()

        (plugin, success) = QInputDialog.getText(
            self,
            self.tr("Plugin class name"),
            self.tr("Enter fully-qualified name of plugin class"),
            text=initial_plugin,
        )

        if success:
            # Make sure the plugin isn't already in the list of plugins.
            if plugin in qlistwidget_to_list(self._ui.list_plugins):
                QMessageBox.information(
                    self,
                    self.tr("Plugin already included"),
                    self.tr('Plugin "{0}" already in plugin list').format(plugin),
                )
                return

            if existing_item is not None:
                # Update the list-widget item with the updated plugin.
                existing_item.setText(plugin)

            else:
                # Add the plugin to the list widget.
                item = QListWidgetItem(plugin)
                item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                self._ui.list_plugins.addItem(item)

    def _on_del_plugin(self, checked=False):
        # Find the plugin(s) to be removed
        plugins: List[Tuple[int, str]] = qlistwidget_selections(self._ui.list_plugins)

        # Get user confirmation
        if len(plugins) == 1:
            msg = self.tr("Remove this plugin?") + f"\n\n{plugins[0][1]}"
        else:
            msg = self.tr("Remove these plugins?") + "\n" + "\n".join([p[1] for p in plugins])

        result = QMessageBox.question(self, self.tr("Remove plugins?"), msg)
        if result != QMessageBox.Yes:
            return  # User decided not to remove the plugin(s).

        # Delete all affected plugins.  Delete in decreasing index order,
        # so that indexes aren't shifted/invalidated by deleting lower-index
        # entries.
        for i in sorted([p[0] for p in plugins], reverse=True):
            self._ui.list_plugins.takeItem(i)

        # Now, no plugins should be selected
        self._ui.btn_del_plugin.setEnabled(False)

    def _on_verify_plugins(self, checked=False):
        if self._ui.list_plugins.count() == 0:
            QMessageBox.information(self, self.tr("No plugins"), self.tr("No plugins have been specified."))
            return

        # Create a new list of system paths with the updated plugin paths.

        paths = self._initial_sys_path[:]
        for p in self._initial_plugin_paths:
            try:
                paths.remove(p)
            except ValueError:
                pass

        for p in qlistwidget_to_list(self._ui.list_plugin_paths):
            if p not in paths:
                paths.append(p)

        sys.path = paths

        # Try to instantiate each plugin class, and verify that it is of the
        # correct type

        issues = []
        for p in qlistwidget_to_list(self._ui.list_plugins):
            try:
                inst = plugutils.instantiate(p)
                if not plugutils.is_plugin(inst):
                    msg = self.tr('Class "{0}" isn\'t a recognized plugin type')
                    issues.append(msg.format(p))

            except Exception as e:
                msg = self.tr('Can\'t instantiate plugin "{0}":  {1}')
                issues.append(msg.format(p, e))

        if issues:
            QMessageBox.warning(
                self,
                self.tr("Plugin issues found"),
                self.tr("Found these plugin issues:") + "\n\n" + "\n".join(issues),
            )

        else:
            QMessageBox.information(
                self,
                self.tr("No plugin issues found"),
                self.tr("No plugin issues found!"),
            )

        # Restore the original system path
        sys.path = self._initial_sys_path[:]

    # ========================================================================
    # OTHER OPERATIONS
    # ========================================================================

    def accept(self):
        # =======================================================================
        # Verify values

        # ==============================
        # New Spectra group-box

        aavg_x = int(self._ui.ledit_aavg_x.text())
        aavg_y = int(self._ui.ledit_aavg_y.text())

        if aavg_x % 2 != 1 or aavg_y % 2 != 1:
            QMessageBox.critical(
                self,
                self.tr("Default area-average values"),
                self.tr("Default area-average values must be odd."),
                QMessageBox.Ok,
            )
            return

        # ==============================
        # Plugin details group-box

        # TODO(donnie):  Validate that plugins can be loaded?  Users have the
        #     ability to do this already, but maybe we want to kick this off
        #     if they have added plugins or removed paths, have not validated,
        #     and then accepted this dialog.

        # =======================================================================
        # Apply values

        # ==============================
        # Error-Reporting group-box

        self._app_state.set_config(
            "general.online_bug_reporting",
            self._ui.ckbox_online_bug_reporting.isChecked(),
        )

        # ==============================
        # Visible-Light group-box

        self._app_state.set_config("general.red_wavelength_nm", int(self._ui.ledit_red_wavelength.text()))

        self._app_state.set_config("general.green_wavelength_nm", int(self._ui.ledit_green_wavelength.text()))

        self._app_state.set_config("general.blue_wavelength_nm", int(self._ui.ledit_blue_wavelength.text()))

        # ==============================
        # Raster Display group-box

        self._app_state.set_config(
            "raster.viewport_highlight_color",
            self._ui.ledit_viewport_highlight_color.text(),
        )

        cursor = self._ui.cbox_pixel_cursor_type.currentData()
        self._app_state.set_config("raster.pixel_cursor_type", cursor)

        self._app_state.set_config("raster.pixel_cursor_color", self._ui.ledit_pixel_cursor_color.text())

        # ==============================
        # New Spectra group-box

        self._app_state.set_config("spectra.default_area_avg_x", aavg_x)
        self._app_state.set_config("spectra.default_area_avg_y", aavg_y)

        mode = self._ui.cbox_default_avg_mode.currentData()
        self._app_state.set_config("spectra.default_area_avg_mode", mode)

        # ==============================
        # Plugin details group-box

        plugin_paths = qlistwidget_to_list(self._ui.list_plugin_paths)
        self._app_state.set_config("plugin_paths", plugin_paths)

        plugins = qlistwidget_to_list(self._ui.list_plugins)
        self._app_state.set_config("plugins", plugins)

        if plugin_paths != self._initial_plugin_paths or plugins != self._initial_plugins:
            QMessageBox.information(
                self,
                self.tr("Plugin changes detected"),
                self.tr(
                    "Changes to plugin configuration will not be\n"
                    + "applied until the next time WISER is started."
                ),
                QMessageBox.Ok,
            )

        # ==============================
        # All done!

        super().accept()
