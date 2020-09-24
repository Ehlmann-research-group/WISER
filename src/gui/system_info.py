import platform, pkg_resources, sys

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

import version

from .generated.system_info_ui import Ui_SystemInfoDialog


class SystemInfoDialog(QDialog):
    '''
    A dialog that shows various important system details in a plain text box,
    so that the user can see what libraries and versions the app is using.
    There is also a "copy to clipboard" button to simplify bug reporting.
    '''

    def __init__(self, parent=None):
        super().__init__(parent=parent)

        # Set up the UI state
        self._ui = Ui_SystemInfoDialog()
        self._ui.setupUi(self)

        # Populate the system information window
        self._ui.te_system_info.setPlainText(self._generate_sysinfo())

        # Hook up the "copy to clipboard" button
        self._ui.btn_copy_to_clipboard.clicked.connect(self._on_copy_to_clipboard)


    def _generate_sysinfo(self):
        # Application info

        info  = 'WISER:  Workbench for Imaging Spectroscopy Exploration and Research\n'
        info += 'Version:  {0}\n'.format(version.VERSION)
        info += 'Release Date:  {0}\n'.format(version.RELEASE_DATE)
        info += '\n'

        # OS platform and Python platform info

        info += 'Platform:  {0}\n'.format(platform.platform())
        info += 'Python Version:  {0}\n'.format(platform.python_version())

        python_build = platform.python_build()
        info += 'Python Interpreter:  {0} {1} {2}\n'.format(
            platform.python_implementation(), python_build[0], python_build[1])

        info += '\n'

        # Installed packages and versions

        installed_packages = [(d.project_name, d.version)
                              for d in pkg_resources.working_set]

        installed_packages.sort(key=lambda t : t[0].upper())

        info += 'Installed Packages:\n'
        for p in installed_packages:
            info += f'    {p[0]} {p[1]}\n'

        return info


    def _on_copy_to_clipboard(self):
        text = self._ui.te_system_info.toPlainText()
        clipboard = QGuiApplication.clipboard()
        clipboard.setText(text)


# Some test code so we can check out if it works!
if __name__ == '__main__':
    app = QApplication(sys.argv)
    dlg = SystemInfoDialog()
    dlg.show()
    sys.exit(app.exec_())
