import json
import pprint

from typing import Optional

import bugsnag

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

import version

from .generated.bug_reporting_dialog_ui import Ui_BugReportingDialog


class BugReportingDialog(QDialog):
    '''
    This dialog asks the user if they would like to opt in to online bug
    reporting.
    '''

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self._ui = Ui_BugReportingDialog()
        self._ui.setupUi(self)


    def user_wants_bug_reporting(self) -> bool:
        '''
        Returns True if the user wants bug reporting, or False otherwise.
        '''
        return self._ui.rbtn_send_crashes.isChecked()


def initialize(config):

    auto_notify = config.get('general.online_bug_reporting')

    bugsnag.configure(
        api_key='29bf39226c3071461f3d0630c9ced4b6',
        app_version=version.VERSION,
        auto_notify=auto_notify,
    )

    bugsnag.before_notify(before_bugsnag_notify)


def set_enabled(enable):
    bugsnag.configure(auto_notify=enable)


def before_bugsnag_notify(event):
    payload = json.loads(event._payload())
    pprint.pprint(payload)
