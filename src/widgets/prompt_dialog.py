# Python modules
import sys
from PyQt5 import QtWidgets

from src.ui.prompt import Ui_PromptDialog

class PromptDialog(QtWidgets.QDialog, Ui_PromptDialog):
    def __init__(self, parent=None):
        super().__init__()
        self.setupUi(self)
    
    def setErrorMsg(self, err_msg):
      self.prompt_msg.setText(err_msg)