# Python modules
import sys
from PyQt5 import QtWidgets

# Project modules
from src.ui.tf_window import Ui_tf_window

import ast
import scipy.signal as signal

class ResponseDialog(QtWidgets.QDialog, Ui_tf_window):
    def __init__(self, parent=None):
        super().__init__()
        self.setupUi(self)

        self.tf_title.textChanged.connect(self.enableResponseFunction)
        self.check_btn.clicked.connect(self.processResponseValues)


    
    def setResponseHelp(self):
        try:
            canvas = self.expr_plot.canvas
            canvas.ax.clear()
            canvas.ax.set_axis_off()
            canvas.ax.text("AYUDA \n")
            canvas.draw()
        except:
            pass

    def getResponseTitle(self):
        return self.tf_title.text()

    def getResponseExpression(self):
        return self.tf_raw.text()
    
    def getTimeDomain(self):
        return np.arange(self.minbox.value(), self.maxbox.value(), self.stepbox.value())

    def enableResponseFunction(self, txt):
        if txt != '':
            self.tf_raw.setEnabled(True)
    
    def validateResponse(self):
        try:
            ast.parse(self.tf_raw.text())
            if (self.minbox >= self.maxbox):
                return False
        except SyntaxError:
            return False
        return True

    def processResponseValues(self):
        if self.validateResponse():
            self.error_label.clear()
        else:
            self.error_label.setText("La expresión y/o los límites no son válidos")