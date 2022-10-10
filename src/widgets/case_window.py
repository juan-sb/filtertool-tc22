import sys
from PyQt5 import QtWidgets, QtCore

from src.ui.case_window import Ui_case_dialog

class CaseDialog(QtWidgets.QDialog, Ui_case_dialog):
    def __init__(self, parent=None):
        super().__init__()
        self.setupUi(self)
        self.color = '#505050'
        self.case_pickcol_btn.clicked.connect(self.openColorPicker)

        self.COLOR_LIST = [
            ["#FF1F5B","#00CD6C","#009ADE","#AF58BA","#FFC61E","#F28522"],
            ["#E4F1F7","#C5E1EF","#9EC9E2","#6CB0D6","#3C93C2","#226E9C","#0D4A70"],
            ["#E1F2E3","#CDE5D2","#9CCEA7","#6CBA7D","#40AD5A","#228B3B","#06592A"],
            ["#F9D8E6","#F2ACCA","#ED85B0","#E95694","#E32977","#C40F5B","#8F003B"],
            ["#B7E6A5","#7CCBA2","#46AEA0","#089099","#00718B","#045275","#003147"],
            ["#FCE1A4","#FABF7B","#F08F6E","#E05C5C","#D12959","#AB1866","#6E005F"],
            ["#009392","#39B185","#9CCB86","#E9E29C","#EEB479","#E88471","#CF597E"],
            ["#045275","#089099","#7CCBA2","#FCDE9C","#F0746E","#DC3977","#7C1D6F"]
        ]

    def populate(self, ds={}):
        if(ds is None):
            return
        self.case_first_cb.clear()
        self.case_last_cb.clear()
        self.case_xdata_cb.clear()
        self.case_ydata_cb.clear()

        self.case_xdata_cb.addItems(ds.fields)
        self.case_ydata_cb.addItems(ds.fields)
        if(self.case_ydata_cb.count() > 1):
            self.case_ydata_cb.setCurrentIndex(1)
        for x in range(len(ds.data)):
            self.case_first_cb.addItem(str(x))
            self.case_last_cb.addItem(str(x))
        self.case_last_cb.setCurrentIndex(self.case_last_cb.count() - 1)

    def openColorPicker(self):
        dialog = QtWidgets.QColorDialog(self)
        dialog.setCurrentColor(QtCore.Qt.red)
        dialog.setOption(QtWidgets.QColorDialog.ShowAlphaChannel)
        dialog.open()
        dialog.currentColorChanged.connect(self.updateSingleColor)
        
    def updateSingleColor(self, color):
        self.case_pickcol_btn.setStyleSheet(f'background-color: {color.name()}')
        self.color = color.name()