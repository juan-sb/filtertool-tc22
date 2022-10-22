from PyQt5 import QtWidgets, QtGui, QtCore

class CheckableComboBox(QtWidgets.QComboBox):
    def __init__(self, arg):
        super(CheckableComboBox, self).__init__()
        self.view().pressed.connect(self.handleItemPressed)
        self.setModel(QtGui.QStandardItemModel(self))
        self.checkedIndexes = []

    def handleItemPressed(self, index):
        item = self.model().itemFromIndex(index)
        actual_human_index = self.findText(item.text())

        if item.checkState() == QtCore.Qt.Checked:
            item.setCheckState(QtCore.Qt.Unchecked)
            self.checkedIndexes.pop(self.checkedIndexes.index(actual_human_index))
        else:
            item.setCheckState(QtCore.Qt.Checked)
            self.checkedIndexes.append(actual_human_index)
    
    def currentIndexes(self):
        return self.checkedIndexes
    def setCurrentIndexes(self, indexes):
        self.checkedIndexes = indexes