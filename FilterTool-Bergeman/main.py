from src.mainwindow import *

if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    app.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
    window = MainWindow()
    window.show()
    app.exec_()