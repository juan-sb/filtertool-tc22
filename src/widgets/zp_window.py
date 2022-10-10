from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QWidget
import sys
import numpy
from src.widgets.mplwidget import MplWidget

import sip

class ZPWindow(QWidget):
    def __init__(self, zeros = [], poles = [], title = '', *args, **kwargs):
        super(ZPWindow, self).__init__()
        self.layout = QVBoxLayout()
        self.zp_plot = MplWidget()
        self.zp_plot.focusWidget()
        self.layout.addWidget(self.zp_plot)
        self.setLayout(self.layout)

        self.setWindowTitle(f'Poles and zeros of function {title}')

        canvas = self.zp_plot.canvas
        canvas.ax.clear()
        canvas.ax.axhline(0, color="black", alpha=0.1)
        canvas.ax.axvline(0, color="black", alpha=0.1)
        (min, max) = self.getRelevantFrequencies(zeros, poles)
        (multiplier, prefix) = self.getMultiplierAndPrefix(max)
        canvas.ax.scatter(zeros.real/multiplier, zeros.imag/multiplier, marker='o')
        canvas.ax.scatter(poles.real/multiplier, poles.imag/multiplier, marker='x')
        canvas.ax.set_xlabel(f'$\sigma$ (${prefix}rad/s$)')
        canvas.ax.set_ylabel(f'$j\omega$ (${prefix}rad/s$)')
        canvas.ax.set_xlim(left=-max*1.2/multiplier, right=max*1.2/multiplier)
        canvas.ax.set_ylim(bottom=-max*1.2/multiplier, top=max*1.2/multiplier)
        canvas.ax.grid(True, which="both", linestyle=':')

        canvas.draw()

    
    def getRelevantFrequencies(self, zeros, poles):
        singularitiesNorm = numpy.append(numpy.abs(zeros), numpy.abs(poles))
        singularitiesNormWithoutZeros = singularitiesNorm[singularitiesNorm!=0]
        if(len(singularitiesNormWithoutZeros) == 0):
            return (1,1)
        return (numpy.min(singularitiesNormWithoutZeros), numpy.max(singularitiesNormWithoutZeros))
    
    def getMultiplierAndPrefix(self, val):
        multiplier = 1
        prefix = ''
        if(val < 1e-7):
            multiplier = 1e9
            prefix = 'n'
        elif(val < 1e-4):
            multiplier = 1e-6
            prefix = 'μ'
        elif(val < 1e-1):
            multiplier = 1e-3
            prefix = 'm'
        elif(val < 1e2):
            multiplier = 1
            prefix = ''
        elif(val < 1e5):
            multiplier = 1e3
            prefix = 'k'
        elif(val < 1e8):
            multiplier = 1e6
            prefix = 'M'
        elif(val > 1e11):
            multiplier = 1e9
            prefix = 'G'
        return (multiplier, prefix)