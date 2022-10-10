from src.ui.mainwindow import *
import numpy as np
from numpy import linspace, logspace, cos, sin, heaviside, log10, floor, zeros, ones, pi, diff, unwrap
import sys
import os
from PyQt5 import QtGui, QtWidgets
from PyQt5.QtWidgets import QMainWindow, QFileDialog, QApplication, QWidget, QPushButton, QAction, QLineEdit, \
    QMessageBox, QRadioButton, QInputDialog
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import pyqtSlot

import scipy.signal as signal

from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

from src.DesignConfig import *
from src.Aproximations import *
from src.FilterStage import *
from src.FilterDesign import *


class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self, *args, **kwargs):
        QtWidgets.QMainWindow.__init__(self, *args, **kwargs)
        self.setupUi(self)

        # Init
        self.setFixedSize(1290, 700)
        self.stackedWidget.setCurrentIndex(0)
        self.tabPlots.setCurrentIndex(0)
        self.tabPlots_2.setCurrentIndex(0)
        self.updateType()
        self.updateAprox()

        # Variables
        self.filter_design = FilterDesign()
        self.page = 0
        self.max_page = 1
        self.designconfig = DesignConfig()
        self.num_plots = self.tabPlots.count()
        self.plot_layouts = [self.plotlayout_1, self.plotlayout_2, self.plotlayout_3, self.plotlayout_4,
                             self.plotlayout_5, self.plotlayout_6, self.plotlayout_7]
        self.num_plots2 = self.tabPlots_2.count()
        self.plot_layouts2 = [self.plotlayout2_1, self.plotlayout2_2, self.plotlayout2_3]
        self.filter_stages = dict()
        self.editingStage = False
        self.editingStageIndex = 0
        self.current_template = 0
        self.gain_remaining = 0

        # Signals/Slots
        # General
        self.btn_next.clicked.connect(self.nextPage)
        self.btn_prev.clicked.connect(self.prevPage)
        self.btn_save.clicked.connect(self.saveFile)
        self.btn_open.clicked.connect(self.openFile)
        self.btn_export.clicked.connect(self.exportFile)
        # Etapa 1
        self.btn_plot.clicked.connect(lambda: self.plotAll(self.designconfig))
        self.combo_tipo.currentIndexChanged.connect(self.updateType)
        self.combo_aprox.currentIndexChanged.connect(self.updateAprox)
        # Etapa 2
        self.btn_new_stage.clicked.connect(self.newStage)
        self.btn_edit_stage.clicked.connect(self.editStage)
        self.btn_plot_stage.clicked.connect(self.plotStage)
        self.btn_delete_stage.clicked.connect(self.deleteStage)
        self.stage_list.itemClicked.connect(self.updateStageView)

        # Plots
        self.plot_types = {'Atenuación': 0, 'Fase': 1, 'Retardo de Grupo': 2, 'Polos y Ceros': 3, 'Impulso': 4,
                           'Escalón': 5, 'Máximo Q': 6}
        self.plot_types2 = {'Polos y Ceros 2': 0, 'Respuesta Total': 1, 'Respuesta Etapa': 2}
        self.figure = [Figure() for x in range(self.num_plots)]
        self.canvas = [FigureCanvas(self.figure[x]) for x in range(self.num_plots)]
        self.axes = [self.figure[x].subplots() for x in range(self.num_plots)]
        for x, layout in enumerate(self.plot_layouts):
            layout.addWidget(NavigationToolbar(self.canvas[x], self))
            layout.addWidget(self.canvas[x])
            self.figure[x].tight_layout()
            self.axes[x].grid(True, which='both')
        self.axes[0].set_xscale('log')
        self.axes[1].set_xscale('log')
        self.axes[2].set_xscale('log')

        # Polos y ceros Etapa 2
        self.figure2 = [Figure() for x in range(self.num_plots2)]
        self.canvas2 = [FigureCanvas(self.figure2[x]) for x in range(self.num_plots2)]
        self.axes2 = [self.figure2[x].subplots() for x in range(self.num_plots2)]
        for x, layout in enumerate(self.plot_layouts2):
            layout.addWidget(NavigationToolbar(self.canvas2[x], self))
            layout.addWidget(self.canvas2[x])
            self.figure2[x].tight_layout()
            self.axes2[x].grid(True, which='both')

    def nextPage(self):
        self.page = min(self.page + 1, self.max_page)
        self.stackedWidget.setCurrentIndex(self.page)
        return

    def prevPage(self):
        self.page = max(self.page - 1, 0)
        self.stackedWidget.setCurrentIndex(self.page)
        return

    def set_A_Template(self):
        for i in range(self.combo_aprox.count()):
            self.combo_aprox.model().item(i).setEnabled(i != 5 and i != 6)

        if self.combo_aprox.currentIndex() == 5 or self.combo_aprox.currentIndex() == 6:
            self.combo_aprox.setCurrentIndex(0)

        self.GD_tau.setEnabled(False)
        self.GD_wrg.setEnabled(False)
        self.GD_gamma.setEnabled(False)

        type = self.combo_tipo.currentText()
        w_band = type == 'Pasa Banda' or type == 'Rechaza Banda'
        self.spin_Ap.setEnabled(True)
        self.spin_ripple.setEnabled(True)
        self.spin_Aa.setEnabled(True)
        self.spin_wp.setEnabled(True)
        self.spin_wa.setEnabled(True)
        self.plantilla_box.setCurrentIndex(0)
        return

    def set_GD_Template(self):
        for i in range(self.combo_aprox.count()):
            self.combo_aprox.model().item(i).setEnabled(i == 5 or i == 6)

        if self.combo_aprox.currentIndex() != 5 and self.combo_aprox.currentIndex() != 6:
            self.combo_aprox.setCurrentIndex(5)

        self.GD_tau.setEnabled(True)
        self.GD_wrg.setEnabled(True)
        self.GD_gamma.setEnabled(True)
        self.spin_Ap.setEnabled(False)
        self.spin_ripple.setEnabled(False)
        self.spin_Aa.setEnabled(False)
        self.spin_wp.setEnabled(False)
        self.spin_wp_2.setEnabled(False)
        self.spin_wa.setEnabled(False)
        self.spin_wa_2.setEnabled(False)
        self.plantilla_box.setCurrentIndex(1)
        return

    def saveFile(self):
        self.filter_design.setFilterStages(self.filter_stages)
        filename = QFileDialog.getSaveFileName(self, "Guardar Archivo", "filter.fdt", "Filter Design Tool (*.fdt)", "Filter Design Tool (*.fdt)")[0]
        if len(filename) > 0:
            try:
                self.filter_design.save(filename)
            except:
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Critical)
                msg.setWindowTitle("Error!")
                msg.setText("Error crítico intentando guardar el archivo!")
                msg.exec_()
        return

    def openFile(self):
        msgbox = QMessageBox(QMessageBox.Question, "Confirmación", "¿Seguro que quiere abrir un archivo?\nSe perderá el progreso actual.")
        msgbox.addButton(QMessageBox.Yes)
        msgbox.addButton(QMessageBox.No)
        msgbox.setDefaultButton(QMessageBox.No)
        reply = msgbox.exec()
        if reply == QMessageBox.Yes:
            filename = QFileDialog.getOpenFileName(self, "Abrir Archivo", "", "Filter Design Tool (*.fdt)", "Filter Design Tool (*.fdt)")[0]
            if filename:
                #try:
                self.filter_design.open(filename)
                self.designconfig = self.filter_design.dc
                self.combo_tipo.setCurrentIndex(self.designconfig.type)
                self.combo_aprox.setCurrentIndex(self.designconfig.aprox)
                self.spin_denorm.setValue(self.designconfig.denorm)
                self.spin_minord.setValue(self.designconfig.minord)
                self.spin_maxord.setValue(self.designconfig.maxord)
                self.spin_qmax.setValue(self.designconfig.qmax)
                self.spin_Ap.setValue(self.designconfig.Ap)
                self.spin_ripple.setValue(self.designconfig.ripple)
                self.spin_Aa.setValue(self.designconfig.Aa)
                self.spin_wp.setValue(self.designconfig.wp)
                self.spin_wa.setValue(self.designconfig.wa)
                self.spin_wp_2.setValue(self.designconfig.wp2)
                self.spin_wa_2.setValue(self.designconfig.wa2)
                self.GD_tau.setValue(self.designconfig.tau)
                self.GD_wrg.setValue(self.designconfig.wrg)
                self.GD_gamma.setValue(self.designconfig.gamma)

                self.plotAll(self.designconfig)

                self.gain_remaining = self.filter_design.gain

                self.filter_stages = self.filter_design.stages
                self.stage_list.clear()
                for label, stage in self.filter_stages.items():
                    self.gain_remaining -= stage.gain
                    self.stage_list.addItem(label)
                    self.combo_polo1.model().item(stage.pole1 + 1).setEnabled(False)
                    self.combo_polo2.model().item(stage.pole1 + 1).setEnabled(False)
                    self.combo_polo1.model().item(stage.pole2 + 1).setEnabled(False)
                    self.combo_polo2.model().item(stage.pole2 + 1).setEnabled(False)
                    self.combo_cero1.model().item(stage.zero1 + 1).setEnabled(False)
                    self.combo_cero2.model().item(stage.zero1 + 1).setEnabled(False)
                    self.combo_cero1.model().item(stage.zero2 + 1).setEnabled(False)
                    self.combo_cero2.model().item(stage.zero2 + 1).setEnabled(False)
                self.label_gain_total.setText('Restante: {:.3f} dB'.format(self.gain_remaining))
                '''except:
                    msg = QMessageBox()
                    msg.setIcon(QMessageBox.Critical)
                    msg.setWindowTitle("Error!")
                    msg.setText("Error crítico intentando abrir el archivo!")
                    msg.exec_()
                return True'''
        return False

    def exportFile(self):
        self.updateDesignConfig()
        self.filter_design.setDesignConfig(self.designconfig)
        self.filter_design.setFilterStages(self.filter_stages)
        filename = QFileDialog.getSaveFileName(self, "Exportar Archivo", "filter.pdf", "PDF (*.pdf)", "PDF (*.pdf)")[0]
        if len(filename) > 0:
            try:
                self.filter_design.export(filename)
            except:
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Critical)
                msg.setWindowTitle("Error!")
                msg.setText("Error crítico intentando exportar el archivo!")
                msg.exec_()
        return

    def updateType(self):
        type = self.combo_tipo.currentText()
        w_band = type == 'Pasa Banda' or type == 'Rechaza Banda'
        g_delay = type == 'Retardo de Grupo'
        if not g_delay:
            self.set_A_Template()
            self.label_wp_2.setEnabled(w_band)
            self.spin_wp_2.setEnabled(w_band)
            self.label_wa_2.setEnabled(w_band)
            self.spin_wa_2.setEnabled(w_band)

            self.label_wp.setText('Frecuencia ωp+' if w_band else 'Frecuencia ωp')
            self.label_wa.setText('Frecuencia ωa+' if w_band else 'Frecuencia ωa')
        else:
            self.set_GD_Template()

        return

    def updateAprox(self):
        return

    def updateDesignConfig(self):
        self.designconfig.setType(self.combo_tipo.currentText())
        self.designconfig.setAprox(self.combo_aprox.currentText())
        self.designconfig.denorm = self.spin_denorm.value()
        self.designconfig.minord = self.spin_minord.value()
        self.designconfig.maxord = self.spin_maxord.value()
        self.designconfig.qmax = self.spin_qmax.value()
        self.designconfig.Ap = self.spin_Ap.value()
        self.designconfig.ripple = self.spin_ripple.value()
        self.designconfig.Aa = self.spin_Aa.value()
        self.designconfig.wp = self.spin_wp.value()
        self.designconfig.wa = self.spin_wa.value()
        self.designconfig.wp2 = self.spin_wp_2.value()
        self.designconfig.wa2 = self.spin_wa_2.value()
        self.designconfig.tau = self.GD_tau.value()
        self.designconfig.wrg = self.GD_wrg.value()
        self.designconfig.gamma = self.GD_gamma.value()

    def plotAll(self, designconfig):
        type = self.combo_tipo.currentText()
        aprox = self.combo_aprox.currentText()
        denorm = self.spin_denorm.value()
        minord = self.spin_minord.value()
        maxord = self.spin_maxord.value()
        qmax = self.spin_qmax.value()
        Ap = self.spin_Ap.value()
        ripple = self.spin_ripple.value()
        Aa = self.spin_Aa.value()
        wp = self.spin_wp.value()
        wa = self.spin_wa.value()
        wp2 = self.spin_wp_2.value()
        wa2 = self.spin_wa_2.value()
        tau = self.GD_tau.value()
        wrg = self.GD_wrg.value()
        gamma = self.GD_gamma.value()

        # Mensaje advertencia: Parametros invalidos
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Warning)
        msg.setWindowTitle("Advertencia!")
        warning_msg = ''
        w_band = type == 'Pasa Banda' or type == 'Rechaza Banda'
        g_delay = type == 'Retardo de Grupo'
        if not g_delay:
            if (wp == 0 or wa == 0) or \
                (w_band and (wp2 == 0 or wa2 == 0 or wp2 >= wp or wa2 >= wa)) or \
                (type == 'Pasa Banda' and (wp >= wa or wp2 <= wa2)) or \
                (type == 'Rechaza Banda' and (wp <= wa or wp2 >= wa2)) or \
                (type == 'Pasa Bajos' and wa <= wp) or \
                (type == 'Pasa Altos' and wp <= wa):
                    warning_msg += "Los parametros para ωp y ωa no son válidos.\n"
            if ripple <= 0:
                warning_msg += "El valor máximo de ripple debe ser mayor a 0.\n"
            if Aa <= Ap:
                warning_msg += "El valor de Aa debe superar al valor de Ap.\n"
            if minord > maxord:
                warning_msg += "El orden mínimo debe ser inferior al orden máximo.\n"
        if len(warning_msg) > 0:
            msg.setText(warning_msg)
            msg.exec_()
        else:
            for x, ax in enumerate(self.axes):
                ax.clear()
                ax.grid()
            for x, ax in enumerate(self.axes2):
                ax.clear()
                ax.grid()

            self.filter_design.gain = ripple - Ap
            self.gain_remaining = self.filter_design.gain
            self.label_gain_total.setText('Restante: {:.3f} dB'.format(self.gain_remaining))

            self.designconfig.setParameters(type, aprox, denorm, minord, maxord, qmax, Ap, ripple, Aa, wp, wa, wp2, wa2, tau, wrg, gamma)

            if (self.check_plantilla.isChecked() and not g_delay) or (self.check_plantilla_GD.isChecked() and g_delay):
                self.plotTemplate(type, Ap, ripple, Aa, wp, wa, wp2, wa2, tau, gamma, wrg)

            # Calcular aproximación here
            # Falta el tema del orden min max, max q, etc
            if aprox == 'Butterworth':
                z, p, k = Butterworth(designconfig)
            elif aprox == 'Chebyshev I':
                z, p, k = ChebyshevI(designconfig)
            elif aprox == 'Chebyshev II':
                z, p, k = ChebyshevII(designconfig)
            elif aprox == 'Bessel':
                z, p, k = Bessel(designconfig)
            elif aprox == 'Cauer':
                z, p, k = Cauer(designconfig)
            elif aprox == 'Gauss':
                z, p, k = Gauss(designconfig)
            elif aprox == 'Legendre':
                z, p, k = Legendre(designconfig)

            try:
                if Ap <= 0:
                    k *= 10**(self.filter_design.gain / 20)
                filter_system = signal.ZerosPolesGain(z, p, k)
                # Atenuacion y Fase
                if type == 'Pasa Bajos' or type == 'Pasa Altos':
                    lowerfreq = wp / 10
                    higherfreq = wa * 10
                elif type == 'Pasa Banda' or type == 'Rechaza Banda':
                    lowerfreq = min(wa, wp, wa2, wp2) / 10
                    higherfreq = max(wa, wp, wa2, wp2) * 10
                elif aprox == 'Bessel' or aprox == 'Gauss':
                    lowerfreq = wrg/10
                    higherfreq = wrg * 10

                x = np.logspace((log10(lowerfreq)), (log10(higherfreq)), num=1000)
                Gain = signal.bode(filter_system, x)
                Attenuation = signal.bode(signal.ZerosPolesGain(p, z, 1 / k), x)
                self.getPlotAxes('Atenuación').semilogx(Attenuation[0], Attenuation[1], 'k')
                self.getPlotAxes('Fase').semilogx(Gain[0], Gain[2], 'k')
                self.getPlotAxes('Respuesta Total').semilogx(Gain[0], Gain[1], 'k')

                # Retardo de Grupo
                w, h = signal.freqs_zpk(z, p, k, x)
                gd = -np.diff(np.unwrap(np.angle(h))) / np.diff(w)
                self.getPlotAxes('Retardo de Grupo').semilogx(w[1:], gd, 'k')

                # Polos y Ceros
                self.stage_list.clear()
                self.combo_polo1.clear()
                self.combo_polo2.clear()
                self.combo_cero1.clear()
                self.combo_cero2.clear()
                self.combo_polo1.addItem('-')
                self.combo_polo2.addItem('-')
                self.combo_cero1.addItem('-')
                self.combo_cero2.addItem('-')
                self.plotPolesAndZeros(z, p)

                # Respuestas temporales

                t, y = signal.impulse(filter_system, N = 1000)
                self.getPlotAxes('Impulso').plot(t, y, 'k')
                t, y = signal.step(filter_system, N = 1000)
                self.getPlotAxes('Escalón').plot(t, y, 'k')

                # Máximo Q
                self.plotQFactor(z, p)

                # Guardar Datos
                self.filter_design.setDesignConfig(self.designconfig)
                self.filter_design.setPolesAndZeros(p, z)

            except:
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Critical)
                msg.setWindowTitle("Error!")
                msg.setText("Error crítico intentando generar gráficos!")
                msg.exec_()

            for x, canv in enumerate(self.canvas):
                self.figure[x].tight_layout()
                canv.draw()
            for x, canv in enumerate(self.canvas2):
                self.figure2[x].tight_layout()
                canv.draw()

        pass  # TO-DO

    def plotTemplate(self, type, Ap, ripple, Aa, wp, wa, wp2, wa2, tau, gamma, wrg):
        if type == 'Pasa Bajos':
            x = [wp / 10, wp, wp]
            y = [Ap, Ap, Aa + 10]
            if Ap <= 0:
                yR = [Ap-ripple, Ap-ripple]
            else:
                yR = [ripple, ripple]
            self.axes[0].semilogx(x, y, 'b--', color='#28658a', linewidth=2)
            self.axes[0].semilogx(x[:-1], yR, 'b--', color='#28658a', linewidth=2)
            self.axes[0].fill_between(x, y, np.max(y), facecolor="none", edgecolor='#539ecd', hatch='X', linewidth=0)
            x = [wa, wa, wa * 10]
            y = [Ap - 10, Aa, Aa]
            self.axes[0].semilogx(x, y, 'b--', color='#28658a', linewidth=2)
            self.axes[0].fill_between(x, y, np.min(y), facecolor="none", edgecolor='#539ecd', hatch='X', linewidth=0)
        elif type == 'Pasa Altos':
            x = [wa / 10, wa, wa]
            y = [Aa, Aa, Ap - 10]
            self.axes[0].semilogx(x, y, 'b--', color='#28658a', linewidth=2)
            self.axes[0].fill_between(x, y, np.min(y), facecolor="none", edgecolor='#539ecd', hatch='X', linewidth=0)
            x = [wp, wp, wp * 10]
            y = [Aa + 10, Ap, Ap]
            if Ap <= 0:
                yR = [Ap-ripple, Ap-ripple]
            else:
                yR = [ripple, ripple]
            self.axes[0].semilogx(x[1:], yR, 'b--', color='#28658a', linewidth=2)
            self.axes[0].semilogx(x, y, 'b--', color='#28658a', linewidth=2)
            self.axes[0].fill_between(x, y, np.max(y), facecolor="none", edgecolor='#539ecd', hatch='X', linewidth=0)
        elif type == 'Pasa Banda':
            x = [wa2 / 10, wa2, wa2]
            y = [Aa, Aa, Ap - 10]
            self.axes[0].semilogx(x, y, 'b--', color='#28658a', linewidth=2)
            self.axes[0].fill_between(x, y, np.min(y), facecolor="none", edgecolor='#539ecd', hatch='X', linewidth=0)
            x = [wp2, wp2, wp, wp]
            y = [Aa + 10, Ap, Ap, Aa + 10]
            if Ap <= 0:
                yR = [Ap-ripple, Ap-ripple]
            else:
                yR = [ripple, ripple]
            self.axes[0].semilogx(x[1:-1], yR, 'b--', color='#28658a', linewidth=2)
            self.axes[0].semilogx(x, y, 'b--', color='#28658a', linewidth=2)
            self.axes[0].fill_between(x, y, np.max(y), facecolor="none", edgecolor='#539ecd', hatch='X', linewidth=0)
            x = [wa, wa, wa * 10]
            y = [Ap - 10, Aa, Aa]
            self.axes[0].semilogx(x, y, 'b--', color='#28658a', linewidth=2)
            self.axes[0].fill_between(x, y, np.min(y), facecolor="none", edgecolor='#539ecd', hatch='X', linewidth=0)
        elif type == 'Rechaza Banda':
            x = [wp2 / 10, wp2, wp2]
            y = [Ap, Ap, Aa + 10]
            if Ap <= 0:
                yR = [Ap-ripple, Ap-ripple]
            else:
                yR = [ripple, ripple]
            self.axes[0].semilogx(x[:-1], yR, 'b--', color='#28658a', linewidth=2)
            self.axes[0].semilogx(x, y, 'b--', color='#28658a', linewidth=2)
            self.axes[0].fill_between(x, y, np.max(y), facecolor="none", edgecolor='#539ecd', hatch='X', linewidth=0)
            x = [wa2, wa2, wa, wa]
            y = [Ap - 10, Aa, Aa, Ap - 10]
            self.axes[0].semilogx(x, y, 'b--', color='#28658a', linewidth=2)
            self.axes[0].fill_between(x, y, np.min(y), facecolor="none", edgecolor='#539ecd', hatch='X', linewidth=0)
            x = [wp, wp, wp * 10]
            y = [Aa + 10, Ap, Ap]
            if Ap <= 0:
                yR = [Ap-ripple, Ap-ripple]
            else:
                yR = [ripple, ripple]
            self.axes[0].semilogx(x[1:], yR, 'b--', color='#28658a', linewidth=2)
            self.axes[0].semilogx(x, y, 'b--', color='#28658a', linewidth=2)
            self.axes[0].fill_between(x, y, np.max(y), facecolor="none", edgecolor='#539ecd', hatch='X', linewidth=0)
        elif type == 'Retardo de Grupo':
            x = [wrg/10, wrg, wrg]
            y = [tau - (tau * gamma/100), tau - (tau * gamma/100), 0]
            self.getPlotAxes('Retardo de Grupo').semilogx(x, y, 'b--', color='#28658a', linewidth=2)
            self.getPlotAxes('Retardo de Grupo').fill_between(x, y, np.min(y), facecolor="none", edgecolor='#539ecd', hatch='X', linewidth=0)

        return

    def plotPolesAndZeros(self, z, p):
        self.getPlotAxes('Polos y Ceros').axhline(linewidth=1, color='k')
        self.getPlotAxes('Polos y Ceros').axvline(linewidth=1, color='k')

        poles_labels = dict()
        zeros_labels = dict()
        for i, pole in enumerate(p):
            self.combo_polo1.addItem('Polo ' + str(i + 1))
            self.combo_polo2.addItem('Polo ' + str(i + 1))
            self.getPlotAxes('Polos y Ceros').plot(pole.real, pole.imag, 'rx', markersize=10)
            xy = (pole.real, pole.imag)
            poles_labels[xy] = 'Polo ' + str(i + 1) if xy not in poles_labels else poles_labels[xy] + ', ' + str(i + 1)
        for i, zero in enumerate(z):
            self.combo_cero1.addItem('Cero ' + str(i + 1))
            self.combo_cero2.addItem('Cero ' + str(i + 1))
            self.getPlotAxes('Polos y Ceros').plot(zero.real, zero.imag, 'bo', markersize=10, fillstyle='none')
            xy = (zero.real, zero.imag)
            zeros_labels[xy] = 'Cero ' + str(i + 1) if xy not in zeros_labels else zeros_labels[xy] + ', ' + str(i + 1)

        for polexy in poles_labels:
            self.getPlotAxes('Polos y Ceros').annotate(poles_labels[polexy], polexy, textcoords="offset points", xytext=(0, 10), ha='center')
        for zeroxy in zeros_labels:
            self.getPlotAxes('Polos y Ceros').annotate(zeros_labels[zeroxy], zeroxy, textcoords="offset points", xytext = (0, 10), ha = 'center')

        return

    def plotQFactor(self, z, p):
        self.getPlotAxes('Máximo Q').axhline(linewidth=1, color='k')
        self.getPlotAxes('Máximo Q').axvline(linewidth=1, color='k')
        self.getPlotAxes('Polos y Ceros 2').axhline(linewidth=1, color='k')
        self.getPlotAxes('Polos y Ceros 2').axvline(linewidth=1, color='k')

        poles_labels = dict()
        zeros_labels = dict()
        for i, pole in enumerate(p):
            self.getPlotAxes('Máximo Q').plot(pole.real, pole.imag, 'rx', markersize=10)
            self.getPlotAxes('Polos y Ceros 2').plot(pole.real, pole.imag, 'rx', markersize=10)
            Q = -abs(pole)/(2*pole.real) if pole.real < 0 else float('inf')
            xy = (pole.real, pole.imag)
            poles_labels[xy] = 'Q = {:.3f} (Polo {}'.format(Q, i+1) if xy not in poles_labels else poles_labels[xy]+', '+ str(i+1)
        for i, zero in enumerate(z):
            self.getPlotAxes('Máximo Q').plot(zero.real, zero.imag, 'bo', markersize=10, fillstyle='none')
            self.getPlotAxes('Polos y Ceros 2').plot(zero.real, zero.imag, 'bo', markersize=10, fillstyle='none')
            xy = (zero.real, zero.imag)
            zeros_labels[xy] = 'Cero ' + str(i + 1) if xy not in zeros_labels else zeros_labels[xy] + ', ' + str(i + 1)

        for polexy in poles_labels:
            self.getPlotAxes('Máximo Q').annotate(poles_labels[polexy]+')', polexy, textcoords="offset points", xytext=(0, 10), ha='center')
            self.getPlotAxes('Polos y Ceros 2').annotate(poles_labels[polexy]+')', polexy, textcoords="offset points", xytext=(0, 10), ha='center')
        for zeroxy in zeros_labels:
            self.getPlotAxes('Máximo Q').annotate(zeros_labels[zeroxy], zeroxy, textcoords="offset points", xytext=(0, 10), ha='center')
            self.getPlotAxes('Polos y Ceros 2').annotate(zeros_labels[zeroxy], zeroxy, textcoords="offset points", xytext=(0, 10), ha='center')

        return

    def newStage(self):
        new_stage = self.getStageParameters()
        if new_stage is not None:
            self.stage_list.addItem(new_stage.getLabel())
            self.filter_stages[new_stage.getLabel()] = new_stage
            return True
        return False

    def getStageParameters(self):
        pole1 = self.combo_polo1.currentText()
        pole2 = self.combo_polo2.currentText()
        pole1 = int(pole1[5:]) - 1 if pole1.startswith('Polo') else -1
        pole2 = int(pole2[5:]) - 1 if pole2.startswith('Polo') else -1

        zero1 = self.combo_cero1.currentText()
        zero2 = self.combo_cero2.currentText()
        zero1 = int(zero1[5:]) - 1 if zero1.startswith('Cero') else -1
        zero2 = int(zero2[5:]) - 1 if zero2.startswith('Cero') else -1

        gain = self.spin_gain.value()

        msg = QMessageBox()
        msg.setIcon(QMessageBox.Warning)
        msg.setWindowTitle("Advertencia!")
        if pole1 == pole2 and pole1 >= 0:
            msg.setText("Ambos polos deben ser distintos.")
            msg.exec_()
        elif zero1 == zero2 and zero1 >= 0:
            msg.setText("Ambos ceros deben ser distintos.")
            msg.exec_()
        elif pole1 == pole2 and zero1 == zero2 and pole1 < 0 and zero1 < 0:
            msg.setText("Se debe seleccionar al menos un polo o un cero.")
            msg.exec_()
        elif pole1 >= 0 and pole2 >= 0 and not np.isclose(self.filter_design.poles[pole1].real, self.filter_design.poles[pole2].real):
            print(self.filter_design.poles[pole1].real, self.filter_design.poles[pole2].real)
            msg.setText("Los polos deben ser complejos conjugados.")
            msg.exec_()
        else:
            pole = None
            if pole1 >= 0:
                self.combo_polo1.model().item(pole1 + 1).setEnabled(False)
                self.combo_polo2.model().item(pole1 + 1).setEnabled(False)
                pole = self.filter_design.poles[pole1]
            if pole2 >= 0:
                self.combo_polo1.model().item(pole2 + 1).setEnabled(False)
                self.combo_polo2.model().item(pole2 + 1).setEnabled(False)
                pole = self.filter_design.poles[pole2]
            if zero1 >= 0:
                self.combo_cero1.model().item(zero1 + 1).setEnabled(False)
                self.combo_cero2.model().item(zero1 + 1).setEnabled(False)
            if zero2 >= 0:
                self.combo_cero1.model().item(zero2 + 1).setEnabled(False)
                self.combo_cero2.model().item(zero2 + 1).setEnabled(False)
            self.combo_polo1.setCurrentIndex(0)
            self.combo_polo2.setCurrentIndex(0)
            self.combo_cero1.setCurrentIndex(0)
            self.combo_cero2.setCurrentIndex(0)

            self.gain_remaining -= gain
            self.label_gain_total.setText('Restante: {:.3f} dB'.format(self.gain_remaining))

            Q = -abs(pole) / (2 * pole.real) if pole.real < 0 else float('inf')
            return FilterStage(pole1, pole2, zero1, zero2, gain, Q)

        return None

    def editStage(self):
        if not self.editingStage:
            selection = self.stage_list.selectedItems()
            if len(selection) > 0:
                self.btn_edit_stage.setText('Hecho')
                self.editingStage = True
                self.editingStageIndex = self.stage_list.row(selection[0])
                self.btn_new_stage.setEnabled(False)
                self.btn_plot_stage.setEnabled(False)
                self.btn_delete_stage.setEnabled(False)
                self.stage_list.setEnabled(False)
                index = self.editingStageIndex
                current_stage = self.filter_stages[self.stage_list.item(index).text()]
                pole1 = current_stage.pole1
                pole2 = current_stage.pole2
                zero1 = current_stage.zero1
                zero2 = current_stage.zero2
                gain = current_stage.gain
                self.spin_gain.setValue(gain)
                self.gain_remaining += gain
                if pole1 >= 0:
                    self.combo_polo1.model().item(pole1 + 1).setEnabled(True)
                    self.combo_polo2.model().item(pole1 + 1).setEnabled(True)
                    self.combo_polo1.setCurrentIndex(pole1 + 1)
                if pole2 >= 0:
                    self.combo_polo1.model().item(pole2 + 1).setEnabled(True)
                    self.combo_polo2.model().item(pole2 + 1).setEnabled(True)
                    self.combo_polo2.setCurrentIndex(pole2 + 1)
                if zero1 >= 0:
                    self.combo_cero1.model().item(zero1 + 1).setEnabled(True)
                    self.combo_cero2.model().item(zero1 + 1).setEnabled(True)
                    self.combo_cero1.setCurrentIndex(zero1 + 1)
                if zero2 >= 0:
                    self.combo_cero1.model().item(zero2 + 1).setEnabled(True)
                    self.combo_cero2.model().item(zero2 + 1).setEnabled(True)
                    self.combo_cero2.setCurrentIndex(zero2 + 1)
                return True
            else:
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Warning)
                msg.setWindowTitle("Advertencia!")
                msg.setText("Debe seleccionar una etapa para editar.")
                msg.exec_()
                return False
        else:
            new_stage = self.getStageParameters()
            if new_stage is not None:
                self.btn_edit_stage.setText('Editar')
                self.editingStage = False
                self.btn_new_stage.setEnabled(True)
                self.btn_plot_stage.setEnabled(True)
                self.btn_delete_stage.setEnabled(True)
                self.stage_list.setEnabled(True)

                index = self.editingStageIndex
                del self.filter_stages[self.stage_list.item(index).text()]
                self.stage_list.item(index).setText(new_stage.getLabel())
                self.filter_stages[new_stage.getLabel()] = new_stage
                return True
            else:
                return False

    def plotStage(self):
        selection = self.stage_list.selectedItems()
        if len(selection) > 0:
            index = self.stage_list.row(selection[0])
            stage = self.filter_stages[self.stage_list.item(index).text()]
            poles = []
            zeros = []
            if stage.pole1 >= 0:
                poles.append(self.filter_design.poles[stage.pole1])
            if stage.pole2 >= 0:
                poles.append(self.filter_design.poles[stage.pole2])

            if stage.zero1 >= 0:
                zeros.append(self.filter_design.zeros[stage.zero1])
            if stage.zero2 >= 0:
                zeros.append(self.filter_design.zeros[stage.zero2])

            Gain = signal.bode(signal.ZerosPolesGain(zeros, poles, 10**(stage.gain/20)))

            self.getPlotAxes('Respuesta Etapa').clear()
            self.getPlotAxes('Respuesta Etapa').grid()
            self.getPlotAxes('Respuesta Etapa').semilogx(Gain[0], Gain[1], 'k')
            self.figure2[2].tight_layout()
            self.canvas2[2].draw()
            return True
        else:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setWindowTitle("Advertencia!")
            msg.setText("Debe seleccionar una etapa para graficar.")
            msg.exec_()
            return False

    def deleteStage(self):
        selection = self.stage_list.selectedItems()
        if len(selection) > 0:
            index = self.stage_list.row(selection[0])
            old_stage = self.filter_stages[self.stage_list.takeItem(index).text()]
            pole1 = old_stage.pole1
            pole2 = old_stage.pole2
            zero1 = old_stage.zero1
            zero2 = old_stage.zero2
            gain = old_stage.gain
            self.gain_remaining += gain
            self.label_gain_total.setText('Restante: {:.3f} dB'.format(self.gain_remaining))
            if pole1 >= 0:
                self.combo_polo1.model().item(pole1 + 1).setEnabled(True)
                self.combo_polo2.model().item(pole1 + 1).setEnabled(True)
            if pole2 >= 0:
                self.combo_polo1.model().item(pole2 + 1).setEnabled(True)
                self.combo_polo2.model().item(pole2 + 1).setEnabled(True)
            if zero1 >= 0:
                self.combo_cero1.model().item(zero1 + 1).setEnabled(True)
                self.combo_cero2.model().item(zero1 + 1).setEnabled(True)
            if zero2 >= 0:
                self.combo_cero1.model().item(zero2 + 1).setEnabled(True)
                self.combo_cero2.model().item(zero2 + 1).setEnabled(True)
            del old_stage
            return True
        else:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setWindowTitle("Advertencia!")
            msg.setText("Debe seleccionar una etapa para borrar.")
            msg.exec_()
            return False

    def updateStageView(self):
        pass  # TO-DO

    def getPlotAxes(self, type):
        if type in self.plot_types:
            return self.axes[self.plot_types[type]]
        elif type in self.plot_types2:
            return self.axes2[self.plot_types2[type]]
