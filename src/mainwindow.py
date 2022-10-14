# PyQt5 modules
from PyQt5.QtWidgets import QMainWindow, QListWidgetItem, QColorDialog, QFileDialog, QDialog
from PyQt5.QtCore import Qt

# Project modules
from src.ui.mainwindow import Ui_MainWindow
from src.package.Dataset import Dataset
import src.package.Filter as Filter
from src.package.Filter import AnalogFilter
from src.widgets.tf_dialog import TFDialog
from src.widgets.case_window import CaseDialog
from src.widgets.zp_window import ZPWindow
from src.widgets.response_dialog import ResponseDialog

from scipy.signal import savgol_filter
import scipy.signal as signal
from scipy.interpolate import splrep, splev, splprep
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import random
from pyparsing.exceptions import ParseSyntaxException

MARKER_STYLES = { 'None': '', 'Point': '.',  'Pixel': ',',  'Circle': 'o',  'Triangle down': 'v',  'Triangle up': '^',  'Triangle left': '<',  'Triangle right': '>',  'Tri down': '1',  'Tri up': '2',  'Tri left': '3',  'Tri right': '4',  'Octagon': '8',  'Square': 's',  'Pentagon': 'p',  'Plus (filled)': 'P',  'Star': '*',  'Hexagon': 'h',  'Hexagon alt.': 'H',  'Plus': '+',  'x': 'x',  'x (filled)': 'X',  'Diamond': 'D',  'Diamond (thin)': 'd',  'Vline': '|',  'Hline': '_' }
LINE_STYLES = { 'None': '', 'Solid': '-', 'Dashed': '--', 'Dash-dot': '-.', 'Dotted': ':' }

class MainWindow(QMainWindow, Ui_MainWindow):

    def __init__(self):
        super(MainWindow, self).__init__()
        self.setupUi(self)
        self.droppedFiles = []
        self.datasets = []
        self.datalines = []
        self.selected_dataset_widget = {}
        self.selected_dataline_widget = {}
        self.selected_dataset_data = {}
        self.selected_dataline_data = {}
        self.zpWindow = type('ZPWindow', (), {})()
        
        self.import_file_btn.clicked.connect(self.importFiles)

        self.populateSelectedDatasetDetails({}, {})
        self.populateSelectedDatalineDetails({}, {})
        
        self.dataset_list.currentItemChanged.connect(self.populateSelectedDatasetDetails)
        self.ds_title_edit.textEdited.connect(self.updateSelectedDataset)
        self.ds_addline_btn.clicked.connect(self.createDataline)
        self.ds_remove_btn.clicked.connect(self.removeSelectedDataset)
        self.ds_poleszeros_btn.clicked.connect(self.showZPWindow)

        self.dataline_list.currentItemChanged.connect(self.populateSelectedDatalineDetails)
        self.dl_name_edit.textEdited.connect(self.updateSelectedDataline)
        self.dl_render_cb.activated.connect(self.updateSelectedDataline)
        self.dl_transform_cb.activated.connect(self.updateSelectedDataline)
        self.dl_xdata_cb.activated.connect(self.updateSelectedDataline)
        self.dl_xscale_sb.valueChanged.connect(self.updateSelectedDataline)
        self.dl_xoffset_sb.valueChanged.connect(self.updateSelectedDataline)
        self.dl_ydata_cb.activated.connect(self.updateSelectedDataline)
        self.dl_yscale_sb.valueChanged.connect(self.updateSelectedDataline)
        self.dl_yoffset_sb.valueChanged.connect(self.updateSelectedDataline)
        self.dl_color_edit.textEdited.connect(self.updateSelectedDataline)
        self.dl_style_cb.activated.connect(self.updateSelectedDataline)
        self.dl_linewidth_sb.valueChanged.connect(self.updateSelectedDataline)
        self.dl_marker_cb.activated.connect(self.updateSelectedDataline)
        self.dl_markersize_sb.valueChanged.connect(self.updateSelectedDataline)
        self.dl_remove_btn.clicked.connect(self.removeSelectedDataline)
        self.dl_savgol_wlen.valueChanged.connect(self.updateSelectedDataline)
        self.dl_savgol_ord.valueChanged.connect(self.updateSelectedDataline)

        self.dl_color_pickerbtn.clicked.connect(self.openColorPicker)

        self.respd = ResponseDialog()
        self.resp_btn.clicked.connect(self.openResponseDialog)
        self.respd.accepted.connect(self.resolveResponseDialog)

        self.tfd = TFDialog()
        self.function_btn.clicked.connect(self.openTFDialog)
        self.tfd.accepted.connect(self.resolveTFDialog)

        self.csd = CaseDialog()
        self.ds_caseadd_btn.clicked.connect(self.openCaseDialog)
        self.csd.accepted.connect(self.resolveCSDialog)
        
        self.plt_labelsize_sb.valueChanged.connect(self.updatePlots)
        self.plt_legendsize_sb.valueChanged.connect(self.updatePlots)
        self.plt_ticksize_sb.valueChanged.connect(self.updatePlots)
        self.plt_titlesize_sb.valueChanged.connect(self.updatePlots)
        self.plt_autoscale.clicked.connect(self.autoscalePlots)
        self.plt_legendpos.activated.connect(self.updatePlots)
        self.plt_grid.stateChanged.connect(self.updatePlots)
        self.tabbing_plots.currentChanged.connect(self.updatePlots)
        
        self.plots_canvases = [
            [ self.plot_1 ],
            [ self.plot_2_1, self.plot_2_2 ],
            [ self.plot_3 ],
            [ self.plot_4_1, self.plot_4_2 ],
            [ self.plot_5 ],
        ]

        self.filter = AnalogFilter()
        self.new_filter_btn.clicked.connect(self.resolveFilterDialog)
        self.chg_filter_btn.clicked.connect(self.updateSelectedFilter)
        self.tipo_box.currentIndexChanged.connect(self.updateFilterParametersAvailable)
        # self.aprox_box.currentIndexChanged.connect(self.updateSelectedFilter)
        # self.N_min_box.valueChanged.connect(self.updateSelectedFilter)
        # self.N_max_box.valueChanged.connect(self.updateSelectedFilter)
        # self.denorm_box.valueChanged.connect(self.updateSelectedFilter)
        # self.gp_box.valueChanged.connect(self.updateSelectedFilter)
        # self.ga_box.valueChanged.connect(self.updateSelectedFilter)
        # self.fp_box.valueChanged.connect(self.updateSelectedFilter)
        # self.fa_box.valueChanged.connect(self.updateSelectedFilter)
        # self.define_with_box.currentIndexChanged.connect(self.updateFilterParametersAvailable)
        self.updateFilterParametersAvailable()

    def dragEnterEvent(self, event):
        if(event.mimeData().hasUrls()):
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        self.statusbar.showMessage('Loading files')
        files = [u.toLocalFile() for u in event.mimeData().urls()]
        self.processFiles(files)

    def importFiles(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        files, _ = QFileDialog.getOpenFileNames(self,"Select files", "","All Files (*);;CSV files (*.csv);;SPICE output files (*.raw)", options=options)
        self.processFiles(files)

    def processFiles(self, filenamearray):
        for f in filenamearray:
            try:
                ds = Dataset(filepath=f)
                dataset_items_origin = [
                    self.dataset_list.item(x).data(Qt.UserRole).origin
                    for x in range(self.dataset_list.count())
                ]
                if(ds.origin not in dataset_items_origin):
                    self.droppedFiles.append(ds.origin)
                    qlwt = QListWidgetItem()
                    qlwt.setData(Qt.UserRole, ds)
                    qlwt.setText(ds.title)
                    self.dataset_list.addItem(qlwt)
                    self.datasets.append(ds)
                    self.datalines.append([])
            except(ValueError):
                print('Wrong file config')
        self.dataset_list.setCurrentRow(self.dataset_list.count() - 1)
        self.statusbar.clearMessage()

    def openResponseDialog(self):
        self.respd.open()
        self.respd.tf_title.setFocus()

    def resolveResponseDialog(self):
        if not(self.respd.validateResponse()):
            return
        if(self.respd.getResponseExpression() == 'step'):
            t, ss = signal.step(self.selected_dataset_data.tf.getND())
            title = self.respd.getResponseTitle()
            time_title = title + "_timebase"
            ans_title = title + "_ans"
            self.selected_dataset_data.data[0][time_title] = t
            self.selected_dataset_data.data[0][ans_title] = ss
            self.selected_dataset_data.fields.append(time_title)
            self.selected_dataset_data.fields.append(ans_title)

            self.updateSelectedDataset()
            self.updateSelectedDataline()
            return
        t = np.linspace(0, 30e-6, 100)
        x = eval(self.respd.getResponseExpression())
        response = signal.lsim(self.selected_dataset_data.tf , U = x , T = t)
        title = self.respd.getResponseTitle()
        time_title = title + "_timebase"
        ans_title = title + "_ans"

        self.selected_dataset_data.data[0][time_title] = t
        self.selected_dataset_data.data[0][title] = x
        self.selected_dataset_data.data[0][ans_title] = response[1]
        self.selected_dataset_data.fields.append(time_title)
        self.selected_dataset_data.fields.append(title)
        self.selected_dataset_data.fields.append(ans_title)

        self.updateSelectedDataset()
        self.updateSelectedDataline()

    def openTFDialog(self):
        self.tfd.open()
        self.tfd.tf_title.setFocus()

    def resolveTFDialog(self):
        if not self.tfd.validateTF():
            return
        ds = Dataset(filepath='', origin=self.tfd.tf, title=self.tfd.getTFTitle())
        qlwt = QListWidgetItem()
        qlwt.setData(Qt.UserRole, ds)
        qlwt.setText(ds.title)
        self.dataset_list.addItem(qlwt)
        self.datasets.append(ds)
        self.datalines.append([])
        self.dataset_list.setCurrentRow(self.dataset_list.count() - 1)
    
    def resolveFilterDialog(self):
        if self.tipo_box.currentIndex() in [Filter.BAND_PASS, Filter.BAND_REJECT]:
            wa = [2 * np.pi * self.fa_min_box.value(), 2 * np.pi * self.fa_max_box.value()]
            wp = [2 * np.pi * self.fp_min_box.value(), 2 * np.pi * self.fp_max_box.value()]
        else:
            wa = 2 * np.pi * self.fa_box.value()
            wp = 2 * np.pi * self.fp_box.value()

        params =         {
            "name": self.filtername_box.text(),
            "filter_type": self.tipo_box.currentIndex(),
            "approx_type": self.aprox_box.currentIndex(),
            "define_with": self.define_with_box.currentIndex(),
            "N_min": self.N_min_box.value(),
            "N_max": self.N_max_box.value(),
            "Q_max": self.Q_max_box.value(),
            "gain": self.gain_box.value(),
            "denorm": self.denorm_box.value(),
            "ga_dB": self.ga_box.value(),
            "gp_dB": self.gp_box.value(),
            "wa": wa,
            "wp": wp,
            "w0": 2 * np.pi * self.f0_box.value(),
            "bw": [2 * np.pi * self.bw_min_box.value(), 2 * np.pi * self.bw_max_box.value()],
            "gamma": self.tol_box.value(),
            "tau0": self.tau0_box.value(),
            "wrg": 2 * np.pi * self.frg_box.value(),
        }
        
        newFilter = AnalogFilter(**params) #CREO EL OBJETO FILTRO
        valid, msg = newFilter.validate()
        if not valid:
            # mostrar el error
            return
        self.filter = newFilter

        ds = Dataset(filepath='', origin=newFilter, title=self.filtername_box.text())
        qlwt = QListWidgetItem()
        qlwt.setData(Qt.UserRole, ds)
        qlwt.setText(ds.title)
        self.dataset_list.addItem(qlwt)
        self.datasets.append(ds)
        self.datalines.append([])
        self.dataset_list.setCurrentRow(self.dataset_list.count() - 1)
        self.updateFilterPlots()
    
    def updateSelectedFilter(self):
        if self.tipo_box.currentIndex() == Filter.BAND_PASS or self.tipo_box.currentIndex() == Filter.BAND_REJECT:
            wa = [2 * np.pi * self.fa_min_box.value(), 2 * np.pi * self.fa_max_box.value()]
            wp = [2 * np.pi * self.fp_min_box.value(), 2 * np.pi * self.fp_max_box.value()]
        else:
            wa = 2 * np.pi * self.fa_box.value()
            wp = 2 * np.pi * self.fp_box.value()

        params =         {
            "name": self.filtername_box.text(),
            "filter_type": self.tipo_box.currentIndex(),
            "approx_type": self.aprox_box.currentIndex(),
            "define_with": self.define_with_box.currentIndex(),
            "N_min": self.N_min_box.value(),
            "N_max": self.N_max_box.value(),
            "Q_max": self.Q_max_box.value(),
            "gain": self.gain_box.value(),
            "denorm": self.denorm_box.value(),
            "ga_dB": self.ga_box.value(),
            "gp_dB": self.gp_box.value(),
            "wa": wa,
            "wp": wp,
            "w0": 2 * np.pi * self.f0_box.value(),
            "bw": [2 * np.pi * self.bw_min_box.value(), 2 * np.pi * self.bw_max_box.value()],
            "gamma": self.tol_box.value(),
            "tau0": self.tau0_box.value(),
            "wrg": 2 * np.pi * self.frg_box.value(),
        }
        
        newFilter = AnalogFilter(**params) #CREO EL OBJETO FILTRO
        valid, msg = newFilter.validate()
        if not valid:
            # mostrar el error
            return
        self.filter = newFilter

        ds = Dataset(filepath='', origin=newFilter, title=self.filtername_box.text())
        
        self.selected_dataset_widget.setText(self.filtername_box.text())
        self.selected_dataset_data = ds
        self.populateSelectedDatasetDetails(self.selected_dataset_widget, None)
        self.updateFilterPlots()

    def updateFilterParametersAvailable(self):
        if self.tipo_box.currentIndex() == Filter.LOW_PASS or self.tipo_box.currentIndex() == Filter.HIGH_PASS:
            for i in range(Filter.LEGENDRE + 1):
                self.aprox_box.model().item(i).setEnabled(True)
            for i in range(Filter.BESSEL, Filter.GAUSS + 1):
                self.aprox_box.model().item(i).setEnabled(False)
            if not self.aprox_box.model().item(self.aprox_box.currentIndex()).isEnabled():
                self.aprox_box.setCurrentIndex(Filter.BUTTERWORTH)
            self.define_with_box.setEnabled(False)
            self.gp_box.setEnabled(True)
            self.ga_box.setEnabled(True)
            self.fp_box.setEnabled(True)
            self.fa_box.setEnabled(True)
            self.fa_min_box.setEnabled(False)
            self.fa_max_box.setEnabled(False)
            self.fp_min_box.setEnabled(False)
            self.fp_max_box.setEnabled(False)
            self.f0_box.setEnabled(False)
            self.bw_min_box.setEnabled(False)
            self.bw_max_box.setEnabled(False)
            self.tau0_box.setEnabled(False)
            self.frg_box.setEnabled(False)
            self.tol_box.setEnabled(False)

        if self.tipo_box.currentIndex() == Filter.BAND_PASS or self.tipo_box.currentIndex() == Filter.BAND_REJECT:
            for i in range(Filter.LEGENDRE + 1):
                self.aprox_box.model().item(i).setEnabled(True)
            for i in range(Filter.BESSEL, Filter.GAUSS + 1):
                self.aprox_box.model().item(i).setEnabled(False)
            if not self.aprox_box.model().item(self.aprox_box.currentIndex()).isEnabled():
                self.aprox_box.setCurrentIndex(Filter.BUTTERWORTH)
            self.define_with_box.setEnabled(True)
            self.fp_box.setEnabled(False)
            self.fa_box.setEnabled(False)
            self.tau0_box.setEnabled(False)
            self.frg_box.setEnabled(False)
            self.tol_box.setEnabled(False)

            if self.define_with_box.currentIndex() == Filter.TEMPLATE_FREQS:
                self.fa_min_box.setEnabled(True)
                self.fa_max_box.setEnabled(True)
                self.fp_min_box.setEnabled(True)
                self.fp_max_box.setEnabled(True)
                self.f0_box.setEnabled(False)
                self.bw_min_box.setEnabled(False)
                self.bw_max_box.setEnabled(False)
            if self.define_with_box.currentIndex() == Filter.F0_BW:
                self.fa_min_box.setEnabled(False)
                self.fa_max_box.setEnabled(False)
                self.fp_min_box.setEnabled(False)
                self.fp_max_box.setEnabled(False)
                self.f0_box.setEnabled(True)
                self.bw_min_box.setEnabled(True)
                self.bw_max_box.setEnabled(True)
        
        if self.tipo_box.currentIndex() == Filter.GROUP_DELAY:
            for i in range(Filter.LEGENDRE + 1):
                self.aprox_box.model().item(i).setEnabled(False)
            for i in range(Filter.BESSEL, Filter.GAUSS + 1):
                self.aprox_box.model().item(i).setEnabled(True)
            if not self.aprox_box.model().item(self.aprox_box.currentIndex()).isEnabled():
                self.aprox_box.setCurrentIndex(Filter.BESSEL)
            self.gp_box.setEnabled(False)
            self.ga_box.setEnabled(False)
            self.fp_box.setEnabled(False)
            self.fa_box.setEnabled(False)
            self.fa_min_box.setEnabled(False)
            self.fa_max_box.setEnabled(False)
            self.fp_min_box.setEnabled(False)
            self.fp_max_box.setEnabled(False)
            self.f0_box.setEnabled(False)
            self.bw_min_box.setEnabled(False)
            self.bw_max_box.setEnabled(False)
            self.tau0_box.setEnabled(True)
            self.frg_box.setEnabled(True)
            self.tol_box.setEnabled(True)

    
    def getRelevantFrequencies(self, zeros, poles):
        singularitiesNorm = np.append(np.abs(zeros), np.abs(poles))
        singularitiesNormWithoutZeros = singularitiesNorm[singularitiesNorm!=0]
        if(len(singularitiesNormWithoutZeros) == 0):
            return (1,1)
        return (np.min(singularitiesNormWithoutZeros), np.max(singularitiesNormWithoutZeros))
    
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

    def updateFilterPlots(self):
        attcanvas = self.fplot_att.canvas
        gaincanvas = self.fplot_gain.canvas
        magcanvas = self.fplot_mag.canvas
        phasecanvas = self.fplot_phase.canvas
        groupdelaycanvas = self.fplot_gd.canvas
        pzcanvas = self.fplot_pz.canvas
        stepcanvas = self.fplot_step.canvas
        impulsecanvas = self.fplot_impulse.canvas

        attcanvas.ax.clear()
        attcanvas.ax.grid(True, which="both", linestyle=':')
        gaincanvas.ax.clear()
        gaincanvas.ax.grid(True, which="both", linestyle=':')
        magcanvas.ax.clear()
        magcanvas.ax.grid(True, which="both", linestyle=':')
        magcanvas.ax.set_xscale('log')
        phasecanvas.ax.clear()
        phasecanvas.ax.grid(True, which="both", linestyle=':')
        phasecanvas.ax.set_xscale('log')
        phasecanvas.ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins='auto', steps=[1.8,2.25,4.5,9]))
        groupdelaycanvas.ax.clear()
        groupdelaycanvas.ax.grid(True, which="both", linestyle=':')
        groupdelaycanvas.ax.set_xscale('log')
        pzcanvas.ax.clear()
        pzcanvas.ax.grid(True, which="both", linestyle=':')
        stepcanvas.ax.clear()
        stepcanvas.ax.grid(True, which="both", linestyle=':')
        impulsecanvas.ax.clear()
        impulsecanvas.ax.grid(True, which="both", linestyle=':')

        filtds = Dataset(filepath='', origin=self.filter, title=self.filtername_box.text())
        
        tstep, stepres = signal.step(filtds.tf.getND())
        timp, impres = signal.impulse(filtds.tf.getND())
        
        f = np.array(filtds.data[0]['f'])
        g = 20 * np.log10(np.abs(np.array(filtds.data[0]['g'])))
        ph = np.array(filtds.data[0]['ph'])
        gd = np.array(filtds.data[0]['gd'])
        z, p = self.filter.tf.getZP()

        attline, = attcanvas.ax.plot(f, -g)
        gainline, = gaincanvas.ax.plot(f, g)
        magline, = magcanvas.ax.plot(f, g)
        phaseline, = phasecanvas.ax.plot(f, ph)
        gdline, = groupdelaycanvas.ax.plot(f, gd)
        stepline, = stepcanvas.ax.plot(tstep, stepres)
        impulseline, = impulsecanvas.ax.plot(timp, impres)
        


        if self.filter.filter_type == Filter.LOW_PASS:
            fp = self.filter.wp/(2*np.pi)
            fa = self.filter.wa/(2*np.pi)
            gp = self.filter.gp_dB
            ga = self.filter.ga_dB
            bw = fa - fp
            x = [fp - bw/3, fp]
            y = [-gp, -gp]
            attcanvas.ax.fill_between(x, y, -ga*1.5, facecolor='#ffcccb', edgecolor='#ef9a9a', hatch='\\', linewidth=0)
            x = [fa, fa + bw/3]
            y = [-ga, -ga]
            attcanvas.ax.fill_between(x, y, 0, facecolor='#ffcccb', edgecolor='#ef9a9a', hatch='\\', linewidth=0)
            attcanvas.ax.set_xlim([fp - bw/3, fa + bw/3])
            attcanvas.ax.set_ylim([0, -ga*1.5])

        elif self.filter.filter_type == Filter.HIGH_PASS:
            fp = self.filter.wp/(2*np.pi)
            fa = self.filter.wa/(2*np.pi)
            gp = self.filter.gp_dB
            ga = self.filter.ga_dB
            bw = fp - fa
            x = [fp, fp + bw/3]
            y = [-gp, -gp]
            attcanvas.ax.fill_between(x, y, -ga*1.5, facecolor='#ffcccb', edgecolor='#ef9a9a', hatch='\\', linewidth=0)
            x = [fa - bw/3, fa]
            y = [-ga, -ga]
            attcanvas.ax.fill_between(x, y, 0, facecolor='#ffcccb', edgecolor='#ef9a9a', hatch='\\', linewidth=0)
            attcanvas.ax.set_xlim([fa - bw/3, fp + bw/3])
            attcanvas.ax.set_ylim([0, -ga*1.5])
        elif self.filter.filter_type == Filter.BAND_PASS:
            fp = [w/(2*np.pi) for w in self.filter.wp]
            fa = [w/(2*np.pi) for w in self.filter.wa]
            gp = self.filter.gp_dB
            ga = self.filter.ga_dB
            print(fp, fa)
            bw = fa[0] - fa[1]
            # x = [fp, fp + bw/3]
            # y = [-gp, -gp]
            # attcanvas.ax.fill_between(x, y, -ga*1.5, facecolor='#ffcccb', edgecolor='#ef9a9a', hatch='\\', linewidth=0)
            # x = [fa - bw/3, fa]
            # y = [-ga, -ga]
            # attcanvas.ax.fill_between(x, y, 0, facecolor='#ffcccb', edgecolor='#ef9a9a', hatch='\\', linewidth=0)
            attcanvas.ax.set_xlim([fa[0] - bw/3, fa[1] + bw/3])
            attcanvas.ax.set_ylim([0, -ga*1.5])        # elif type == 'Pasa Banda':
        #     x = [wa2 / 10, wa2, wa2]
        #     y = [Aa, Aa, Ap - 10]
        #     plt.semilogx(x, y, 'b--', color='#28658a', linewidth=2)
        #     plt.fill_between(x, y, np.min(y), facecolor="none", edgecolor='#539ecd', hatch='X', linewidth=0)
        #     x = [wp2, wp2, wp, wp]
        #     y = [Aa + 10, Ap, Ap, Aa + 10]
        #     if Ap <= 0:
        #         yR = [Ap - ripple, Ap - ripple]
        #     else:
        #         yR = [ripple, ripple]
        #     plt.semilogx(x[1:-1], yR, 'b--', color='#28658a', linewidth=2)
        #     plt.semilogx(x, y, 'b--', color='#28658a', linewidth=2)
        #     plt.fill_between(x, y, np.max(y), facecolor="none", edgecolor='#539ecd', hatch='X', linewidth=0)
        #     x = [wa, wa, wa * 10]
        #     y = [Ap - 10, Aa, Aa]
        #     plt.semilogx(x, y, 'b--', color='#28658a', linewidth=2)
        #     plt.fill_between(x, y, np.min(y), facecolor="none", edgecolor='#539ecd', hatch='X', linewidth=0)
        # elif type == 'Rechaza Banda':
        #     x = [wp2 / 10, wp2, wp2]
        #     y = [Ap, Ap, Aa + 10]
        #     if Ap <= 0:
        #         yR = [Ap - ripple, Ap - ripple]
        #     else:
        #         yR = [ripple, ripple]
        #     plt.semilogx(x[:-1], yR, 'b--', color='#28658a', linewidth=2)
        #     plt.semilogx(x, y, 'b--', color='#28658a', linewidth=2)
        #     plt.fill_between(x, y, np.max(y), facecolor="none", edgecolor='#539ecd', hatch='X', linewidth=0)
        #     x = [wa2, wa2, wa, wa]
        #     y = [Ap - 10, Aa, Aa, Ap - 10]
        #     plt.semilogx(x, y, 'b--', color='#28658a', linewidth=2)
        #     plt.fill_between(x, y, np.min(y), facecolor="none", edgecolor='#539ecd', hatch='X', linewidth=0)
        #     x = [wp, wp, wp * 10]
        #     y = [Aa + 10, Ap, Ap]
        #     if Ap <= 0:
        #         yR = [Ap - ripple, Ap - ripple]
        #     else:
        #         yR = [ripple, ripple]
        #     plt.semilogx(x[1:], yR, 'b--', color='#28658a', linewidth=2)
        #     plt.semilogx(x, y, 'b--', color='#28658a', linewidth=2)
        #     plt.fill_between(x, y, np.max(y), facecolor="none", edgecolor='#539ecd', hatch='X', linewidth=0)


        pzcanvas.ax.axis('equal')
        pzcanvas.ax.axhline(0, color="black", alpha=0.1)
        pzcanvas.ax.axvline(0, color="black", alpha=0.1)
        (min, max) = self.getRelevantFrequencies(z, p)
        (multiplier, prefix) = self.getMultiplierAndPrefix(max)
        pzcanvas.ax.scatter(z.real/multiplier, z.imag/multiplier, marker='o')
        pzcanvas.ax.scatter(p.real/multiplier, p.imag/multiplier, marker='x')
        pzcanvas.ax.set_xlabel(f'$\sigma$ (${prefix}rad/s$)')
        pzcanvas.ax.set_ylabel(f'$j\omega$ (${prefix}rad/s$)')
        pzcanvas.ax.set_xlim(left=-max*1.2/multiplier, right=max*1.2/multiplier)
        pzcanvas.ax.set_ylim(bottom=-max*1.2/multiplier, top=max*1.2/multiplier)

        attcanvas.draw()
        gaincanvas.draw()
        magcanvas.draw()
        phasecanvas.draw()
        groupdelaycanvas.draw()
        pzcanvas.draw()
        stepcanvas.draw()
        impulsecanvas.draw()

    def removeSelectedDataset(self, event):
        selected_row = self.dataset_list.currentRow()
        first_dataline_index = 0
        last_dataline_index = len(self.datalines[0])
        for x in range(self.dataset_list.count()):
            if(x == 0):
                first_dataline_index = 0
            else:    
                first_dataline_index += len(self.datalines[x - 1])
            if(x == selected_row):
                break
        last_dataline_index = first_dataline_index + len(self.datalines[selected_row])

        for x in range(first_dataline_index, last_dataline_index):
            self.dataline_list.takeItem(first_dataline_index)
        self.dataset_list.takeItem(selected_row)
        self.datalines.pop(selected_row)
        self.updatePlots()

    def getInternalDataIndexes(self, datalineRow):
        i = self.dataline_list.currentRow()
        for x in range(self.dataset_list.count()):
            ds = self.dataset_list.item(x).data(Qt.UserRole)
            if(i >= len(self.datalines[x])):
                i = i - len(self.datalines[x])
            else:
                return (x, i)
        return (x, i)

    def removeSelectedDataline(self, event):
        selected_row = self.dataline_list.currentRow()
        try:
            dsi, dli = self.getInternalDataIndexes(selected_row)
            del self.datalines[dsi][dli]
            self.dataline_list.takeItem(selected_row).data(Qt.UserRole)
            del self.dataset_list.item(dsi).data(Qt.UserRole).datalines[dli]
            if(self.dataline_list.currentRow() == -1):
                self.dataline_list.setCurrentRow(self.dataline_list.count() - 1)
        except AttributeError:
            pass
        self.updatePlots()

    def openCaseDialog(self):
        self.csd.open()
        self.csd.populate(ds = self.selected_dataset_data)
        # self.tfd.tf_title.setFocus()

    def resolveCSDialog(self):
        first_case = int(self.csd.case_first_cb.currentIndex())
        last_case = int(self.csd.case_last_cb.currentIndex())
        casenum = len(self.selected_dataset_data.data)
        dli = 0
        for x in range(self.dataset_list.count()):
            ds = self.dataset_list.item(x).data(Qt.UserRole)
            dli = dli + len(self.datalines[x])
            if(ds.origin == self.selected_dataset_data.origin):
                break
        color_iter = 0
        for case in range(first_case, min(last_case + 1, casenum)):
            dl = self.selected_dataset_data.create_dataline(case)
            qlwt = QListWidgetItem()
            dl.plots = self.csd.case_render_cb.currentIndex()
            dl.transform = self.csd.case_transform_cb.currentIndex()
            dl.xsource = self.csd.case_xdata_cb.currentText()
            dl.xscale = self.csd.case_xscale_sb.value()
            dl.xoffset = self.csd.case_xoffset_sb.value()
            dl.ysource = self.csd.case_ydata_cb.currentText()
            dl.yscale = self.csd.case_yscale_sb.value()
            dl.yoffset = self.csd.case_yoffset_sb.value()
            if(self.csd.case_randomcol_rb.isChecked()):
                dl.color = ["#"+''.join([random.choice('0123456789ABCDEF') for i in range(6)])][0]
            elif(self.csd.case_presetcol_rb.isChecked()):
                colorpalette_i = self.csd.case_palettecol_cb.currentIndex()
                colorpalette = self.csd.COLOR_LIST[colorpalette_i]
                dl.color = colorpalette[color_iter]
                color_iter += 1
                if(color_iter == len(colorpalette)):
                    color_iter = 0
            else:
                dl.color = self.csd.color
            dl.linestyle = self.csd.case_style_cb.currentText()
            dl.linewidth = self.csd.case_linewidth_sb.value()
            dl.markerstyle = self.csd.case_marker_cb.currentText()
            dl.markersize = self.csd.case_markersize_sb.value()
            if(self.csd.case_inforname_rb.isChecked()):
                dstitle = self.selected_dataset_data.title 
                dscases = self.selected_dataset_data.casenames
                if(case < len(dscases)):
                    dl.name = dstitle + ' ' + dscases[case]
            qlwt.setText(dl.name)
            dl.name = dl.name if self.csd.case_addlegend_cb.isChecked() else '_' + dl.name
            qlwt.setData(Qt.UserRole, dl)
            self.dataline_list.insertItem(dli, qlwt)
            self.datalines[self.dataset_list.currentRow()].append(dl)
        self.updatePlots()

    def populateSelectedDatasetDetails(self, listitemwidget, qlistwidget):
        if(not listitemwidget):
            self.setDatasetControlsStatus(False)
            self.ds_title_edit.setText('')
            self.ds_casenum_lb.setText('0')
            self.ds_info_lb.setText('')
            return
        self.setDatasetControlsStatus(True)
        self.ds_title_edit.setText(listitemwidget.text())
        self.selected_dataset_widget = listitemwidget
        self.selected_dataset_data = listitemwidget.data(Qt.UserRole)
        isTF = self.selected_dataset_data.type in ['TF', 'filter']
        self.ds_poleszeros_btn.setEnabled(isTF)
        self.resp_btn.setEnabled(isTF)
        self.ds_casenum_lb.setText(str(len(self.selected_dataset_data.data)))
        self.ds_caseadd_btn.setEnabled(len(self.selected_dataset_data.data) > 1)
        self.ds_info_lb.setText(self.selected_dataset_data.miscinfo)

        #relleno las cajas del filtro
        if(self.selected_dataset_data.type == 'filter'):
            self.filtername_box.setText(self.selected_dataset_data.title)
            self.tipo_box.setCurrentIndex(self.selected_dataset_data.origin.filter_type)
            self.aprox_box.setCurrentIndex(self.selected_dataset_data.origin.approx_type)
            self.gain_box.setValue(self.selected_dataset_data.origin.gain)
            self.ga_box.setValue(self.selected_dataset_data.origin.ga_dB)
            self.gp_box.setValue(self.selected_dataset_data.origin.gp_dB)
            self.N_label.setText(str(self.selected_dataset_data.origin.N))
            self.N_min_box.setValue(self.selected_dataset_data.origin.N_min)
            self.N_max_box.setValue(self.selected_dataset_data.origin.N_max)
            self.Q_max_box.setValue(self.selected_dataset_data.origin.Q_max)

            if self.selected_dataset_data.origin.filter_type in [Filter.BAND_PASS, Filter.BAND_REJECT]:
                self.fp_box.setValue(0)
                self.fa_box.setValue(0)
                self.fa_min_box.setValue(self.selected_dataset_data.origin.wa[0] / (2 * np.pi))
                self.fa_max_box.setValue(self.selected_dataset_data.origin.wa[1] / (2 * np.pi))
                self.fp_min_box.setValue(self.selected_dataset_data.origin.wp[0] / (2 * np.pi))
                self.fp_max_box.setValue(self.selected_dataset_data.origin.wp[1] / (2 * np.pi)) 
            elif self.selected_dataset_data.origin.filter_type in [Filter.LOW_PASS, Filter.HIGH_PASS]:
                self.fp_box.setValue(self.selected_dataset_data.origin.wp / (2 * np.pi))
                self.fa_box.setValue(self.selected_dataset_data.origin.wa / (2 * np.pi))
                self.fa_min_box.setValue(0)
                self.fa_max_box.setValue(0)
                self.fp_min_box.setValue(0)
                self.fp_max_box.setValue(0)
            else:
                self.fp_box.setValue(0)
                self.fa_box.setValue(0)
                self.fa_min_box.setValue(0)
                self.fa_max_box.setValue(0)
                self.fp_min_box.setValue(0)
                self.fp_max_box.setValue(0)
                
        
            self.f0_box.setValue(self.selected_dataset_data.origin.w0 / (2 * np.pi))
            self.bw_min_box.setValue(self.selected_dataset_data.origin.bw[0] / (2 * np.pi))
            self.bw_max_box.setValue(self.selected_dataset_data.origin.bw[1] / (2 * np.pi))
            self.tol_box.setValue(self.selected_dataset_data.origin.gamma)
            self.tau0_box.setValue(self.selected_dataset_data.origin.tau0)
            self.frg_box.setValue(self.selected_dataset_data.origin.wrg / (2* np.pi))

    def updateSelectedDataset(self):
        new_title = self.ds_title_edit.text()
        self.selected_dataset_widget.setText(new_title)
        self.selected_dataset_data.title = new_title
        self.populateSelectedDatasetDetails(self.selected_dataset_widget, None)

    def createDataline(self):
        if(not self.selected_dataset_data):
            return
        dl = self.selected_dataset_data.create_dataline()
        qlwt = QListWidgetItem()
        qlwt.setData(Qt.UserRole, dl)
        qlwt.setText(dl.name)
        dli = 0
        for x in range(self.dataset_list.count()):
            ds = self.dataset_list.item(x).data(Qt.UserRole)
            dli += len(self.datalines[x])
            if(ds.origin == self.selected_dataset_data.origin):
                break
        self.dataline_list.insertItem(dli, qlwt)
        self.dataline_list.setCurrentRow(dli)
        self.datalines[self.dataset_list.currentRow()].append(dl)
        self.updateSelectedDataline()
        self.updatePlots()

    def populateSelectedDatalineDetails(self, listitemwidget, qlistwidget):
        if(not listitemwidget):
            self.setDatalineControlsStatus(False)
            self.dl_name_edit.setText('')
            return
        self.setDatalineControlsStatus(True)
        self.dl_xscale_sb.blockSignals(True)
        self.dl_yscale_sb.blockSignals(True)
        self.dl_xoffset_sb.blockSignals(True)
        self.dl_yoffset_sb.blockSignals(True)
        self.dl_linewidth_sb.blockSignals(True)
        self.dl_markersize_sb.blockSignals(True)
        self.dl_savgol_wlen.blockSignals(True)
        self.dl_savgol_ord.blockSignals(True)

        self.selected_dataline_widget = listitemwidget
        self.selected_dataline_data = listitemwidget.data(Qt.UserRole)
        self.dl_name_edit.setText(self.selected_dataline_widget.text())
        self.dl_render_cb.setCurrentIndex(self.selected_dataline_data.plots)
        
        self.dl_xdata_cb.clear()
        self.dl_ydata_cb.clear()
        self.dl_xdata_cb.addItems(self.selected_dataline_data.dataset.fields)
        self.dl_ydata_cb.addItems(self.selected_dataline_data.dataset.fields)
        
        self.dl_transform_cb.setCurrentIndex(self.selected_dataline_data.transform)
        self.dl_xdata_cb.setCurrentText(self.selected_dataline_data.xsource)
        self.dl_xscale_sb.setValue(self.selected_dataline_data.xscale)
        self.dl_xoffset_sb.setValue(self.selected_dataline_data.xoffset)
        self.dl_ydata_cb.setCurrentText(self.selected_dataline_data.ysource)
        self.dl_yscale_sb.setValue(self.selected_dataline_data.yscale)
        self.dl_yoffset_sb.setValue(self.selected_dataline_data.yoffset)
        self.dl_color_edit.setText(self.selected_dataline_data.color)
        self.dl_style_cb.setCurrentText(self.selected_dataline_data.linestyle)
        self.dl_linewidth_sb.setValue(self.selected_dataline_data.linewidth)
        self.dl_marker_cb.setCurrentText(self.selected_dataline_data.markerstyle)
        self.dl_markersize_sb.setValue(self.selected_dataline_data.markersize)
        self.dl_color_label.setStyleSheet(f'background-color: {self.selected_dataline_data.color}')
        self.dl_savgol_wlen.setValue(self.selected_dataline_data.savgolwindow)
        self.dl_savgol_ord.setValue(self.selected_dataline_data.savgolord)
        
        self.dl_xscale_sb.blockSignals(False)
        self.dl_yscale_sb.blockSignals(False)
        self.dl_xoffset_sb.blockSignals(False)
        self.dl_yoffset_sb.blockSignals(False)
        self.dl_linewidth_sb.blockSignals(False)
        self.dl_markersize_sb.blockSignals(False)
        self.dl_savgol_wlen.blockSignals(False)
        self.dl_savgol_ord.blockSignals(False)

    def updateSelectedDataline(self):
        if(not self.selected_dataline_widget):
            return
        new_name = self.dl_name_edit.text()
        self.selected_dataline_widget.setText(new_name)
        self.selected_dataline_data.name = new_name
        self.selected_dataline_data.plots = self.dl_render_cb.currentIndex()
        self.selected_dataline_data.transform = self.dl_transform_cb.currentIndex()
        self.selected_dataline_data.xsource = self.dl_xdata_cb.currentText()
        self.selected_dataline_data.xscale = self.dl_xscale_sb.value()
        self.selected_dataline_data.xoffset = self.dl_xoffset_sb.value()
        self.selected_dataline_data.ysource = self.dl_ydata_cb.currentText()
        self.selected_dataline_data.yscale = self.dl_yscale_sb.value()
        self.selected_dataline_data.yoffset = self.dl_yoffset_sb.value()
        self.selected_dataline_data.color = self.dl_color_edit.text()
        self.selected_dataline_data.linestyle = self.dl_style_cb.currentText()
        self.selected_dataline_data.linewidth = self.dl_linewidth_sb.value()
        self.selected_dataline_data.markerstyle = self.dl_marker_cb.currentText()
        self.selected_dataline_data.markersize = self.dl_markersize_sb.value()
        self.selected_dataline_data.savgolwindow = self.dl_savgol_wlen.value()
        self.selected_dataline_data.savgolord = self.dl_savgol_ord.value()
        self.populateSelectedDatalineDetails(self.selected_dataline_widget, None)
        self.updatePlots()

    def openColorPicker(self):
        dialog = QColorDialog(self)
        dialog.setCurrentColor(Qt.red)
        dialog.setOption(QColorDialog.ShowAlphaChannel)
        dialog.open()
        dialog.currentColorChanged.connect(self.updateDatalineColor)

    def updateDatalineColor(self, color):
        self.dl_color_edit.setText(color.name())
        self.dl_color_label.setStyleSheet(f'background-color: {color.name()}')
        self.selected_dataline_data.color = color.name()
        self.updatePlots()

    def setDatasetControlsStatus(self, enabled=True):
        self.ds_title_edit.setEnabled(enabled)
        self.ds_addline_btn.setEnabled(enabled)
        self.ds_caseadd_btn.setEnabled(enabled)
        self.ds_remove_btn.setEnabled(enabled)

    def setDatalineControlsStatus(self, enabled=True):
        self.dl_name_edit.setEnabled(enabled)
        self.dl_render_cb.setEnabled(enabled)
        self.dl_transform_cb.setEnabled(enabled)
        self.dl_xdata_cb.setEnabled(enabled)
        self.dl_xscale_sb.setEnabled(enabled)
        self.dl_xoffset_sb.setEnabled(enabled)
        self.dl_ydata_cb.setEnabled(enabled)
        self.dl_yscale_sb.setEnabled(enabled)
        self.dl_yoffset_sb.setEnabled(enabled)
        self.dl_color_edit.setEnabled(enabled)
        self.dl_color_pickerbtn.setEnabled(enabled)
        self.dl_style_cb.setEnabled(enabled)
        self.dl_linewidth_sb.setEnabled(enabled)
        self.dl_marker_cb.setEnabled(enabled)
        self.dl_markersize_sb.setEnabled(enabled)
        self.dl_savgol_wlen.setEnabled(enabled)
        self.dl_savgol_ord.setEnabled(enabled)
        self.dl_remove_btn.setEnabled(enabled)

    def getPlotFromIndex(self, plotnum):
        x = plotnum
        tab = 0
        for tab_plots in self.plots_canvases:
            if(x - len(tab_plots) >= 0):
                tab += 1
                x -= len(tab_plots)
            else:
                break
        return self.plots_canvases[tab][x]

    def autoscalePlots(self):
        processedCanvas = [x.canvas for x in self.plots_canvases[self.tabbing_plots.currentIndex()]]
        for canvas in processedCanvas:
            canvas.ax.margins(self.plt_marginx.value(), self.plt_marginy.value())
            canvas.ax.relim()
            canvas.ax.autoscale()
        self.updatePlots()

    def changeLabelSize(self):
        # plt.rcParams.update({'font.size': self.plt_labelsize_sb.value()})
        self.updatePlots()

    def updatePlots(self):
        processedCanvas = [x.canvas for x in self.plots_canvases[self.tabbing_plots.currentIndex()]]
        for canvas in processedCanvas:
            plotlist = []
            for artist in canvas.ax.lines + canvas.ax.collections:
                artist.remove()
            for x in range(self.dataset_list.count()):
                ds = self.dataset_list.item(x).data(Qt.UserRole)
                for dl in ds.datalines:
                    dl_canvas = self.getPlotFromIndex(dl.plots).canvas
                    if(dl_canvas == canvas):
                        
                        for label in (canvas.ax.get_xticklabels() + canvas.ax.get_yticklabels()):
                            label.set_fontsize(self.plt_ticksize_sb.value())
                        canvas.ax.xaxis.label.set_size(self.plt_labelsize_sb.value())
                        canvas.ax.yaxis.label.set_size(self.plt_labelsize_sb.value())
                        canvas.ax.title.set_size(self.plt_titlesize_sb.value())

                        x, y = ds.get_datapoints(dl.xsource, dl.ysource, dl.casenum)

                        if(dl.transform == 1):
                            y = np.abs(y)
                        elif(dl.transform == 2):
                            y = np.angle(y, deg=True)
                        elif(dl.transform == 3):
                            y = np.unwrap(np.angle(y, deg=True), period=360)
                        elif(dl.transform == 4):
                            y = 20 * np.log10(y)
                        elif(dl.transform == 5):
                            y = 20 * np.log10(np.abs(y))
                        elif(dl.transform == 6):
                            y = np.unwrap(y, period=360)
                        else:
                            y = np.real(y)

                        if(dl.transform in [2,3,6]):
                            canvas.ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins='auto', steps=[1.8,2.25,4.5,9]))
                        else:
                            canvas.ax.yaxis.set_major_locator(ticker.AutoLocator())

                        try:
                            savgolw = int(dl.savgolwindow)
                            if(savgolw <= len(x)):
                                savgolo = int(dl.savgolord)
                                y = y if savgolw <= savgolo else savgol_filter(y, savgolw, savgolo)
                        except ValueError:
                            pass

                        try:
                            line, = canvas.ax.plot(
                                x * dl.xscale + dl.xoffset,
                                y * dl.yscale + dl.yoffset,
                                linestyle = LINE_STYLES[dl.linestyle],
                                linewidth = dl.linewidth,
                                marker = MARKER_STYLES[dl.markerstyle],
                                markersize = dl.markersize,
                                color = dl.color,
                                label = dl.name,
                            )
                            if(dl.name != '' and dl.name[0] != '_'):
                                plotlist.append(line)
                        except ValueError:
                            self.statusbar.showMessage('Wrong data source matching', 2000)
            if(self.plt_legendpos.currentText() == 'None'):
                canvas.ax.get_legend().remove()
            else:
                canvas.ax.legend(handles=plotlist, fontsize=self.plt_legendsize_sb.value(), loc=self.plt_legendpos.currentIndex())
            if(self.plt_grid.isChecked()):
                canvas.ax.grid(True, which="both", linestyle=':')
            else:
                canvas.ax.grid(False)

            try:
                canvas.draw()
            except ParseSyntaxException or ValueError:
                pass

    def showZPWindow(self):
        zeros = self.selected_dataset_data.zeros[0]
        poles = self.selected_dataset_data.poles[0]
        self.zpWindow = ZPWindow(zeros, poles, self.selected_dataset_data.title)
        self.zpWindow.show()