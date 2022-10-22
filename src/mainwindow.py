# PyQt5 modules
from math import inf
from PyQt5.QtWidgets import QMainWindow, QListWidgetItem, QColorDialog, QFileDialog, QDialog
from PyQt5.QtCore import Qt

# Project modules
from src.ui.mainwindow import Ui_MainWindow
from src.package.Dataset import Dataset
import src.package.Filter as Filter
from src.package.Filter import AnalogFilter
from src.widgets.exprwidget import MplCanvas
from src.widgets.tf_dialog import TFDialog
from src.widgets.case_window import CaseDialog
from src.widgets.zp_window import ZPWindow
from src.widgets.response_dialog import ResponseDialog
from src.widgets.prompt_dialog import PromptDialog

from scipy.signal import savgol_filter
import scipy.signal as signal
from scipy.interpolate import splrep, splev, splprep
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from mplcursors import HoverMode, cursor, Selection
from matplotlib.patches import Rectangle

import numpy as np
import random
from pyparsing.exceptions import ParseSyntaxException

import pickle

MARKER_STYLES = { 'None': '', 'Point': '.',  'Pixel': ',',  'Circle': 'o',  'Triangle down': 'v',  'Triangle up': '^',  'Triangle left': '<',  'Triangle right': '>',  'Tri down': '1',  'Tri up': '2',  'Tri left': '3',  'Tri right': '4',  'Octagon': '8',  'Square': 's',  'Pentagon': 'p',  'Plus (filled)': 'P',  'Star': '*',  'Hexagon': 'h',  'Hexagon alt.': 'H',  'Plus': '+',  'x': 'x',  'x (filled)': 'X',  'Diamond': 'D',  'Diamond (thin)': 'd',  'Vline': '|',  'Hline': '_' }
LINE_STYLES = { 'None': '', 'Solid': '-', 'Dashed': '--', 'Dash-dot': '-.', 'Dotted': ':' }

POLE_COLOR = '#FF0000'
POLE_SEL_COLOR = '#00FF00'
ZERO_COLOR = '#0000FF'
ZERO_SEL_COLOR = '#00FF00'

TEMPLATE_FACE_COLOR = '#ffcccb'
TEMPLATE_EDGE_COLOR = '#ef9a9a'
ADD_TEMPLATE_FACE_COLOR = '#c8e6c9'
ADD_TEMPLATE_EDGE_COLOR = '#a5d6a7'

SHOW_PZ_IN_HZ = True
PZ_XLABEL = f'$\sigma/2\pi$ [1/s]' if SHOW_PZ_IN_HZ else '$\sigma$ ($rad/s$)'
PZ_YLABEL = f'$jf$ [Hz]' if SHOW_PZ_IN_HZ else '$j\omega$ ($rad/s$)'
F_TO_W = 2*np.pi
W_TO_F = 1/F_TO_W
SING_B_TO_F = W_TO_F if SHOW_PZ_IN_HZ else 1
SING_F_TO_B = F_TO_W if SHOW_PZ_IN_HZ else 1

def stage_to_str(stage):
    stage_str = 'Z={'
    for z in stage.z:
        stage_str += str(z)
        stage_str += ', '
    stage_str += '} , P={'
    for p in stage.p:
        stage_str += str(p)
        stage_str += ', '
    stage_str += '} , K='
    stage_str+= str(stage.k)
    return stage_str

class MainWindow(QMainWindow, Ui_MainWindow):

    def __init__(self):
        super(MainWindow, self).__init__()
        self.setupUi(self)
        self.droppedFiles = []
        self.datasets = []
        self.datalines = []
        # self.stage_datasets = [] capaz mas adelante lo ponga para serializar también las etapas, pero creo que para ahora es mucho
        self.selected_dataset_widget = {}
        self.selected_dataline_widget = {}
        self.selected_dataset_data = {}
        self.selected_dataline_data = {}
        self.zpWindow = type('ZPWindow', (), {})()
        
        self.import_file_btn.clicked.connect(self.importFiles)
        
        self.dataset_list.currentItemChanged.connect(self.populateSelectedDatasetDetails)
        self.ds_title_edit.textEdited.connect(self.updateSelectedDatasetName)
        self.ds_addline_btn.clicked.connect(self.addDataline)
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

        self.pmptd = PromptDialog()
        
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

        self.new_filter_btn.clicked.connect(self.addFilter)
        self.chg_filter_btn.clicked.connect(self.changeSelectedFilter)
        self.tipo_box.currentIndexChanged.connect(self.updateFilterParametersAvailable)
        self.define_with_box.currentIndexChanged.connect(self.updateFilterParametersAvailable)
        self.updateFilterParametersAvailable()

        self.new_stage_btn.clicked.connect(self.addFilterStage)
        self.remove_stage_btn.clicked.connect(self.removeFilterStage)

        self.actionLoad_2.triggered.connect(self.loadFile)
        self.actionSave_2.triggered.connect(self.saveFile)

        self.stageCursorZer = {}
        self.stageCursorPol = {}

        self.poles_list.itemSelectionChanged.connect(self.stage_sel_changed)
        self.zeros_list.itemSelectionChanged.connect(self.stage_sel_changed)
        self.stages_list.itemSelectionChanged.connect(self.updateStagePlots)

        self.filters = []
        self.selfil_cb.currentIndexChanged.connect(self.populateSelectedFilterDetails)
        self.stages_selfil_cb.currentIndexChanged.connect(self.populateSelectedFilterDetails)
        self.symmetrize_btn.clicked.connect(self.makeFilterTemplateSymmetric)


    def addDataset(self, ds):
        qlwt = QListWidgetItem()
        qlwt.setData(Qt.UserRole, ds)
        qlwt.setText(ds.title)
        self.dataset_list.addItem(qlwt)
        self.datasets.append(ds)
        #self.stage_datasets.append([])
        self.datalines.append([])
        self.dataset_list.setCurrentRow(self.dataset_list.count() - 1)

    def removeDataset(self, i):
        #Saco los datalines
        first_dataline_index = 0
        last_dataline_index = len(self.datalines[0])
        for x in range(self.dataset_list.count()):
            if(x == 0):
                first_dataline_index = 0
            else:    
                first_dataline_index += len(self.datalines[x - 1])
            if(x == i):
                break
        last_dataline_index = first_dataline_index + len(self.datalines[i])

        for x in range(first_dataline_index, last_dataline_index):
            self.dataline_list.takeItem(first_dataline_index)
        self.datalines.pop(i)

        ds = self.dataset_list.item(i).data(Qt.UserRole)
        if(ds.type == 'filter'):
            fi = self.filters.index(ds)
            self.filters.pop(fi)
            self.selfil_cb.removeItem(fi)
            self.stages_selfil_cb.removeItem(fi)
            if(self.selfil_cb.count() == 0):
                self.chg_filter_btn.setEnabled(False)
        self.dataset_list.takeItem(i)
        self.updatePlots()

    def addDataline(self):
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

    def removeDataline(self, i):        
        try:
            dsi, dli = self.getInternalDataIndexes(i)
            del self.datalines[dsi][dli]
            self.dataline_list.takeItem(i).data(Qt.UserRole)
            del self.dataset_list.item(dsi).data(Qt.UserRole).datalines[dli]
            if(self.dataline_list.currentRow() == -1):
                self.dataline_list.setCurrentRow(self.dataline_list.count() - 1)
        except AttributeError:
            pass
        self.updatePlots()
    
    def removeSelectedDataline(self, event):
        selected_row = self.dataline_list.currentRow()
        self.removeDataline(selected_row)

    def removeSelectedDataset(self, event):
        selected_row = self.dataset_list.currentRow()
        self.removeDataset(selected_row)

    def getInternalDataIndexes(self, datalineRow):
        i = self.dataline_list.currentRow()
        for x in range(self.dataset_list.count()):
            ds = self.dataset_list.item(x).data(Qt.UserRole)
            if(i >= len(self.datalines[x])):
                i = i - len(self.datalines[x])
            else:
                return (x, i)
        return (x, i)

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
        # options |= QFileDialog.DontUseNativeDialog
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
                    self.addDataset(ds)
            except(ValueError):
                print('Wrong file config')
        self.statusbar.clearMessage()

    def openTFDialog(self):
        self.tfd.open()
        self.tfd.tf_title.setFocus()

    def resolveTFDialog(self):
        if not self.tfd.validateTF():
            return
        ds = Dataset(filepath='', origin=self.tfd.tf, title=self.tfd.getTFTitle())
        self.addDataset(ds)

    def openResponseDialog(self):
        self.respd.open()
        self.respd.input_txt.setFocus()

    def resolveResponseDialog(self):
        if not(self.respd.validateResponse()):
            return

        t = self.respd.getTimeDomain()
        expression = self.respd.getResponseExpression()
        title = self.respd.getResponseTitle()
        time_title = title + "_timebase"
        ans_title = title + "_ans"
        
        if expression == 'step':
            x = np.heaviside(t, 0.5)
            _, response = signal.step(self.selected_dataset_data.tf.tf_object, T=t)
        elif expression == 'delta':
            delta = lambda t, eps: (1 / (np.sqrt(np.pi) *eps)) * np.exp(-(t/eps)**2) #para plotear la delta
            x = delta(t, (t[1] - t[0]))
            _, response = signal.delta(self.selected_dataset_data.tf.tf_object, T=t)
        else:
            x = eval(expression)
            response = signal.lsim(self.selected_dataset_data.tf.tf_object , U = x , T = t)[1]
        
        self.selected_dataset_data.data[0][time_title] = t
        self.selected_dataset_data.data[0][title] = x
        self.selected_dataset_data.data[0][ans_title] = response

        self.selected_dataset_data.fields.append(time_title)
        self.selected_dataset_data.fields.append(title)
        self.selected_dataset_data.fields.append(ans_title)
        
        self.updateSelectedDataset()
        self.updateSelectedDataline()

    def buildFilterFromParams(self):
        if self.tipo_box.currentIndex() in [Filter.BAND_PASS, Filter.BAND_REJECT]:
            wa = [F_TO_W * self.fa_min_box.value(), F_TO_W * self.fa_max_box.value()]
            wp = [F_TO_W * self.fp_min_box.value(), F_TO_W * self.fp_max_box.value()]
        else:
            wa = F_TO_W * self.fa_box.value()
            wp = F_TO_W * self.fp_box.value()
            
        params = {
            "name": self.filtername_box.text(),
            "filter_type": self.tipo_box.currentIndex(),
            "approx_type": self.aprox_box.currentIndex(),
            "helper_approx": self.compareapprox_cb.currentIndex(),
            "helper_N": self.comp_N_box.value(),
            "is_helper": False,
            "define_with": self.define_with_box.currentIndex(),
            "N_min": self.N_min_box.value(),
            "N_max": self.N_max_box.value(),
            "gain": self.gain_box.value(),
            "denorm": self.denorm_box.value(),
            "aa_dB": self.aa_box.value(),
            "ap_dB": self.ap_box.value(),
            "wa": wa,
            "wp": wp,
            "w0": F_TO_W * self.f0_box.value(),
            "bw": [F_TO_W * self.bw_min_box.value(), F_TO_W * self.bw_max_box.value()],
            "gamma": self.tol_box.value(),
            "tau0": self.tau0_box.value(),
            "wrg": F_TO_W * self.frg_box.value(),
        }
        return AnalogFilter(**params)
    
    def addFilter(self):
        newFilter = self.buildFilterFromParams()
        valid, msg = newFilter.validate()
        if not valid:
            self.pmptd.setErrorMsg(msg)
            self.pmptd.open()
            self.pmptd.setFocus()
            return
        ds = Dataset(filepath='', origin=newFilter, title=self.filtername_box.text())
        self.filters.append(ds)
        self.selfil_cb.blockSignals(True)
        self.stages_selfil_cb.blockSignals(True)
        self.selfil_cb.addItem(ds.title, ds)
        self.stages_selfil_cb.addItem(ds.title, ds)
        self.selfil_cb.blockSignals(False)
        self.stages_selfil_cb.blockSignals(False)
        self.chg_filter_btn.setEnabled(True)
        self.addDataset(ds)
    
    def makeFilterTemplateSymmetric(self):
        fa = [self.fa_min_box.value(), self.fa_max_box.value()]
        fp = [self.fp_min_box.value(), self.fp_max_box.value()]
        f0 = 0
        bw = [0, 0]
        if(self.tipo_box.currentIndex() == Filter.BAND_PASS):
            f0 = np.sqrt(fp[0] * fp[1])
            bw[0] = fp[1] - fp[0]
            if(fa[0] * fa[1] != f0**2):
                famincalc = f0**2 / fa[1]
                famaxcalc = f0**2 / fa[0]
                if(famincalc > fa[0]):
                    fa[0] = famincalc
                elif(famaxcalc < fa[1]):
                    fa[1] = famaxcalc
            bw[1] = fa[1] - fa[0]

        elif(self.tipo_box.currentIndex() == Filter.BAND_REJECT):
            f0 = np.sqrt(fa[0] * fa[1])
            bw[0] = fa[1] - fa[0]
            if(fp[0] * fp[1] != f0**2):
                fpmincalc = f0**2 / fp[1]
                fpmaxcalc = f0**2 / fp[0]
                if(fpmincalc > fp[0]):
                    fp[0] = fpmincalc
                elif(fpmaxcalc < fp[1]):
                    fp[1] = fpmaxcalc    
            bw[1] = fp[1] - fp[0]

        self.fa_min_box.setValue(fa[0])
        self.fa_max_box.setValue(fa[1])
        self.fp_min_box.setValue(fp[0])
        self.fp_max_box.setValue(fp[1])
        self.f0_box.setValue(f0)
        self.bw_min_box.setValue(bw[0])
        self.bw_max_box.setValue(bw[1])

    def changeSelectedFilter(self):
        newFilter = self.buildFilterFromParams()
        valid, msg = newFilter.validate()
        if not valid:
            self.pmptd.setErrorMsg(msg)
            self.pmptd.setFocus()
            return
        temp_datalines = self.selected_dataset_data.datalines
        ds = Dataset('', self.filtername_box.text(), newFilter)
        ds.datalines = temp_datalines
        ds.title = self.filtername_box.text()
        self.selected_dataset_widget.setText(self.filtername_box.text())
        self.selfil_cb.setItemText(self.selfil_cb.currentIndex(), ds.title)
        self.stages_selfil_cb.setItemText(self.selfil_cb.currentIndex(), ds.title)
        self.selected_dataset_widget.setData(Qt.UserRole, ds)
        self.filters[self.selfil_cb.currentIndex()] = ds
        self.populateSelectedDatasetDetails(self.selected_dataset_widget, None)

    def updateFilterParametersAvailable(self):
        if self.tipo_box.currentIndex() == Filter.LOW_PASS or self.tipo_box.currentIndex() == Filter.HIGH_PASS:
            for i in range(Filter.LEGENDRE + 1):
                self.aprox_box.model().item(i).setEnabled(True)
                self.compareapprox_cb.model().item(i).setEnabled(True)
            for i in range(Filter.BESSEL, Filter.GAUSS + 1):
                self.aprox_box.model().item(i).setEnabled(False)
                self.compareapprox_cb.model().item(i).setEnabled(False)
            if not self.aprox_box.model().item(self.aprox_box.currentIndex()).isEnabled():
                self.aprox_box.setCurrentIndex(Filter.BUTTERWORTH)
                self.compareapprox_cb.setCurrentIndex(Filter.BUTTERWORTH)
            self.define_with_box.setVisible(False)
            self.label_definewith.setVisible(False)
            self.ap_box.setVisible(True)
            self.label_ap.setVisible(True)
            self.aa_box.setVisible(True)
            self.label_aa.setVisible(True)
            self.fp_box.setVisible(True)
            self.label_fp.setVisible(True)
            self.fa_box.setVisible(True)
            self.label_fa.setVisible(True)
            self.fa_min_box.setVisible(False)
            self.symmetrize_btn.setVisible(False)
            self.label_famin.setVisible(False)
            self.fa_max_box.setVisible(False)
            self.label_famax.setVisible(False)
            self.fp_min_box.setVisible(False)
            self.label_fpmin.setVisible(False)
            self.fp_max_box.setVisible(False)
            self.label_fpmax.setVisible(False)
            self.f0_box.setVisible(False)
            self.label_f0.setVisible(False)
            self.bw_min_box.setVisible(False)
            self.label_bwmin.setVisible(False)
            self.bw_max_box.setVisible(False)
            self.label_bwmax.setVisible(False)
            self.tau0_box.setVisible(False)
            self.label_tau0.setVisible(False)
            self.frg_box.setVisible(False)
            self.label_fRG.setVisible(False)
            self.tol_box.setVisible(False)
            self.label_tolerance.setVisible(False)

        elif self.tipo_box.currentIndex() == Filter.BAND_PASS or self.tipo_box.currentIndex() == Filter.BAND_REJECT:
            for i in range(Filter.LEGENDRE + 1):
                self.aprox_box.model().item(i).setEnabled(True)
                self.compareapprox_cb.model().item(i).setEnabled(True)
            for i in range(Filter.BESSEL, Filter.GAUSS + 1):
                self.aprox_box.model().item(i).setEnabled(False)
                self.compareapprox_cb.model().item(i).setEnabled(False)
            if not self.aprox_box.model().item(self.aprox_box.currentIndex()).isEnabled():
                self.aprox_box.setCurrentIndex(Filter.BUTTERWORTH)
                self.compareapprox_cb.setCurrentIndex(Filter.BUTTERWORTH)
            self.define_with_box.setVisible(True)
            self.label_definewith.setVisible(True)
            self.ap_box.setVisible(True)
            self.label_ap.setVisible(True)
            self.aa_box.setVisible(True)
            self.label_aa.setVisible(True)
            self.fp_box.setVisible(False)
            self.label_fp.setVisible(False)
            self.fa_box.setVisible(False)
            self.label_fa.setVisible(False)
            self.tau0_box.setVisible(False)
            self.label_tau0.setVisible(False)
            self.frg_box.setVisible(False)
            self.label_fRG.setVisible(False)
            self.tol_box.setVisible(False)
            self.label_tolerance.setVisible(False)

            if self.define_with_box.currentIndex() == Filter.TEMPLATE_FREQS:
                self.fa_min_box.setVisible(True)
                self.label_famin.setVisible(True)
                self.fa_max_box.setVisible(True)
                self.label_famax.setVisible(True)
                self.fp_min_box.setVisible(True)
                self.label_fpmin.setVisible(True)
                self.fp_max_box.setVisible(True)
                self.label_fpmax.setVisible(True)
                self.f0_box.setVisible(False)
                self.label_f0.setVisible(False)
                self.bw_min_box.setVisible(False)
                self.label_bwmin.setVisible(False)
                self.bw_max_box.setVisible(False)
                self.label_bwmax.setVisible(False)
                self.symmetrize_btn.setVisible(True)

            if self.define_with_box.currentIndex() == Filter.F0_BW:
                self.fa_min_box.setVisible(False)
                self.label_famin.setVisible(False)
                self.fa_max_box.setVisible(False)
                self.label_famax.setVisible(False)
                self.fp_min_box.setVisible(False)
                self.label_fpmin.setVisible(False)
                self.fp_max_box.setVisible(False)
                self.label_fpmax.setVisible(False)
                self.f0_box.setVisible(True)
                self.label_f0.setVisible(True)
                self.bw_min_box.setVisible(True)
                self.label_bwmin.setVisible(True)
                self.bw_max_box.setVisible(True)
                self.label_bwmax.setVisible(True)
                self.symmetrize_btn.setVisible(False)
        
        elif self.tipo_box.currentIndex() == Filter.GROUP_DELAY:
            for i in range(Filter.LEGENDRE + 1):
                self.aprox_box.model().item(i).setEnabled(False)
                self.compareapprox_cb.model().item(i).setEnabled(False)
            for i in range(Filter.BESSEL, Filter.GAUSS + 1):
                self.aprox_box.model().item(i).setEnabled(True)
                self.compareapprox_cb.model().item(i).setEnabled(True)
            if not self.aprox_box.model().item(self.aprox_box.currentIndex()).isEnabled():
                self.aprox_box.setCurrentIndex(Filter.BESSEL)
                self.compareapprox_cb.setCurrentIndex(Filter.BESSEL)
            
            self.define_with_box.setVisible(False)
            self.label_definewith.setVisible(False)
            self.ap_box.setVisible(False)
            self.label_ap.setVisible(False)
            self.aa_box.setVisible(False)
            self.label_aa.setVisible(False)
            self.fp_box.setVisible(False)
            self.label_fp.setVisible(False)
            self.fa_box.setVisible(False)
            self.label_fa.setVisible(False)
            self.fa_min_box.setVisible(False)
            self.label_famin.setVisible(False)
            self.fa_max_box.setVisible(False)
            self.label_famax.setVisible(False)
            self.fp_min_box.setVisible(False)
            self.label_fpmin.setVisible(False)
            self.fp_max_box.setVisible(False)
            self.label_fpmax.setVisible(False)
            self.f0_box.setVisible(False)
            self.label_f0.setVisible(False)
            self.bw_min_box.setVisible(False)
            self.label_bwmin.setVisible(False)
            self.bw_max_box.setVisible(False)
            self.label_bwmax.setVisible(False)
            self.tau0_box.setVisible(True)
            self.label_tau0.setVisible(True)
            self.frg_box.setVisible(True)
            self.label_fRG.setVisible(True)
            self.tol_box.setVisible(True)
            self.label_tolerance.setVisible(True)

    def condition_canvas(self, canvas, xlabel, ylabel, xscale='linear', yscale='linear', grid=True):
        canvas.ax.clear()
        canvas.ax.grid(grid, which="both", linestyle=':')
        canvas.ax.set_xlabel(xlabel)
        canvas.ax.set_ylabel(ylabel)
        canvas.ax.set_xscale(xscale)
        canvas.ax.set_yscale(yscale)
        canvas.ax.xaxis.label.set_size(self.plt_labelsize_sb.value())
        canvas.ax.yaxis.label.set_size(self.plt_labelsize_sb.value())
        for label in (canvas.ax.get_xticklabels() + canvas.ax.get_yticklabels()):
            label.set_fontsize(self.plt_ticksize_sb.value())

    def updateFilterPlots(self):
        attcanvas = self.fplot_att.canvas
        gaincanvas = self.fplot_gain.canvas
        magcanvas = self.fplot_mag.canvas
        phasecanvas = self.fplot_phase.canvas
        groupdelaycanvas = self.fplot_gd.canvas
        pzcanvas = self.fplot_pz.canvas
        stepcanvas = self.fplot_step.canvas
        impulsecanvas = self.fplot_impulse.canvas

        self.condition_canvas(attcanvas, 'Frecuencia [Hz]', 'Atenuación [dB]')
        self.condition_canvas(gaincanvas, 'Frecuencia [Hz]', 'Ganancia [dB]')
        self.condition_canvas(magcanvas, 'Frecuencia [Hz]', 'Magnitud [dB]', 'log')
        self.condition_canvas(phasecanvas, 'Frecuencia [Hz]', 'Fase [$^o$]', 'log')
        phasecanvas.ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins='auto', steps=[1.8,2.25,4.5,9]))
        self.condition_canvas(groupdelaycanvas, 'Frecuencia [Hz]', 'Retardo de grupo [s]', 'log')
        self.condition_canvas(pzcanvas, '', '')
        self.condition_canvas(stepcanvas, 'Tiempo [s]', 'Respuesta [V]')
        self.condition_canvas(impulsecanvas, 'Tiempo [s]', 'Respuesta [V]')
        
        filtds = self.selected_dataset_data
        
        tstep, stepres = signal.step(filtds.tf.tf_object, N=5000)
        timp, impres = signal.impulse(filtds.tf.tf_object, N=5000)
        
        f = np.array(filtds.data[0]['f'])
        g = 20 * np.log10(np.abs(np.array(filtds.data[0]['g'])))
        ph = np.array(filtds.data[0]['ph'])
        gd = np.array(filtds.data[0]['gd'])
        z, p = filtds.origin.tf.getZP(SHOW_PZ_IN_HZ)

        gainline, = gaincanvas.ax.plot(f, g)
        magline, = magcanvas.ax.plot(f, g)
        phaseline, = phasecanvas.ax.plot(f, ph)
        gdline, = groupdelaycanvas.ax.plot(f, gd)
        stepline, = stepcanvas.ax.plot(tstep, stepres)
        impulseline, = impulsecanvas.ax.plot(timp, impres)

        ap = filtds.origin.ap_dB
        aa = filtds.origin.aa_dB

        xmin = 0
        xmax = 0
        xmin = 0
        ymax = aa*2
        if filtds.origin.filter_type == Filter.LOW_PASS:
            fp = filtds.origin.wp * W_TO_F
            fa = filtds.origin.wa * W_TO_F
            deltaf = (fa - fp)/2
            xmax = fa + deltaf
            xmin = max([0, fp - deltaf])
            attcanvas.ax.fill_between([xmin, fp], [ap, ap], ymax, facecolor=TEMPLATE_FACE_COLOR, edgecolor=TEMPLATE_EDGE_COLOR, hatch='\\', linewidth=0)
            attcanvas.ax.fill_between([fa, xmax], [aa, aa], 0, facecolor=TEMPLATE_FACE_COLOR, edgecolor=TEMPLATE_EDGE_COLOR, hatch='\\', linewidth=0)
            attcanvas.ax.set_ylim([0, ymax])

        elif filtds.origin.filter_type == Filter.HIGH_PASS:
            fp = filtds.origin.wp * W_TO_F
            fa = filtds.origin.wa * W_TO_F
            deltaf = (fp - fa)/2
            xmax = fp + deltaf
            xmin = max([0, fa - deltaf])
            attcanvas.ax.fill_between([fp, xmax], [ap, ap], ymax, facecolor=TEMPLATE_FACE_COLOR, edgecolor=TEMPLATE_EDGE_COLOR, hatch='\\', linewidth=0)
            attcanvas.ax.fill_between([xmin, fa], [aa, aa], 0, facecolor=TEMPLATE_FACE_COLOR, edgecolor=TEMPLATE_EDGE_COLOR, hatch='\\', linewidth=0)
            attcanvas.ax.set_ylim([0, ymax])

        elif filtds.origin.filter_type == Filter.BAND_PASS:
            fp = [w * W_TO_F for w in filtds.origin.wp]
            fa = [w * W_TO_F for w in filtds.origin.wa]
            reqfa = [w * W_TO_F for w in filtds.origin.reqwa]
            deltaf = (fa[1] - fa[0])/2
            xmax = fa[1] + deltaf
            xmin = max([0, fa[0] - deltaf])
            
            attcanvas.ax.fill_between([xmin,  reqfa[0]], [aa, aa], 0, facecolor=TEMPLATE_FACE_COLOR, edgecolor=TEMPLATE_EDGE_COLOR, hatch='\\', linewidth=0)
            attcanvas.ax.fill_between([reqfa[1], xmax ], [aa, aa], 0, facecolor=TEMPLATE_FACE_COLOR, edgecolor=TEMPLATE_EDGE_COLOR, hatch='\\', linewidth=0)
            
            if self.define_with_box.currentIndex() == Filter.TEMPLATE_FREQS:
                if(fa[0] != reqfa[0]):
                    attcanvas.ax.fill_between([fa[0],  reqfa[0]], [aa, aa], 0, facecolor=ADD_TEMPLATE_FACE_COLOR, edgecolor=ADD_TEMPLATE_EDGE_COLOR, hatch='//', linewidth=0)
                elif(fa[1] != reqfa[1]):
                    attcanvas.ax.fill_between([reqfa[1], fa[1] ], [aa, aa], 0, facecolor=ADD_TEMPLATE_FACE_COLOR, edgecolor=ADD_TEMPLATE_EDGE_COLOR, hatch='//', linewidth=0)
                else:
                    print("WTF1")
            attcanvas.ax.fill_between([fp[0], fp[1]], [ap, ap], ymax, facecolor=TEMPLATE_FACE_COLOR, edgecolor=TEMPLATE_EDGE_COLOR, hatch='\\', linewidth=0)
            attcanvas.ax.set_ylim([0, ymax])

        elif filtds.origin.filter_type == Filter.BAND_REJECT:
            fp = [w * W_TO_F for w in filtds.origin.wp]
            fa = [w * W_TO_F for w in filtds.origin.wa]
            deltaf = (fp[1] - fp[0])/2
            xmax = fp[1] + deltaf
            xmin = max([0, fp[0] - deltaf])
            
            if self.define_with_box.currentIndex() == Filter.TEMPLATE_FREQS:
                reqfp = [w * W_TO_F for w in filtds.origin.reqwp]
                if(fp[0] != reqfp[0]):
                    attcanvas.ax.fill_between([xmin,  reqfp[0]], [ap, ap], ymax, facecolor='#555555', edgecolor='#121212', hatch='//', linewidth=0)
                elif(fp[1] != reqfp[1]):
                    attcanvas.ax.fill_between([reqfp[1], xmax ], [ap, ap], ymax, facecolor='#555555', edgecolor='#121212', hatch='//', linewidth=0)
                else:
                    print("WTF")
            attcanvas.ax.fill_between([xmin,  fp[0]], [ap, ap], ymax, facecolor=TEMPLATE_FACE_COLOR, edgecolor=TEMPLATE_EDGE_COLOR, hatch='\\', linewidth=0)
            attcanvas.ax.fill_between([fp[1], xmax ], [ap, ap], ymax, facecolor=TEMPLATE_FACE_COLOR, edgecolor=TEMPLATE_EDGE_COLOR, hatch='\\', linewidth=0)
            attcanvas.ax.fill_between([fa[0], fa[1]], [aa, aa], 0, facecolor=TEMPLATE_FACE_COLOR, edgecolor=TEMPLATE_EDGE_COLOR, hatch='\\', linewidth=0)
            attcanvas.ax.set_ylim([0, ymax])
        
        elif filtds.origin.filter_type == Filter.GROUP_DELAY:
            frg = filtds.origin.wrg * W_TO_F
            xmax = 2 * frg
            xmin = 0

        attcanvas.ax.set_xlim(xmin, xmax)
        fa, ga, pa, gda = filtds.origin.tf_template.getBode(linear=True, start=0.5*xmin, stop=2*xmax, num=15000)
        attline, = attcanvas.ax.plot(fa, -20*np.log10(ga), label = str(filtds.origin))
        fa, ga, pa, gda = filtds.origin.helperFilters.tf_template.getBode(linear=True, start=0.5*xmin, stop=2*xmax, num=15000)
        attline, = attcanvas.ax.plot(fa, -20*np.log10(ga), label = str(filtds.origin.helperFilters))
        attcanvas.ax.legend()
        
        pzcanvas.ax.axis('equal')
        pzcanvas.ax.axhline(0, color="black", alpha=0.1)
        pzcanvas.ax.axvline(0, color="black", alpha=0.1)
        minf, maxf = self.getRelevantFrequencies(z, p)
        zx = z.real
        zy = z.imag
        px = p.real
        py = p.imag
        zeroes = pzcanvas.ax.scatter(zx, zy, marker='o')
        poles = pzcanvas.ax.scatter(px, py, marker='x')
        pzcanvas.ax.set_xlabel(PZ_XLABEL)
        pzcanvas.ax.set_ylabel(PZ_YLABEL)
        pzcanvas.ax.set_xlim(left=-maxf*1.2, right=maxf*1.2)
        pzcanvas.ax.set_ylim(bottom=-maxf*1.2, top=maxf*1.2)
        cursor(zeroes, multiple=True, highlight=True).connect("add", self.formatZeroAnnotation)
        cursor(poles, multiple=True, highlight=True).connect("add", self.formatPoleAnnotation)

        attcanvas.draw()
        gaincanvas.draw()
        magcanvas.draw()
        phasecanvas.draw()
        groupdelaycanvas.draw()
        pzcanvas.draw()
        stepcanvas.draw()
        impulsecanvas.draw()

        self.updateStagePlots()

    def updateFilterStages(self):
        self.stages_list.clear()
        self.zeros_list.clear()
        self.poles_list.clear()
        self.remaining_gain_text.clear()
        if self.selected_dataset_data.type == 'filter':
            self.new_stage_btn.setEnabled(True)
            self.remove_stage_btn.setEnabled(True)
            zeros, poles = self.selected_dataset_data.origin.tf.getZP(SHOW_PZ_IN_HZ)
            for p in poles:
                qlwt = QListWidgetItem()
                qlwt.setData(Qt.UserRole, p)
                qlwt.setText(str(p))
                if(p not in np.array(self.selected_dataset_data.origin.remainingPoles)*SING_B_TO_F):
                    qlwt.setFlags(Qt.ItemFlag.NoItemFlags)
                self.poles_list.addItem(qlwt)
            for z in zeros:
                qlwt = QListWidgetItem()
                qlwt.setData(Qt.UserRole, z)
                qlwt.setText(str(z))
                if(z not in np.array(self.selected_dataset_data.origin.remainingZeros)*SING_B_TO_F):
                    qlwt.setFlags(Qt.ItemFlag.NoItemFlags)
                self.zeros_list.addItem(qlwt)
            for implemented_stage in self.selected_dataset_data.origin.stages:
                qlwt = QListWidgetItem()
                qlwt.setData(Qt.UserRole, Dataset(origin=implemented_stage))
                qlwt.setText(stage_to_str(implemented_stage))
                self.stages_list.addItem(qlwt)
            self.remaining_gain_text.setText(str(self.selected_dataset_data.origin.remainingGain))
            self.stage_gain_box.setValue(self.selected_dataset_data.origin.remainingGain)
        else:  
            self.new_stage_btn.setEnabled(False)
            self.remove_stage_btn.setEnabled(False)
        
        self.stages_list.setCurrentRow(self.stages_list.count() - 1)

    def stage_sel_changed(self):
        selected_pol_indexes = [sel.index for sel in self.stageCursorPol.selections]
        selected_zer_indexes = [sel.index for sel in self.stageCursorZer.selections]
        
        for x in range(self.poles_list.count()):
            pole = self.poles_list.item(x)
            poledata = pole.data(Qt.UserRole)
            if(pole.isSelected()):
                if(x not in selected_pol_indexes):
                    sel = Selection(
                        artist=self.splot_fpz.canvas.ax,
                        target_=[poledata.real, poledata.imag],
                        index=self.poles_list.row(pole),
                        dist=0,
                        annotation=None,
                        extras=[]
                    )
                    self.stageCursorPol.add_selection(sel)
            else:
                if(x in selected_pol_indexes):
                    for sel in self.stageCursorPol.selections:
                        if(sel.index == x):
                            self.stageCursorPol.remove_selection(sel)

        for x in range(self.zeros_list.count()):
            zero = self.zeros_list.item(x)
            zerodata = zero.data(Qt.UserRole)
            if(zero.isSelected()):
                if(x not in selected_zer_indexes):
                    sel = Selection(
                        artist=self.splot_fpz.canvas.ax,
                        target_=[zerodata.real, zerodata.imag],
                        index=self.zeros_list.row(zero),
                        dist=0,
                        annotation=None,
                        extras=[]
                    )
                    self.stageCursorZer.add_selection(sel)
            else:
                if(x in selected_zer_indexes):
                    for sel in self.stageCursorZer.selections:
                        if(sel.index == x):
                            self.stageCursorZer.remove_selection(sel)

                
    def updateSelectedPolesFromPlot(self, s):
        self.poles_list.blockSignals(True)
        selected_pole_indexes = [sel.index for sel in self.stageCursorPol.selections]
        for x in range(self.poles_list.count()):
            self.poles_list.item(x).setSelected(x in selected_pole_indexes)
        self.poles_list.blockSignals(False)

    def updateSelectedZerosFromPlot(self, s):
        self.zeros_list.blockSignals(True)
        selected_zero_indexes = [sel.index for sel in self.stageCursorZer.selections]
        for x in range(self.zeros_list.count()):
            self.zeros_list.item(x).setSelected(x in selected_zero_indexes)
        self.zeros_list.blockSignals(False)


    def addFilterStage(self):
        selected_poles = [x.data(Qt.UserRole) for x in self.poles_list.selectedIndexes()]
        selected_zeros = [x.data(Qt.UserRole) for x in self.zeros_list.selectedIndexes()]
        selected_gain = self.stage_gain_box.value()

        selected_poles_idx = [x.row() for x in self.poles_list.selectedIndexes()]
        selected_zeros_idx = [x.row() for x in self.zeros_list.selectedIndexes()]
        selected_poles_idx.sort(reverse=True)
        selected_zeros_idx.sort(reverse=True)

        if self.selected_dataset_data.origin.addStage(selected_zeros, selected_poles, selected_gain, SHOW_PZ_IN_HZ):
            for z in selected_zeros_idx:
                self.zeros_list.item(z).setFlags(Qt.ItemFlag.NoItemFlags)
            for p in selected_poles_idx:
                self.poles_list.item(p).setFlags(Qt.ItemFlag.NoItemFlags)
        
            [self.stageCursorPol.remove_selection(sel) for sel in self.stageCursorPol.selections]
            [self.stageCursorZer.remove_selection(sel) for sel in self.stageCursorZer.selections]
            qlwt = QListWidgetItem()
            qlwt.setData(Qt.UserRole, Dataset(origin=self.selected_dataset_data.origin.stages[-1]))
            qlwt.setText(stage_to_str(self.selected_dataset_data.origin.stages[-1]))
            self.stages_list.addItem(qlwt)
            self.remaining_gain_text.setText(str(self.selected_dataset_data.origin.remainingGain))
            self.stage_gain_box.setValue(self.selected_dataset_data.origin.remainingGain)
        
            self.stages_list.setCurrentRow(self.stages_list.count() - 1)
            
            self.updateStagePlots()
        else:
            print('Error al crear STAGE')

    def removeFilterStage(self):
        i = self.stages_list.currentRow()
        if i < 0:
            return

        self.zeros_list.clear()
        self.poles_list.clear()

        self.selected_dataset_data.origin.removeStage(i)
        
        zeros, poles = self.selected_dataset_data.origin.tf.getZP(SHOW_PZ_IN_HZ)
        for p in poles:
            qlwt = QListWidgetItem()
            qlwt.setData(Qt.UserRole, p)
            qlwt.setText(str(p))
            if(p not in np.array(self.selected_dataset_data.origin.remainingPoles)*SING_B_TO_F):
                qlwt.setFlags(Qt.ItemFlag.NoItemFlags)
            self.poles_list.addItem(qlwt)
        for z in zeros:
            qlwt = QListWidgetItem()
            qlwt.setData(Qt.UserRole, z)
            qlwt.setText(str(z))
            if(z not in np.array(self.selected_dataset_data.origin.remainingZeros)*SING_B_TO_F):
                qlwt.setFlags(Qt.ItemFlag.NoItemFlags)
            self.zeros_list.addItem(qlwt)
        self.stages_list.takeItem(i)
        self.remaining_gain_text.setText(str(self.selected_dataset_data.origin.remainingGain))
        self.stages_list.setCurrentRow(self.stages_list.count() - 1)

    def formatPoleAnnotation(self, sel):
        sel.annotation.set_text('Pole {:d}\n{:.2f}+j{:.2f}\nQ={:.2f}'.format(sel.index, sel.target[0], sel.target[1], self.calcQ(sel.target)))

    def formatZeroAnnotation(self, sel):
        if(True or sel.target[0] == 0 and sel.target[1] == 0):
            sel.annotation.set_text('Zero {:d}\n{:.2f}+j{:.2f}'.format(sel.index, sel.target[0], sel.target[1]))
        else:
            sel.annotation.set_text('Zero {:d}\n{:.2f}+j{:.2f}\nQ={:.2f}'.format(sel.index, sel.target[0], sel.target[1], self.calcQ(sel.target)))

    def calcQ(self, singRe, singIm):
        return self.calcQ(singRe + singIm*1j)

    def calcQ(self, sing):
        if(isinstance(sing, (list, tuple, np.ndarray))):
            sing = sing[0] + sing[1]*1j
        if(sing.real == 0):
            return inf
        elif(sing.real > 0):
            return -1
        else:
            return np.abs(sing)/(- 2 * sing.real)

    def updateStagePlots(self):
        spzcanvas = self.splot_pz.canvas
        sgaincanvas = self.splot_sgain.canvas
        sphasecanvas = self.splot_sphase.canvas
        tgaincanvas = self.splot_tgain.canvas
        tphasecanvas = self.splot_tphase.canvas
        tpzcanvas = self.splot_tpz.canvas
        fpzcanvas = self.splot_fpz.canvas

        self.condition_canvas(spzcanvas, PZ_XLABEL, PZ_YLABEL)
        self.condition_canvas(fpzcanvas, PZ_XLABEL, PZ_YLABEL)
        self.condition_canvas(tpzcanvas, PZ_XLABEL, PZ_YLABEL)
        self.condition_canvas(sgaincanvas, 'Frecuencia [Hz]', 'Magnitud [dB]', 'log')
        self.condition_canvas(sphasecanvas, 'Frecuencia [Hz]', 'Fase [$^o$]', 'log')
        self.condition_canvas(tgaincanvas, 'Frecuencia [Hz]', 'Magnitud [dB]', 'log')
        self.condition_canvas(tphasecanvas, 'Frecuencia [Hz]', 'Fase [$^o$]', 'log')

        zf, pf = self.selected_dataset_data.origin.tf.getZP(SHOW_PZ_IN_HZ)
        mint, maxt = self.getRelevantFrequencies(zf, pf)

        fpzcanvas.ax.axis('equal')
        fpzcanvas.ax.axhline(0, color="black", alpha=0.1)
        fpzcanvas.ax.axvline(0, color="black", alpha=0.1)
        fpzcanvas.ax.set_xlim(left=-maxt*1.2, right=maxt*1.2)
        fpzcanvas.ax.set_ylim(bottom=-maxt*1.2, top=maxt*1.2)

        polcol = [POLE_COLOR if pole in np.array(self.selected_dataset_data.origin.remainingPoles)*SING_B_TO_F else POLE_SEL_COLOR for pole in pf]
        zercol = [ZERO_COLOR if zero in np.array(self.selected_dataset_data.origin.remainingZeros)*SING_B_TO_F else ZERO_SEL_COLOR for zero in zf]
        zeroes_f = fpzcanvas.ax.scatter(zf.real, zf.imag, c=zercol, marker='o')
        poles_f = fpzcanvas.ax.scatter(pf.real, pf.imag, c=polcol, marker='x')
        self.stageCursorZer = cursor(zeroes_f, multiple=True, highlight=True)
        self.stageCursorZer.connect("add", self.formatZeroAnnotation)
        self.stageCursorZer.connect("add", self.updateSelectedZerosFromPlot)
        self.stageCursorZer.connect("remove", self.updateSelectedZerosFromPlot)
        self.stageCursorPol = cursor(poles_f, multiple=True, highlight=True)
        self.stageCursorPol.connect("add", self.formatPoleAnnotation)
        self.stageCursorPol.connect("add", self.updateSelectedPolesFromPlot)
        self.stageCursorPol.connect("remove", self.updateSelectedPolesFromPlot)

        accumulated_ds = Dataset(origin=self.selected_dataset_data.origin.implemented_tf)

        f = accumulated_ds.data[0]['f']
        g = accumulated_ds.data[0]['g']
        ph = accumulated_ds.data[0]['ph']
        tgaincanvas.ax.plot(f, g)
        tphasecanvas.ax.plot(f, ph)
        
        tpzcanvas.ax.axis('equal')
        tpzcanvas.ax.axhline(0, color="black", alpha=0.1)
        tpzcanvas.ax.axvline(0, color="black", alpha=0.1)
        tpzcanvas.ax.set_xlim(left=-maxt*1.2, right=maxt*1.2)
        tpzcanvas.ax.set_ylim(bottom=-maxt*1.2, top=maxt*1.2)
        zt, pt = self.selected_dataset_data.origin.implemented_tf.getZP(SHOW_PZ_IN_HZ)
        
        zeroes_t = tpzcanvas.ax.scatter(zt.real, zt.imag, c='#0000FF', marker='o')
        poles_t = tpzcanvas.ax.scatter(pt.real, pt.imag, c='#FF0000', marker='x')
        cursor(zeroes_t, multiple=True, highlight=True).connect("add", self.formatZeroAnnotation)
        cursor(poles_t, multiple=True, highlight=True).connect("add", self.formatPoleAnnotation)

        fpzcanvas.draw()
        tpzcanvas.draw()
        tgaincanvas.draw()
        tphasecanvas.draw()

        if(self.stages_list.currentItem()):
            accumulated_ds = self.stages_list.currentItem().data(Qt.UserRole)

            f = accumulated_ds.data[0]['f']
            g = accumulated_ds.data[0]['g']
            ph = accumulated_ds.data[0]['ph']
            z, p = accumulated_ds.origin.getZP(SHOW_PZ_IN_HZ)

            gline, = sgaincanvas.ax.plot(f, g)
            phline, = sphasecanvas.ax.plot(f, ph)

            (min, max) = self.getRelevantFrequencies(z, p)
            spzcanvas.ax.axis('equal')
            spzcanvas.ax.axhline(0, color="black", alpha=0.1)
            spzcanvas.ax.axvline(0, color="black", alpha=0.1)
            spzcanvas.ax.set_xlim(left=-max*1.2, right=max*1.2)
            spzcanvas.ax.set_ylim(bottom=-max*1.2, top=max*1.2)

            zeroes_f = spzcanvas.ax.scatter(z.real, z.imag, marker='o')
            poles_f = spzcanvas.ax.scatter(p.real, p.imag, marker='x')

            cursor(zeroes_f).connect("add", self.formatZeroAnnotation)
            cursor(poles_f).connect("add", self.formatPoleAnnotation)
        spzcanvas.draw()
        sgaincanvas.draw()
        sphasecanvas.draw()



    def clearCanvas(self, canvas):
        canvas.ax.clear()
        canvas.ax.grid(True, which="both", linestyle=':')

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
        self.ds_caseadd_btn.setVisible(len(self.selected_dataset_data.data) > 1)
        self.ds_info_lb.setText(self.selected_dataset_data.miscinfo)

        #relleno las cajas del filtro
        if(self.selected_dataset_data.type == 'filter'):
            self.populateSelectedFilterDetails()

    
    def populateSelectedFilterDetails(self, index=-2):
        if(index == -2):
            for i, fds in enumerate(self.filters):
                if(fds.origin == self.selected_dataset_data.origin):
                    self.selfil_cb.blockSignals(True)
                    self.stages_selfil_cb.blockSignals(True)
                    self.selfil_cb.setCurrentIndex(i)
                    self.stages_selfil_cb.setCurrentIndex(i)
                    self.selfil_cb.blockSignals(False)
                    self.stages_selfil_cb.blockSignals(False)
                    break
        elif(index != -1):
            if(index == self.selfil_cb.currentIndex()):
                filtds = self.selfil_cb.currentData()
                self.stages_selfil_cb.blockSignals(True)
                self.stages_selfil_cb.setCurrentIndex(index)
                self.stages_selfil_cb.blockSignals(False)
            else:
                filtds = self.stages_selfil_cb.currentData()
                self.selfil_cb.blockSignals(True)
                self.selfil_cb.setCurrentIndex(index)
                self.selfil_cb.blockSignals(False)
            if(self.selected_dataset_data.origin != filtds.origin):
                for x in range(self.dataset_list.count()):
                    item = self.dataset_list.item(x)
                    ds = item.data(Qt.UserRole)
                    if(ds.origin == filtds.origin):
                        self.dataset_list.setCurrentRow(x)
                        self.populateSelectedDatasetDetails(item, None)
                        return
        else:
            return
        self.filtername_box.setText(self.selected_dataset_data.title)
        self.tipo_box.setCurrentIndex(self.selected_dataset_data.origin.filter_type)
        self.aprox_box.setCurrentIndex(self.selected_dataset_data.origin.approx_type)
        self.compareapprox_cb.setCurrentIndex(self.selected_dataset_data.origin.helper_approx)
        self.comp_N_box.setValue(self.selected_dataset_data.origin.helper_N)
        self.gain_box.setValue(self.selected_dataset_data.origin.gain)
        self.aa_box.setValue(self.selected_dataset_data.origin.aa_dB)
        self.ap_box.setValue(self.selected_dataset_data.origin.ap_dB)
        self.N_label.setText(str(self.selected_dataset_data.origin.N))
        self.N_min_box.setValue(self.selected_dataset_data.origin.N_min)
        self.N_max_box.setValue(self.selected_dataset_data.origin.N_max)
        Qs = [self.calcQ(p) for p in self.selected_dataset_data.origin.tf.getZP()[1]]
        self.max_Q_label.setText("{:.2f}".format(max(Qs)))
        self.define_with_box.setCurrentIndex(self.selected_dataset_data.origin.define_with)
        if self.selected_dataset_data.origin.filter_type in [Filter.BAND_PASS, Filter.BAND_REJECT]:
            self.fp_box.setValue(0)
            self.fa_box.setValue(0)
            
            if self.define_with_box.currentIndex() == Filter.TEMPLATE_FREQS:
                fa = []
                fp = []
                if(self.selected_dataset_data.origin.filter_type == Filter.BAND_PASS):
                    fp = [w * W_TO_F for w in self.selected_dataset_data.origin.wp]
                    fa = [w * W_TO_F for w in self.selected_dataset_data.origin.reqwa]
                else:
                    fp = [w * W_TO_F for w in self.selected_dataset_data.origin.reqwp]
                    fa = [w * W_TO_F for w in self.selected_dataset_data.origin.wa]
                self.fa_min_box.setValue(fa[0])
                self.fa_max_box.setValue(fa[1])
                self.fp_min_box.setValue(fp[0])
                self.fp_max_box.setValue(fp[1])
            else:
                self.bw_max_box.setValue(self.selected_dataset_data.origin.bw[1])
                self.bw_min_box.setValue(self.selected_dataset_data.origin.bw[0])
                self.f0_box.setValue(self.selected_dataset_data.origin.w0 * W_TO_F)
        elif self.selected_dataset_data.origin.filter_type in [Filter.LOW_PASS, Filter.HIGH_PASS]:
            self.fp_box.setValue(self.selected_dataset_data.origin.wp * W_TO_F)
            self.fa_box.setValue(self.selected_dataset_data.origin.wa * W_TO_F)
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
        self.updateFilterStages()
        self.updateFilterPlots()
        self.updateFilterParametersAvailable()

    def updateSelectedDatasetName(self):
        new_title = self.ds_title_edit.text()
        self.selected_dataset_widget.setText(new_title)
        self.selected_dataset_data.title = new_title
        if(self.selected_dataset_data.type == 'filter'):
            self.selfil_cb.setItemText(self.selfil_cb.currentIndex(), new_title)
            self.stages_selfil_cb.setItemText(self.stages_selfil_cb.currentIndex(), new_title)
            self.filtername_box.setText(new_title)

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
        self.saveFile(True)
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
        self.saveFile(True)
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

    def newFile(self):
        self.droppedFiles = []
        self.datasets = []
        self.datalines = []
        self.selected_dataset_widget = {}
        self.selected_dataline_widget = {}
        self.selected_dataset_data = {}
        self.selected_dataline_data = {}
        self.zpWindow = type(ZPWindow, (), {})()
        self.updateAll()
    
    def saveFile(self, noprompt=False):
        if(noprompt):
            filename = 'temp.fto'
        else:
            filename, _ = QFileDialog.getSaveFileName(self,"Save File", "","Filter tool file (*.fto)")
        if(not filename): return
        with open(filename, 'wb') as f:
            flat_plots_canvas = [item.canvas for sublist in self.plots_canvases for item in sublist]
            plots_data = []
            for canv in flat_plots_canvas:
                plots_data.append(canv.get_properties())
            general_config = {
                'labelsize_sb': self.plt_labelsize_sb.value(),
                'legendsize_sb': self.plt_legendsize_sb.value(),
                'ticksize_sb': self.plt_ticksize_sb.value(),
                'titlesize_sb': self.plt_titlesize_sb.value(),
                'legendpos': self.plt_legendpos.currentIndex(),
                'grid': self.plt_grid.isChecked(),
                'marginx': self.plt_marginx.value(),
                'marginy': self.plt_marginy.value()      
            }
            d = [self.datasets, self.datalines, plots_data, general_config]
            pickle.dump(d, f, pickle.HIGHEST_PROTOCOL)

    def loadFile(self):
        filename, _ = QFileDialog.getOpenFileName(self,"Select files", "","Filter tool file (*.fto)")
        if(not filename): return
        with open(filename, 'rb') as f:
            f.seek(0)
            self.datasets, self.datalines, plotdata, general_config = pickle.load(f)            
            self.plt_labelsize_sb.setValue(general_config['labelsize_sb'])
            self.plt_legendsize_sb.setValue(general_config['legendsize_sb'])
            self.plt_ticksize_sb.setValue(general_config['ticksize_sb'])
            self.plt_titlesize_sb.setValue(general_config['titlesize_sb'])
            self.plt_legendpos.setCurrentIndex(general_config['legendpos'])
            self.plt_grid.setChecked(general_config['grid'])
            self.plt_marginx.setValue(general_config['marginx'])
            self.plt_marginy.setValue(general_config['marginy'])  
            acc = 0   
            for s in self.plots_canvases:
                for p in s:
                    p.canvas.restore_properties(plotdata[acc])
                    acc += 1
            for ds in self.datasets:
                qlwt = QListWidgetItem()
                qlwt.setData(Qt.UserRole, ds)
                qlwt.setText(ds.title)
                self.dataset_list.addItem(qlwt)
                for dl in ds.datalines:
                    qlwt = QListWidgetItem()
                    qlwt.setData(Qt.UserRole, dl)
                    qlwt.setText(dl.name)
                    self.dataline_list.addItem(qlwt)
                if(ds.type == 'filter'):
                    self.filters.append(ds)
                    self.selfil_cb.blockSignals(True)
                    self.stages_selfil_cb.blockSignals(True)
                    self.selfil_cb.addItem(ds.title, ds)
                    self.stages_selfil_cb.addItem(ds.title, ds)
                    self.selfil_cb.blockSignals(False)
                    self.stages_selfil_cb.blockSignals(False)

            self.dataset_list.setCurrentRow(self.dataset_list.count() - 1)
            self.updateAll()
    
    def updateAll(self):
        self.updatePlots()
        self.updateSelectedDataline()
        self.updateFilterParametersAvailable()
    
    def getRelevantFrequencies(self, zeros, poles):
        singularitiesNorm = np.append(np.abs(zeros), np.abs(poles))
        singularitiesNormWithoutZeros = singularitiesNorm[singularitiesNorm!=0]
        if(len(singularitiesNormWithoutZeros) == 0):
            return (1, 1)
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