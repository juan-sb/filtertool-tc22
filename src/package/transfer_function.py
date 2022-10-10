import sympy as sym
import scipy.signal as signal
import numpy as np

class TFunction():
    def __init__(self, *args):
        self.s = sym.Symbol('s')
        self.tf_object = {}
        
        if(len(args) == 2):
            self.N, self.D = args[0], args[1]
            self.z, self.p, self.k = signal.tf2zpk(self.N, self.D)
            self.tf_object = signal.TransferFunction(self.N, self.D)
        if(len(args) == 3):
            self.z, self.p, self.k = args[0], args[1], args[2]
            self.N = np.poly1d(z, r=True) * k
            self.D = np.poly1d(p, r=True)
            self.tf_object = signal.ZerosPolesGain(self.z, self.p, self.k)

    def setExpression(self, txt):
        try:
            self.tf_object = sym.parsing.sympy_parser.parse_expr(txt, transformations = 'all')
            self.tf_object = sym.simplify(self.tf_object)
            self.tf_object = sym.fraction(self.tf_object)
            if self.s not in self.tf_object[0].free_symbols:
                self.N = np.array(self.tf_object[0].evalf(), dtype=float)
            else:
                self.N = np.array(sym.Poly(self.tf_object[0]).all_coeffs(), dtype=float)

            if self.s not in self.tf_object[1].free_symbols:
                self.D = np.array(self.tf_object[1].evalf(), dtype=float)
            else:
                self.D = np.array(sym.Poly(self.tf_object[1]).all_coeffs(), dtype=float)

            self.tf_object = signal.TransferFunction(self.N, self.D)
            self.z, self.p, self.k = signal.tf2zpk(self.N, self.D)

            return True
        except:
            return False
    
    def at(self, s):
        return self.N(s) / self.D(s)
    
    def gd_at(self, w0):
        w = np.linspace(w0*0.9, w0*1.1, 1000)
        w, h = signal.freqs(self.N, self.D, w)
        index = np.where(w >= w0)[0][0]
        return (-np.diff(np.angle(h)) / np.diff(w))[index]
        
    def getZP(self):
        return self.tf_object.zeros, self.tf_object.poles

    def getBode(self, start=-2, stop=9, num=2222):
        ws = np.logspace(start, stop, num)
        w, g, ph = signal.bode(self.tf_object, w=ws)
        gd = - np.diff(ph) / np.diff(w)
        f = w / (2 * np.pi)
        return f, np.power(10, g/20), ph, gd

    def getLatex(self, txt):
        return sym.latex(sym.parsing.sympy_parser.parse_expr(txt, transformations = 'all'))