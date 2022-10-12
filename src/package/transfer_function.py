import sympy as sym
import scipy.signal as signal
import numpy as np

# Evaluate a polynomial in reverse order using Horner's Rule,
# for example: a3*x^3+a2*x^2+a1*x+a0 = ((a3*x+a2)x+a1)x+a0
def poly_at(p, x):
    total = 0
    for a in p:
        total = total*x+a
    return total

class TFunction():
    def __init__(self, *args):
        self.s = sym.Symbol('s')
        self.tf_object = {}

        if(len(args) == 1):
            self.setExpression(args[0])
        if(len(args) == 2):
            self.setND(args[0], args[1])
        if(len(args) == 3):
            self.setZPK(args[0], args[1], args[2])

    def setExpression(self, txt):
        try:
            expression = sym.parsing.sympy_parser.parse_expr(txt, transformations = 'all')
            expression = sym.simplify(expression)
            expression = sym.fraction(expression)

            self.N = sym.Poly(expression[0]).all_coeffs() if (self.s in expression[0].free_symbols) else [expression[0].evalf()]
            self.D = sym.Poly(expression[1]).all_coeffs() if (self.s in expression[1].free_symbols) else [expression[1].evalf()]
            self.z, self.p, self.k = signal.tf2zpk(self.N, self.D)
            self.tf_object = signal.TransferFunction(self.N, self.D)
            return True
        except:
            return False

    def setND(self, N, D):
        self.N, self.D = N, D
        self.z, self.p, self.k = signal.tf2zpk(self.N, self.D)
        self.tf_object = signal.TransferFunction(self.N, self.D)

    def setZPK(self, z, p, k):
        self.z, self.p, self.k = z, p, k
        self.N = np.poly1d(z, r=True).coef * self.k
        self.D = np.poly1d(p, r=True).coef
        self.tf_object = signal.ZerosPolesGain(self.z, self.p, self.k)
    
    def at(self, s):
        return poly_at(self.N, s) / poly_at(self.D, s)
    
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