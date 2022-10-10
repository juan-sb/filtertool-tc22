import sympy
import scipy
import numpy

class TFunction():
    def __init__(self, *args):
        self.s = sympy.Symbol('s')
        self.tf_object = {}
        
        if(len(args) == 2):
            self.tf_object = scipy.signal.TransferFunction(args[0], args[1])
        if(len(args) == 3):
            self.z, self.p, self.k = args[0], args[1], args[2]
            self.N = numpy.poly1d(z, r=True) * k
            self.D = numpy.poly1d(p, r=True)
            self.tf_object = scipy.signal.ZerosPolesGain(self.z, self.p, self.k)

    def setExpression(self, txt):
        try:
            self.tf_object = sympy.parsing.sympy_parser.parse_expr(txt, transformations = 'all')
            self.calcTFunction()
            return True
        except:
            return False

    def calcTFunction(self):
        self.tf_object = sympy.simplify(self.tf_object)
        self.tf_object = sympy.fraction(self.tf_object)

        if self.s not in self.tf_object[0].free_symbols:
            self.N = numpy.array(self.tf_object[0].evalf(), dtype=float)
        else:
            self.N = numpy.array(sympy.Poly(self.tf_object[0]).all_coeffs(), dtype=float)

        if self.s not in self.tf_object[1].free_symbols:
            self.D = numpy.array(self.tf_object[1].evalf(), dtype=float)
        else:
            self.D = numpy.array(sympy.Poly(self.tf_object[1]).all_coeffs(), dtype=float)

        self.tf_object = scipy.signal.TransferFunction(self.N, self.D)
        self.z, self.p, self.k = scipy.signal.tf2zpk(self.N, self.D)
    
    def at(self, s):
        return self.N(s) / self.D(s)
    
    def gd_at(self, w0):
        f, g, ph, gd = self.getBode(start=self.w*0.9, stop=self.w*1.1, num=100)
        w = f * 2 * np.pi
        index = np.where(w >= w0)[0][0]
        return gd[index]
        
    def getZP(self):
        return self.tf_object.zeros, self.tf_object.poles

    def getBode(self, start=-2, stop=9, num=2222):
        ws = numpy.logspace(start, stop, num)
        w, g, ph = scipy.signal.bode(self.tf_object, w=ws)
        gd = - np.diff(ph) / np.diff(w)
        f = w / (2 * numpy.pi)
        return f, numpy.power(10, g/20), ph, gd

    def getLatex(self, txt):
        return sympy.latex(sympy.parsing.sympy_parser.parse_expr(txt, transformations = 'all'))