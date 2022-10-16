import sympy as sym
import scipy.signal as signal
from scipy.optimize import basinhopping
import numpy as np

# Evaluate a polynomial in reverse order using Horner's Rule,
# for example: a3*x^3+a2*x^2+a1*x+a0 = ((a3*x+a2)x+a1)x+a0
def poly_at(p, x):
    total = 0
    for a in p:
        total = total*x+a
    return total

def poly_diff_at(p, x):
    total = 0
    j = len(p) - 1
    for a in p:
        prod = 1
        for i in range(j-1):
            prod *= x
        total += a * j * prod
        j -= 1
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

            N = sym.Poly(expression[0]).all_coeffs() if (self.s in expression[0].free_symbols) else [expression[0].evalf()]
            D = sym.Poly(expression[1]).all_coeffs() if (self.s in expression[1].free_symbols) else [expression[1].evalf()]
            self.setND(N, D)
            return True
        except:
            return False

    def setND(self, N, D):
        self.N, self.D = np.array(N, dtype=np.complex128), np.array(D, dtype=np.complex128)
        self.z, self.p, self.k = signal.tf2zpk(self.N, self.D) #aún tengo que chequear si signal devuelve la ganancia bien (normalizada)
        self.tf_object = signal.TransferFunction(self.N, self.D)
    
    def getND(self):
        return self.N, self.D

    def setZPK(self, z, p, k):
        self.z, self.p = np.array(z, dtype=np.complex128), np.array(p, dtype=np.complex128)
        a = 1
        for zero in z:
            a *= zero
        for pole in p:
            a /= pole
        self.k = k/a
        self.N, self.D = signal.zpk2tf(self.z, self.p, self.k)
        self.tf_object = signal.TransferFunction(self.N, self.D)
    
    def getZPK(self):
        return self.z, self.p, self.k

    def at(self, s):
        return poly_at(self.N, s) / poly_at(self.D, s)

    def deriv_at(self, s):
        N = poly_at(self.N, s)
        D = poly_at(self.D, s)
        dN = poly_diff_at(self.N, s)
        dD = poly_diff_at(self.D, s)
        return (dN*D - N*dD)/(D*D)
    
    def minFunctionMod(self, w):
        return abs(self.at(1j*w))
    
    def maxFunctionMod(self, w):
        return -abs(self.at(1j*w))
    
    #como ln(H) = ln(G) + j phi --> H'/H = G'/G + j phi'
    def gd_at(self, w0):
        return -np.imag(1j*self.deriv_at(1j*w0)/self.at(1j*w0)) #'1j*..' --> regla de la cadena
        
    def getZP(self):
        return self.z, self.p

    def getBode(self, start=-2, stop=9, num=2222):
        ws = np.logspace(start, stop, num)
        w, g, ph = signal.bode(self.tf_object, w=ws)
        gd = self.gd_at(w) # * 2 *np.p1 --> no hay que hacer regla de cadena porque se achica tmb la escala de w
        f = w / (2 * np.pi)
        return f, np.power(10, g/20), ph, gd

    def optimize(self, start, stop, maximize = False):
        # rewrite the bounds in the way required by L-BFGS-B
        bounds = [(start, stop)]
        w0 = 0.5*(start + stop)

        if not maximize:
            f = lambda w : self.minFunctionMod(w)
        else:
            f = lambda w : self.maxFunctionMod(w)

        # use method L-BFGS-B because the problem is smooth and bounded
        minimizer_kwargs = dict(method="L-BFGS-B", bounds=bounds)
        res = basinhopping(f, w0, minimizer_kwargs=minimizer_kwargs)
        return res.x, (res.fun if not maximize else -res.fun)

    
    def getLatex(self, txt):
        return sym.latex(sym.parsing.sympy_parser.parse_expr(txt, transformations = 'all'))