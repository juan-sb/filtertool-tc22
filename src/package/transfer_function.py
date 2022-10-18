import sympy as sym
import scipy.signal as signal
from scipy.optimize import basinhopping
import numpy as np
from numpy.polynomial import Polynomial

# Evaluate a polynomial in reverse order using Horner's Rule,
# for example: a3*x^3+a2*x^2+a1*x+a0 = ((a3*x+a2)x+a1)x+a0
def poly_at(p, x):
    total = 0
    for a in p:
        total = total*x+a
    return total

class TFunction():
    def __init__(self, *args, normalize=False):
        self.s = sym.Symbol('s')
        self.tf_object = {}

        if(len(args) == 1):
            self.setExpression(args[0], normalize=normalize)
        if(len(args) == 2):
            self.setND(args[0], args[1], normalize=normalize)
        if(len(args) == 3):
            self.setZPK(args[0], args[1], args[2], normalize=normalize)

    def setExpression(self, txt, normalize=False):
        try:
            expression = sym.parsing.sympy_parser.parse_expr(txt, transformations = 'all')
            expression = sym.simplify(expression)
            expression = sym.fraction(expression)

            N = sym.Poly(expression[0]).all_coeffs() if (self.s in expression[0].free_symbols) else [expression[0].evalf()]
            D = sym.Poly(expression[1]).all_coeffs() if (self.s in expression[1].free_symbols) else [expression[1].evalf()]
            self.setND(N, D, normalize=normalize)
            return True
        except:
            return False

    def setND(self, N, D, normalize=False):
        self.z, self.p, self.k = signal.tf2zpk(np.array(N, dtype=np.complex128), np.array(D, dtype=np.complex128))
        self.setZPK(self.z, self.p, self.k, normalize=normalize)
    
    def getND(self):
        return self.N, self.D

    #Nota: signal NO normaliza la transferencia, por lo que k multiplica pero no es la ganancia en s=0
    def setZPK(self, z, p, k, normalize=False):
        self.z, self.p = np.array(z, dtype=np.complex128), np.array(p, dtype=np.complex128)
        self.k = k
        if normalize:
            self.normalize()
        self.N, self.D = signal.zpk2tf(self.z, self.p, self.k)
        self.tf_object = signal.TransferFunction(self.N, self.D)
        self.computedDerivatives = False

    def getZPK(self):
        return self.z, self.p, self.k

    def getDerivatives(self):
        N = Polynomial(np.flip(self.N))
        D = Polynomial(np.flip(self.D))
        self.derivN = np.flip(((N.deriv())*D - N*(D.deriv())).coef)
        self.derivD = np.flip((D * D).coef)
        self.computedDerivatives = True

    def normalize(self):
        a = 1 #lo voy a usar para normalizar, los zpk que da numpy no vienen normalizados
        for zero in self.z:
            a *= -zero
        for pole in self.p:
            a /= -pole
        self.k /= a
        self.computedDerivatives = False
    
    def denormalize(self):
        a = 1
        for zero in self.z:
            a *= -zero
        for pole in self.p:
            a /= -pole
        self.k *= a
        self.computedDerivatives = False

    def at(self, s):
        return poly_at(self.N, s) / poly_at(self.D, s)

    def deriv_at(self, s):
        if not self.computedDerivatives:
            self.getDerivatives()
        return poly_at(self.derivN, s)/poly_at(self.derivD, s)
    
    def minFunctionMod(self, w):
        return abs(self.at(1j*w))
    
    def maxFunctionMod(self, w):
        return -abs(self.at(1j*w))
    
    #como ln(H) = ln(G) + j phi --> H'/H = G'/G + j phi'
    def gd_at(self, w0):
        return -np.imag(1j*self.deriv_at(1j*w0)/self.at(1j*w0)) #'1j*..' --> regla de la cadena
        
    def getZP(self):
        return self.z, self.p

    def getBode(self, start=-2, stop=7, num=3312):
        ws = np.logspace(start, stop, num)
        w, g, ph = signal.bode(self.tf_object, w=ws)
        gd = self.gd_at(w) #/ (2 * np.pi) #--> no hay que hacer regla de cadena porque se achica tmb la escala de w
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

    def appendStage(self, tf):
        self.setZPK(np.append(self.z, tf.z), np.append(self.p, tf.p), self.k*tf.k)

    def removeStage(self, tf):
        self.setZPK([i for i in self.z if i not in tf.z], [i for i in self.p if i not in tf.p], self.k/tf.k)
        
    def getLatex(self, txt):
        return sym.latex(sym.parsing.sympy_parser.parse_expr(txt, transformations = 'all'))