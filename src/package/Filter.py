import sys
import traceback
import math
from src.package.transfer_function import TFunction
import scipy.signal as signal
import scipy.special as special
import numpy as np
import sympy as sym

pi = np.pi

MAX_ORDER = 50
LOW_PASS, HIGH_PASS, BAND_PASS, BAND_REJECT, GROUP_DELAY = range(5)
BUTTERWORTH, CHEBYSHEV, CHEBYSHEV2, CAUER, LEGENDRE, BESSEL, GAUSS = range(7)
TEMPLATE_FREQS, F0_BW = range(2)

def get_Leps(n, eps):
    k = int(n / 2 - 1) if (n % 2 == 0) else int((n - 1) / 2)
        
    a = []
    for i in range(k + 1):
        if n % 2 == 0:
            if k % 2 != 0:
                if i == 0:
                    a.append(1 / (np.sqrt(((k + 1) * (k + 2)))))
                elif i % 2 == 0:
                    a.append((2 * i + 1) * a[0])
                else:
                    a.append(0)
            else:
                if i == 1:
                    a.append(3 / (np.sqrt((k + 1) * (k + 2))))
                elif i % 2 == 0:
                    a.append(0)
                else:
                    a.append((2 * i + 1) * a[1] / 3)
        else:
            if i == 0:
                a.append(1 / (np.sqrt(2) * (k + 1)))
            else:
                a.append((2 * i + 1) * a[0])
    
    sum_prod_pol = np.poly1d([0])
    for i in range(len(a)):
        sum_prod_pol += a[i] * special.legendre(i)
    
    sum_prod_pol **= 2
    if n % 2 == 0:
        sum_prod_pol *= np.poly1d([1, 1]) #multiplico por x+1
    
    sum_prod_pol = np.polyint(sum_prod_pol) # primitiva
    sum_prod_pol = sum_prod_pol(np.poly1d([2, 0, -1])) - sum_prod_pol(-1) # evalúo
    return np.poly1d([1]) + sum_prod_pol*eps*eps

def select_roots(p):
    roots = np.roots(p)*(-1j) #vuelvo desde w al dominio de s
    valid_roots = []
    for root in roots:
        if root.real <= 0:
            valid_roots.append(root)
    return valid_roots

def is_conjugate(z1, z2):
    return np.isclose(np.imag(z1), -np.imag(z2), rtol=1e-5)

def is_complex(z):
    return not np.isclose(np.imag(z), 0, rtol=1e-5)

def is_equal(z1, z2):
    return abs(np.real(z1) - np.real(z2)) + abs(np.imag(z1) - np.imag(z2)) < 1e-5


class AnalogFilter():
    def __init__(self, **kwargs):
        self.tf = TFunction()
        self.tf_norm = TFunction()
        for k, v in kwargs.items():
            setattr(self, k, v) #Seteo todos los atributos de 1
        self.stages = []
        self.remainingZeros = []
        self.remainingPoles = []
        self.remainingGain = np.nan
        
    def validate(self):
        try:
            assert self.N_max <= MAX_ORDER
            assert self.N_min >= 1
            assert self.N_min <= self.N_max

            if self.filter_type == LOW_PASS:
                assert self.gp_dB > self.ga_dB
                assert self.wp < self.wa

            if self.filter_type == HIGH_PASS:
                assert self.gp_dB > self.ga_dB
                assert self.wp > self.wa

            if self.filter_type == BAND_PASS:
                assert self.gp_dB > self.ga_dB 
                if self.define_with == TEMPLATE_FREQS:
                    assert self.wa[0] < self.wp[0]
                    assert self.wp[0] < self.wp[1]
                    assert self.wp[1] < self.wa[1]
                    self.w0 = np.sqrt(self.wp[0]*self.wp[1]) # me quedo con las frecuencias centrales
                    self.bw[0] = self.wp[1] - self.wp[0] # y los anchos de banda
                    self.bw[1] = self.wa[1] - self.wa[0]
                elif self.define_with == F0_BW:
                    assert self.bw[0] < self.bw[1]
                    self.wp[0] = 0.5 * (-self.bw[0] + np.sqrt(self.bw[0]**2 + 4*(self.w0**2))) #defino las frecuencias centrales a partir del ancho de banda
                    self.wp[1] = self.wp[0] + self.bw[0]
                self.wa[0] = 0.5 * (-self.bw[1] + np.sqrt(self.bw[1]**2 + 4*(self.w0**2))) #defino las frecuencias de afuera tal que haya simetría geométrica
                self.wa[1] = self.wa[0] + self.bw[1]

            if self.filter_type == BAND_REJECT:
                assert self.gp_dB > self.ga_dB 
                if self.define_with == TEMPLATE_FREQS:
                    assert self.wp[0] < self.wa[0]
                    assert self.wa[0] < self.wa[1]
                    assert self.wa[1] < self.wp[1]
                    self.w0 = np.sqrt(self.wa[0]*self.wa[1]) # me quedo con las frecuencias centrales
                    self.bw[0] = self.wa[1] - self.wa[0] # y los anchos de banda
                    self.bw[1] = self.wp[1] - self.wp[0]
                elif self.define_with == F0_BW:
                    assert self.bw[0] < self.bw[1]
                    self.wa[0] = 0.5 * (-self.bw[0] + np.sqrt(self.bw[0]**2 + 4*(self.w0**2))) #defino las frecuencias centrales a partir del ancho de banda
                    self.wa[1] = self.wa[0] + self.bw[0]
                self.wp[0] = 0.5 * (-self.bw[1] + np.sqrt(self.bw[1]**2 + 4*(self.w0**2))) #defino las frecuencias de afuera tal que haya simetría geométrica
                self.wp[1] = self.wp[0] + self.bw[1]

            if self.filter_type == GROUP_DELAY:
                assert self.gamma > 0 and self.gamma < 100
                assert self.tau0 > 0
                assert self.wrg > 0

            self.compute_normalized_parameters(init=True)
            self.tf = None
            self.tf_norm = None
            self.get_tf_norm()
            assert self.tf_norm
            self.compute_denormalized_parameters()
            self.resetStages()
            
        except:
            a, err, tb = sys.exc_info()
            tb_info = traceback.extract_tb(tb)
            err_msg = ''
            for tb_item in tb_info:
                err_msg += tb_item.filename + ' - ' + tb_item.line + '\n'
            err_msg += str(a) + ' ' + str(err) + '\n'
            print(err_msg)
            return False, err_msg
        return True, "OK"

    def get_tf_norm(self):
            
            if self.approx_type == BUTTERWORTH:
                self.N, self.wc = signal.buttord(1, self.wan, -self.gp_dB, -self.ga_dB, analog=True)
                assert self.N <= self.N_max
                if self.N < self.N_min:
                    self.N = self.N_min
                z, p, k = signal.butter(self.N, self.wc, analog=True, output='zpk')
                self.tf_norm = TFunction(z, p, k)

            elif self.approx_type == CHEBYSHEV:
                self.N, self.wc = signal.cheb1ord(1, self.wan, -self.gp_dB, -self.ga_dB, analog=True)
                assert self.N <= self.N_max
                if self.N < self.N_min:
                    self.N = self.N_min
                z, p, k = signal.cheby1(self.N, -self.gp_dB, self.wc, analog=True, output='zpk')
                self.tf_norm = TFunction(z, p, k)

            elif self.approx_type == CHEBYSHEV2:
                self.N, self.wc = signal.cheb2ord(1, self.wan, -self.gp_dB, -self.ga_dB, analog=True)
                assert self.N <= self.N_max
                if self.N < self.N_min:
                    self.N = self.N_min
                z, p, k = signal.cheby2(self.N, -self.ga_dB, self.wc, analog=True, output='zpk')
                self.tf_norm = TFunction(z, p, k)
            
            elif self.approx_type == CAUER:
                self.N, self.wc = signal.ellipord(1, self.wan, -self.gp_dB, -self.ga_dB, analog=True)
                assert self.N <= self.N_max
                if self.N < self.N_min:
                    self.N = self.N_min
                z, p, k = signal.ellip(self.N, -self.gp_dB, -self.ga_dB, self.wc, analog=True, output='zpk')
                self.tf_norm = TFunction(z, p, k)
            
            
            elif self.approx_type == LEGENDRE:
                self.N = self.N_min
                eps = np.sqrt(((10 ** (-0.1 * self.gp_dB)) - 1))

                while True:
                    L_eps = get_Leps(self.N, eps)
                    z = []
                    p = select_roots(L_eps)
                    p0 = np.prod(p) * (1 if self.N % 2 == 0 else -1) #en N tengo N polos y yo quiero obtener el producto de los polos negados para normalizar
                    tf2 = TFunction(z, p, p0)
                    #wmin, tf2_wmin = tf2.optimize(0.1, 1)
                    #wmax, tf2_wmax = tf2.optimize(0.1, 1, True)
                    #wmax2, tf2_wmax2 = tf2.optimize(self.wan, 10*self.wan, True)
                    #no anda la optimización, evalúo sólo en 1 y wan
                    tf2_wmin = tf2.at(1)
                    tf2_wmax = tf2.at(wan)
                    if tf2_wmin >= self.gp and tf2_wmax <= self.ga:
                        self.tf_norm = TFunction(z, p, p0)
                        break
                    self.N += 1
                    assert self.N <= self.N_max 
            
            elif self.approx_type == BESSEL:
                self.N = self.N_min
                while True:
                    z, p, k = signal.bessel(self.N, 1, analog=True, output='zpk', norm='delay') #produce un delay de 1/1 seg (cambiar el segundo parámetro)
                    tf2 = TFunction(z, p, k)
                    if 1 - tf2.gd_at(self.wrg_n) <= self.gamma/100: #si el gd es menor-igual que el esperado, estamos
                        self.tf_norm = TFunction(z, p, k)
                        break
                    self.N += 1
                    assert self.N <= self.N_max

            if self.approx_type == GAUSS:
                self.N = 1
                gauss_poly = [1, 0, 1] # producirá un delay de 1 segundo
                fact_prod = 1
                while True:
                    if self.N >= self.N_min:
                        z = []
                        p = select_roots(np.poly1d(gauss_poly))
                        p0 = np.prod(p)
                        tf2 = TFunction(z, p, p0)
                        if 1 - tf2.gd_at(self.wrg_n) <= self.gamma/100: #si el gd es menor-igual que el esperado, estamos
                            g0 = tf2.gd_at(0)                       
                            p = [r * g0 for r in p]
                            self.tf_norm = TFunction(z, p, p0)
                            break
                    self.N += 1
                    assert self.N <= self.N_max
                    fact_prod *= self.N
                    gauss_poly.insert(0, 0)
                    gauss_poly.insert(0, 1/fact_prod)
    
    def compute_normalized_parameters(self, init=False):
        if self.filter_type < GROUP_DELAY:
            self.ga = np.power(10, (self.ga_dB / 20))
            self.gp = np.power(10, (self.gp_dB / 20))

            if self.filter_type == LOW_PASS:
                self.wan = self.wa / self.wp
            elif self.filter_type == HIGH_PASS:
                self.wan = self.wp / self.wa
            elif self.filter_type == BAND_PASS:
                self.wan = (self.wa[1] - self.wa[0]) / (self.wp[1] - self.wp[0])
            elif self.filter_type == BAND_REJECT:
                self.wan = (self.wp[1] - self.wp[0]) / (self.wa[1] - self.wa[0])
        elif self.filter_type == GROUP_DELAY and init:
            self.wrg_n = self.wrg * self.tau0
        else: #debería calcular las ganancias del group delay, pero cuáles si nunca definí la plantilla?
            pass
    
    def compute_denormalized_parameters(self):
        # no es necesario (por ahora) desnormalizar las ganancias
        s = sym.symbols('s')
        h_norm = sym.Poly(self.tf_norm.N, s)/sym.Poly(self.tf_norm.D, s)
        
        if self.filter_type == LOW_PASS:
            transformation = s / self.wp
        elif self.filter_type == HIGH_PASS:
            transformation = self.wp / s
        elif self.filter_type == BAND_PASS:
            transformation = (self.w0 / (self.wp[1] - self.wp[0])) * ((s / self.w0) + (self.w0 / s))
        elif self.filter_type == BAND_REJECT:
            transformation = ((self.wa[1] - self.wa[0]) / self.w0) / ((s / self.w0) + (self.w0 / s))
        elif self.filter_type == GROUP_DELAY:
            transformation = s * self.tau0
        
        h_denorm = h_norm.subs(s, transformation) 
        h_denorm = sym.simplify(h_denorm)
        h_denorm = sym.fraction(h_denorm)

        N = sym.Poly(h_denorm[0]).all_coeffs() if (s in h_denorm[0].free_symbols) else [h_denorm[0].evalf()]
        D = sym.Poly(h_denorm[1]).all_coeffs() if (s in h_denorm[1].free_symbols) else [h_denorm[1].evalf()]

        self.tf = TFunction([a * self.gain for a in N], D)

    def resetStages(self):
        self.remainingGain = self.gain
        self.remainingZeros = self.tf.z.tolist()
        self.remainingPoles = self.tf.p.tolist()
        self.stages = []

    def addStage(self, z_arr, p_arr, gain):
        if len(z_arr) > 2 or len(p_arr) > 2 or len(p_arr) == 0 or len(z_arr) > len(p_arr):
            return False
        
        if is_complex(p_arr[0]) and (len(p_arr) < 2 or not is_conjugate(p_arr[0], p_arr[1])):
            return False
        if(len(z_arr) > 0):
            if is_complex(z_arr[0]) and (len(z_arr) < 2 or not is_conjugate(z_arr[0], z_arr[1])):
                return False

        newRemainingZeros = len(self.remainingZeros) - len(z_arr)
        newRemainingPoles = len(self.remainingPoles) - len(p_arr)

        if newRemainingZeros > newRemainingPoles:
            return False

        append_gain = self.remainingGain if newRemainingPoles == 0 else gain
        a = 1 #lo voy a usar para normalizar, los zpk que da numpy no vienen normalizados
        for zero in z:
            a *= -zero
        for pole in p:
            a /= -pole
        self.stages.append(TFunction(z_arr, p_arr, append_gain/a))
        self.remainingGain /= append_gain
        for z in z_arr:
            self.remainingZeros.remove(z)
        for p in p_arr:
            self.remainingPoles.remove(p)
        return True
        

    def removeStage(self, i):
        self.remainingGain *= self.stages[i].k
        for z in self.stages[i].z:
            self.remainingZeros.append(z)
        for p in self.stages[i].p:
            self.remainingPoles.append(p)
        self.stages.pop(i)

