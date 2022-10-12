import sys
import traceback
import math
from src.package.transfer_function import TFunction
import scipy.signal as signal
import scipy.special as special
import numpy as np
import sympy as sp

pi = np.pi

LOW_PASS, HIGH_PASS, BAND_PASS, BAND_REJECT, GROUP_DELAY = range(5)
BUTTERWORTH, CHEBYSHEV, CHEBYSHEV2, CAUER, LEGENDRE, BESSEL, GAUSS = range(7)
TEMPLATE_FREQS, F0_BW = range(2)
filter_types = ['lowpass', 'highpass', 'bandpass', 'bandstop']

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
                    a.append(3 / (((k + 1) * (k + 2)) ** (1 / 2)))
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
    roots = np.roots(p)
    valid_roots = []
    for root in roots:
        if root.real <= 0:
            valid_roots.append(root)
    return valid_roots



class Filter():
    def __init__(self, **kwargs):
        self.limits_x = []
        self.limits_y = []
        self.tf = type('TFunction', (), {})()
        self.tf_norm = type('TFunction', (), {})()
        for k, v in kwargs.items():
            setattr(self, k, v) #Seteo todos los atributos de 1
        
        
    def validate(self):
        try:
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
                assert self.gamma > 0 and self.gamma < 1
                assert self.tau0 > 0
                assert self.wrg > 0

            self.tf = None
            self.compute_normalized_parameters(init=True)
            self.get_filter_tf()
            assert self.tf
            self.compute_denormalized_parameters()
            self.get_template_limits()
            
        except:
            _, _, tb = sys.exc_info()
            tb_info = traceback.extract_tb(tb)
            filename, line, func, text = tb_info[-1]
            return False, text
        return True, "OK"

    def get_filter_tf(self):
            
            if self.approx_type == BUTTERWORTH:
                self.N, self.wc = signal.buttord(1, self.wan, -self.gp_dB, -self.ga_dB, analog=True)
                if self.N > self.N_max:
                    return
                elif self.N < self.N_min:
                    self.N = self.N_min
                z, p, k = signal.butter(self.N, self.wc, analog=True, output='zpk')
                self.tf_norm = TFunction(z, p, k)

            if self.approx_type == CHEBYSHEV:
                self.N, self.wc = signal.cheb1ord(1, self.wan, -self.gp_dB, -self.ga_dB, analog=True)
                if self.N > self.N_max:
                    return
                elif self.N < self.N_min:
                    self.N = self.N_min
                z, p, k = signal.cheby1(self.N, -self.gp_dB, self.wc, analog=True, output='zpk')
                self.tf_norm = TFunction(z, p, k)

            if self.approx_type == CHEBYSHEV2:
                self.N, self.wc = signal.cheb2ord(1, self.wan, -self.gp_dB, -self.ga_dB, analog=True)
                if self.N > self.N_max:
                    return
                elif self.N < self.N_min:
                    self.N = self.N_min
                z, p, k = signal.cheby2(self.N, -self.ga_dB, self.wc, analog=True, output='zpk')
                self.tf_norm = TFunction(z, p, k)
            
            if self.approx_type == CAUER:
                self.N, self.wc = signal.ellipord(1, self.wan, -self.gp_dB, -self.ga_dB, analog=True)
                if self.N > self.N_max:
                    return
                elif self.N < self.N_min:
                    self.N = self.N_min
                z, p, k = signal.ellip(self.N, -self.gp_dB, -self.ga_dB, self.wc, analog=True, output='zpk')
                self.tf_norm = TFunction(z, p, k)
            
            
            if self.approx_type == LEGENDRE:
                self.N = self.N_min
                eps = np.sqrt(((10 ** (-0.1 * self.gp_dB)) - 1))

                while True:
                    L_eps = get_Leps(self.N, eps)
                    z = []
                    p = select_roots(L_eps)
                    tf2 = TFunction(z, p, 1)
                    if abs(tf2.at(1j*self.wan)) < self.ga:
                        self.tf_norm = TFunction(z, p, 1)
                        break
                    self.N += 1
                    if self.N > 25: #excedí el límite
                        break
            
            if self.approx_type == BESSEL:
                self.N = self.N_min
                while True:
                    z, p, k = signal.bessel(self.N, self.wrg_n, analog=True, output='zpk', norm='delay')
                    tf2 = TFunction(z, p, k)
                    if abs(tf2.gd_at(self.wrg_n) - 1) <= self.gamma: #si el gd es menor-igual que el esperado, estamos
                        self.tf_norm = TFunction(z, p, k)
                        break
                    self.N += 1
                    if self.N > 25: #excedí el límite
                        break

            if self.approx_type == GAUSS:
                self.N = 1
                gauss_poly = [-1, 0, 1]
                fact_prod = -1
                while True:
                    if self.N >= self.N_min:
                        z = []
                        p = select_roots(np.poly1d(gauss_poly))
                        tf2 = TFunction(z, p, 1)
                        if abs(tf2.gd_at(self.wrg_n) - 1) <= self.gamma: #si el gd es menor-igual que el esperado, estamos
                            self.tf_norm = TFunction(z, p, 1)
                            break
                    self.N += 1
                    if self.N > 25: #excedí el límite
                        break
                    fact_prod *= -self.N
                    gauss_poly.insert(0, 0)
                    gauss_poly.insert(0, 1/fact_prod)

        
    def get_template_limits(self):
        #### FALTA MULTIPLICAR TODO POR self.gain (es la ganancia 'extra' que se le pone al filtro)
        limits = []
        self.compute_normalized_parameters()
        if self.filter_type == LOW_PASS:
            limits.extend([[1e-9, self.gp], [self.wp, self.gp]]) # Extiende tus límites!
            limits.extend([[self.wp, self.gp], [self.wp, 1e-9]])
            limits.extend([[self.wa, self.ga], [self.wa, 1e9]])
            limits.extend([[self.wa, self.ga], [1e9, self.ga]])
        if self.filter_type == HIGH_PASS:
            limits.extend([[1e-9, self.ga], [self.wa, self.ga]])
            limits.extend([[self.wa, self.ga], [self.wa, 1e9]])
            limits.extend([[self.wp, self.gp], [self.wp, 1e-9]])
            limits.extend([[self.wp, self.gp], [1e9, self.gp]])
        if self.filter_type == BAND_PASS:
            limits.extend([[1e-9, self.ga], [self.wa[0], self.ga]])
            limits.extend([[self.wa[0], self.ga], [self.wa[0], 1e9]])
            limits.extend([[self.wp[0], self.gp], [self.wp[0], 1e-9]])
            limits.extend([[self.wp[0], self.gp], [self.wp[1], self.gp]])

            limits.extend([[self.wp[1], self.gp], [self.wp[1], 1e-9]])
            limits.extend([[self.wa[1], self.ga], [self.wa[1], 1e9]])
            limits.extend([[self.wa[1], self.ga], [1e9, self.ga]])
        if self.filter_type == BAND_REJECT:
            limits.extend([[1e-9, self.gp], [self.wp[0], self.gp]])
            limits.extend([[self.wp[0], self.gp], [self.wp[0], 1e-9]])
            limits.extend([[self.wa[0], self.ga], [self.wa[0], 1e9]])
            limits.extend([[self.wa[0], self.ga], [self.wa[1], self.ga]])

            limits.extend([[self.wa[1], self.ga], [self.wa[1], 1e9]])
            limits.extend([[self.wp[1], self.gp], [self.wa[1], 1e-9]])
            limits.extend([[self.wp[1], self.gp], [1e9, self.gp]])
        
        self.limits_x = [limit[0] for limit in limits]
        self.limits_y = [limit[1] for limit in limits]
    
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
            elif self.filter_type == BAND_STOP:
                self.wan = (self.wp[1] - self.wp[0]) / (self.wa[1] - self.wa[0])
        elif self.filter_type == GROUP_DELAY and init:
            self.wrg_n = self.wrg * self.tau0
        else: #debería calcular las ganancias del group delay, pero cuáles si nunca definí la plantilla?
            pass
    
    def compute_denormalized_parameters(self):
        # no es necesario (por ahora) desnormalizar las ganancias
        s = sym.symbols('s')
        h_norm = sym.Poly(self.tf_norm.N, s)/sym.Poly(self.tf_norm.D, s)

        if self.filter_type == LOW_PASS or self.filter_type == GROUP_DELAY:
            transformation = s / self.wp
        elif self.filter_type == HIGH_PASS:
            transformation = self.wp / s
        elif self.filter_type == BAND_PASS:
            transformation = (self.w0 / (self.wp[1] - self.wp[0])) * ((s / self.w0) + (self.w0 / s))
        elif self.filter_type == BAND_STOP:
            transformation = ((self.wa[1] - self.wa[0]) / self.w0) / ((s / self.w0) + (self.w0 / s))
        
        h_denorm = h_norm(transformation) 
        h_denorm = sym.simplify(h_denorm)
        h_denorm = sym.fraction(h_denorm)

        N = sym.Poly(h_denorm[0]).all_coeffs() if (s in h_denorm[0].free_symbols) else [h_denorm[0].evalf()]
        D = sym.Poly(h_denorm[1]).all_coeffs() if (s in h_denorm[1].free_symbols) else [h_denorm[1].evalf()]

        self.tf = TFunction(N * self.gain, D)
