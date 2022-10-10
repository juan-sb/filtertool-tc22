import sys
import traceback
import math
from src.package.transfer_function import TFunction
import scipy.signal as signal
import scipy.special as special
import numpy as np

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
        self.tf = type('TransferFunction', (), {})()
        self.limits_x = []
        self.limits_y = []
        for k, v in kwargs.items():
            setattr(self, k, v) #Seteo todos los atributos de 1
        
        
    def validate(self):
        try:
            assert self.N_min <= self.N_max
            if self.filter_type == LOW_PASS:
                assert self.gp_dB > self.ga_dB
                assert self.fp < self.fa
            if self.filter_type == HIGH_PASS:
                assert self.gp_dB > self.ga_dB
                assert self.fp > self.fa
            if self.filter_type == BAND_PASS:
                assert self.gp_dB > self.ga_dB 
                if self.define_with == TEMPLATE_FREQS:
                    assert self.fa_min < self.fp_min
                    assert self.fp_min < self.fp_max
                    assert self.fp_max < self.fa_max
                    self.f0 = np.sqrt(self.fp_min*self.fp_max) # me quedo con las frecuencias centrales
                    self.bw_min = self.fp_max - self.fp_min # y los anchos de banda
                    self.bw_max = self.fa_max - self.fa_min
                elif self.define_with == F0_BW:
                    assert self.bw_min < self.bw_max
                    self.fp_min = 0.5 * (-self.bw_min + np.sqrt(self.bw_min**2 + 4*(self.f0**2))) #defino las frecuencias centrales a partir del ancho de banda
                    self.fp_max = self.fp_min + self.bw_min
                self.fa_min = 0.5 * (-self.bw_max + np.sqrt(self.bw_max**2 + 4*(self.f0**2))) #defino las frecuencias de afuera tal que haya simetría geométrica
                self.fa_max = self.fa_min + self.bw_max

            if self.filter_type == BAND_REJECT:
                assert self.gp_dB > self.ga_dB 
                if self.define_with == TEMPLATE_FREQS:
                    assert self.fp_min < self.fa_min
                    assert self.fa_min < self.fa_max
                    assert self.fa_max < self.fp_max
                    self.f0 = np.sqrt(self.fa_min*self.fa_max) # me quedo con las frecuencias centrales
                    self.bw_min = self.fa_max - self.fa_min # y los anchos de banda
                    self.bw_max = self.fp_max - self.fp_min
                elif self.define_with == F0_BW:
                    assert self.bw_min < self.bw_max
                    self.fa_min = 0.5 * (-self.bw_min + np.sqrt(self.bw_min**2 + 4*(self.f0**2))) #defino las frecuencias centrales a partir del ancho de banda
                    self.fa_max = self.fa_min + self.bw_min
                self.fp_min = 0.5 * (-self.bw_max + np.sqrt(self.bw_max**2 + 4*(self.f0**2))) #defino las frecuencias de afuera tal que haya simetría geométrica
                self.fp_max = self.fp_min + self.bw_max

            if self.filter_type == GROUP_DELAY:
                assert self.gamma > 0 and self.gamma < 1
                assert self.filter_type in [LOW_PASS, HIGH_PASS]

            self.tf = None
            self.get_filter_tf()
            assert self.tf
            self.get_template_limits()
            
        except:
            _, _, tb = sys.exc_info()
            tb_info = traceback.extract_tb(tb)
            filename, line, func, text = tb_info[-1]
            return False, text
        return True, "OK"

    def get_filter_tf(self):
        if self.filter_type < GROUP_DELAY:
            wp, wa = None, None
            if self.filter_type == BAND_PASS or self.filter_type == BAND_REJECT:
                wp, wa = [2*pi*self.fp_min, 2*pi*self.fp_max], [2*pi*self.fa_min, 2*pi*self.fa_max]
                wan = max(self.wa, self.wp) / min(self.wa, self.wp)
            else:
                wp, wa = 2*pi*self.fp, 2*pi*self.fa
                self.w0 = np.sqrt(wp[1]*wp[0])
                self.B = abs(wp[1] - wp[0])
                w_test_arr = [abs(wa[1] - wa[0]), abs(wp[1] - wp[0])]
                wan = max(w_test_arr) / min(w_test_arr)
            
            if self.approx_type == BUTTERWORTH:
                self.N, self.wn = signal.buttord(wp, wa, -self.gp_dB, -self.ga_dB, analog=True)
                z, p, k = signal.butter(self.N, self.wn, btype=filter_types[self.filter_type], analog=True, output='zpk')
                self.tf = TFunction(z, p, self.gain)

            if self.approx_type == CHEBYSHEV:
                self.N, self.wn = signal.cheb1ord(wp, wa, -self.gp_dB, -self.ga_dB, analog=True)
                z, p, k = signal.cheby1(self.N, self.rp, self.wn, btype=filter_types[self.filter_type], analog=True, output='zpk')
                self.tf = TFunction(z, p, self.gain)

            if self.approx_type == CHEBYSHEV2:
                self.N, self.wn = signal.cheb2ord(wp, wa, -self.gp_dB, -self.ga_dB, analog=True)
                z, p, k = signal.cheby2(self.N, self.ra, self.wn, btype=filter_types[self.filter_type], analog=True, output='zpk')
                self.tf = TFunction(z, p, self.gain)
            
            if self.approx_type == CAUER:
                self.N, self.wn = signal.ellipord(wp, wa, -self.gp_dB, -self.ga_dB, analog=True)
                z, p, k = signal.ellip(self.N, self.rp, self.ra, self.wn, btype=filter_types[self.filter_type], analog=True, output='zpk')
                self.tf = TFunction(z, p, self.gain)
            
            
            if self.approx_type == LEGENDRE:
                self.compute_normalized_gains()
                self.N = 1
                eps = np.sqrt(((10 ** (self.gp_dB / 10)) - 1))

                while True:
                    L_eps = get_Leps(self.N, eps)
                    z = []
                    p = select_roots(L_eps)
                    tf2 = TFunction(z, p, 1)
                    ok = True
                    for w_pass in wp: #chequeo si cumple con la plantilla normalizada
                        ok = ok and abs(tf2.at(1j*w_pass)) > self.gp
                    for w_ate in wa:
                        ok = ok and abs(tf2.at(1j*w_ate)) < self.ga
                    if ok: #si el gd es menor-igual que el esperado, estamos
                        self.tf = TFunction(z, p, self.gain)
                        break
                    self.N += 1
                    if self.N > 25: #excedí el límite
                        break
                
                #AHORA HAY QUE DESNORMALIZAR
            
            if self.approx_type == BESSEL:
                self.N = 1
                self.wrg_n = self.tau0 * 2 * pi * self.frg
                while True:
                    z, p, k = signal.bessel(self.N, self.wrg_n, btype=filter_types[self.filter_type], analog=True, output='zpk', norm='delay')
                    tf2 = TFunction(z, p, k)
                    if abs(tf2.gd_at(self.wrg_n) - 1) <= self.gamma: #si el gd es menor-igual que el esperado, estamos
                        self.tf = TFunction(z, p, self.gain)
                        break
                    self.N += 1
                    if self.N > 25: #excedí el límite
                        break

            if self.approx_type == GAUSS:
                self.N = 1
                self.wrg_n = self.tau0 * 2 * pi * self.frg
                gauss_poly = [self.gamma, 0, 1]
                fact_prod = 1
                while True:
                    z = []
                    p = select_roots(np.poly1d(gauss_poly))
                    tf2 = TFunction(z, p, 1)
                    if abs(tf2.gd_at(self.wrg_n) - 1) <= self.gamma: #si el gd es menor-igual que el esperado, estamos
                        self.tf = TFunction(z, p, self.gain)
                        break
                    self.N += 1
                    if self.N > 25: #excedí el límite
                        break
                    fact_prod *= self.N
                    gauss_poly.insert(0, 0)
                    gauss_poly.insert(0, (self.gamma**self.N)/fact_prod)

        
    def get_template_limits(self):
        limits = []
        self.compute_normalized_gains()
        if self.filter_type == LOW_PASS:
            limits.extend([[1e-9, self.gp], [self.fp, self.gp]]) # Extiende tus límites!
            limits.extend([[self.fp, self.gp], [self.fp, 1e-9]])
            limits.extend([[self.fa, self.ga], [self.fa, 1e9]])
            limits.extend([[self.fa, self.ga], [1e9, self.ga]])
        if self.filter_type == HIGH_PASS:
            limits.extend([[1e-9, self.ga], [self.fa, self.ga]])
            limits.extend([[self.fa, self.ga], [self.fa, 1e9]])
            limits.extend([[self.fp, self.gp], [self.fp, 1e-9]])
            limits.extend([[self.fp, self.gp], [1e9, self.gp]])
        if self.filter_type == BAND_PASS:
            limits.extend([[1e-9, self.ga], [self.fa_min, self.ga]])
            limits.extend([[self.fa_min, self.ga], [self.fa_min, 1e9]])
            limits.extend([[self.fp_min, self.gp], [self.fp_min, 1e-9]])
            limits.extend([[self.fp_min, self.gp], [self.fp_max, self.gp]])

            limits.extend([[self.fp_max, self.gp], [self.fp_max, 1e-9]])
            limits.extend([[self.fa_max, self.ga], [self.fa_max, 1e9]])
            limits.extend([[self.fa_max, self.ga], [1e9, self.ga]])
        if self.filter_type == BAND_REJECT:
            limits.extend([[1e-9, self.gp], [self.fp_min, self.gp]])
            limits.extend([[self.fp_min, self.gp], [self.fp_min, 1e-9]])
            limits.extend([[self.fa_min, self.ga], [self.fa_min, 1e9]])
            limits.extend([[self.fa_min, self.ga], [self.fa_max, self.ga]])

            limits.extend([[self.fa_max, self.ga], [self.fa_max, 1e9]])
            limits.extend([[self.fp_max, self.gp], [self.fa_max, 1e-9]])
            limits.extend([[self.fp_max, self.gp], [1e9, self.gp]])
        
        self.limits_x = [limit[0] for limit in limits]
        self.limits_y = [limit[1] for limit in limits]
    
    def compute_normalized_gains(self):
        self.ga = np.power(10, (self.ga_dB / 20))
        self.gp = np.power(10, (self.gp_dB / 20))
                   