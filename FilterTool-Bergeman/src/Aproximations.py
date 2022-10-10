from scipy import signal, special
import numpy as np
from src.DesignConfig import *


def Butterworth(designconfig):
    Ap, Aa = designconfig.getNormalAttenuations()

    type = designconfig.getType()
    signaltypes = {'Pasa Bajos': 'lowpass', 'Pasa Altos': 'highpass', 'Pasa Banda': 'bandpass',
                   'Rechaza Banda': 'bandstop'}
    if type == 'Pasa Bajos' or type == 'Pasa Altos':
        N, Wn = signal.buttord(designconfig.wp, designconfig.wa, Ap, Aa, True)
    elif type == 'Pasa Banda' or type == 'Rechaza Banda':
        N, Wn = signal.buttord([designconfig.wp2, designconfig.wp], [designconfig.wa2, designconfig.wa], Ap, Aa, True)
    else:
        return [[], [], 0]
    N = max(min(N, designconfig.maxord), designconfig.minord)

    z, p, k = signal.butter(N, Wn, btype=signaltypes[type], analog=True, output='zpk')

    p = setMaxQ(designconfig.qmax, p)

    return z, p, k


def ChebyshevI(designconfig):
    Ap, Aa = designconfig.getNormalAttenuations()

    type = designconfig.getType()
    signaltypes = {'Pasa Bajos': 'lowpass', 'Pasa Altos': 'highpass', 'Pasa Banda': 'bandpass',
                   'Rechaza Banda': 'bandstop'}
    if type == 'Pasa Bajos' or type == 'Pasa Altos':
        N, Wn = signal.cheb1ord(designconfig.wp, designconfig.wa, Ap, Aa, True)
    elif type == 'Pasa Banda' or type == 'Rechaza Banda':
        N, Wn = signal.cheb1ord([designconfig.wp2, designconfig.wp], [designconfig.wa2, designconfig.wa], Ap, Aa, True)
    else:
        return [[], [], 0]
    N = max(min(N, designconfig.maxord), designconfig.minord)

    z, p, k = signal.cheby1(N, Ap, Wn, btype=signaltypes[type], analog=True, output='zpk')

    p = setMaxQ(designconfig.qmax, p)

    return z, p, k


def ChebyshevII(designconfig):
    Ap, Aa = designconfig.getNormalAttenuations()

    type = designconfig.getType()
    signaltypes = {'Pasa Bajos': 'lowpass', 'Pasa Altos': 'highpass', 'Pasa Banda': 'bandpass',
                   'Rechaza Banda': 'bandstop'}
    if type == 'Pasa Bajos' or type == 'Pasa Altos':
        N, Wn = signal.cheb2ord(designconfig.wp, designconfig.wa, Ap, Aa, True)
    elif type == 'Pasa Banda' or type == 'Rechaza Banda':
        N, Wn = signal.cheb2ord([designconfig.wp2, designconfig.wp], [designconfig.wa2, designconfig.wa], Ap, Aa, True)
    else:
        return [[], [], 0]
    N = max(min(N, designconfig.maxord), designconfig.minord)

    z, p, k = signal.cheby2(N, Ap, Wn, btype=signaltypes[type], analog=True, output='zpk')

    p = setMaxQ(designconfig.qmax, p)

    return z, p, k


def Bessel(designconfig):
    n = 1
    wrgN = designconfig.tau * designconfig.wrg
    tb, ta = signal.bessel(N=n, Wn=1, btype='low', analog=True, output='ba', norm='delay')
    w = np.linspace(wrgN - 0.5, wrgN + 0.5, 1000)
    w, th = signal.freqs(tb, ta, w)
    gd = -np.diff(np.unwrap(np.angle(th))) / np.diff(w)
    index = np.where(w >= wrgN)[0][0]
    while gd[index] < (1 - (designconfig.gamma / 100)):
        n += 1
        tb, ta = signal.bessel(N=n, Wn=1, btype='low', analog=True, output='ba', norm='delay')
        w = np.linspace(wrgN - 0.5, wrgN + 0.5, 1000)
        w, th = signal.freqs(tb, ta, w)
        gd = -np.diff(np.unwrap(np.angle(th))) / np.diff(w)
        index = np.where(w >= wrgN)[0][0]

    n = max(min(n, designconfig.maxord), designconfig.minord)

    z, p, k = signal.bessel(n, 1, 'low', analog=True, output='zpk', norm='delay')

    for i in range(len(p)):
        p[i] = p[i] / designconfig.tau

    for i in range(len(z)):
        z[i] = z[i] / designconfig.tau

    k = k / (designconfig.tau ** (n - len(z)))

    p = setMaxQ(designconfig.qmax, p)

    return z, p, k


def Cauer(designconfig):
    Ap, Aa = designconfig.getNormalAttenuations()

    type = designconfig.getType()
    signaltypes = {'Pasa Bajos': 'low', 'Pasa Altos': 'high', 'Pasa Banda': 'pass',
                   'Rechaza Banda': 'stop'}
    if type == 'Pasa Bajos' or type == 'Pasa Altos':
        N, Wn = signal.ellipord(designconfig.wp, designconfig.wa, Ap, Aa, True)
    elif type == 'Pasa Banda' or type == 'Rechaza Banda':
        N, Wn = signal.ellipord([designconfig.wp2, designconfig.wp], [designconfig.wa2, designconfig.wa], Ap, Aa, True)
    else:
        return [[], [], 0]
    N = max(min(N, designconfig.maxord), designconfig.minord)

    z, p, k = signal.ellip(N, Ap, Aa, Wn, btype=signaltypes[type], analog=True, output='zpk')

    p = setMaxQ(designconfig.qmax, p)

    return z, p, k


def Gauss(designconfig):
    n = 1
    wrgN = designconfig.wrg * designconfig.tau
    z, p, k = gauss_poly(n, 1)
    b, a = signal.zpk2tf(z, p, k)
    w = np.linspace(wrgN - 0.5, wrgN + 0.5, 1000)
    w, th = signal.freqs(b, a, w)
    gd = -np.diff(np.unwrap(np.angle(th))) / np.diff(w)
    index = np.where(w >= wrgN)[0][0]
    while gd[index] < (1 - (designconfig.gamma / 100)):
        n += 1
        z, p, k = gauss_poly(n, 1)
        b, a = signal.zpk2tf(z, p, k)
        w = np.linspace(wrgN - 0.5, wrgN + 0.5, 1000)
        w, th = signal.freqs(b, a, w)
        gd = -np.diff(np.unwrap(np.angle(th))) / np.diff(w)
        index = np.where(w >= wrgN)[0][0]

    n = max(min(n, designconfig.maxord), designconfig.minord)

    z, p, k = gauss_poly(n, 1)

    for i in range(len(p)):
        p[i] = p[i] / designconfig.tau

    for i in range(len(z)):
        z[i] = z[i] / designconfig.tau

    k = k / (designconfig.tau ** (n - len(z)))

    p = setMaxQ(designconfig.qmax, p)

    return z, p, k


def gauss_poly(n, tau):
    a = []
    wtow2 = np.poly1d([1, 0, 0])
    for i in range(n, 0, -1):
        a.append(((-tau) ** i) / np.math.factorial(i))  # desarrollo del polinomio e**x

    a.append(1)
    poly = np.poly1d(a)(wtow2)  # cambio de variable x => w**2
    p = []
    roots = np.roots(poly)  # saco las raices
    for i in range(len(roots)):
        c_root = complex(roots[i])
        if np.sign(c_root.real) == -1:  # me quedo con aquellas de parte real negativa
            p.append(c_root)

    k = 1

    z = []

    w = np.logspace(-5, -4, 1000)
    b, a = signal.zpk2tf(z, p, k)
    w, th = signal.freqs(b, a, w)
    gd = -np.diff(np.unwrap(np.angle(th))) / np.diff(w)

    for i in range(len(p)):
        p[i] = p[i] * gd[0]
    for i in range(len(p)):
        k = p[i] * k
    return z, p, k


def Legendre(designconfig):
    Ap, Aa = designconfig.getNormalAttenuations()

    w0 = 0
    B = 0

    type = designconfig.getType()
    if type == 'Pasa Bajos' or type == 'Pasa Altos':
        waN = max(designconfig.wa, designconfig.wp) / min(designconfig.wa, designconfig.wp)
    elif type == 'Pasa Banda' or type == 'Rechaza Banda':
        waN = max(abs(designconfig.wa2 - designconfig.wa), abs(designconfig.wp2 - designconfig.wp)) / min(abs(designconfig.wa2 - designconfig.wa), abs(designconfig.wp2 - designconfig.wp))
        w0 = np.sqrt(designconfig.wp2*designconfig.wp)
        B = abs(designconfig.wp2 - designconfig.wp)
    else:
        return [[], [], 0]

    eps = ((10 ** (Ap / 10)) - 1) ** (1 / 2)

    n = 1
    z, p, k = Legendre_poly(n, eps)
    H = signal.ZerosPolesGain(z, p, k)
    w = np.linspace(waN - 0.5, waN + 0.5, 5000)
    w, bode, fase = signal.bode(H, w)
    index = np.where(w >= waN)[0][0]
    while abs(bode[index]) < Aa:
        n += 1
        z, p, k = Legendre_poly(n, eps)
        H = signal.ZerosPolesGain(z, p, k)
        w = np.linspace(waN - 0.5, waN + 0.5, 5000)
        w, bode, fase = signal.bode(H, w)
        index = np.where(w >= waN)[0][0]

    n = max(min(n, designconfig.maxord), designconfig.minord)
    z, p, k = Legendre_poly(n, eps)

    if type == 'Pasa Bajos':
        z, p, k = signal.lp2lp_zpk(z, p, k, designconfig.wp)
    elif type == 'Pasa Altos':
        z, p, k = signal.lp2hp_zpk(z, p, k, designconfig.wp)
    elif type == 'Pasa Banda':
        z, p, k = signal.lp2bp_zpk(z, p, k, w0, B)
    elif type == 'Rechaza Banda':
        z, p, k = signal.lp2bs_zpk(z, p, k, w0, B)

    p = setMaxQ(designconfig.qmax, p)

    return z, p, k

def sumprod(a, p):
    a_p = []

    for i in range(len(p)):
        a_p.append(a[i] * p[i])

    ps = np.poly1d([0])
    for i in range(len(p)):
        ps = ps + a_p[i]

    return ps ** 2


def get_L(n):
    # valor de k
    if n % 2 == 0:
        k = int(n / 2 - 1)
    else:
        k = int((n - 1) / 2)
    # polinomios de legendre
    P = []
    for i in range(k + 1):
        P.append(special.legendre(i))
    # coeficientes de la suma de los polinomios
    a = []
    for i in range(k + 1):
        if n % 2 == 0:
            if k % 2 == 0:
                if i == 0:
                    a.append(1 / (((k + 1) * (k + 2)) ** (1 / 2)))
                elif i % 2 == 0:
                    a.append((2 * i + 1) * a[0])
                else:
                    a.append(0)
            else:
                if i % 2 == 0:
                    a.append(0)
                elif i == 1:
                    a.append(3 / (((k + 1) * (k + 2)) ** (1 / 2)))
                else:
                    a.append((2 * i + 1) * a[1] / 3)
        else:
            if i == 0:
                a.append(1 / ((2 ** (1 / 2)) * (k + 1)))
            else:
                a.append((2 * i + 1) * a[0])
    # suma de polinomios
    p_sum = sumprod(a, P)
    if n % 2 == 0:
        p_sum = np.poly1d([1, 1]) * p_sum

    # integral entre -1 y 2*w**2 - 1
    x_to_2w2m1 = np.poly1d([2, 0, -1])
    integ = np.polyint(p_sum)

    return integ(x_to_2w2m1) - integ(-1)


def Legendre_poly(n, eps):
    poly = 1 + (eps**2) * get_L(n)
    p = []
    roots = 1j*np.roots(poly)  # saco las raices
    for i in range(len(roots)):
        c_root = complex(roots[i])
        if np.sign(c_root.real) == -1:  # me quedo con aquellas de parte real negativa
            p.append(c_root)

    z = []
    k = 1
    for i in range(len(p)):
        k = k * p[i]

    return z, p, k


def setMaxQ(maxQ, poles):
    for i in range(len(poles)):
        w0 = abs(poles[i])
        a = abs(poles[i].real)
        Q = w0 / (2 * a)
        if Q > maxQ:
            w0st = (maxQ-0.05) * 2 * a
            ast = w0 / (2 * (maxQ-0.05))
            pst1 = [-a, np.sign(poles[i].imag) * ((w0st ** 2 - a ** 2) ** (1 / 2))]
            pst2 = [-ast, poles[i].imag]

            m = (pst2[1] - pst1[1]) / (pst2[0] - pst1[0])
            line = lambda x: m * (x - pst1[0]) + pst1[1]
            xs = np.linspace(min(-a, -ast), max(-a, -ast), 1000)
            ys = line(xs)
            difs = np.zeros(len(xs))

            min_dif = w0
            value = 0
            for n in range(len(xs)):
                difs[n] = ((xs[n] - poles[i].real) ** 2 + (ys[n] - poles[i].imag) ** 2) ** (1 / 2)
                if difs[n] < min_dif:
                    value = n
                    min_dif = difs[n]

            poles[i] = xs[value] + 1j*ys[value]

    return poles
