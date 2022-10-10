class DesignConfig:
    def __init__(self, type=0, aprox=0, denorm=0, minord=0, maxord=0, qmax=0, Ap=0, ripple=0, Aa=0, wp=0, wa=0, wp2=0, wa2=0, tau=0, wrg=0, gamma=0):
        self.filter_types = ['Pasa Bajos', 'Pasa Altos', 'Pasa Banda', 'Rechaza Banda', 'Retardo de Grupo']
        self.aprox_types = ['Butterworth', 'Chebyshev I', 'Chebyshev II', 'Bessel', 'Cauer', 'Legendre', 'Gauss']

        self.type = type
        self.aprox = aprox
        self.denorm = denorm
        self.minord = minord
        self.maxord = maxord
        self.qmax = qmax
        self.Ap = Ap
        self.ripple = ripple
        self.Aa = Aa
        self.wp = wp
        self.wa = wa
        self.wp2 = wp2
        self.wa2 = wa2
        self.tau = tau
        self.wrg = wrg
        self.gamma = gamma

    def setType(self, type):
        self.type = self.type if type not in self.filter_types else self.filter_types.index(type)

    def setAprox(self, aprox):
        self.aprox = self.aprox if aprox not in self.aprox_types else self.aprox_types.index(aprox)

    def getType(self):
        return self.filter_types[self.type]

    def getAprox(self):
        return self.filter_types[self.aprox]

    def getNormalAttenuations(self):
        Ap = self.ripple
        Aa = self.Aa + self.ripple - self.Ap
        return Ap, Aa

    def setParameters(self, type, aprox, denorm, minord, maxord, qmax, Ap, ripple, Aa, wp, wa, wp2, wa2, tau, wrg, gamma):
        self.setType(type)
        self.setAprox(aprox)
        self.denorm = denorm
        self.minord = minord
        self.maxord = maxord
        self.qmax = qmax
        self.Ap = Ap
        self.ripple = ripple
        self.Aa = Aa
        self.wp = wp
        self.wa = wa
        self.wp2 = wp2
        self.wa2 = wa2
        self.tau = tau
        self.wrg = wrg
        self.gamma = gamma

    def __str__(self):
        return str(self.type) + '\n' + \
                str(self.aprox) + '\n' + \
                str(self.denorm) + '\n' + \
                str(self.minord) + '\n' + \
                str(self.maxord) + '\n' + \
                str(self.qmax) + '\n' + \
                str(self.Ap) + '\n' + \
                str(self.ripple) + '\n' + \
                str(self.Aa) + '\n' + \
                str(self.wp) + '\n' + \
                str(self.wa) + '\n' + \
                str(self.wp2) + '\n' + \
                str(self.wa2) + '\n' + \
                str(self.tau) + '\n' + \
                str(self.wrg) + '\n' + \
                str(self.gamma) + '\n'

    def export_names(self):
        txt = 'Tipo de Filtro:\n'
        txt += 'Aproximación:\n'
        txt += 'Rango de Desnormalización:\n'
        txt += 'Orden Mínimo:\n'
        txt += 'Orden Máximo:\n'
        txt += 'Q Máximo:\n'
        w_band = self.filter_types[self.type] == 'Pasa Banda' or self.filter_types[self.type] == 'Rechaza Banda'
        if self.filter_types[self.type] != 'Retardo de Grupo':
            txt += 'Banda de paso (Ap):\n'
            txt += 'Máximo Ripple:\n'
            txt += 'Banda de atenuación (Aa):\n'
            if w_band:
                txt += 'Frecuencia ωp+:\n'
                txt += 'Frecuencia ωp-:\n'
                txt += 'Frecuencia ωa+:\n'
                txt += 'Frecuencia ωa-:\n'
            else:
                txt += 'Frecuencia ωp:\n'
                txt += 'Frecuencia ωa:\n'
        else:
            txt += 'Retardo en banda de paso (𝜏(0)):\n'
            txt += 'Frecuencia ωRG:\n'
            txt += 'Retardo máximo en ωRG (𝛾%):\n'
        return txt

    def export_values(self):
        txt = '{}\n'.format(self.filter_types[self.type])
        txt += '{}\n'.format(self.aprox_types[self.aprox])
        txt += '{} %\n'.format(self.denorm)
        txt += '{}\n'.format(self.minord)
        txt += '{}\n'.format(self.maxord)
        txt += '{}\n'.format(self.qmax)
        w_band = self.filter_types[self.type] == 'Pasa Banda' or self.filter_types[self.type] == 'Rechaza Banda'
        if self.filter_types[self.type] != 'Retardo de Grupo':
            txt += '{} dB\n'.format(self.Ap)
            txt += '{} dB\n'.format(self.ripple)
            txt += '{} dB\n'.format(self.Aa)
            if w_band:
                txt += '{}\n'.format(self.wp)
                txt += '{}\n'.format(self.wp2)
                txt += '{}\n'.format(self.wa)
                txt += '{}\n'.format(self.wa2)
            else:
                txt += '{}\n'.format(self.wp)
                txt += '{}\n'.format(self.wa)
        else:
            txt += '{}\n'.format(self.tau)
            txt += '{}\n'.format(self.wrg)
            txt += '{}\n'.format(self.gamma)
        return txt