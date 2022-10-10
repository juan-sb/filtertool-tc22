class FilterStage:
    def __init__(self, p1=-1, p2=-1, z1=-1, z2=-1, g=-1, Q=-1):
        self.pole1 = p1
        self.pole2 = p2
        self.zero1 = z1
        self.zero2 = z2
        self.gain = g
        self.Q = Q

    def setPoles(self, p1, p2):
        self.pole1 = p1
        self.pole2 = p2

    def getPoles(self):
        return self.pole1, self.pole2

    def getLabel(self):
        pole_text = ''
        zero_text = ''
        if self.pole1 < 0:
            pole_text = ' Polo {}\n'.format(self.pole2+1)
        elif self.pole2 < 0:
            pole_text = ' Polo {}\n'.format(self.pole1+1)
        else:
            pole_text = ' Polo {}, Polo {}\n'.format(self.pole1+1, self.pole2+1)
        if self.zero1 < 0:
            if self.zero2 >= 0:
                zero_text = ' Cero {}\n'.format(self.zero2+1)
        elif self.zero2 < 0:
            zero_text = ' Cero {}\n'.format(self.zero1+1)
        else:
            zero_text = ' Cero {}, Cero {}\n'.format(self.zero1+1, self.zero2+1)

        return '{}{} Ganancia = {:.3f} dB\n Q = {:.3f}'.format(pole_text, zero_text, self.gain, self.Q)

    def __str__(self):
        return '{}\n{}\n{}\n{}\n{}\n{}\n'.format(self.pole1, self.pole2, self.zero1, self.zero2, self.gain, self.Q)