import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal
from package.transfer_function import TFunction
import matplotlib.pyplot as plt

N, wc = signal.buttord(1, 1.591549431, 3, 20, analog=True)
z, p, k = signal.butter(N, wc, analog=True, output='zpk')
tf_norm = TFunction(z, p, k)

print(tf_norm.z)
print(tf_norm.p)
denorm_z = []
denorm_p = []

for z in tf_norm.z:
    denorm_z.append(z/2 + np.sqrt(np.power(z/2,2) - 1))
    denorm_z.append(z/2 - np.sqrt(np.power(z/2,2) - 1))
for p in tf_norm.p:
    denorm_p.append(p/2 + np.sqrt(np.power(p/2,2) - 1))
    denorm_p.append(p/2 - np.sqrt(np.power(p/2,2) - 1))
denorm_z += [0 for i in range(0, len(tf_norm.p) - len(tf_norm.z))]

print(denorm_z)
print(denorm_p)

# tf = TFunction(denorm_z, denorm_p, 1)
N, wc = signal.buttord(2*np.pi*np.array([975.5, 1025.3]), 2*np.pi*np.array([905, 1105]), 3, 20, analog=True)
z, p, k = signal.butter(N, wc, analog=True, btype="bandpass", output='zpk')

tf = TFunction(z, p, k)
f, g, ph, gd = tf.getBode()
plt.figure()
plt.semilogx(f, g)    # Bode magnitude pl
plt.show()


