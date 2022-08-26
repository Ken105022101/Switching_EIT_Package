import numpy as np
import matplotlib.pyplot as plt
import EIT_class_test as EIT
import time

def switch_eff(Omega_c, Omega_s, OD, g2):
    P, Switching = EIT.Theory(Omega_c = Omega_c, shift = 0, Omega_s = Omega_s, OD = OD, gamma_2 = g2, T = 46).Data()
    return Switching[int(len(P)/2)]
    
from joblib import Parallel, delayed
start = time.time()
result_same = Parallel(n_jobs=25)(delayed(switch_eff)(4.4, Omega_s, 90, 0.22) for Omega_s in np.arange(0, 60, 0.1))
result_opposite = Parallel(n_jobs=25)(delayed(switch_eff)(4.4, -Omega_s, 90, 0.22) for Omega_s in np.arange(0, 60, 0.1))

print('time = ' + str(time.time() - start))

plt.figure(dpi = 600)
plt.plot(np.arange(0, 60, 0.1), result_same, label = "Same with probe")
plt.plot(np.arange(0, 60, 0.1), result_opposite, label = "Opposite with probe")
plt.xlabel('$\Omega_{Switching}$')
plt.ylabel('Transparency at $\Delta\omega_{P} = 0$')
plt.legend(loc = 'best')
#plt.title('Switching Transparency with different $\Omega_{Switching}$ (MOT)')
plt.show()

    
    