import numpy as np
import matplotlib.pyplot as plt
import EIT_class_test as EIT
import time
from tqdm import tqdm

#Test = EIT.ReadData('E:/Ken/Study/專題/熱原子/Two_photon_detune/', 'tek0565', 'CH2', 'tek0575', 'CH2', 'tek0566', 'CH2')
Test = EIT.ReadData('E:/Ken/Study/專題/熱原子/Switching/20220307/', 'tek0441', 'CH2', 'tek0451', 'CH2', 'tek0444', 'CH2')
#Test = EIT.ReadData('E:/Ken/Study/專題/熱原子/Switching/20220307/', 'tek0470', 'CH2', 'tek0477', 'CH2', 'tek0473', 'CH2')
#Test = EIT.ReadData('E:/Ken/Study/專題/熱原子/Switching/20220307/', 'tek0496', 'CH2', 'tek0509', 'CH2', 'tek0504', 'CH2')
P_data = Test.P_data
Data = Test.normalized

P_con = []
T_con = []
for i in range(len(Data) - 10):
    P_con.append(sum(P_data[i:i+10])/10)
    T_con.append(sum(Data[i:i+10])/10)
P_con = np.array(P_con)
T_con = np.array(T_con)

OD_data = EIT.Theory().FindOD(np.median(T_con))+1.5
def loss(shift, C, g2):
    P, Switching = EIT.Theory(Omega_c = C, shift = shift, Omega_s = 0, shift_s = 0, OD = OD_data, gamma_2 = g2, T = 42).FittingData(P_con)
    #plt.plot(P, Switching)
    #plt.plot(P, T_con)
    #plt.plot(P, Switching - T_con)
    #plt.plot(P, abs(Switching - T_con))
    #plt.show()
    return [np.sum(abs(Switching - T_con)), shift, C, g2]

from joblib import Parallel, delayed
start = time.time()
result = Parallel(n_jobs=-2)(delayed(loss)(shift, C, g2) for shift in tqdm(np.arange(0, 1.5, 0.1)) for C in np.arange(4.0, 4.4, 0.05) for g2 in np.arange(0.28, 0.31, 0.005)) 

print(time.time() - start)
print(len(result))
parameter = min(result[:])
print(parameter)
print('OD = ' + str(OD_data))

temp_data = []
temp_P = []
temp_fit = []
temp_P_fit = []
temp_label = []

temp_label.append(parameter)
temp_P.append(P_con/(2 * np.pi * 10**6))
temp_data.append(T_con)
P, data = EIT.Theory(Omega_c = parameter[2], shift = parameter[1], Omega_s = 0, OD = OD_data, gamma_2 = parameter[3], T = 42).Data()
temp_P_fit.append(P)
temp_fit.append(data)
'''
plt.figure(dpi = 600)
plt.plot(P_con/(2 * np.pi * 10**6), T_con, label = '42°C Experiment')
P, data = EIT.Theory(Omega_c = parameter[2], shift = parameter[1], Omega_s = 0, OD = OD_data, gamma_2 = parameter[3], T = 42).Data()
plt.plot(P, data, label = '42°C Fit, OD = ' + str(round(OD_data, 2)))
#plt.title("EIT fitting " + "tek0444 with" + "$\Omega_{C}$ = " + str(round(parameter[2], 2)) + ", Shift = " + str(round(parameter[1], 2)) + ", $\gamma_{2}$ = " + str(round(parameter[3], 2)))
#P, data = EIT.Theory(Omega_c = 4.0, shift = 0.4, Omega_s = 0, OD = OD_data, gamma_2 = 0.24).Data()
plt.title("EIT in different temperature")
#plt.plot(P, data, label = 'TEST, OD = ' + str(round(OD_data, 2)))
plt.ylabel("Transparency")
plt.xlabel("$\Delta p$")
plt.legend(loc = 'best')
'''
#plt.show()
'''
Test = EIT.ReadData('E:/Ken/Study/專題/熱原子/Switching/20220307/', 'tek0496', 'CH2', 'tek0509', 'CH2', 'tek0506', 'CH2')
P_data = Test.P_data
Data = Test.normalized
#OD_data = EIT.Theory().FindOD(np.median(Data))
def loss(s):
    P, Switching = EIT.Theory(Omega_c = parameter[1], shift = 1.5, Omega_s = -s, OD = OD_data, gamma_2 = parameter[2]).FittingData(P_data)
    return [np.sum(abs(Switching - Data)), s]

from joblib import Parallel, delayed
start = time.time()
result = Parallel(n_jobs=32)(delayed(loss)(s) for s in np.arange(0.8, 2.1, 0.1))

print(time.time() - start)
print(len(result))
parameter_s = min(result[:])
print(parameter_s)
print('OD = ' + str(OD_data))
plt.figure(dpi = 600)
plt.plot(P_data/(2 * np.pi * 10**6), Test.normalized, label = 'Experiment')
P, data = EIT.Theory(Omega_c = parameter[1], shift = 1.5, Omega_s = -parameter_s[1], OD = OD_data, gamma_2 = parameter[2]).Data()
plt.plot(P, data, label = 'Fit, OD = ' + str(round(OD_data, 2)))
plt.ylabel("Transparency")
plt.xlabel("$\Delta p$")
plt.legend(loc = 'best')
plt.show()'''