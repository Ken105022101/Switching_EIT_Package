import numpy as np
from numba import float64, complex128
from numba.experimental import jitclass

spec = [
    ('e0', float64),
    ('hbar', float64),
    ('c', float64),
    ('j', complex128),
    ('M', float64),
    ('kb', float64),
    ('mu_31', float64),
    ('L', float64),
    ('T', float64),
    ('Gamma', float64),
    ('gamma', float64),
    ('omega_c0', float64),
    ('omega_p0', float64),
    ('omega_s0', float64),
    ('gamma_3', float64),
    ('gamma_24', float64),
    ('Gamma_D', float64),
    ('CS', float64),
    ('Omega_c', float64),
    ('Omega_s', float64),
    ('shift', float64),
    ('shift_s', float64),
    ('OD', float64),
    ('gamma_2', float64)
]

@jitclass(spec)
class Theory:
    def __init__(self, Omega_c = 0, shift = 0, Omega_s = 0, shift_s = 0, OD = 0, gamma_2 = 0, T = 46):
        '''
        Parameters
        ----------
        Omega_c : (float64) The input value will be multiplied by Gamma
            Rabifrequency of Coupling laser, minus value will be regarded as traveling opposite with respect to probe laser.
            
        shift : (float64) Unit is MHz
            The frequency shift of Coupling laser.
            
        Omega_s : (float64) The input value will be multiplied by Gamma
            Rabifrequency of Switching laser, minus value will be regarded as traveling opposite with respect to probe laser.
            
        OD : (float64)
            Optical depth of the atoms.
            
        gamma_2 : (float64) The input value will be multiplied by Gamma
            Dephasing rate of the system, can only get from experiments.
            
        T : (float64) Unit is Celsius, optional
            Temperature of the atom. The default is 46.

        Returns
        -------
        None.

        '''
        'Coefficients'
        self.e0 = 8.85*10**-12
        self.hbar = 1.054571800*10**-34
        self.c = 299792458
        mole = 6.022 * 10**23
        amu = 10**-3 / mole
        self.j = complex(0,1)
        
        THz = 10**12; GHz = 10**9; MHz = 10**6;
        self.M = 86.9 * amu;
        self.kb = 1.38 * 10**-23
        self.mu_31 = 1.26885 * 10**-29
        self.L = 0.05
        self.T = 273.5 + T
        
        self.Gamma = 2*np.pi*5.75*MHz
        self.gamma = 0.225 * 2*np.pi*5.75*MHz
        
        self.omega_c0 = 2 * np.pi * (377.107463540*THz - 2.563005979*GHz + 306.246*MHz)
        self.omega_p0 = 2 * np.pi * (377.107463540*THz + 4.271676631*GHz + 306.246*MHz)
        self.omega_s0 = 2 * np.pi * (384.230484468*THz - 2.563005979*GHz + 193.741*MHz)
        self.gamma_3 = 0.5 * 2*np.pi*5.75*MHz
        self.gamma_24 = 2*np.pi*6.06*MHz*1
        self.Gamma_D = np.sqrt(2 * self.kb * self.T * (self.omega_p0 / self.c) ** 2 / self.M)
        self.CS = (2*(self.omega_p0/self.c)*self.mu_31**2)/(self.e0 * self.hbar * self.Gamma)
        
        self.Omega_c = Omega_c * 2*np.pi*5.75*MHz
        self.Omega_s = Omega_s * 2*np.pi*5.75*MHz
        self.shift = shift
        self.shift_s = shift_s
        self.OD = OD #* 2.05
        self.gamma_2 = gamma_2 * 2*np.pi*5.75*MHz
        

    # @jit(nopython = True)
    def d_p(self, omega, v):
        return (self.omega_p0 + omega) * (1 - v/self.c) - self.omega_p0
    
    # @jit(nopython = True)
    def d_c(self, shift, v):
        if self.Omega_c >= 0:
            return (self.omega_c0 + shift*2*np.pi*10**6) * (1 - v/self.c) - self.omega_c0
        if self.Omega_c < 0:
            return (self.omega_c0 + shift*2*np.pi*10**6) * (1 + v/self.c) - self.omega_c0
       
    # @jit(nopython = True)
    def d_s(self, shift_s, v):
        
        if self.Omega_s >= 0:
            return (self.omega_s0 + shift_s*2*np.pi*10**6) * (1 - v/self.c) - self.omega_s0
        if self.Omega_s < 0:
            return (self.omega_s0 + shift_s*2*np.pi*10**6) * (1 + v/self.c) - self.omega_s0
        '''
        if self.Omega_s >= 0:
            return self.omega_s0 * (1 - v/self.c) - self.omega_s0
        if self.Omega_s < 0:
            return self.omega_s0 * (1 + v/self.c) - self.omega_s0
        '''
    # @jit(nopython = True)
    def d(self, omega, v, shift):
        return self.d_p(omega, v) - self.d_c(shift, v)
    
    
    # @jit(nopython = True)
    def X(self, omega, v):
        up = (4*(self.OD/self.L/self.CS)*(self.mu_31**2)*(self.d(omega, v, self.shift) + self.gamma_2 * self.j))
        down = ((self.Omega_c**2 - 4*(self.d(omega, v, self.shift) + self.gamma_2 * self.j)*(self.d_p(omega, v) + self.gamma_3 * self.j)) * self.hbar * self.e0)
        return (up / down)
    
    # @jit(nopython = True)
    def X_switching(self, omega, v):
        dwp = self.d_p(omega, v) + self.gamma_3 * self.j
        dwc = self.d(omega, v, self.shift) + self.gamma_2 * self.j
        #dws = self.d_s(v, self.shift_s) + self.d(omega, v, self.shift) + self.gamma_24 * self.
        dws = self.d_s(self.shift_s, v) + self.d(omega, v, self.shift) + self.gamma_24 * self.j
        pre = (self.OD/self.L/self.CS)*(self.mu_31**2) / (self.hbar * self.e0)
        up = self.Omega_s**2 - 4 * dwc * dws
        down = 4 * dwp * dwc * dws - self.Omega_c**2 * dws - self.Omega_s**2 * dwp
        return (pre * up / down)
    
    # @jit(nopython = True)
    def f(self, v):
        return np.sqrt(self.M/(2*np.pi*self.kb*self.T)) * np.exp((-self.M*v**2)/(2*self.kb*self.T))
    
    # @jit(nopython = True)
    def FindOD(self, Base):
        return -(self.Gamma_D/self.Gamma) * (2/np.sqrt(np.pi)) * np.log(Base)
    
    def Data(self):
        dn = 0.1
        range_dp = 2 * np.pi * 40 * 10**6
        P = np.linspace(-range_dp, range_dp, 1000)
        V = np.arange(-600, 600, dn) #12000)
        #Tran_E = []
        #Tran_S = []
        Tran_S_test = []
        #Tran_S_T_test = []
        
        v_f = self.f
        v_X_switching = self.X_switching
        #v_X_EIT = self.X
        
        for p in P:
            #dvE = v_f(V) * v_X_EIT(p, V) * dn
            #dvS = v_f(V) * v_X_switching(p, V) * dn
            dvS_test = v_f(V) * (v_X_switching(p, V)).imag * dn
            '''
            T_test.append(dvS_test)
            T_test = np.array(T_test)
            T_test_tr = T_test.transpose()
            '''
            #dvs_T_test = dvS_test[6205] / dn
            #int_dvE = np.sum(dvE)
            #int_dvS = np.sum(dvS)
            int_dvS_test = np.sum(dvS_test)
            
            ##Tran_E.append(np.exp(-(int_dvE/2).imag * (self.omega_c0 / self.c) * 0.05))
            #Tran_S.append(np.exp(-(int_dvS/2).imag * (self.omega_c0 / self.c) * 0.05))
            Tran_S_test.append(np.exp(-(int_dvS_test/1) * (self.omega_c0 / self.c) * 0.05))
            #Tran_S_T_test.append(np.exp(-(dvs_T_test/1) * (self.omega_c0 / self.c) * 0.05))
        
        return P/(2 * np.pi * 10**6), Tran_S_test
        
    def FittingData(self, P_data):
        dn = 0.1
        P = P_data#* 4000 * 2 * np.pi * 10**6
        V = np.arange(-600, 600, dn) #12010)
        
        Tran_S = []
        
        v_f = self.f
        v_X_switching = self.X_switching
        
        for p in P:
            dvS = v_f(V) * v_X_switching(p, V) * dn
            int_dvS = np.sum(dvS)
            Tran_S.append(np.exp(-(int_dvS).imag * (self.omega_c0 / self.c) * 0.05))
        
        return P/(2 * np.pi * 10**6), Tran_S
    
    def different_V(self, V):
        dV = 0.1
        range_dp = 2 * np.pi * 40 * 10**6
        P = np.linspace(-range_dp, range_dp, 1000)
        #V = np.arange(-600, 600, dV) #12000)
        Tran_S_T = []
        
        v_f = self.f
        v_X_switching = self.X_switching
        #v_X_EIT = self.X
            
        for p in P:
            #dvE = v_f(V) * v_X_EIT(p, V) * dn
            #dvS = v_f(V) * v_X_switching(p, V) * dn
            dvS = v_f(V) * (v_X_switching(p, V)).imag# * dV
            #dvs_T = dvS[index_num]
            
            ##Tran_E.append(np.exp(-(int_dvE/2).imag * (self.omega_c0 / self.c) * 0.05))
            Tran_S_T.append(np.exp(-(dvS) * (self.omega_c0 / self.c) * 0.05))
        
        return P/(2 * np.pi * 10**6), Tran_S_T
            
            
            
        
        


import pandas as pd
import matplotlib.pyplot as plt

class ReadData:
    
    def __init__(self, filepath, Background_filename, Background_label, Trans_filename, Trans_label, Data_filename, Data_label, convolution_num = 0, Trans_time_label = 'TIME', Data_time_label = 'TIME'):
        self.filepath = filepath
        #self.filename = filename
        self.convolution_num = convolution_num
        print('FilePath = ' + str(self.filepath))
        print('Reading ' + str(np.array(Data_filename)) + ' file')
        
        self.Background_data = self.Background(Background_filename, Background_label)
        self.Trans_data = self.Transparent(Trans_filename, Trans_label, Trans_time_label)
        self.P_data, self.EIT_data = self.Data(Data_filename, Data_label, Data_time_label)
        self.normalized = self.Process()
        
        
        
    def Background(self, Background_filename, Background_label):
        file = str(self.filepath) + str(Background_filename) + '.csv'
        df = pd.read_csv(file)
        Background_Data = pd.DataFrame(df, columns = [str(Background_label)])
        Background_Data = np.array(Background_Data[str(Background_label)])
               
        return np.mean(Background_Data)
    
    def Transparent(self, Data_filename, Data_label, Time_label = 'TIME'):
        file = str(self.filepath) + str(Data_filename) + '.csv'
        df = pd.read_csv(file)
        Trans_data = pd.DataFrame(df, columns = [str(Data_label)])
        Trans_data = np.array(Trans_data[str(Data_label)])
        
        return np.mean(Trans_data)
    
    def Data(self, Data_filename, Data_label, Time_label = 'TIME'):
        file = str(self.filepath) + str(Data_filename) + '.csv'
        df = pd.read_csv(file)
        EIT_data = pd.DataFrame(df, columns = [str(Data_label)])
        EIT_data = np.array(EIT_data[str(Data_label)])
        Time = pd.DataFrame(df, columns = [str(Time_label)])
        Time = np.array(Time[str(Time_label)])
        P_data = 4000 * 2 * np.pi * 10**6 * Time
        
        return P_data, EIT_data
    
    def Process(self):
        Background = self.Background_data
        Trans = self.Trans_data
        Data = self.EIT_data
        normalized_EIT = (Data - Background) / (Trans - Background)
        
        return normalized_EIT
   



if __name__ == "__main__":
    Test = Theory(Omega_c = 4.4, shift = 0, Omega_s = 30, shift_s=10, OD = 90, gamma_2 = 0.22, T = 46)
    import time
    start = time.time()
    #P, Switching = Test.Data()
    end = time.time()
    print(end - start)
    plt.figure(dpi = 600)
    #plt.plot(P, EIT, label = "EIT")
    #P, T = Test.different_V(-10)
    #T = np.array(T).transpose
    #temp = np.ones(1000)
    #for i in range(12000):
        #temp *= T[i]
        
    #plt.plot(P, Switching, label = "EIT")
    
    #plt.plot(P, Switching, label = "Switching")
    P, T = Test.different_V(0)
    plt.plot(P, T, label = "V = 0")
    P, T = Test.different_V(10)
    plt.plot(P, T, label = "V = 10")
    P, T = Test.different_V(20)
    plt.plot(P, T, label = "V = 20")
    P, T = Test.different_V(30)
    plt.plot(P, T, label = "V = 30")
    P, T = Test.different_V(40)
    plt.plot(P, T, label = "V = 40")
    P, T = Test.different_V(50)
    plt.plot(P, T, label = "V = 50")
    P, T = Test.different_V(60)
    plt.plot(P, T, label = "V = 60")
    P, T = Test.different_V(70)
    plt.plot(P, T, label = "V = 70")
    plt.title('Switching with same propagation $\Omega_{S}$ = 30')
    #plt.plot(P, temp, label = "T_sum")
    
    plt.ylabel("Transparency")
    plt.xlabel("$\Delta P$")
    plt.legend(loc = "best")
    plt.ylim(0.8, 1)
    plt.show()
    
    plt.figure(dpi = 600)
    for s in [0,0.5,1,1.5,2]:
        Test = Theory(Omega_c = 1.0, shift = 0, Omega_s = -s, OD = 90, gamma_2 = 0.022, T = -273.14997)
        P, T = Test.different_V(20)
        plt.plot(P, T, label = "$\Omega_{Switching}$ = " + str(s))
        
    plt.ylabel("Transparency")
    plt.xlabel("$\Delta P$")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.title('Contribution of V = 20 atoms of different $\Omega_{Switching}$ opposite')
    plt.ylim(0, 1)
    plt.show()
