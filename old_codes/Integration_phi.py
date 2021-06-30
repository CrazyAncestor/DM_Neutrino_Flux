## This module aims to calculate the dphi/dT result of Torsten Bringmann1 et.al.
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

e = 1.
M_chi = 0.1e09
sigma_chi = 1e-30
def differential_intensity_array(R,i):
    split = 0
    for k in range(len(R)):
        if R[k]>1.:
            split = k
            break
    Ra = R[:split]
    Rb = R[split:]
    
    if i==0:#Proton
        DIa = 94.1 -831*Ra + 0*(Ra**2) + 16700*(Ra**3) - 10200*(Ra**4) - 0*(Ra**5)
        DIb = 10800 + 8590/Rb -4230000/(3190+Rb) +274000/(17.4+Rb) -39400/(0.464+Rb) + 0*Rb

    elif i==1:#Helium
        DIa = 1.14 +0*Ra -118*(Ra**2) + 578*(Ra**3) + 0*(Ra**4) - 87*(Ra**5)
        DIb = 3120 -5530/Rb +3370/(1.29+Rb) +134000/(88.5+Rb) -1170000/(861+Rb) + 0.03*Rb
                
    return np.concatenate((DIa,DIb))/(R**2.7)

def differential_intensity(R,i):
    if i==0:#Proton
        if R<=1:
            Ra = R
            DIa = 94.1 -831*Ra + 0*(Ra**2) + 16700*(Ra**3) - 10200*(Ra**4) - 0*(Ra**5)
            return DIa/(R**2.7)
        else:
            Rb = R
            DIb = 10800 + 8590/Rb -4230000/(3190+Rb) +274000/(17.4+Rb) -39400/(0.464+Rb) + 0*Rb
            return DIb/(R**2.7)

    elif i==1:#Helium
        if R<=1:
            Ra = R
            DIa = 1.14 +0*Ra -118*(Ra**2) + 578*(Ra**3) + 0*(Ra**4) - 87*(Ra**5)
            return DIa/(R**2.7)
        else:
            Rb = R
            DIb = 3120 -5530/Rb +3370/(1.29+Rb) +134000/(88.5+Rb) -1170000/(861+Rb) + 0.03*Rb
            return DIb/(R**2.7)
def dR_dI(R):
    return differential_intensity(R,0)
            
def phi_T(T,i):
    m = 0
    Z = 0
    if i==0:#Proton
        m = 938.08e06
        Z =  1
    elif i==1:#Helium
        m = 3727.6e06
        Z =  2
    R = (T**2+2*m*T)**0.5/(Z*e)
    #print(R)
    DI_DR = differential_intensity(R*1e-9,i)
    return 4*np.pi*(T+m)/(Z*e*((T**2+2*m*T)**0.5))*DI_DR

def int_base(T,i):

    if i==0:#Proton
        m = 938.08e06
        Z =  1
    elif i==1:#Helium
        m = 3727.6e06
        Z =  2

    T_chi_max = (T**2+2*m*T)/(T+ (m+M_chi)**2/(2*M_chi) )
    return phi_T(T,i)/T_chi_max

def integration(T_chi,i):
    T_min = (T_chi/2-m)* (1-(1+ 2*T_chi*((m+M_chi)**2) /(M_chi* ((2*m-T_chi)**2)) )**0.5)
    result = quad(int_base, T_min, T_min*1000, args=(i))
    
    #print(T_min)
    #print(result)
    return result[0]

def phi_chi_dt(T_chi,i,lumda):
    def G(q2):
        return 1/((1+q2/(lumda**2))**2)
    return sigma_chi*(G(2*M_chi*T_chi)**2)*integration(T_chi,i)

def phi_result(T_chi):
    
    proton = 0
    helium = 1
    
    lumda_proton = 770e06
    lumda_helium = 410e06
    mem1 = phi_chi_dt(T_chi,proton,lumda_proton)
    mem2 = phi_chi_dt(T_chi,helium,lumda_helium)
    return T_chi*(mem1+mem2)

def show_plot(logx,func):
    
    x = np.exp(logx*np.log(10))
    y = []
    for k in range(len(x)):
        y.append(func(x[k]))
    y = np.array(y)
    logy = np.log(y/np.log(10))
    
    plt.plot(logx, logy, color ='blue')
    plt.show()

if __name__== '__main__':

    logT_chi = np.linspace(4,8,100)

    logR = np.linspace(-1,2,100)
    #show_plot(logT_chi,phi_result)
    show_plot(logR,dR_dI)
    #integration(T_chi,proton)
