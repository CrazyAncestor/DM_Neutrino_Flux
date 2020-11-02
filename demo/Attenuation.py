## This module aims to analyze the attenuation effect of electron
import numpy as np
import math as mt
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import Kinematics as kim
from pynverse import inversefunc

#Particle Property
#Neutrino
M_nu = 0.32 # Unit:ev/c2
E_total_nu = 1e53*6.24150913e11 #Total energy of neutrinos #Transfer 2e51erg to unit of MeV
E_per_nu = 10e6 #Mean energy of each neutrino #estimated value

#Electron
M_e = 0.511e06
ne = 5.5/2.*6e23

#DM
M_DM = 1e03

#NFW Parameter
rho_s = 0.184e9
rs=24.42*3.08567758e21

#cross section (Electron and DM)
cs = 1e-28

T0 = kim.energy_kicked_by_neutrino(E_per_nu, M_nu,M_DM)-M_DM
def depth(T):
    a = 2*M_DM
    b = (M_DM+M_e)**2/(M_e*2)
    u= -( (a-b)*np.log((a+T)/(a+T0)) + b*np.log((T)/(T0)) )/a
    return u/cs/ne

def energy_attenuation(z):
    inv = inversefunc(depth, y_values=z,domain=[T0*0.0000001,T0])
    return inv

if __name__== '__main__':
    print(T0)
    print(depth(T0/2))
    print(energy_attenuation(1e5)/T0)
