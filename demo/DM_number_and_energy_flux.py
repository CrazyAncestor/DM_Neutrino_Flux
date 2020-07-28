## This module aims to compute the number and energy flux of DM particles from a supernova
import numpy as np
import matplotlib.pyplot as plt
import DM_p_distribution as dmp
import scipy.integrate as integrate
import pynverse as vs

#Particle Property
#Neutrino
M_nu = 0.32 # Unit:ev/c2
E_total_nu = 1e53*6.24150913e11 #Total energy of neutrinos #Transfer 2e51erg to unit of MeV
E_per_nu = 10e6 #Mean energy of each neutrino #estimated value
Alpha = 2.0 #Energy Distribution Parameter

#DM
M_DM = 1e03
E_max = 60e06

#NFW Parameter
rho_s = 0.184e9
rs=24.42*3.08567758e21

#Burst Time
t_burst = 10*3e10

#cross section
cs = 1e-30


def DM_flux(m_dm,e_max,e_per_nu,alpha,start,end,n_total):
    k = 1/(8*np.pi*m_dm)

    
    L = (np.sum((start-end)**2))**0.5
    l = end -start
    s1 = rho_s

    
    s2 = n_total/(4*np.pi*t_burst)
    
    r=(np.sum((start)**2))**0.5
    x= r/rs
    
    result = cs*k*s1*s2/(x*(1+x)*(1+x)) *( 1/1e7)

    r_nu=n_total/(4*np.pi*L*L)
    print("DM Flux:"+str(result))
    print("Neutrino Flux:"+str(r_nu))
    
if __name__== '__main__':
    
    
    print("Total number of neutrino:"+str(E_total_nu/E_per_nu))

    start=np.array([1*3.08567758e21,0*3.08567758e21,0])
    end =np.array([10*3.08567758e21,0,0])
    
    #DM_flux(M_DM,E_max,E_per_nu ,Alpha,start,end,E_total_nu/E_per_nu)

    gamma = dmp.energy_kicked_by_neutrino(E_per_nu, M_nu,M_DM)/M_DM
    beta = (1-gamma**(-2))**0.5
    time_delay = np.sum((start-end)**2)**0.5*(1/beta-1)/(3e10)
    print(time_delay)
    