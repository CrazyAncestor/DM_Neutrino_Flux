## This module aims to compute the number and energy flux of DM particles from a supernova
import numpy as np
import matplotlib.pyplot as plt
import Kinematics as kim
import scipy.integrate as integrate


#Particle Property
#Neutrino
M_nu = 0.32 # Unit:ev/c2
E_total_nu = 1e53*6.24150913e11 #Total energy of neutrinos #Transfer 2e51erg to unit of MeV
E_per_nu = 10e6 #Mean energy of each neutrino #estimated value
Alpha = 2.0 #Energy Distribution Parameter

#DM
M_DM = 10e06
E_max = 60e06

#NFW Parameter
rho_s = 0.184e9
rs=24.42*3.08567758e21

#Burst Time
t_burst = 10*3e10

#cross section
cs = 1e-28

#SN
Ri=1e7

def DM_number(m_dm,e_max,e_per_nu,alpha,start,end,n_total):
    gamma = kim.energy_kicked_by_neutrino(E_per_nu, M_nu,m_dm)/m_dm
    beta = (1-gamma**(-2))**0.5
    time_delay = np.sum((start-end)**2)**0.5*(1/beta-1)/(3e10)
    print("time delay:"+str(time_delay))

    R = (np.sum((start-end)**2))**0.5
    l = end -start
    def f(t):
        r=(np.sum((start+l*t)**2))**0.5
        x= r/rs
        return 1/(x*(1+x)*(1+x))
    k = n_total*rho_s*cs/m_dm/(4*np.pi) /R
    
    phi_dm = integrate.nquad(f, [[0,1.]])[0]*k
    phi_nu = n_total/(4*np.pi*R*R)
    
    print("DM Number(1/s*cm^2):"+str(phi_dm/(86400*36525)))
    print("Neutrino Number:"+str(phi_nu))
    
if __name__== '__main__':
    
    
    print("Total number of neutrino:"+str(E_total_nu/E_per_nu))

    start=np.array([0.87*3.08567758e21,0,2.4*3.08567758e18])
    end =np.array([8.7*3.08567758e21,0,24*3.08567758e18])
    
    DM_number(M_DM,E_max,E_per_nu ,Alpha,start,end,E_total_nu/E_per_nu)

    
    