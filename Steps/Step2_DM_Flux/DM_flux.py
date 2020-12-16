## This module aims to compute the number and energy flux of DM particles from a supernova
import numpy as np
import matplotlib.pyplot as plt
from ..Step1_Kinematics import Kinematics
import scipy.integrate as integrate


#Particle Property
#Neutrino
M_nu = 0.32 # Unit:eV/c2
E_total_nu = 2e51*6.24150913e11 #Total energy of neutrinos #Transfer 2e51erg to unit of eV
E_per_nu = 10e6 #Mean energy of each neutrino #estimated value

#DM
M_DM = 10e06

#NFW Parameter
rho_s = 0.184e9
rs=24.42*3.08567758e21

#cross section (Neutrino and DM)
cs = 1e-28

def DM_flux(m_dm,e_per_nu,start,end,n_total):
    gamma = Kinematics.energy_kicked_by_neutrino(E_per_nu, M_nu,m_dm)/m_dm
    beta = (1-gamma**(-2))**0.5
    R = (np.sum((start-end)**2))**0.5
    pos = -start+end
    time_delay = R*(1/beta-1)/3e10
    print("time delay(s):"+str(time_delay))

    T = R/3e10
    t = np.linspace(0,time_delay,100)
    
    def n_ori(l):
        
        def get_mod(x):
            x2 = x**2
            return (x2[:,0] + x2[:,1] +x2[:,2])**0.5
        
        r= get_mod(np.tensordot(l,pos,axes=0)/R+np.tensordot(np.ones(l.shape),start,axes=0))
        
        x= r/rs
        return 1/(x*(1+x)*(1+x))

    def n(t):
        l = 3e10*(T - beta*t/(1-beta))
        l[-1]= 0.
        return n_ori(l)
    phi = cs *n_total*rho_s/m_dm/(4*np.pi*(R**2))*n(t) *beta/(1-beta)
    plt.plot(t, phi, color ='blue', label = 'DM Flux')
    plt.xlabel('Time (s)')
    plt.ylabel('Flux (#/cm^2*s)')
    plt.legend(loc='upper right')
    
    plt.show()
    

def DM_number(m_dm,e_per_nu,start,end,n_total):
    gamma = Kinematics.energy_kicked_by_neutrino(E_per_nu, M_nu,m_dm)/m_dm
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
"""
if __name__== '__main__':
    
    
    print("Total number of neutrino:"+str(E_total_nu/E_per_nu))

    start=np.array([0.87*3.08567758e21,0,2.4*3.08567758e18])
    end =np.array([8.7*3.08567758e21,0,24*3.08567758e18])
    
    DM_number(M_DM,E_per_nu ,start,end,E_total_nu/E_per_nu)
    DM_flux(M_DM,E_per_nu ,start,end,E_total_nu/E_per_nu)
    
    """