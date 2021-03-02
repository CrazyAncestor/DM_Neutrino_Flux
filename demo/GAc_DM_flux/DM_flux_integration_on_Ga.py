## This module aims to compute the intergrated number flux of DM particles from SNe across the entire Milky Way
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate


#Particle Property
#Neutrino
M_nu = 0.32 # Unit:ev/c2
E_total_nu = 3.6e53*6.24150913e11 #Total energy of neutrinos #Transfer 2e51erg to unit of MeV
E_per_nu = 10e6 #Mean energy of each neutrino #estimated value

#DM
M_DM = 10e06

#NFW Parameter
rho_s = 0.184e9
rs=24.42*3.08567758e21

#cross section (Neutrino and DM)
cs = 1e-30


#Galactic Property  (ref: ADAMS, SCOTT et al. arXiv:1306.0559 [astro-ph.HE])

Rd = 2.9e03#kpc
H = 95 #pc
r_s =24.2e03

def DM_flux(m_dm,e_per_nu,start,end,n_total):
    R = (np.sum((start-end)**2))**0.5
    l = end -start
    def f(t):
        r=(np.sum((start+l*t)**2))**0.5
        x= r/rs
        return 1/(x*(1+x)*(1+x))
    k = n_total*rho_s*cs/m_dm/(4*np.pi) /R
    
    result_spec = integrate.nquad(f, [[0,1.]])[0]*k
    result_ave = rho_s*n_total*cs /( 4*np.pi*m_dm)* 0.000488329 /3.08567758e18
    print("DM Flux(1/s*cm^2) SPEC:"+str(result_spec/(86400*36525)))
    print("DM Flux(1/s*cm^2) AVE:"+str(result_ave/(86400*36525)))
    
    
if __name__== '__main__':
    
    
    print("Total number of neutrino:"+str(E_total_nu/E_per_nu))

    start=np.array([0.87*3.08567758e21,0,2.4*3.08567758e18])
    end =np.array([8.7*3.08567758e21,0,24*3.08567758e18])
    
    DM_flux(M_DM,E_per_nu ,start,end,E_total_nu/E_per_nu)

    