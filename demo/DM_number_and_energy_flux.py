## This module aims to compute the number and energy flux of DM particles from a supernova
import numpy as np
import matplotlib.pyplot as plt
import DM_p_distribution as dmp
import scipy.integrate as integrate
import pynverse as vs

M_nu = 0.32 # Unit:ev/c2
E_total_nu = 2e51/6.24150913e11 #Total energy of neutrinos #Transfer 2e51erg to unit of MeV
E_per_nu = 12e6 #Mean energy of each neutrino #estimated value

def f_energy_nu(e,alpha):
    e  = np.array(e)
    return (e**alpha)*np.exp(-(alpha+1)*e)

def f_evolution(m_dm,e_max,e_per_nu,alpha,start,end):
    E = np.linspace(m_dm/e_per_nu,e_max/e_per_nu,1000)
    f = f_energy_nu(E,alpha)

    def normalize(y):
        m = len(y)
        sum = np.sum(y)
        return y*m/sum

    def opacity(start,end):
        cross_section = 10e-40
        rho_s = 0.184e9
        rs=24.42*3.08567758e21
        L = (np.sum((start-end)**2))**0.5
        l = end -start
        def f(t):
            r=(np.sum((start+l*t)**2))**0.5
            x= r/rs
            return 1/(x*(1+x)*(1+x))
        r = integrate.quad(f,0,1)[0]*L*rho_s*cross_section
        return r

    r_decay = np.exp(-opacity(start,end))
    print(r_decay)

    def energy_conv(E):
        return E+m_dm-dmp.energy_kicked_by_neutrino(E,M_nu,m_dm)
    f_n = f_energy_nu(vs.inversefunc(energy_conv, y_values = E*e_per_nu,domain=[M_nu,e_max*10])/e_per_nu,alpha)*(1-r_decay)
    f_n += f*r_decay 
    return E,f_n
    
def DM_flux(m_dm,e_max,e_per_nu,alpha,start,end,n_total):
    

    
    cross_section = 1e-30
    k = cross_section/(8*np.pi*m_dm)

    rho_s = 0.184e9
    rs=24.42*3.08567758e21
    L = (np.sum((start-end)**2))**0.5
    l = end -start
    s1 = rho_s

    t_burst = 1e-3*3e10
    s2 = n_total/(4*np.pi*(L**2)*t_burst)
    def f(t):
        r=(np.sum((start+l*t)**2))**0.5
        x= r/rs
        return 1/(x*(1+x)*(1+x)*(t**2))
    result = integrate.quad(f,(1e-6) ,1)[0]*L*k*s1*s2

    r=(np.sum((start)**2))**0.5
    x= r/rs
    result+= L*k*s1*s2/(x*(1+x)*(1+x)) *( L/1e7-1e6)

    r_nu=n_total/(4*np.pi*L*L)
    print("DM Flux:"+str(result))
    print("Neutrino Flux:"+str(r_nu))
    


if __name__== '__main__':
    M_DM = 1e03
    E_max = 30e06
    Alpha = 2.0
    
    print("Total number of nu:"+str(E_total_nu/E_per_nu))

    start=np.array([7*3.08567758e21,4*3.08567758e21,0])
    end =np.array([10*3.08567758e21,0,0])
    
    DM_flux(M_DM,E_max,E_per_nu,Alpha,start,end,E_total_nu/E_per_nu)
    


