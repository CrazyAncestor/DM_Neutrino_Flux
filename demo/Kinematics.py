## This module is to compute the energy and momentum kicked of DM particle, kicked by a neutrino of energy E
import numpy as np
import matplotlib.pyplot as plt

M_nu = 0.32 # Unit:ev/c2
def p_kicked_by_neutrino(E,m_nu,m_DM,n0):
    E = np.array(E)
    beta = (((E**2)-(m_nu**2))**0.5)/(E+m_DM)
    b = beta/(1-beta**2)
    return m_DM*np.tensordot(b,n0,axes=0) 

def energy_kicked_by_neutrino(E,m_nu,m_DM):
    E = np.array(E)
    beta = (((E**2)-(m_nu**2))**0.5)/(E+m_DM)
    
    return m_DM/(1-beta**2)

if __name__== '__main__':
    M_DM = 1e07
    E_max = 10e06
    
    E = np.linspace(M_nu,E_max,1000)
    n0 = np.array([1,0,0])

    p = p_kicked_by_neutrino(E,M_nu,M_DM,n0)[:,0]
    eng = energy_kicked_by_neutrino(E,M_nu,M_DM)

    plt.plot(E,p, color ='blue', linewidth =1, linestyle ='-', label = 'momentum mag')
    plt.plot(E,eng, color ='red', linewidth =1, linestyle ='-', label = 'energy')
    plt.legend(loc='upper left')
    plt.show()
    
