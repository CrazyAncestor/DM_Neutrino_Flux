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

#DM
M_DM = 10e06

#NFW Parameter
rho_s = 0.184e9
rs=24.42*3.08567758e21

#cross section (Neutrino and DM)
cs = 1e-28

def Standard_DM_flux(m_dm,e_per_nu,start,end,n_total):
    gamma = kim.energy_kicked_by_neutrino(E_per_nu, M_nu,m_dm)/m_dm
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

def Smeared_DM_flux(m_dm,e_per_nu,start,end,n_total,alpha):
    gamma = kim.energy_kicked_by_neutrino(E_per_nu, M_nu,m_dm)/m_dm
    beta = (1-gamma**(-2))**0.5
    R = (np.sum((start-end)**2))**0.5
    pos = -start+end
    time_delay = R*(1/beta-1)/3e10
    print("time delay(s):"+str(time_delay))

    T = R/3e10
    t = np.linspace(0,time_delay,100)
    
    def phi_E(tp):
        def dphi_de(E):
            beta_E = (1-(M_DM/E)**2)**0.5
            def f_energy_nu(E):
                e  = np.array(E)/E_per_nu
                return (e**alpha)*np.exp(-(alpha+1)*e)

            def special_ratio(E):
                return beta_E/(1-beta_E)
            
            def n(E):
                l = 3e10*(T - beta_E*tp/(1-beta_E))
                if l<0 :
                    l =0.
                r= np.sum((pos*l/R+start)**2)**0.5
                x= r/rs
                return 1/(x*(1+x)*(1+x))
            return f_energy_nu(E)*special_ratio(E)*n(E)
            
        E_DM_ave = kim.energy_kicked_by_neutrino(E_per_nu, M_nu,m_dm)
        delta_E = E_DM_ave - m_dm
        result = integrate.nquad(dphi_de, [[E_DM_ave-delta_E,E_DM_ave+delta_E]])[0]
        k = cs *n_total*rho_s/m_dm/(4*np.pi*(R**2))
        return k*result
    
    phi=[]
    for m in range(len(t)):
        phi.append(phi_E(t[m]))
    phi=np.array(phi)

    
    
    
    
    
    plt.plot(t, phi, color ='blue', label = 'DM Flux')
    plt.xlabel('Time (s)')
    plt.ylabel('Flux (#/cm^2*s)')
    plt.legend(loc='upper right')
    
    plt.show()
    


    
if __name__== '__main__':
    
    
    print("Total number of neutrino:"+str(E_total_nu/E_per_nu))

    start=np.array([0.87*3.08567758e21,0,2.4*3.08567758e18])
    end =np.array([8.7*3.08567758e21,0,24*3.08567758e18])
    
    
    Smeared_DM_flux(M_DM,E_per_nu ,start,end,E_total_nu/E_per_nu,2.)
    
    