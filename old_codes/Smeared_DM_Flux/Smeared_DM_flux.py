## This module aims to compute the number and energy flux of DM particles from a supernova
import numpy as np
import matplotlib.pyplot as plt
import Kinematics as kim
import scipy.integrate as integrate


#Particle Property
#Neutrino
M_nu = 0.32 # Unit:ev/c2
E_total_nu = 3.6e53*6.24150913e11 #Total energy of neutrinos #Transfer 2e51erg to unit of MeV
E_per_nu = 10e6 #Mean energy of each neutrino #estimated value

#DM
M_DM_light = 1e03
M_DM_heavy = 1e07

#NFW Parameter
rho_s = 0.184e9
rs=24.42*3.08567758e21

#cross section (Neutrino and DM)
cs = 1e-30

Alpha = 3.

def Standard_DM_flux(m_dm,e_per_nu,start,end,n_total,beta):
    
    R = (np.sum((start-end)**2))**0.5
    pos = -start+end
    time_delay = R*(1/beta-1)/3e10

    T = R/3e10
    t = np.linspace(0,time_delay,200)
    
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
    c = 3e10 #in unit of cm/s
    phi = cs *n_total*rho_s/m_dm/(4*np.pi*(R**2))*n(t) *beta/(1-beta)*c
    return t,phi


def Smeared_DM_flux(m_dm,e_per_nu,start,end,n_total,beta,alpha):
    
    R = (np.sum((start-end)**2))**0.5
    pos = -start+end
    time_delay = R*(1/beta-1)/3e10

    T = R/3e10
    t = np.linspace(0,4*time_delay,200)
    
    E_DM_ave = kim.energy_kicked_by_neutrino(E_per_nu, M_nu,m_dm)
    delta_E = E_DM_ave - m_dm
    def phi_E(tp):
        def dphi_de(E):
            beta_E = (1-(m_dm/E)**2)**0.5
            def f_energy_nu(E):
                e  = np.array(E)/E_DM_ave
                return (e**alpha)*np.exp(-(alpha+1)*e)

            def special_ratio(E):
                return beta_E/(1-beta_E)
            
            def n(E):
                l = 3e10*(T - beta_E*tp/(1-beta_E))
                if l<0 :
                    return 0.
                r= np.sum((pos*l/R+start)**2)**0.5
                x= r/rs
                return 1/(x*(1+x)*(1+x))
            return f_energy_nu(E)*special_ratio(E)*n(E)
        def base(E):
            def f_energy_nu(E):
                e  = np.array(E)/E_DM_ave
                return (e**alpha)*np.exp(-(alpha+1)*e)
            return f_energy_nu(E)
            
        scope = [E_DM_ave-delta_E,E_DM_ave+100*delta_E]
        result = integrate.nquad(dphi_de, [scope])[0]
        basis = integrate.nquad(base, [scope])[0]
        c = 3e10 #in unit of cm/s
        k = cs *n_total*rho_s/m_dm/(4*np.pi*(R**2))*c
        return k*result/basis
    
    phi=[]
    for m in range(len(t)):
        phi.append(phi_E(t[m]))
    phi=np.array(phi)

    return t,phi
    
if __name__== '__main__':
    gamma_light = kim.energy_kicked_by_neutrino(E_per_nu, M_nu,M_DM_light)/M_DM_light
    beta_light = (1-gamma_light**(-2))**0.5

    gamma_heavy = kim.energy_kicked_by_neutrino(E_per_nu, M_nu,M_DM_heavy)/M_DM_heavy
    beta_heavy = (1-gamma_heavy**(-2))**0.5
    
    print("Total number of neutrino:"+str(E_total_nu/E_per_nu))

    start=np.array([0.87*3.08567758e21,0,2.4*3.08567758e18])
    end =np.array([8.7*3.08567758e21,0,24*3.08567758e18])
    
    t_light,phi_light = Smeared_DM_flux(M_DM_light,E_per_nu ,start,end,E_total_nu/E_per_nu,beta_light,Alpha)
    #t_heavy,phi_heavy = Smeared_DM_flux(M_DM_heavy,E_per_nu ,start,end,E_total_nu/E_per_nu,beta_heavy,Alpha)
    t_std,phi_std = Standard_DM_flux(M_DM_light,E_per_nu ,start,end,E_total_nu/E_per_nu,beta_light)
    
    
    plt.plot(t_std, phi_std, color ='red', label = 'One Energy DM Flux')
    
    plt.plot(t_light, phi_light, color ='blue', label = '1keV DM')
    #plt.plot(t_heavy, phi_heavy, color ='red', label = '10MeV DM')
    plt.xlabel('Time (s)')
    plt.ylabel('Flux (#/cm^2*s)')
    plt.legend(loc='upper right')
    plt.savefig("KeV1.png")
    plt.show()

    
    

    
