## This module aims to analyze the effect of angle distribuiton
import numpy as np
import math as mt
import matplotlib.pyplot as plt
import Kinematics 
import scipy.integrate as integrate


#Particle Property
#Neutrino
M_nu = 0.32 # Unit:ev/c2
E_total_nu = 3.6e53*6.24150913e11 #Total energy of neutrinos #Transfer 2e51erg to unit of MeV
E_per_nu = 10e6 #Mean energy of each neutrino #estimated value

#DM
M_DM = 1e05

#NFW Parameter
rho_s = 0.184e9
rs=24.42*3.08567758e21

#cross section (Neutrino and DM)
cs = 1e-30

def angle_norm(beta,r,R):
    def f(psi):
        rp = (R**2+r**2-2*R*r*mt.cos(psi))**0.5
        cos_theta_psi = ((R**2+(r*mt.cos(psi))**2-2*R*r*mt.cos(psi))/(R**2+r**2-2*R*r*mt.cos(psi)))**0.5
        
        def g(psi):
            def theta(psi):
                return mt.atan(1/(1/(mt.tan(psi))-r/(R*mt.sin(psi))))
            
            theta = theta(psi)
            sec = 1/mt.cos(theta)
            
            return 4*mt.tan(theta)*(sec**2)*(1-beta*beta)/((sec**2-beta**2)**2)/mt.sin(theta)
        return mt.sin(psi)*g(psi)*cos_theta_psi/(rp**2)/2.
    
    theta_max = np.pi/2
    def psi(theta):
        return theta - mt.asin(r/R*mt.sin(theta))
    psi_max = psi(theta_max)
    return integrate.nquad(f, [[0,psi_max]])[0] 

def DM_number(m_dm,e_per_nu,start,end,n_total):
    beta = (E_per_nu**2- M_nu**2)**0.5/(E_per_nu+M_DM)
    gamma = (1-beta**(2))**0.5  
    time_delay = np.sum((start-end)**2)**0.5*(1/beta-1)/(3e10)
    

    R = (np.sum((start-end)**2))**0.5
    l = end -start
    def f(r,theta):
        psi = theta - mt.asin(r/R*mt.sin(theta))
        dpsi_dtheta = 1- r/R*mt.cos(theta) /(1-(r/R*mt.sin(theta))**2)**0.5
        x=(np.sum((start + np.array([r*mt.cos(psi),0,r*mt.sin(psi)]))**2))**0.5/rs
        cos_theta_psi=((R**2+(r*mt.cos(psi))**2-2*R*r*mt.cos(psi))/(R**2+r**2-2*R*r*mt.cos(psi)))**0.5
        theta = mt.atan(1/(1/(mt.tan(psi))-r/(R*mt.sin(psi))))
        sec = 1/mt.cos(theta)
        dn_domega= (sec**3)*(1-beta*beta)/((sec**2-beta**2)**2)/np.pi
        
        return mt.sin(psi)*cos_theta_psi *dpsi_dtheta *R/(R**2+r**2-2*R*r*mt.cos(psi))*dn_domega/(x*(1+x)*(1+x))
        
    k = n_total*rho_s*cs/m_dm/(4*np.pi)
    result =integrate.dblquad(f, 0, np.pi/2., lambda theta: 0, lambda theta: R)
    print(result[1])
    L_dm = result[0]*2*np.pi*k/R
    print("DM Number(1/cm^2):"+str(L_dm))
    

def DM_number_original(m_dm,e_per_nu,start,end,n_total):
    beta = (E_per_nu**2- M_nu**2)**0.5/(E_per_nu+M_DM)
    gamma = (1-beta**(2))**0.5  
    time_delay = np.sum((start-end)**2)**0.5*(1/beta-1)/(3e10)
    

    R = (np.sum((start-end)**2))**0.5
    l = end -start
    def f(t):
        r=(np.sum((start+l*t)**2))**0.5
        x= r/rs
        return 1/(x*(1+x)*(1+x))
    k = n_total*rho_s*cs/m_dm/(4*np.pi) /R
    
    L_dm = integrate.nquad(f, [[0,1.]])[0]*k
    print("DM Number original(1/cm^2):"+str(L_dm))
    

def Spectrum(m_dm,e_per_nu,start,end,n_total):
    beta = (E_per_nu**2- M_nu**2)**0.5/(E_per_nu+M_DM)
    gamma = (1-beta**(2))**0.5  
    time_delay = np.sum((start-end)**2)**0.5*(1/beta-1)/(3e10)
    

    R = (np.sum((start-end)**2))**0.5
    l = end -start
    def dL_dEdR(r,theta):
        
        dE_dtheta = M_DM *(beta**2)*mt.sin(theta)/(1-beta**2)
        def f(r):
            psi = theta - mt.asin(r/R*mt.sin(theta))
            x=(np.sum((start + np.array([r*mt.cos(psi),0,r*mt.sin(psi)]))**2))**0.5/rs
            cos_theta_psi= mt.cos(theta-psi)
            sec = 1/mt.cos(theta)
            dpsi_dtheta = 1- r/R*mt.cos(theta) /(1-(r/R*mt.sin(theta))**2)**0.5
            dn_domega= (sec**3)*(1-beta*beta)/((sec**2-beta**2)**2)/np.pi
            
            return mt.sin(psi)*cos_theta_psi*R /(R**2+r**2-2*R*r*mt.cos(psi))*dn_domega*dpsi_dtheta/(x*(1+x)*(1+x))
        return  f(r)

    
    E_0 = M_DM*1.001
    E_f = M_DM/(1-beta**2)*(1+(beta**2))*0.999
    
    norm = integrate.dblquad(dL_dEdR, 0, np.pi/2, lambda theta: 0, lambda theta: R)[0]*2*np.pi  
    print(norm)
    def dL_dE(theta):
        
        sec = 1/mt.cos(theta)
        dE_dtheta = M_DM *(beta**2)/(1-beta**2) *4*(gamma**2)*mt.tan(theta)*(sec**2) /(1+(gamma**2)*(mt.tan(theta)**2))
        def f(r):
            psi = theta - mt.asin(r/R*mt.sin(theta))
            x=(np.sum((start + np.array([r*mt.cos(psi),0,r*mt.sin(psi)]))**2))**0.5/rs
            cos_theta_psi= mt.cos(theta-psi)
            sec = 1/mt.cos(theta)
            dpsi_dtheta = 1- r/R*mt.cos(theta) /(1-(r/R*mt.sin(theta))**2)**0.5
            dn_domega= (sec**3)*(1-beta*beta)/((sec**2-beta**2)**2)/np.pi
        
            return mt.sin(psi)*cos_theta_psi*R /(R**2+r**2-2*R*r*mt.cos(psi))*dn_domega*dpsi_dtheta/(x*(1+x)*(1+x))
        return  integrate.nquad(f, [[0,R]])[0]  /dE_dtheta
    
    theta =np.linspace(0.001,np.pi/2,1000)
    E = [M_DM/(1-beta**2)*(1+(beta**2)*(1-(gamma**2)*(mt.tan(theta[i])**2)) /(1+(gamma**2)*(mt.tan(theta[i])**2)))for i in range(0,1000)]
    print()
    spec = [dL_dE(theta[i])/norm for i in range(0,1000)]
   
    plt.plot(E, spec, color ='blue')
    plt.xlabel('E(eV)')
    plt.ylabel('dL_chi/dE(1/cm**2 eV)')
    
    plt.show()
    

if __name__== '__main__':
    
    beta = (E_per_nu**2- M_nu**2)**0.5/(E_per_nu+M_DM)
    gamma = (1-beta**(2))**0.5      
    print(beta)
    print("Angle Norm:"+str(angle_norm(beta,0.5,1)))

    start=np.array([0.87*3.08567758e21,0,2.4*3.08567758e18])
    end =np.array([8.7*3.08567758e21,0,24*3.08567758e18])
    
    DM_number(M_DM,E_per_nu ,start,end,E_total_nu/E_per_nu)

    DM_number_original(M_DM,E_per_nu ,start,end,E_total_nu/E_per_nu)
    Spectrum(M_DM,E_per_nu ,start,end,E_total_nu/E_per_nu)
