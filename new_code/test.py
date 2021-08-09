import numpy as np
import math as mt
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import scipy.optimize as opt

#Particle Property
#kpc
kpc_in_cm = 3.08567758e21
#light speed
vc = 3e10

#Neutrino
M_nu = 0.32 # Unit:ev/c2
E_total_nu = 3.6e53*6.24150913e11 #Total energy of neutrinos #Transfer 2e51erg to unit of MeV
E_per_nu = 10e6 #Mean energy of each neutrino #estimated value
n_tot = E_total_nu/E_per_nu
#DM
M_DM = 1e03

#NFW Parameter
rho_s = 0.184e9
rs=24.42*kpc_in_cm

#cross section (Neutrino and DM)
cs = 1e-30
def norm(start,end):
    R = (np.sum((start-end)**2))**0.5
    l = end -start
    beta = 0.3
    def f(r,theta):
        psi = theta - np.arcsin(r/R*np.sin(theta))
        dpsi_dtheta = 1- r/R*np.cos(theta) /(1-(r/R*np.sin(theta))**2)**0.5
        x=(np.sum((start + np.array([r*np.cos(psi),0,r*np.sin(psi)]))**2))**0.5/rs
        cos_theta_psi=((R**2+(r*np.cos(psi))**2-2*R*r*np.cos(psi))/(R**2+r**2-2*R*r*np.cos(psi)))**0.5
        theta = np.arctan(1/(1/(np.tan(psi))-r/(R*np.sin(psi))))
        sec = 1/np.cos(theta)
        dn_domega= (sec**3)*(1-beta*beta)/((sec**2-beta**2)**2)/np.pi
        
        return np.sin(psi) *dpsi_dtheta *R/(R**2+r**2-2*R*r*np.cos(psi))*np.cos(theta)*dn_domega
    
    def gl(l,theta): 
        r = (l**2+R**2-2*R*l*np.cos(theta))**0.5
        alpha = np.arcsin(R/r*np.sin(theta))
        psi = alpha - theta
        sec = 1/np.cos(alpha)
        
        dn_domega= (sec**3)*(1-beta*beta)/((sec**2-beta**2)**2)*2
        if alpha>=np.pi/2 or np.isnan(alpha):
            dn_domega= 0
        L = n_tot/4/np.pi
        def rho(r):
            x = r/R
            return rho_s/(x*(1+x)**2)
        return np.sin(theta) *np.cos(theta)/R/np.pi*dn_domega*3/4*rho(r)*L/r**2*cs/M_DM
    
    def g(r,psi): 
        l = (R**2+r**2-2*R*r*np.cos(psi))**0.5
        theta = np.arcsin(r/l*np.sin(psi))
        alpha = psi+theta
        sec = 1/np.cos(alpha)

        L = n_tot/4/np.pi
        def rho(r):
            x = r/R
            return rho_s/(x*(1+x)**2)
        dn_domega= (sec**3)*(1-beta*beta)/((sec**2-beta**2)**2)*4
        if alpha>=np.pi/2 or np.isnan(alpha):
            dn_domega= 0
        
        return np.sin(psi) *r**2/l**2*np.cos(theta)/R /np.pi*3/4*rho(r)*L/r**2*cs/M_DM*dn_domega

    def h(r,alpha): 
        theta = np.arcsin(r/R*np.sin(alpha))
        psi = alpha - theta
        l = (R**2+r**2-2*R*r*np.cos(psi))**0.5
        sec = 1/np.cos(alpha)
        dn_domega= (sec**3)*(1-beta*beta)/((sec**2-beta**2)**2)*2
        dpsi_dalpha = 1- r/R*np.cos(alpha) /(1-(r/R*np.sin(alpha))**2)**0.5
        return np.sin(psi) *r**2/l**2*np.cos(theta)*dpsi_dalpha/R*dn_domega

    #result =integrate.dblquad(gl, 0, np.pi/2, lambda theta: 0, lambda theta: R*2*np.cos(theta))[0]*R
    result =integrate.dblquad(g, 0, np.pi, lambda theta: 0.0, lambda theta: R)[0]*R
    #result =integrate.dblquad(h, 0, np.pi/2, lambda alpha: 0, lambda alpha: R)[0]
    #print(result[1])
    L_dm = result*2*np.pi
    print("DM Number(1/cm^2):"+str(L_dm))
    return L_dm
def DM_number(m_dm,e_per_nu,start,end,n_total):
    beta = (E_per_nu**2- M_nu**2)**0.5/(E_per_nu+M_DM)
    gamma = (1-beta**(2))**0.5  
    time_delay = np.sum((start-end)**2)**0.5*(1/beta-1)/(vc)
    

    R = (np.sum((start-end)**2))**0.5
    l = end -start
    def f(r,theta):
        psi = theta - np.arcsin(r/R*np.sin(theta))
        dpsi_dtheta = 1- r/R*np.cos(theta) /(1-(r/R*np.sin(theta))**2)**0.5
        x=(np.sum((start + np.array([r*np.cos(psi),0,r*np.sin(psi)]))**2))**0.5/rs
        cos_theta_psi=((R**2+(r*np.cos(psi))**2-2*R*r*np.cos(psi))/(R**2+r**2-2*R*r*np.cos(psi)))**0.5
        theta = np.arctan(1/(1/(np.tan(psi))-r/(R*np.sin(psi))))
        sec = 1/np.cos(theta)
        dn_domega= (sec**3)*(1-beta*beta)/((sec**2-beta**2)**2)/np.pi
        
        return np.sin(psi) *dpsi_dtheta *R/(R**2+r**2-2*R*r*np.cos(psi))*dn_domega/(x*(1+x)*(1+x))
        
    k = n_total*rho_s*cs/m_dm/(4*np.pi)
    r0 = 0.01*rs
    result =integrate.dblquad(f, 0, np.pi/2., lambda theta: r0, lambda theta: R)
    print(result[1])
    L_dm = result[0]*2*np.pi*k/R
    print("DM Number(1/cm^2):"+str(L_dm))
    return L_dm
    

    

if __name__== '__main__':
    
    beta = (E_per_nu**2- M_nu**2)**0.5/(E_per_nu+M_DM)
    gamma = (1-beta**(2))**0.5      
    print(beta)
    start=np.array([8.7*0.0*kpc_in_cm,0,0.*3.08567758e18])
    end =np.array([8.7*kpc_in_cm,0,0.*3.08567758e18])
    
    #ref = DM_number(M_DM,E_per_nu ,start,end,E_total_nu/E_per_nu)
    ref = norm(start,end)