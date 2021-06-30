## This module aims to analyze the effect of angle distribuiton
import numpy as np
import math as mt
import matplotlib.pyplot as plt
import Kinematics 
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

#DM
M_DM = 1e03

#NFW Parameter
rho_s = 0.184e9
rs=24.42*kpc_in_cm

#cross section (Neutrino and DM)
cs = 1e-30

def angle_norm(beta,r,R):
    def f(psi):
        rp = (R**2+r**2-2*R*r*np.cos(psi))**0.5
        cos_theta_psi = ((R**2+(r*np.cos(psi))**2-2*R*r*np.cos(psi))/(R**2+r**2-2*R*r*np.cos(psi)))**0.5
        
        def g(psi):
            def theta(psi):
                return np.arctan(1/(1/(np.tan(psi))-r/(R*np.sin(psi))))
            
            theta = theta(psi)
            sec = 1/np.cos(theta)
            
            return 4*np.tan(theta)*(sec**2)*(1-beta*beta)/((sec**2-beta**2)**2)/np.sin(theta)
        return np.sin(psi)*g(psi)*cos_theta_psi/(rp**2)/2.
    
    theta_max = np.pi/2
    def psi(theta):
        return theta - np.arcsin(r/R*np.sin(theta))
    psi_max = psi(theta_max)
    return integrate.nquad(f, [[0,psi_max]])[0] 

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
    
def DM_flux(m_dm,e_per_nu,start,end,n_total,t):
    beta = (E_per_nu**2- M_nu**2)**0.5/(E_per_nu+M_DM)
    a =(vc)*t
    #print(beta)
    gamma = (1-beta**(2))**0.5  
    time_delay = np.sum((start-end)**2)**0.5*(1/beta-1)/(vc)
    R = (np.sum((start-end)**2))**0.5
    l = end -start
    def get_dtheta_dt(r,t):
        def get_theta():
            def f(theta):
                #v = vc *beta *((1-beta**2)/(np.cos(theta)**(-2)-beta**2))**0.5
                eta  = 2* np.arctan(np.tan(theta)/(1-beta**2)**0.5)
                root = ((1+np.cos(eta))**2 + (np.sin(eta)**2) *(1-beta**2))**0.5
                v = vc *beta *root/(1+beta**2*np.cos(eta))
                l = v*(t-r/vc)
                return l**2+ r**2 - R**2 + 2*r*l*np.cos(theta)
            sol = opt.fsolve(f, np.pi/4)
            return sol[0]
        theta = np.abs(get_theta())
        psi = theta - np.arcsin(r/R*np.sin(theta))
        eta  = 2* np.arctan(np.tan(theta)/(1-beta**2)**0.5)
        l = R*np.sin(psi)/np.sin(theta)
        #v = vc *beta *((1-beta**2)/(np.cos(theta)**(-2)-beta**2))**0.5
        eta  = 2* np.arctan(np.tan(theta)/(1-beta**2)**0.5)
        root = ((1+np.cos(eta))**2 + (np.sin(eta)**2) *(1-beta**2))**0.5

        v = vc *beta *root/(1+beta**2*np.cos(eta))
        sec = np.cos(theta)**(-1)
        dl_dtheta = r*np.sin(theta)-r**2*np.sin(2*theta)/((R**2-r**2*np.sin(theta)**2)**0.5)
        #dv_dtheta = -vc*beta*np.sin(eta/2.)/2. *2*sec**2*(1-beta*beta)**0.5/(sec**2-beta**2)
        d_eta_dtheta = 2*sec**2*(1-beta*beta)**0.5/(sec**2-beta**2)
        
        eba = (1+beta**2*np.cos(eta))
        dv_dtheta = vc*beta *d_eta_dtheta* (  root*(beta**2*np.sin(eta)) - eba/root*(np.sin(eta)+beta**2*np.sin(eta)*np.cos(eta))  )/eba**2 
       

        dt_dtheta = dl_dtheta/v -l/(v**2)*dv_dtheta
        
        return 1/dt_dtheta, theta, psi
    
    
    def f(r):
        dtheta_dt, theta, psi = get_dtheta_dt(r,t)
        dpsi_dtheta = 1- r/R*np.cos(theta) /(1-(r/R*np.sin(theta))**2)**0.5
        x=(np.sum((start + np.array([r*np.cos(psi),0,r*np.sin(psi)]))**2))**0.5/rs
        cos_theta_psi=((R**2+(r*np.cos(psi))**2-2*R*r*np.cos(psi))/(R**2+r**2-2*R*r*np.cos(psi)))**0.5
        
        sec = 1/np.cos(theta)
        dn_domega= (sec**3)*(1-beta*beta)/((sec**2-beta**2)**2)/np.pi
        
        return np.sin(psi) *dpsi_dtheta *R/(R**2+r**2-2*R*r*np.cos(psi))*dn_domega/(x*(1+x)*(1+x)) *dtheta_dt
    
    k = n_total*rho_s*cs/m_dm/(4*np.pi)
    r0 = 0.01*rs
    result = integrate.quad(f,r0,R)[0]
    if result<0:
        return 0
    L_dm = result*2*np.pi*k/R
    return L_dm
    

def DM_number_original(m_dm,e_per_nu,start,end,n_total):
    beta = (E_per_nu**2- M_nu**2)**0.5/(E_per_nu+M_DM)
    gamma = (1-beta**(2))**0.5  
    time_delay = np.sum((start-end)**2)**0.5*(1/beta-1)/(vc)
    

    R = (np.sum((start-end)**2))**0.5
    l = end -start
    def f(t):
        r=(np.sum((start+l*t)**2))**0.5
        x= r/rs
        return 1/(x*(1+x)*(1+x))
    k = n_total*rho_s*cs/m_dm/(4*np.pi) /R
    r0 = 0.01*rs/R
    L_dm = integrate.nquad(f, [[r0,1.]])[0]*k
    print("DM Number original(1/cm^2):"+str(L_dm))
    return  L_dm
    

def Spectrum(m_dm,e_per_nu,start,end,n_total):
    beta = (E_per_nu**2- M_nu**2)**0.5/(E_per_nu+M_DM)
    gamma = (1-beta**(2))**0.5  
    time_delay = np.sum((start-end)**2)**0.5*(1/beta-1)/(vc)
    

    R = (np.sum((start-end)**2))**0.5
    l = end -start
    def dL_dEdR(r,theta):
        
        dE_dtheta = M_DM *(beta**2)*np.sin(theta)/(1-beta**2)
        def f(r):
            psi = theta - np.arcsin(r/R*np.sin(theta))
            x=(np.sum((start + np.array([r*np.cos(psi),0,r*np.sin(psi)]))**2))**0.5/rs
            cos_theta_psi= np.cos(theta-psi)
            sec = 1/np.cos(theta)
            dpsi_dtheta = 1- r/R*np.cos(theta) /(1-(r/R*np.sin(theta))**2)**0.5
            dn_domega= (sec**3)*(1-beta*beta)/((sec**2-beta**2)**2)/np.pi
            
            return np.sin(psi)*cos_theta_psi*R /(R**2+r**2-2*R*r*np.cos(psi))*dn_domega*dpsi_dtheta/(x*(1+x)*(1+x))
        return  f(r)

    
    E_0 = M_DM*1.001
    E_f = M_DM/(1-beta**2)*(1+(beta**2))*0.999
    
    norm = integrate.dblquad(dL_dEdR, 0, np.pi/2, lambda theta: 0, lambda theta: R)[0]*2*np.pi  
    print(norm)
    def dL_dE(theta):
        
        sec = 1/np.cos(theta)
        dE_dtheta = M_DM *(beta**2)/(1-beta**2) *4*(gamma**2)*np.tan(theta)*(sec**2) /(1+(gamma**2)*(np.tan(theta)**2))
        def f(r):
            psi = theta - np.arcsin(r/R*np.sin(theta))
            x=(np.sum((start + np.array([r*np.cos(psi),0,r*np.sin(psi)]))**2))**0.5/rs
            cos_theta_psi= np.cos(theta-psi)
            sec = 1/np.cos(theta)
            dpsi_dtheta = 1- r/R*np.cos(theta) /(1-(r/R*np.sin(theta))**2)**0.5
            dn_domega= (sec**3)*(1-beta*beta)/((sec**2-beta**2)**2)/np.pi
        
            return np.sin(psi)*cos_theta_psi*R /(R**2+r**2-2*R*r*np.cos(psi))*dn_domega*dpsi_dtheta/(x*(1+x)*(1+x))
        return  integrate.nquad(f, [[0.01*R,R]])[0]  /dE_dtheta
    
    theta =np.linspace(0.001,np.pi/2,1000)
    E = [M_DM/(1-beta**2)*(1+(beta**2)*(1-(gamma**2)*(np.tan(theta[i])**2)) /(1+(gamma**2)*(np.tan(theta[i])**2)))for i in range(0,1000)]
    
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

    start=np.array([8.7*0.0*kpc_in_cm,0,0.*3.08567758e18])
    end =np.array([8.7*kpc_in_cm,0,0.*3.08567758e18])
    
    ref = DM_number(M_DM,E_per_nu ,start,end,E_total_nu/E_per_nu)

    t0 = kpc_in_cm/(vc)

    mode = 1
    if mode==0:
        N = 100
        s = np.linspace(-3,-0.5,100)
        #t = [8.7*(0.99+10**(s[i]))*t0 for i in range(len(s))]
    
        t = np.linspace(8.7*(0.99+9e-3),8.7*(0.99+1e-2),N)*t0
        flux = [DM_flux(M_DM,E_per_nu ,start,end,E_total_nu/E_per_nu,t[i]) for i in range(len(t))]

        t2 = np.linspace(8.7*(0.99+1e-2),8.7*(0.99+1e-1),N)*t0
        flux2 = [DM_flux(M_DM,E_per_nu ,start,end,E_total_nu/E_per_nu,t2[i]) for i in range(len(t))]

        t3 = np.linspace(8.7*(0.99+1e-1),8.7*(0.99+1e-0),N)*t0
        flux3 = [DM_flux(M_DM,E_per_nu ,start,end,E_total_nu/E_per_nu,t3[i]) for i in range(len(t))]
        #ref = DM_number_original(M_DM,E_per_nu ,start,end,E_total_nu/E_per_nu)

        inter = [(flux[i+1]+flux[i])/2*(t[i+1]-t[i])for i in range(len(t)-1)]
        inter2 = [(flux2[i+1]+flux2[i])/2*(t2[i+1]-t2[i])for i in range(len(t)-1)]
        inter3 = [(flux3[i+1]+flux3[i])/2*(t3[i+1]-t3[i])for i in range(len(t)-1)]

        t=np.concatenate((t, t2), axis=None)
        t=np.concatenate((t, t3), axis=None)
        flux=np.concatenate((flux, flux2), axis=None)
        flux=np.concatenate((flux, flux3), axis=None)
        print("ratio:"+str(np.sum(inter+inter2+inter3)/ref))
        plt.plot(t,flux)
        plt.xscale('log')
        plt.yscale('log')
        plt.show() 
    elif mode==1:
        s = np.linspace(-8,-4,100)
        r0 = 0.01*rs
        delta_t = 8.7*t0 #- r0/vc
        t = [ delta_t+ 8.7*(10**(s[i]))*t0 for i in range(len(s))]
        flux = [DM_flux(M_DM,E_per_nu ,start,end,E_total_nu/E_per_nu,t[i]) for i in range(len(t))]
        plt.plot(t-delta_t*np.ones(len(t)),flux)
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('t(s)')
        plt.ylabel('Flux (#/s/cm^2)')
        
        
        plt.show()
        inter = [(flux[i+1]+flux[i])/2*(t[i+1]-t[i])for i in range(len(t)-1)]
        print("ratio:"+str(np.sum(inter)/ref))
   
