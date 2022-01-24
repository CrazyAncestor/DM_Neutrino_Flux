import numpy as np
import math as mt
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import scipy.optimize as opt

# Particle Property
# kpc
kpc_in_cm = 3.08567758e21
# light speed
vc = 3e10
c = 3e10
# year in seconds
yr = 31536000

# Neutrino
M_nu = 0. # Unit:ev/c2
E_total_nu = 3.6e53*6.24150913e11 # Total energy of neutrinos # Transfer 2e51erg to unit of MeV
E_per_nu = 10 # Mean energy of each neutrino #estimated value
n_tot = E_total_nu/1e7
# DM
M_DM = 10 # MeV

# NFW Parameter
rho_s = 0.184e3
rs=24.42*kpc_in_cm

# cross section (Neutrino and DM)
cs = 1e-45

def f_Ev(Ev):
    
    def fv(Ev,Tnu):
        # Fermi-Dirac distribution
        return (1/18.9686)*(1/Tnu**3)*(Ev**2/(np.exp(Ev/Tnu - 3)+1))
    nue_dist = fv(Ev,2.76)/11
    nueb_dist = fv(Ev,4.01)/16
    # total 4 species for x
    nux_dist = fv(Ev,6.26)/25
    
    return (nue_dist+nueb_dist+4*nux_dist)

def norm(start,end):
    R = (np.sum((start-end)**2))**0.5
    l = end -start
    beta = 0.3
    
    def gl(l,theta): 
        # Geometric Terms
        r = (l**2+R**2-2*R*l*np.cos(theta))**0.5
        alpha = np.arcsin(R/r*np.sin(theta))
        psi = alpha - theta
        sec = 1/np.cos(alpha)
        dn_domega = 1/4/np.pi
        geometric_terms = np.sin(theta) /R *2*np.pi*dn_domega
        
        return  geometric_terms

    result =integrate.dblquad(gl, 0, np.pi/2, lambda theta: 0, lambda theta: R*2*np.cos(theta))[0]

    L_dm = result
    print("Norm:"+str(L_dm))
    return L_dm

def Gamma_Ev(Ev,mx):
    return (Ev + mx)/((mx**2+2*mx*Ev)**0.5)

def g(alpha, Ev,mx):
    gamma = Gamma_Ev(Ev,mx)
    beta = 1/(1-gamma**(-2))**(-0.5)
    sec = 1/np.cos(alpha)

    # Artificial Criterium 1
    # Avoid Backward Scattering

    if alpha<np.pi/2: 
        return sec**3*2*(1-beta**2)/(sec**2-beta**2)**2
    else:

        return 0

def _Ev(Tx,mx,alpha):
    """
    Calculate the neutrino energy to produce DM kinetic energy at
    lab frame scattering angle alpha via analytical expression
    
    Input
    ------
    Tx: DM kinetic energy
    mx: DM mass
    alpha: scattering angle in lab frame
    
    Output
    Ev: the corresponding neutrino energy
    """
    #if alpha==0:
        #return Tx+0.5*(Tx**2+2*Tx*mx)**0.5
    
    Ev = -2*mx + Tx + np.sqrt((2*mx + Tx)**2 + 8*mx*Tx*np.tan(alpha)**(-2)) \
            + np.sqrt(2*Tx*(2*mx + Tx + 4*mx*np.tan(alpha)**(-2)+   \
                    + np.sqrt((2*mx + Tx)**2 + 8*mx*Tx*np.tan(alpha)**(-2))))
    return Ev/4

def dEvdTx(Tx,mx,alpha):
    """
    Calculate dEv/dTx via analytical expression
    
    Input
    ------
    Tx: DM kinetic energy
    mx: DM mass
    alpha: scattering angle in lab frame
    dTx: arbitrarity small number, default 1e-5
    
    Output
    ------
    tuple: dEv/dTx
    """

    v1 = 4*mx*Tx + 8*mx*Tx*np.tan(alpha)**(-2)
    v2 = np.sqrt((2*mx + Tx)**2 + 8*mx*Tx*np.tan(alpha)**(-2))
    v3 = 2*Tx + np.sqrt(2*Tx*(2*mx + Tx + 4*mx*np.tan(alpha)**(-2) + v2))
    v4 = 8*Tx*v2
    return (v1 + (Tx + v2)*v3)/v4

def _Tx(Ev,mx,alpha):
    """
    Calculate DM kinetic energy, Eq. (28), and velocity, Eq. (30),
    in terms of lab frame scattering angle
    
    Input
    ------
    Enu: Neutrino energy
    mx: DM mass
    alpha: lab frame scattering angle in [0,Pi/2]
    
    Output
    ------
    array: DM kinetic energy
           DM velocity in the unit of c
           CM frame scatering angle within [0,Pi]
    """
    # gamma factor in CM frame
    gm = Gamma_Ev(Ev,mx)
    # CM frame scattering angle
    theta_c = 2*np.arctan(gm*np.tan(alpha))
    
    # Tmax in lab frame
    Tmax = Ev**2/(Ev+0.5*mx)
    Tchi = 0.5*Tmax*(1+np.cos(theta_c))
    return Tchi

def Root_Finding(theta, t,vx,R = 8.5):
    cos = np.cos(theta)
    l = (-(4*vx**2*(R**2-(c*t)**2)*(c**2-vx**2) + vx**2*(2*c**2*t-2*R*cos*vx)**2 )**(0.5) + vx*(2*c**2*t-2*R*cos*vx) )/(2*(c**2-vx**2))
    r = (l**2 + R**2 - 2 *l *R*cos)**0.5
    alpha = np.arccos( (r**2 + R**2 - l**2)/(2*R*r) ) + np.arccos(cos)

    # Artificial Criterium 2
    # Avoid Unphysical Geometry

    if l < 0 or r < 0 or np.isnan(alpha):
        flag = False
    else:
        flag = True
    return l,r,alpha,flag

def bdmflux(start,end,Tx,mx,t):
    R = (np.sum((start-end)**2))**0.5
    l = end -start
    vx = (Tx*(Tx + 2*mx))**0.5/(Tx+mx) *c # Natural Unit
    def gl(theta): 
        time = t+R/c
        l,r,alpha,Flag = Root_Finding(theta,time,vx,R) # solve geometry

        # Geometric Terms
        psi = alpha - theta
        sec = 1/np.cos(alpha)
        geometric_terms = np.sin(theta) *2*np.pi 

        # Physical terms
        Ev = _Ev(Tx,mx,alpha)
        if Ev<=0 or np.isnan(Ev):   
            return 0 
        dn_domega= g(alpha,Ev,mx)
        dEv_dTx = dEvdTx(Tx,mx,alpha)
        L = n_tot/4/np.pi
        def rho(r):
            x = r/R
            return rho_s/(x*(1+x)**2)
        physical_terms = rho(r)*L/r**2*dn_domega*dEv_dTx*f_Ev(Ev) *vx

        return geometric_terms*physical_terms

    result = integrate.quad(gl,0,np.pi/2)[0]
    L_dm = result*cs/mx

    return L_dm

def Acc_bdmflux_overtime(start,end,Tx,mx):
    R = (np.sum((start-end)**2))**0.5
    l = end -start
    vx = (Tx*(Tx + 2*mx))**0.5/(Tx+mx) *c # Natural Unit
    def gl(t,theta): 
        time = t+R/c
        l,r,alpha,Flag = Root_Finding(theta,time,vx,R) # solve geometry
        # Geometric Terms
        psi = alpha - theta
        sec = 1/np.cos(alpha)
        geometric_terms = np.sin(theta) *2*np.pi 

        # Physical terms
        Ev = _Ev(Tx,mx,alpha)
        if Ev<=0 or np.isnan(Ev):   
            return 0 
        dn_domega= g(alpha,Ev,mx)
        dEv_dTx = dEvdTx(Tx,mx,alpha)
        L = n_tot/4/np.pi
        def rho(r):
            x = r/R
            return rho_s/(x*(1+x)**2)
        physical_terms = rho(r)*L/r**2*dn_domega*dEv_dTx*f_Ev(Ev) *vx

        return geometric_terms*physical_terms

    result =integrate.dblquad(gl, 0, np.pi/2, lambda theta: 0, lambda theta: 30*yr)[0]
    L_dm = result*cs/mx

    return L_dm

def Acc_bdmnumber(start,end,mx):
    def f(tx):
        return Acc_bdmflux_overtime(start,end,tx,mx)
    return integrate.quad(f,5,100)[0]

if __name__== '__main__':
    
    beta = (E_per_nu**2- M_nu**2)**0.5/(E_per_nu+M_DM)
    gamma = (1-beta**(2))**0.5      

    start=np.array([8.7*0.0*kpc_in_cm,0,0.*3.08567758e18])
    end =np.array([8.7*kpc_in_cm,0,0.*3.08567758e18])
    
    nor = norm(start,end)
    mode = 'accumulated number'

    if mode=='accumulated number':
        mx = 10.
        acc_number = Acc_bdmnumber(start,end,mx)
        print("Accumulated DM number:"+str(acc_number))

        mx = np.logspace(np.log(6),np.log(20),10)
        number = []
        for m in mx:
            number.append(Acc_bdmnumber(start,end,m))
            print(number[-1])
        plt.plot(mx,number)

        #plt.xscale('log')
        #plt.yscale('log')
        plt.xlabel('mx(MeV)')
        plt.ylabel(r'$Acc Number(1/cm^2)$')
        plt.legend(loc='best')
        plt.show()
        

    elif mode=='acc_flux over time':
        
        Tx = np.logspace(np.log(5),2,10)
        mx = 4.0
        flux = []
        
        for tx in Tx:
            flux.append(Acc_bdmflux_overtime(start,end,tx,mx))
            

        plt.plot(Tx,flux,label = 'mx = %.2f'%(mx))

        #plt.xscale('log')
        #plt.yscale('log')
        plt.xlabel('Tx(MeV)')
        plt.ylabel(r'$d\Phi^N/dT_\chi$')
        plt.legend(loc='best')
        plt.show()
        
    
    elif mode=='flux':
        Tx = 10
        mx1 = 10.
        mx2 = 1.
        mx3 = 0.1
        mx4 = 0.01
        # years
        yrls=np.logspace(0,6,200)*yr
        flux1 = []
        flux2 = []
        flux3 = []
        for t in yrls:
            flux1.append(bdmflux(start,end,Tx,mx1,t))
            flux2.append(bdmflux(start,end,Tx,mx2,t))
            flux3.append(bdmflux(start,end,Tx,mx3,t))

        plt.plot(yrls/yr,flux1,label = 'Tx = %.2f, mx = %.2f'%(Tx,mx1))
        plt.plot(yrls/yr,flux2,label = 'Tx = %.2f, mx = %.2f'%(Tx,mx2))
        plt.plot(yrls/yr,flux3,label = 'Tx = %.2f, mx = %.2f'%(Tx,mx3))

        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('time(yr)')
        plt.ylabel(r'$d\Phi^N/dT_\chi$')
        plt.legend(loc='best')
        plt.show()

