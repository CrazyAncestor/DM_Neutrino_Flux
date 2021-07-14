import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar
import scipy.optimize as opt
# Global constants
# light speed, cm
c = 3*1e10
# kpc to cm
kpc2cm = 3.08*1e21
# erg to MeV
erg2MeV = 624150.913
# year in seconds
yr = 31536000

# Physical Parameter

# DM Halo Model
Rho_0 = 184 #MeV
Rs = 24.42*kpc2cm
model_type = 'NFW'

# SN
Lnu = 1e52*erg2MeV 
R = 8.7*kpc2cm

# Cross Section
cs = 1e-45

# First Part: Find g(alpha,Ev,mx)
def Gamma_Ev(Ev,mx):
    return (Ev + mx)/((mx**2+2*mx*Ev)**0.5)

def g(alpha, Ev,mx):
    gamma = Gamma_Ev(Ev,mx)
    beta = 1/(1-gamma**(-2))**(-0.5)
    sec = 1/np.cos(alpha)

    # Artificial Criterium 1
    # Avoid Backward Scattering

    if alpha<np.pi/2: 
        return sec**3/np.pi*(1-beta**2)/(sec**2-beta**2)**2
    else:
        return 0

# Second Part: NFW Number Density Function. n_chi(r,mx,model_type)
def n_chi(r,mx,model_type):
    x = r/Rs
    if model_type =='NFW':
        return Rho_0/mx/(x*((1.+x)**2))
    elif model_type == 'Hernquist':
        return Rho_0/mx/(x*((1.+x)**3))

# Third Part: SN Neutrino Energy Distribution Function. dnv_dEv(r,Ev)
def dnv_dEv(r,Ev):
    L = Lnu/(4*np.pi*r**2*c)

    def fv(Ev,Tnu):
        # Fermi-Dirac distribution
        return (1/18.9686)*(1/Tnu**3)*(Ev**2/(np.exp(Ev/Tnu - 3)+1))
    
    nue_dist = fv(Ev,2.76)/11
    nueb_dist = fv(Ev,4.01)/16
    # total 4 species for x
    nux_dist = fv(Ev,6.26)/25
    
    return L*(nue_dist+nueb_dist+4*nux_dist)



# Fourth Part: As a function of Tchi

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
    if alpha==0:
        return Tx+0.5*(Tx**2+2*Tx*mx)**0.5

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
    if alpha==0:
        return 1+0.5*(Tx+mx)/(Tx**2+2*Tx*mx)**0.5
    v1 = 4*mx*Tx + 8*mx*Tx*np.tan(alpha)**(-2)
    v2 = np.sqrt((2*mx + Tx)**2 + 8*mx*Tx*np.tan(alpha)**(-2))
    v3 = 2*Tx + np.sqrt(2*Tx*(2*mx + Tx + 4*mx*np.tan(alpha)**(-2) + v2))
    v4 = 8*Tx*v2
    return (v1 + (Tx + v2)*v3)/v4

# Fifth Part: Root Finding Fuction of alpha, r when cos, t, and vx are given. Root_Finding(cos, t,vx,R)
def Root_Finding(cos, t,vx,R = 8.5):
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


# Sixth Part: Emmissivity and integrate it. j(r,E_nu,alpha)
def j(r,Ev,alpha,mx):
    print(n_chi(r,mx,model_type),dnv_dEv(r,Ev) ,g(alpha,Ev,mx))
    return c *cs *n_chi(r,mx,model_type) *dnv_dEv(r,Ev) *g(alpha,Ev,mx)



def dbdmflux_dTx(t,R,Tx,mx):
    tau  = 10.
    t = t+R/c
    vx = (Tx*(Tx + 2*mx))**0.5/(Tx+mx) *c # Natural Unit
    def j_base(cos):
        l,r,alpha,Flag = Root_Finding(cos,t,vx,R)
        Ev = _Ev(Tx,mx,alpha)
        dEv_dTx = dEvdTx(Tx,mx,alpha)
        if(Flag==False):
            return 0
        else:
            return vx *cs *n_chi(r,mx,model_type) *dnv_dEv(r,Ev) *g(alpha,Ev,mx)*dEv_dTx
        
    result = 2*np.pi*c*tau*integrate.quad(j_base,0,1)[0]
    
    return result

def bdmflux(t,R,mx):
    def f(Tx):
        return dbdmflux_dTx(t,R,Tx,mx)
    result = integrate.quad(f,0,100)[0]
    print(result)
    return result

def dNum_dTx(R,Tx,mx):
    tau  = 10.
    vx = (Tx*(Tx + 2*mx))**0.5/(Tx+mx) *c # Natural Unit
    fix = 1e-45
    def f_base(cos,r):
        sin_theta = (1.-cos*cos)**0.5
        alpha = np.arcsin(sin_theta/r*R)
        Ev = _Ev(Tx,mx,alpha)
        dEv_dTx = dEvdTx(Tx,mx,alpha)
        #print(alpha, Ev,dEv_dTx)
        if np.isnan(alpha) or np.isnan(Ev) or np.isnan(dEv_dTx):
            return 0
        return vx/c  *fix*n_chi(r,mx,model_type) *dnv_dEv(r,Ev) *g(alpha,Ev,mx)*dEv_dTx

    def integral(r):
        return integrate.quad(f_base,0,1,args=(r))[0]


    result = integrate.quad(integral, 0,R)[0]/fix
    result *= 2*np.pi*tau *Tx*cs      
    print(result)
    return result

def dNum_dTx_Single(R,Tx,mx):
    tau  = 10.

    def f_base(r):
        Ev = _Ev(Tx,mx,0)
        dEv_dTx = dEvdTx(Tx,mx,0)
        if  np.isnan(Ev) or np.isnan(dEv_dTx):
            print(Ev,dEv_dTx)
            return 0
        return n_chi(r,mx,model_type) *dnv_dEv(r,Ev) *dEv_dTx*(r**2)/(R**2)

    result = integrate.quad(f_base, 0.01*R,R)[0]
    result *= 2*np.pi*tau *Tx*cs      
    print(result)
    return result

# Main
mode = 'flux'
# All in MeV
mx1 = 0.01
mx2 = 0.1
mx3 = 1
Tx = 10



if mode=='flux':
    # years
    yrls=np.logspace(0,4,200)

    fluxmx1=[]
    for y in yrls:
        fluxmx1.append(dbdmflux_dTx(y*yr,R ,10.,mx1))

    fluxmx2=[]
    for y in yrls:
        fluxmx2.append(dbdmflux_dTx(y*yr,R ,10.,mx2))
    
    fluxmx3=[]
    for y in yrls:
        fluxmx3.append(dbdmflux_dTx(y*yr,R ,10.,mx3))

    plt.plot(yrls,fluxmx1,label='$m_\chi=0.01$ MeV')
    plt.plot(yrls,fluxmx2,label='$m_\chi=0.1$ MeV')
    plt.plot(yrls,fluxmx3,label='$m_\chi=1$ MeV')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('time(yr)')
    plt.ylabel(r'$d\Phi^N/dT_\chi(1/cm^2/s/MeV)$')
    plt.title(r'$T_\chi = 10\,{\rm MeV}$')
    plt.legend(loc='best')
    #plt.ylim(1,500)
    plt.show()

elif mode=='total_flux':
    # years
    yrls=np.logspace(0,4,200)

    fluxmx1=[]
    for y in yrls:
        fluxmx1.append(bdmflux(y*yr,R ,0.001))# mx = 1keV

    plt.plot(yrls,fluxmx1,label='$m_\chi=1$ keV')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('yr')
    plt.ylabel(r'$Phi^N(1/cm^2/s)$')
    plt.title(r'$Total Flux vs. Time$')
    plt.legend(loc='best')
    plt.show()

elif mode=='num':
    # Tx
    Tx = np.logspace(-4,1,200)
    DMnum=[]
    for y in Tx:
        DMnum.append(dNum_dTx_Single(R,y,300))

    plt.plot(Tx*1e3,DMnum)

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Tx(keV)')
    plt.ylabel(r'$T_\chi d\Phi^N/dT_\chi(cm^-2 s^-1)$')
    plt.title(r'$m_\chi=300$ MeV')

    plt.show()

