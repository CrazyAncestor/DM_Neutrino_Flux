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
        #return np.sin(2*np.arctan(gamma*np.tan(alpha)))*gamma*(sec**2)/(1 + gamma**2*(np.tan(alpha)**2))/(2*np.pi*np.sin(alpha))
        return sec**3/np.pi*(1-beta**2)/(sec**2-beta**2)**2
    else:
        #print(alpha/np.pi*180)
        return 0#sec**3/np.pi*(1-beta**2)/(sec**2-beta**2)**2

def g_new(alpha, Ev,mx):
    gamma = Gamma_Ev(Ev,mx)
    beta = 1/(1-gamma**(-2))**(-0.5)
    sec = 1/np.cos(alpha)

    # Artificial Criterium 1
    # Avoid Backward Scattering

    if alpha<(np.pi/1.25): 
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
    sec = 1/np.cos(alpha)
    enu = (Tx*sec**2 + sec*np.sqrt(Tx*(2*mx + Tx)))/(2 - Tx*np.tan(alpha)**2/mx)
    return enu

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
    sec = 1/np.cos(alpha)
    numerator = mx**2*sec*(2*sec*np.sqrt(Tx*(2*mx + Tx)) + 2*mx + Tx*sec**2 + Tx)
    denominator = (Tx*np.tan(alpha)**2 - 2*mx)**2*np.sqrt(Tx*(2*mx + Tx))
    return numerator/denominator

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
        
    result = integrate.quad(j_base,0.99999,1)[0]
    result += integrate.quad(j_base,0.9999,0.99999)[0]
    result += integrate.quad(j_base,0.999,0.9999)[0]
    result += integrate.quad(j_base,0.99,0.999)[0]
    result += integrate.quad(j_base,0.9,0.99)[0]
    result += integrate.quad(j_base,0,0.9)[0]
    
    return 2*np.pi*c*tau*result

def dbdmflux_dTx_ori(t,R,Tx,mx):
    tau  = 10.
    t = t + R/c
    vx = (Tx*(Tx + 2*mx))**0.5/(Tx+mx) *c # Natural Unit
    def j_base(cos):
        l,r,alpha,Flag = Root_Finding(cos,t,vx,R)
        
        Ev_in = _Ev(Tx,mx,alpha)
        dEv_dTx = dEvdTx(Tx,mx,alpha)
        if(Flag==False):
            return 0
        else:
            return vx *cs *n_chi(r,mx,model_type) *dnv_dEv(r,Ev_in) *g(alpha,Ev_in,mx) *dEv_dTx

    result = 2*np.pi*vx*tau*integrate.quad(j_base,0,1,epsabs=1.49e-30)[0]
    return result

def bdmflux(t,R,mx):
    def f(Tx):
        return dbdmflux_dTx(t,R,Tx,mx)
    result = integrate.quad(f,0,100)[0]
    print(result)
    return result

def check_bdmflux(t,R,Tx,mx):
    tau  = 10.
    t = t + R/c
    vx = (Tx*(Tx + 2*mx))**0.5/(Tx+mx) *c # Natural Unit

    def f(cos):
        l,r,alpha,Flag = Root_Finding(cos,t,vx,R)
        return alpha-np.pi/2
    root = opt.fsolve(f, [0.5])
    
    def j_base(cos):
        l,r,alpha,Flag = Root_Finding(cos,t,vx,R)
        
        Ev_in = _Ev(Tx,mx,alpha)
        dEv_dTx = dEvdTx(Tx,mx,alpha)
        if(Flag==False):
            return 0
        else:
            return vx *cs *n_chi(r,mx,model_type) *dnv_dEv(r,Ev_in)  *dEv_dTx *g(alpha,Ev_in,mx)

    result = 2*np.pi*vx*tau*integrate.quad(j_base,root,1)[0]*kpc2cm
    l,r,alpha,Flag = Root_Finding(root,t,vx,R)
    Ev_in = _Ev(Tx,mx,alpha)
    return result
    #return np.arccos(root)/np.pi*180
    #return dEvdTx(Tx,mx,alpha)
    #return vx *cs *n_chi(r,mx,model_type) *dnv_dEv(r,Ev_in) *g(alpha,Ev_in,mx) *dEv_dTx

def check_bdmflux_new(t,R,Tx,mx):
    tau  = 10.
    t = t + R/c
    vx = (Tx*(Tx + 2*mx))**0.5/(Tx+mx) *c # Natural Unit

    def f(cos):
        l,r,alpha,Flag = Root_Finding(cos,t,vx,R)
        return alpha-np.pi/2
    root = opt.fsolve(f, [0.5])
    
    def j_base(cos):
        l,r,alpha,Flag = Root_Finding(cos,t,vx,R)
        
        Ev_in = _Ev(Tx,mx,alpha)
        dEv_dTx = dEvdTx(Tx,mx,alpha)
        if(Flag==False):
            return 0
        else:
            return vx *cs *n_chi(r,mx,model_type) *dnv_dEv(r,Ev_in)  *dEv_dTx *g_new(alpha,Ev_in,mx)

    result = 2*np.pi*vx*tau*integrate.quad(j_base,root,1)[0]*kpc2cm
    l,r,alpha,Flag = Root_Finding(root,t,vx,R)
    Ev_in = _Ev(Tx,mx,alpha)
    return result

def check_g(cos,t,R,Tx,mx):
    tau  = 10.
    t = t + R/c
    vx = (Tx*(Tx + 2*mx))**0.5/(Tx+mx) *c # Natural Unit

    l,r,alpha,Flag = Root_Finding(cos,t,vx,R)
    Ev_in = _Ev(Tx,mx,alpha)
    #return alpha/np.pi*180
    if Flag == False:
        return 0
    return g(alpha,Ev_in,mx)

    def f(cos):
        l,r,alpha,Flag = Root_Finding(cos,t,vx,R)
        return alpha-np.pi/2
    root = opt.fsolve(f, [0.5])
    
    l,r,alpha,Flag = Root_Finding(cos,t,vx,R)
    dEv_dTx = dEvdTx(Tx,mx,alpha)
    Ev_in = _Ev(Tx,mx,alpha)

    if root>cos:
        return 0
    return g(alpha,Ev_in,mx)
    return g(alpha,Ev_in,mx)
    #return np.arccos(root)/np.pi*180
    #return dEvdTx(Tx,mx,alpha)
    #return vx *cs *n_chi(r,mx,model_type) *dnv_dEv(r,Ev_in) *g(alpha,Ev_in,mx) *dEv_dTx

def check_others(cos,t,R,Tx,mx):
    tau  = 10.
    t = t + R/c
    vx = (Tx*(Tx + 2*mx))**0.5/(Tx+mx) *c # Natural Unit

    l,r,alpha,Flag = Root_Finding(cos,t,vx,R)
    Ev_in = _Ev(Tx,mx,alpha)
    #return alpha
    #return g(alpha,Ev_in,mx)

    def f(cos):
        l,r,alpha,Flag = Root_Finding(cos,t,vx,R)
        return alpha-np.pi/2
    root = opt.fsolve(f, [0.5])
    
    l,r,alpha,Flag = Root_Finding(cos,t,vx,R)
    dEv_dTx = dEvdTx(Tx,mx,alpha)
    Ev_in = _Ev(Tx,mx,alpha)

    if root>cos:
        return 0
    return n_chi(r,mx,model_type) *dnv_dEv(r,Ev_in) *dEv_dTx#*g(alpha,Ev_in,mx)
    return g(alpha,Ev_in,mx)
    #return np.arccos(root)/np.pi*180
    #return dEvdTx(Tx,mx,alpha)
    #return vx *cs *n_chi(r,mx,model_type) *dnv_dEv(r,Ev_in) *g(alpha,Ev_in,mx) *dEv_dTx



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

def D_Plot(cos,t,R,Tx,mx):
    tau  = 10.
    t = t + R/c
    vx = (Tx*(Tx + 2*mx))**0.5/(Tx+mx) *c # Natural Unit

    l,r,alpha,Flag = Root_Finding(cos,t,vx,R)
    return l*(-cos),l*(1-cos**2)**0.5


# Main
mode = 'total_flux'
# All in MeV
mx1 = 0.01
mx2 = 0.1
mx3 = 1
Tx = 10

R = 8.7*kpc2cm

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
        fluxmx1.append(bdmflux(y*yr,R ,0.001))#1keV

    """fluxmx2=[]
    for y in yrls:
        fluxmx2.append(bdmflux(y*yr,R ,mx2))
    
    fluxmx3=[]
    for y in yrls:
        fluxmx3.append(bdmflux(y*yr,R ,mx3))"""

    plt.plot(yrls,fluxmx1,label='$m_\chi=1$ keV')
    #plt.plot(yrls,fluxmx2,label='$m_\chi=0.1$ MeV')
    #plt.plot(yrls,fluxmx3,label='$m_\chi=1$ MeV')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('yr')
    plt.ylabel(r'$Phi^N(1/cm^2/s)$')
    plt.title(r'$Total Flux vs. Time$')
    plt.legend(loc='best')
    #plt.ylim(1,500)
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

    #plt.ylim(1,500)
    plt.show()

elif mode=='check':
    yrls=np.logspace(3,4,200)
    #yrls = np.linspace(0,1.,200)
    theta=[]
    comp = []

    for y in yrls:
        theta.append(check_bdmflux(y*yr,R,10.,1.))
        
        if(y<3000 and y>2900):
            comp.append(30)
        else:
            comp.append(0)
        #comp.append(check_bdmflux_new(y*yr,R,10.,1.))
        #theta.append(check_g(y,2800*yr,R,10.,1.))
    plt.plot(yrls,theta)
    plt.plot(yrls,comp)
    plt.xscale('log')
    plt.yscale('log')
    #plt.xlabel('yr')
    
   
    plt.legend(loc='best')

    plt.show()

elif mode=='check_g':
    #yrls=np.logspace(0,4,2000)
    yrls = np.linspace(0.8,1.,1000)
    theta=[]
    comp=[]
    comp_ot=[]
    comp_ot2=[]
    for y in yrls:
        #theta.append(check_bdmflux(y*yr,R,10.,1.))
        #theta.append(check_g(y,100*yr,R,10.,1.))
        #comp.append(check_g(y,1000*yr,R,10.,1.))
        comp_ot.append(check_others(y,100*yr,R,10.,1.)/800000/500)
        comp_ot2.append(check_others(y,1000*yr,R,10.,1.)/800000/500)
    #plt.plot(yrls,theta,label = 'g,early,t=100yr')
    #plt.plot(yrls,comp,label = 'g,late,t=1000yr')
    plt.plot(yrls,comp_ot,label = 'f,t=100yr')
    plt.plot(yrls,comp_ot2,label = 'f,t=1000yr')
    #plt.xscale('log')
    #plt.yscale('log')
    plt.xlabel(r'$cos\theta$')
    plt.ylabel(r'$\alpha(degrees)$')
    
   
    plt.legend(loc='best')

    plt.show()

elif mode=='SN':
    #yrls=np.logspace(0,4,2000)
    yrls = np.linspace(0.99,1.,1000)
    x=[]
    y=[]
    x2=[]
    y2=[]
    for s in yrls:
        x_tp,y_tp = D_Plot(s,100*yr,R,10,1)#cos,t,R,Tx,mx
        x.append(x_tp)
        y.append(y_tp)
        x_tp,y_tp = D_Plot(s,1400*yr,R,10,1)#cos,t,R,Tx,mx
        x2.append(x_tp)
        y2.append(y_tp)
    plt.plot(x,y, label = 'early,t=100yr')
    plt.plot(x2,y2, label = 'late,t=1400yr')
    plt.axis('equal')
    #plt.plot(yrls,comp,label = 'g,late,t=1500yr')
    #plt.plot(yrls,comp_ot,label = 'f')
    #plt.xscale('log')
    #plt.yscale('log')
    #plt.xlabel(r'$cos\theta$')
    #plt.ylabel('func')
    
   
    plt.legend(loc='best')

    plt.show()