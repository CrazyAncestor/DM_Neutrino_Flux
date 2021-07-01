import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar
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
# DM Mass
Mchi = 1

# NFW Model
Rho_0 = 184
Rs = 24.42*kpc2cm

# SN
Lnu = 1e52

# Cross Section
cs = 1e-45

# First Part: Find g(alpha,Enu)
def Gamma_Enu(Enu):
    return (Enu + Mchi)/((Mchi**2+2*Mchi*Enu)**0.5)

def g(alpha, E_nu):
    gamma = Gamma_Enu(E_nu)
    sec = 1/np.cos(alpha)
    if 1e-6 < alpha <= np.pi/2:
        return np.sin(2*np.arctan(gamma*np.tan(alpha)))*gamma*(sec**2)/(1 + gamma**2*(np.tan(alpha)**2))/(2*np.pi*np.sin(alpha))
    elif 0 <= alpha <= 1e-6:
        return gm**2/np.pi
    else:
        return 0

# Second Part: NFW Number Density Function. n_chi(r)
def n_chi(r):
    x = r/Rs
    return Rho_0/Mchi/(x*((1.+x)**2))

# Third Part: SN Neutrino Energy Distribution Function. dn_nu_dEnu(r,Enu)
def dn_nu_dEnu(r,Enu):
    L=Lnu/(4*np.pi*r**2*c)*erg2MeV 

    def _fv(Enu,Tnu):
        # Fermi-Dirac distribution
        return (1/18.9686)*(1/Tnu**3)*(Enu**2/(np.exp(Enu/Tnu - 3)+1))
    
    nue_dist = _fv(Enu,2.76)/11
    nueb_dist = _fv(Enu,4.01)/16
    # total 4 species for x
    nux_dist = _fv(Enu,6.26)/25

    return L*(nue_dist+nueb_dist+4*nux_dist)



# Fifth Part: As a function of Tchi
def _gamma(Ev,mx):
    """
    Calculate gamma factor in CM frame
    """
    s = mx**2+2*Ev*mx
    Ecm = 0.5*(s+mx**2)/np.sqrt(s)
    return Ecm/mx
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
    gm = _gamma(Ev,mx)
    # CM frame scattering angle
    theta_c = 2*np.arctan(gm*np.tan(alpha))
    
    # Tmax in lab frame
    Tmax = Ev**2/(Ev+0.5*mx)
    Tchi = 0.5*Tmax*(1-np.cos(theta_c))
    return Tchi
def Ev(Tx,mx,alpha):
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
    enu = -2*mx + Tx + np.sqrt((2*mx + Tx)**2 + 8*mx*Tx*np.tan(alpha)**(-2)) \
            + np.sqrt(2*Tx*(2*mx + Tx + 4*mx*np.tan(alpha)**(-2)+   \
                    + np.sqrt((2*mx + Tx)**2 + 8*mx*Tx*np.tan(alpha)**(-2))))
    return enu/4


def Ev_root(Tx,mx,alpha,Emax=1e7):
    """
    Calculate the neutrino energy to produce DM kinetic energy at
    lab frame scattering angle alpha via root finding
    
    Input
    ------
    Tx: DM kinetic energy
    mx: DM mass
    alpha: scattering angle in lab frame
    
    Output
    Ev: the corresponding neutrino energy
    """
    # function for root finding
    _f = lambda ev: _Tx(ev,mx,alpha) - Tx
    # Finding the corresponding Enu
    Enu = root_scalar(_f,bracket=[0,Emax*mx],method='brentq')
    return Enu.root
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

# Fourth Part: Root Finding Fuction of alpha, r when cos_theta, t, and v_chi are given. Root_Finding(cos_theta, t,v_chi,R)
def Root_Finding1(cos_theta, t,v_chi,R=8.5):
    l = (-(4*v_chi**2*(R**2-(c*t)**2)*(c**2-v_chi**2) + v_chi**2*(2*c**2*t-2*R*cos_theta*v_chi)**2 )**(0.5) + v_chi*(2*c**2*t-2*R*cos_theta*v_chi) )/(2*(c**2-v_chi**2))
    r = (l**2 + R**2 - 2 *l *R*cos_theta)**0.5
    alpha = np.arccos( (r*2 + R**2 - l**2)/(2*R*r) ) + np.arccos(cos_theta)



    if l < 0 or r < 0 or np.isnan(alpha):
        flag = False
    else:
        flag = True
    
    return l,r,alpha,flag

def Root_Finding2(cos,t,vx,R=8.5):
    """
    Find l, r and alpha when t, cos\theta, R and vx are given
    
    Input
    ------
    t: time in seconds, 0 starts from the arrival of the SN neutrino at Earth
    cos: cos\theta
    vx: dimensionless DM velocity, v/c
    R: R in kpc, default 8.5 kpc
    
    Output
    ------
    tuple: (l,r,alpha,flag)
           flag is a string label, physical/unphysical, where unphysical
           result should be considered zero contribution to the emissivity
    """
    # convert to CGS unit
    #R = R*kpc2cm
    #t = t + R/c
    #vx = vx*c
    # numerators
    n1 = - np.sqrt(4*vx**2*((c*t)**2-R**2)*(vx**2-c**2)+vx**2*(2*c**2*t-2*R*cos*vx)**2)
    n2 = 2*c**2*t*vx
    n3 = -2*R*cos*vx**2
    # the l
    l = (n1 + n2 + n3)/(2*(c**2 - vx**2))
    # the corresponding r
    r = np.sqrt(l**2 + R**2 - 2*l*R*cos)
    Theta = np.arccos((R**2 + r**2 - l**2)/(2*R*r))
    
    # alpha = Theta + theta
    alpha = Theta + np.arccos(cos)
    
    # Some input, eg., t, will cause l or r is negative, this is not physical and
    # have no contribution to the emissivity.
    # I raise a flag to discrimate such situation and ask the program to set the
    # corresponding emissivity zero manually
    if l < 0 or r < 0 or np.isnan(alpha):
        flag = False
    else:
        flag = True
    return l,r,alpha,flag

# Sixth Part: Emmissivity and integrate it. j(r,E_nu,alpha)
def j(r,Enu,alpha):
    print(n_chi(r),dn_nu_dEnu(r,Enu) ,g(alpha,Enu))
    return c *cs *n_chi(r) *dn_nu_dEnu(r,Enu) *g(alpha,Enu)



def dF_dEnu(t,R,Enu):
    
    def j_base(cos_theta):
        Tchi = 2*Enu**2/(Mchi+2*Enu) *(1-cos_theta)/2.
        v_chi = (Tchi*(Tchi + 2*Mchi))**0.5/(Tchi+Mchi) *c # Natural Unit
        l,r,alpha,Flag = Root_Finding1(cos_theta,t,v_chi,R)
        if(Flag==False):
            return 0
        else:
            return c *cs *n_chi(r) *dn_nu_dEnu(r,Enu) *g(alpha,Enu)

    result = integrate.quad(lambda x: j_base(x), 0, 1.)[0]
    print(result)

def bdmflux(t,R,Tx):
    tau  = 10.
    t = t+R/c
    v_chi = (Tx*(Tx + 2*Mchi))**0.5/(Tx+Mchi) *c # Natural Unit
    def j_base(cos_theta):
        l,r,alpha,Flag = Root_Finding1(cos_theta,t,v_chi,R)
        
        Enu = Ev(Tx,Mchi,alpha)
        dEv_dTx = dEvdTx(Tx,Mchi,alpha)
        if(Flag==False):
            return 0
        else:
            return v_chi *cs *n_chi(r) *dn_nu_dEnu(r,Enu) *g(alpha,Enu) *dEv_dTx

    result = 2*np.pi*v_chi*tau*integrate.quad(j_base,0,1)[0]*kpc2cm#integrate.quad(lambda x: j_base(x), 0, 1.)[0]
    return result




# Main
#bdmflux(31536000*1e2,8.5*kpc2cm,10.)
"""
# Checking analytical expression and root finding method
Tx = np.logspace(-2,2,100)
ev = Ev(Tx,1,0.3*np.pi)
# root finding function does not support numpy vectorization
ev_root = []
for tx in Tx:
    ev_root.append(Ev_root(tx,1,0.3*np.pi))

# plot
plt.plot(Tx,ev,label='analytical')
plt.plot(Tx,ev_root,'.',label='root finding')
plt.xscale('log')
plt.xlabel(r'$T_\chi$ [MeV]')
plt.ylabel(r'$E_\nu$ [MeV]')
plt.title(r'$m_\chi=1$ MeV, $\alpha=0.3\pi$')
plt.legend()
plt.show()"""


#result = integrate.quad(lambda x: g(x,10.)*np.sin(x), 0, np.pi/2)[0]*2*np.pi
#print(result)

#print(dn_nu_dEnu(4*kpc2cm/c,10.))

#print(g(np.pi/4,180.))
#dF_dEnu(8.5*kpc2cm/c+31536000*1e2,8.5*kpc2cm,10.)

# All in MeV
mx1 = 0.01
mx2 = 0.1
mx3 = 1
Tx = 10

R = 8.7*kpc2cm

# years
yrls=np.logspace(0,5,200)

fluxmx1=[]
for y in yrls:
    fluxmx1.append(bdmflux(y*yr,R ,10.))

fluxmx2=[]
for y in yrls:
    fluxmx2.append(bdmflux(y*yr,R ,10.))
    
fluxmx3=[]
for y in yrls:
    fluxmx3.append(bdmflux(y*yr,R ,10.))

plt.plot(yrls,fluxmx1,label='$m_\chi=0.01$ MeV')
plt.plot(yrls,fluxmx2,label='$m_\chi=0.1$ MeV')
plt.plot(yrls,fluxmx3,label='$m_\chi=1$ MeV')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('yr')
plt.ylabel(r'$d\Phi^N/dT_\chi$')
plt.title(r'$T_\chi = 10\,{\rm MeV}$')
plt.legend(loc='best')
plt.ylim(1,500)
plt.show()
