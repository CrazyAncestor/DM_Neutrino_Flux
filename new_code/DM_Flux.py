import numpy as np
import scipy.integrate as integrate
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
Mchi = 1e1

# NFW Model
Rho_0 = 184
Rs = 24.42*kpc2cm

# SN
Lnu = 1e52

# Cross Section
cs = 1e-30

# First Part: Find g(alpha,Enu)
def Gamma_Enu(Enu):
    return (Enu + Mchi)/(Mchi**2+2*Mchi*Enu)

def g(alpha, E_nu):
    gamma = 1
    sec = 1/np.cos(alpha)
    return np.sin(2*np.arctan(gamma*np.tan(alpha)))*gamma*(sec**2)/(1 + gamma**2*(np.tan(alpha)**2))

# Second Part: NFW Number Density Function. n_chi(r)
def n_chi(r):
    x = r/Rs
    return Rho_0/Mchi/(x*(1+x)**2)

# Third Part: SN Neutrino Energy Distribution Function. dn_nu_dEnu(r,Enu)
def dn_nu_dEnu(r,Enu):
    L=Lnu/(4*np.pi*r**2*c)

    def _fv(Enu,Tnu):
        # Fermi-Dirac distribution
        return (1/18.9686)*(1/Tnu**3)*(Enu**2/(np.exp(Enu/Tnu - 3)+1))
    
    nue_dist = _fv(Enu,2.76)/11
    nueb_dist = _fv(Enu,4.01)/16
    # total 4 species for x
    nux_dist = _fv(Enu,6.26)/25

    return L*(nue_dist+nueb_dist+4*nux_dist)

# Fourth Part: Root Finding Fuction of alpha, r when cos_theta, t, and v_chi are given. Root_Finding(cos_theta, t,v_chi,R)
def Root_Finding(cos_theta, t,v_chi,R):
    l = (-(4*v_chi**2*(R**2-(c*t)**2)*(c**2-v_chi**2) + v_chi**2*(2*c**2*t-2*R*cos_theta*v_chi)**2 )**(0.5) + v_chi*(2*c**2*t-2*R*cos_theta*v_chi) )/(2*(c**2-v_chi**2))
    r = (l**2 + R**2 - 2 *l *R*cos_theta)**0.5

    alpha = np.arccos( (r*2 + R**2 - l**2)/(2*R*r) ) + np.arccos(cos_theta)
    
    return l,r,alpha

# Fifth Part: Emmissivity and integrate it. j(r,E_nu,alpha)
def j(r,Enu,alpha):
    return c *cs *n_chi(r) *dn_nu_dEnu(r,Enu) *g(alpha,Enu)



def dF_dEnu(t,R,Enu):
    
    def j_base(cos_theta):
        Tchi = 2*Enu**2/(Mchi+2*Enu) *(1-cos_theta)/2.
        v_chi = (Tchi*(Tchi + 2*Mchi))**0.5/(Tchi+Mchi)
        l,r,alpha = Root_Finding(cos_theta,t,v_chi,R)
        return c *cs *n_chi(r) *dn_nu_dEnu(r,Enu) *g(alpha,Enu)

    result = integrate.quad(lambda x: j_base(x), 0, 1.)[0]
    print(result)

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
plt.show()