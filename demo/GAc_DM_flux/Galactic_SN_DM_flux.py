## This module aims to compute the number and energy flux of DM particles from a supernova
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate


#Particle Property
#Neutrino
M_nu = 0.32 # Unit:ev/c2
E_total_nu = 1e53*6.24150913e11 #Total energy of neutrinos #Transfer 2e51erg to unit of MeV
E_per_nu = 10e6 #Mean energy of each neutrino #estimated value
Alpha = 2.0 #Energy Distribution Parameter

#DM
M_DM = 1e03
E_max = 60e06

#NFW Parameter
rho_s = 0.184e9
rs=24.42*3.08567758e21

#Burst Time
t_burst = 10*3e10

#cross section
cs = 1e-30

#SN
Ri=1e7

#Galactic Property
Rd = 2.9e03#kpc
H = 95 #pc
r_s =24.2e03

def r_2(pos):
    x = pos[0]
    y = pos[1]
    z = pos[2]
    return x**2 + y**2 +z**2
def R_2(pos):
    x = pos[0]
    y = pos[1]
    return x**2 + y**2 
def ang_to_car(angular_pos):
    #angular_pos : r ,theta ,phi
    x = angular_pos[0]*np.sin(angular_pos[1])*np.sin(angular_pos[2])
    y = angular_pos[0]*np.sin(angular_pos[1])*np.cos(angular_pos[2])
    z = angular_pos[0]*np.cos(angular_pos[1])
    pos = np.array([x,y,z])
    return pos
def GN_prob():
    
    def func(r,theta,phi):
        r2 = r**2
        pos = ang_to_car(np.array([r,theta,phi]))
        #pos: Sun centric pos
        pos_GAC = np.array([-8.7e03,0,-24])
        d = pos - pos_GAC
        dV = r2*np.sin(theta)
        x= r_2(d)**0.5/r_s
        return np.exp(-R_2(d)**0.5/Rd)*np.exp(-np.abs(d[2])/H) *dV/(x*(1+x)*(1+x))
    return integrate.nquad(func, [[0,np.pi], [0,2*np.pi], [0,23e03]])[0]

def DM_flux(m_dm,e_max,e_per_nu,alpha,start,end,n_total):
    r=(np.sum((start)**2))**0.5
    x= r/rs
    
    result = rho_s*n_total*cs /( 4*np.pi*m_dm)*0.000488329 /3.08567758e21

    print("DM Flux:"+str(result))
    
    
if __name__== '__main__':
    
    
    print("Total number of neutrino:"+str(E_total_nu/E_per_nu))

    start=np.array([1*3.08567758e21,0*3.08567758e21,0])
    end =np.array([10*3.08567758e21,0,0])
    
    DM_flux(M_DM,E_max,E_per_nu ,Alpha,start,end,E_total_nu/E_per_nu)

    