## This module aims to compute the interaction rate of a steady source of DM particles with water in Super-K
import numpy as np

#Particle Property
#Neutrino
M_nu = 0.32 # Unit:ev/c2
E_total_nu = 1e53*6.24150913e11 #Total energy of neutrinos #Transfer 2e51erg to unit of MeV
E_per_nu = 10e6 #Mean energy of each neutrino #estimated value

#DM
M_DM = 10e06

#cross section (Water and DM)
cs = 1e-30

def interaction_rate(phi_DM,A,L):
    n = 2.* 6.023e23 /18. #free proton density in water (#/cm^3)
    mean_free_path = 1/(n*cs)

    #Interaction Probability P = L(water depth)/mean_free_path
    #While incident DM number rate is phi_DM *A(Area)
    return phi_DM*A*L/mean_free_path


if __name__== '__main__':
    phi_DM = 0.046 # unit: #/cm^2*s
    A = (3930./2.)**2*np.pi
    L = 4140.
    print("Interaction Rate in Super-K(1/s):"+str(interaction_rate(phi_DM,A,L)))