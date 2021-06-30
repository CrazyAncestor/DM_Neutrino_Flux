import numpy as np
import Steps.Step2_DM_Flux.DM_flux as DM

#Particle Property
#Neutrino
M_nu = 0.32 # Unit:eV/c2
E_total_nu = 3.6e53*6.24150913e11 #Total energy of neutrinos #Transfer 2e51erg to unit of eV
E_per_nu = 10e6 #Mean energy of each neutrino in eV #estimated value

#DM
M_DM = 1e03 #eV

#NFW Parameter
rho_s = 0.184e9 #eV/cm^3
rs=24.42*3.08567758e21 #in cm

#cross section (Neutrino and DM)
cs = 1e-30

if __name__== '__main__':
    
    
    print("Total number of neutrino:"+str(E_total_nu/E_per_nu))

    start=np.array([0.87*3.08567758e21,0,2.4*3.08567758e18])
    end =np.array([8.7*3.08567758e21,0,24*3.08567758e18])
    
    DM.DM_number(M_DM,E_per_nu ,start,end,E_total_nu/E_per_nu)
    DM.DM_flux(M_DM,E_per_nu ,start,end,E_total_nu/E_per_nu)