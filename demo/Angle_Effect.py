## This module aims to analyze the effect of angle distribuiton
import numpy as np
import math as mt
import matplotlib.pyplot as plt
import Kinematics as kim
import scipy.integrate as integrate


#Particle Property
#Neutrino
M_nu = 0.32 # Unit:ev/c2
E_total_nu = 1e53*6.24150913e11 #Total energy of neutrinos #Transfer 2e51erg to unit of MeV
E_per_nu = 10e6 #Mean energy of each neutrino #estimated value

#DM
M_DM = 1e07

#NFW Parameter
rho_s = 0.184e9
rs=24.42*3.08567758e21

#cross section (Neutrino and DM)
cs = 1e-28

def angle_ratio(beta,r,R):
    def f(psi):
        rp = (R**2+r**2-2*R*r*mt.cos(psi))**0.5
        cos_theta_psi = (rp**2+R**2-r**2)/(2*R*rp)
        
        def g(psi):
            def theta(psi):
                return mt.atan(1/(1/(mt.tan(psi))-r/(R*mt.sin(psi))))
            
            theta = theta(psi)
            sec = 1/mt.cos(theta)
            #return 1
            return 4*mt.tan(theta)*(sec**2)*(1-beta*beta)/((sec**2-beta**2)**2)/mt.sin(theta)
            # 
            #return 4*x/((1+x**2)**2) /(mt.sin(theta)*mt.cos(theta)*mt.cos(theta)) /((1-beta**2)**0.5)
            # #x = mt.tan(theta)/((1-beta**2)**0.5)

        return mt.sin(psi)*g(psi)*cos_theta_psi/(rp**2)/2.
    
    theta_max = np.pi/2
    def psi(theta):
        return theta - mt.asin(r/R*mt.sin(theta))
    psi_max = psi(theta_max)
    #print(psi_max/np.pi)
    #return integrate.nquad(f, [[0,np.pi]])[0]
    return integrate.nquad(f, [[0,psi_max]])[0] 
def norm(beta):
    
    
    def g(theta):
        x = mt.tan(theta)/((1-beta**2)**0.5)
        return 4*x/((1+x**2)**2) /(mt.sin(theta)*mt.cos(theta)*mt.cos(theta)) /((1-beta**2)**0.5)
    theta_max = mt.atan((1-beta**2)**0.5)
    def f(theta):
        return mt.sin(theta)*g(theta)
    return integrate.nquad(f, [[theta_max,np.pi]])[0]
if __name__== '__main__':
    gamma = kim.energy_kicked_by_neutrino(E_per_nu, M_nu,M_DM)/M_DM
    beta = (1-gamma**(-2))**0.5
    angle_ratio(beta,0.5,1)
    
    print("Angle Ratio:"+str(angle_ratio(beta,0.5,1)))
    print("Norm:"+str(norm(beta)))
    

    r = np.linspace(0.001,0.9,100)
    ratio = []
    for i in range (len(r)):
        ratio.append(angle_ratio(beta,r[i],1))
    ratio = np.array(ratio)
    plt.plot(r, ratio, color ='blue', label = 'Angle ratio')
    plt.xlabel('Distance (0 starts from SN, and 1 ends at the earth)')
    plt.ylabel('Ratio ')
    #plt.legend(loc='upper right')
    plt.savefig("Angle.png")
    plt.show()
    print(ratio)