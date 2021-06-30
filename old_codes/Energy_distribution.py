## This module aims to compute the number and energy flux of DM particles from a supernova
import numpy as np
import matplotlib.pyplot as plt
import Kinematics as kim

import scipy.optimize as opt

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

def f_energy_nu(e,alpha):
    e  = np.array(e)
    return (e**alpha)*np.exp(-(alpha+1)*e)

def f_evolution(m_dm,e_max,e_per_nu,alpha,start,end):
    E = np.linspace(m_dm/e_per_nu,e_max/e_per_nu,1000)
    f = f_energy_nu(E,alpha)
    

    def normalize(y):
        m = len(y)
        sum = np.sum(y)
        
        return y*m/sum
    
    
    def shift_base(E_nu,f,E):
        dE = E[1]-E[0]
        f_new = np.zeros(len(f))
        E_judge = E + dE/2
        
        E_now  = E[0]
        id = 0
        while E_nu[id]<=E_now:
            id = id+1
        
        id_judge = 0
        while (id_judge  <len(f) )and ((id)  <len(f)):
            s = []
            while E_nu[id]<=E_judge[id_judge]:
                s.append( f[id])
                id=id+1
                if id ==len(f): 
                    break
            s = np.array(s)
            f_new[id_judge] = np.average(s)
            id_judge += 1
 
        return f_new

    def energy_conv(E):
        return kim.energy_kicked_by_neutrino(E,M_nu,m_dm)#E+m_dm-
    
    def opacity(start,end,E,f):
        L = (np.sum((start-end)**2))**0.5
        l = end -start
        r=(np.sum((start+l*0.5)**2))**0.5
        x= r/rs
        rho_structure =1/(x*(1+x)*(1+x))
        ratio = np.exp(-rho_structure*L*rho_s*cs/m_dm)
        print(ratio)

        E_nu = energy_conv(E*e_per_nu)/e_per_nu
        f_n = shift_base(E_nu,f,E)*(1-ratio) + f*ratio
        #f_n = normalize(f_n)
        return f_n

    E_nu = energy_conv(E*e_per_nu)/(e_per_nu)
    f_n = shift_base(E_nu,f,E)
    f = normalize(f)
    f_n = normalize(f_n)
    plt.plot(E*E_per_nu/1e6, f, color ='blue', label = 'Neutrino energy distribution')
    plt.plot(E*E_per_nu/1e6, f_n, color ='red', label = 'DM energy distribution')
    plt.xlabel('E (MeV)')
    plt.ylabel('Probability Density')
    plt.legend(loc='upper right')
    plt.savefig('DM distribution.png')
    plt.show()

    def de_zero(E,f):
        id = 0
        while f[id]!=0.:
            id+=1
        f_new = f[:id]
        E_new = E[:id]
        print(f_new)
        return E_new,f_new
    """E,f_n = de_zero(E,f_n)
    def deviation(para):
        a= para[0]
        k = para[1]
        s= para[2]
        def f_energy_dm(E,a,s):
            return (E**a)*np.exp(-(a+s)*E)
        f_std = f_energy_dm(E,a,s)
        f_std = k*f_std
        vyerr = f_n**0.5
        
        return np.sum(((f_std-f_n)/vyerr)**2)
    para =[2.6,1.,1.]
    result = opt.minimize(deviation,para).x
    print(result)
    f_new = result[1]*(E**result[0])*np.exp(-(result[0]+result[2])*E)
    plt.plot(E*E_per_nu/1e6, f_new, color ='green', label = 'fitting')
    plt.show()"""


    """N = 1000
    l = (end-start)/N
    for i in range(N):
        f = opacity(start+l*i,start+l*(i+1),E,f)"""
    
    

if __name__== '__main__':
    
    
    print("Total number of neutrino:"+str(E_total_nu/E_per_nu))

    start=np.array([1*3.08567758e21,0*3.08567758e21,0])
    end =np.array([10*3.08567758e21,0,0])
    
    
    f_evolution(M_DM,E_max,E_per_nu ,Alpha,start,end)
    