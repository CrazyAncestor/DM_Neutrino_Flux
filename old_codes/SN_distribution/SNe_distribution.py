import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
import time

import pandas as pd
data_source= pd.read_csv("profile.csv")#You may input the wanted file name here
data = np.array(data_source)

Rd = 2.9e03#kpc
H = 95 #pc
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
def diff_prob(r):
    r2 = r**2
    def func(theta,phi):
        pos = ang_to_car(np.array([r,theta,phi]))
        #pos: Sun centric pos
        pos_GAC = np.array([-8.7e03,0,-24])
        d = pos - pos_GAC
        dV = r2*np.sin(theta)
        return np.exp(-R_2(d)**0.5/Rd)*np.exp(-np.abs(d[2])/H) *dV
    return integrate.nquad(func, [[0,np.pi], [0,2*np.pi]])[0]

def normalize(x,y):
    sum = np.sum(y)
    dx = x[-1]-x[0]
    print(sum)
    return y/(sum)
if __name__== '__main__':
    print(data)
    r = data[:,0]/1e03#kpc
    prob_dens = data[:,1]
    prob_dens = normalize(r,prob_dens)

    plt.plot(r, prob_dens, color ='blue', label = 'ccSNe distribution')    
    plt.xlabel('r (kpc)')
    plt.ylabel('Probability Density')
    plt.legend(loc='upper right')
        
    plt.show()
    







"""pos = np.array([np.linspace(0e03,15e03,1000),np.linspace(0e03,0e03,1000),np.linspace(24,24,1000)])
pos = np.transpose(pos)
distribution = f(pos)

pos_x = np.linspace(0e03,15e03,1000)
r_los = pos_x/1e03 
"""