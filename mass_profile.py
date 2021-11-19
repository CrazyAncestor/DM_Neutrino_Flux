
import numpy as np
from matplotlib import pyplot as plt
from locale import atof

def column_value(line,col):
   a = line.split(' ')
   idx = 0
   for x in a:
      if x!='':
         idx=idx+1
      if idx==col:
         return atof(x)
   return 0

def Fornax_Dens_readfile(filename):
   fp = open(filename, "r")
   line = fp.readline()
   dens = []
   r = []
   while line:
      line = fp.readline()
      dens.append(column_value(line,2))
      r.append(column_value(line,1))
         
   fp.close()
   r = r[0:-1]
   dens = dens[0:-1]
   return r,dens

def Enclosed_Mass_CDM(r,dens,r_max=-1.):
   r = np.array(r)
   dens=np.array(dens)
   r_avg = (r[1:]+r[0:-1])/2.
   dr    = (r[1:]-r[0:-1])
   dens_avg = (dens[1:]+dens[0:-1])/2.
   if r_max<0. or r_avg[-1]<=r_max:
      return 4* np.pi* np.sum(r_avg**2*dr*dens_avg)
   else:
      id_max = np.where(r_avg>r_max)[0][0]
      r_avg = r_avg[0:id_max]
      dr    = dr[0:id_max]
      dens_avg = dens_avg[0:id_max]

      return 4* np.pi* np.sum(r_avg**2*dr*dens_avg)

def mass_soliton(r, m_22, x_c): # r in unit of kpc
   factor = (2**(1/8)-1.)**0.5
   x = r/x_c
   a = 1
   enclosed_mass =  4.077703890131877e6*a**-1./(m_22*10)**(2.)/x_c/(1.+(factor*x)**2)**7*\
                    (3465*(factor*x)**13.+23100*(factor*x)**11+65373*(factor*x)**9+\
                     101376*(factor*x)**7+92323*(factor*x)**5+48580*(factor*x)**3-\
                     3465*(factor*x)+3465*((factor*x)**2+1)**7.*np.arctan(factor*x))  # in unit of M_sun
   return enclosed_mass/1e6 # in unit of Msun

r_CDM,dens_CDM = Fornax_Dens_readfile('CDM_Dens.txt')
r_PsiDM = r_CDM
mass_CDM = []
mass_PsiDM = []

m_22 = 0.662823805791
x_c = 1

for i in range(len(r_CDM)):
   mass_CDM.append(Enclosed_Mass_CDM(r_CDM,dens_CDM,r_max=r_CDM[i]))
   mass_PsiDM.append(mass_soliton(r_PsiDM[i], m_22, x_c))

#figure(figsize=(10, 6), dpi=80)
#plt.ylim(0,0.4)
plt.plot(r_PsiDM,mass_PsiDM,label = 'PsiDM',linewidth=1)
plt.plot(r_CDM,mass_CDM,label = 'CDM',linewidth=1)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('r (kpc)')
plt.ylabel('mass (1e6Msun)')
plt.legend(loc = 'best')

plt.show()