import numpy as np
import matplotlib.pyplot as plt
from numba import jit
num_sims = 100000  # Number of runs to evaluate

t_init = 0
t_end  =1.0001
dt     = 0.0001
N      = int((t_end - t_init)/dt)  #this many grid points will be calculated
y_init = 0
prdens_time = (1, 50, 150, N )


ts = np.arange(t_init, t_end, dt) #timestep array
#@jit(nopython=True ,fastmath=True)
def loop(dt,N,num_sims,y_init):
    pos = np.zeros(num_sims) +y_init
    Dt=np.sqrt(dt)
   # positivepos=np.zeros(num_sims)
    positivetime= np.zeros(num_sims)
    maxpostime = np.zeros(num_sims)
    maxpostn=np.zeros(num_sims)
    signchangetime = np.zeros(num_sims)
    lastiter = np.zeros(num_sims).astype(bool)
    
    for j in range(N):
        pos += np.random.normal(0,Dt,num_sims)
        thisiter = pos>0
       # temp = (lastiter-thisiter).astype(bool)
        temp = np.logical_xor(lastiter,thisiter)
        signchangetime = np.multiply(signchangetime,~temp) + temp*j
        lastiter = thisiter
        positivetime += thisiter

        temp = maxpostn > pos
        maxpostime = np.multiply(maxpostime,temp) + ~temp*j
        maxpostn = np.multiply(maxpostn,temp) + np.multiply(pos,~temp)
    
    return positivetime,maxpostime,signchangetime

positivetime,maxpostime,signchangetime = loop(dt,N,num_sims,y_init)
positivetime = positivetime*dt
maxpostime = maxpostime*dt
signchangetime = signchangetime*dt

plt.title(r"First arcsine law - $10^4$ samples")
plt.xlabel("Time")
plt.ylabel('Probability')
bins = np.linspace(0,1,100)
hist,_ = np.histogram(positivetime, bins = bins, density=True )
plt.plot((bins[1:] + bins[:-1]) / 2, hist, 'k-', lw=1)
plt.savefig('arcsinepositivetime')


plt.clf()
plt.title(r"Second arcsine law - $10^4$ samples")
plt.xlabel("Time")
plt.ylabel('Probability')
bins = np.linspace(0,1,100)
hist,_ = np.histogram(maxpostime, bins = bins, density=True )
plt.plot((bins[1:] + bins[:-1]) / 2, hist, 'r-', lw=1)
plt.savefig('arcsinemaxpostime')

plt.clf()
plt.title(r"Third arcsine law - $10^4$ samples")
plt.xlabel("Time")
plt.ylabel('Probability')
bins = np.linspace(0,1,100)
hist,_ = np.histogram(signchangetime, bins = bins, density=True )
plt.plot((bins[1:] + bins[:-1]) / 2, hist, 'g-', lw=1)
plt.savefig('arcsinesignchangetime', dpi=600)


