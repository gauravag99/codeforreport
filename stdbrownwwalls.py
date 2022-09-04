import numpy as np
import matplotlib.pyplot as plt
from numba import jit
import time

time0=time.time()

num_sims = 100000  # Number of runs to evaluate

t_init = 0
t_end  =2
dt     = 0.001
wall1 , wall2 = 0.5,-0.5
N      = int((t_end - t_init)/dt)  #this many grid points will be calculated
y_init = 0
prdens_time = (0, int(0.03/dt), int(0.05/dt),int(0.09/dt),int(0.1/dt), int(0.15/dt), N-1)
styledict = ('-','--','-.','-','--','-.', '--' )

ts = np.arange(t_init, t_end, dt) #timestep array
probdenspos = np.zeros((np.size(prdens_time),num_sims))
@jit(nopython=True ,fastmath=True)
def loop(dt,N,num_sims,y_init,prdens_time,probdenspos):
    posonerun = np.zeros(N) + y_init
    variance = np.zeros(N)
    
    y=np.zeros(num_sims)
    Dt = np.sqrt(dt)
    i=0
    for j in range(N):
        y += np.random.normal(0,Dt,num_sims)
        temp = (y<wall2)
        temp2 = np.multiply((2*wall2-np.multiply(y,temp)),temp)
        y = np.multiply(~temp,y) + temp2
        temp = (y>wall1)
        temp2 = np.multiply((2*wall1 - np.multiply(y,temp)),temp)
        y = np.multiply(~temp,y) + temp2
        variance[j] = np.var(y)
        posonerun[j] = y[0]
        if j in prdens_time:
            probdenspos[i,:] = y
            i+=1

    return posonerun,variance,probdenspos

posonerun,variance,probdenspos = loop(dt,N,num_sims,y_init,prdens_time,probdenspos)


# #plot one of the motions:
print(np.shape(ts),np.shape(posonerun))
plt.plot(ts[0:N], posonerun,'g-' ,lw=0.5)


# #--- below variance is plotted ----- #

iteratedlog2 = np.sqrt(np.multiply( 2*ts , np.log(np.log(ts)) ))
# maxt = int(40/dt)
# plt.plot(ts[:maxt],variance[:maxt], lw=0.5, c='black', label=f'Calculated variance from {num_sims} runs.')
plt.plot(ts, iteratedlog2, 'b--', lw=0.7, label =r'$\pm \sqrt{2t\log{\log{t}}}$')
# plt.plot(ts, -iteratedlog2, 'b--', lw=0.7)

plt.legend()
plt.title("Brownian Motion")
plt.xlabel("Time (s)")
plt.ylabel("Position", rotation='vertical')
plt.minorticks_on()
plt.grid(which='both')
plt.savefig('teststd1',dpi=600)

#---- plot probability density below ---- #
plt.clf()

bins = np.linspace(-1, 1, 100)

i=0
for j in prdens_time: #plotting prob density for velocity at times in prdens_time
    hist, _ = np.histogram(np.reshape(probdenspos[i,:],-1), bins=bins, density =True)
    plt.plot((bins[1:] + bins[:-1]) / 2, hist,dict(zip(prdens_time,styledict))[j],label=f"t={j * dt:.2f}")
    i+=1
    
plt.legend()
plt.title("Position distribution at different times")
plt.xlabel("Position")
plt.ylabel("Probability", rotation='vertical')
plt.savefig('teststd2',dpi=600)

plt.clf()

bins = np.linspace(0.2, 0.6, 20)

i=0
for j in prdens_time: #plotting prob density for velocity at times in prdens_time
    hist, _ = np.histogram(np.reshape(probdenspos[i,:],-1), bins=bins, density =True)
    plt.plot((bins[1:] + bins[:-1]) / 2, hist,label=f"t={j * dt:.2f}")
    i+=1
    
plt.legend()
plt.title("Position distribution at different times")
plt.xlabel("Position")
plt.ylabel("Probability", rotation='vertical')
plt.savefig('teststd3')

print(f"done in {time.time()-time0}")