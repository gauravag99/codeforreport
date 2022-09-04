import numpy as np
import matplotlib.pyplot as plt
from numba import jit
num_sims = 10000  # Number of runs to evaluate

t_init = 0
t_end  = 0.2
dt     = 0.0001
N      = int((t_end - t_init)/dt)  #this many grid points will be calculated
y_init = 0



ts = np.arange(t_init, t_end+dt, dt) #timestep array
ys = np.zeros(N + 1)
print(np.shape(ts),np.shape(ys))
ys[0] = y_init
variance = np.zeros(N+1)
allmotions = np.zeros((num_sims,N+1))

#numba is used to compile code into machine language, as python interpreter is slow
@jit(nopython=True,fastmath=True)
def loop(num_sims,ts,t_init, dt,ys,allmotions):
    for j in range(num_sims):
        for i in range(1, ts.size):
            t = t_init + (i - 1) * dt
            y = ys[i - 1]
            ys[i] = y + np.random.normal(loc=0.0, scale=np.sqrt(dt))
        allmotions[j] = ys
    return ys, allmotions

ys, allmotions = loop(num_sims,ts,t_init,dt,ys,allmotions)

#plot one of the motions:
plt.plot(ts, ys, lw=0.5, c='green')

#calculation of variance using all the runs
for j in range(1,ts.size):
    variance[j] = np.var(allmotions[:,j])

#--- below variance is plotted ----- #
maxt= int(0.04243/dt)
print(maxt)
iteratedlog = np.sqrt( 2*np.multiply(ts[1:maxt],np.log(np.log(np.reciprocal(ts[1:maxt])))) )
iteratedlog2 = np.sqrt(np.multiply( 2*ts , np.log(np.log(ts)) ))

plt.plot(ts,variance, lw=0.5, c='black', label=f'Calculated variance from {num_sims} runs.')
plt.plot(ts[1:maxt], iteratedlog, 'b--', lw=0.7, label =r'$\pm \sqrt{2n\log{\log{\frac{1}{t}}}}$')
plt.plot(ts[1:maxt], -iteratedlog, 'b--', lw=0.7)
#plt.plot(ts, iteratedlog2, 'b--', lw=0.7)

plt.legend()
plt.title("Brownian Motion")
plt.xlabel("Time (s)")
plt.ylabel("Velocity", rotation='vertical')
plt.minorticks_on()
plt.grid(which='both')
plt.savefig('stdbrown1',dpi=600)

#---- plot probability density below ---- #
plt.clf()
prdens_time = (1, 50, 150, N )
styledict = ('-','--','-.','-')
bins = np.linspace(-1, 1, 100)

for j in prdens_time: #plotting prob density for velocity at times in prdens_time
    hist, _ = np.histogram(np.reshape(allmotions[:,j],-1), bins=bins, density =True)
    plt.plot((bins[1:] + bins[:-1]) / 2, hist,dict(zip(prdens_time,styledict))[j],label=f"t={j * dt:.2f}")
    
plt.legend()
plt.title("Velocity distribution at different times")
plt.xlabel("Velocity")
plt.ylabel("Probability", rotation='vertical')
plt.savefig('stdbrown2')