import numpy as np
import matplotlib.pyplot as plt
import time
from numba import jit

## -----All physical variables -------#
mass = 1
gamma = 0.1
mu = 2                       # Mean velocity (affects drift)
#(don't put 0 here)
Gamma = 2*((mu*mass)**2) * gamma
##---- All Simulation variables -----#
init_pos = 0
init_vel = 0
wall2, wall1 = -1,1           # Wall coordinates in x
dt = .001                     # Time step.
T = 10                        # Total time in seconds.
ntrials=10000                 # number of brownian motions to do

#------------------------------------#
n = int(T / dt)                # Number of time steps.
time0 = time.time()
print("number of time steps = ",n)
print("number of trials = ", ntrials)
#---------Initialisation-------------#

vel = np.zeros((1,ntrials))+init_vel
postn = init_pos+np.zeros((1,ntrials))
prdens_time = (0, int(0.005/(gamma*dt)), int(0.05/(gamma*dt)), int(0.2/(gamma*dt)), int(T/dt)-1 ) 
#the times at which probability density will be plotted

probdensvel = np.zeros((np.size(prdens_time),ntrials))
probdenspos = np.zeros((np.size(prdens_time),ntrials))

""" 
#LOOP EXPLAINED:
# tempwall returns the boolean array indicating if particle has reached out of wall
# temp holds the array of new negative velocities at the true locations indicated by tempwall, rest are zero
# new vel gets upddated by normal laws, but only the ones which are within limits, i.e. false in tempwall i.e. ~tempwall
# then just add temp to new vel to reverse the required velocities
# position is updated and saved onto file
"""
@jit(nopython=True,fastmath=True)
def loop(vel,postn,probdensvel,probdenspos,gamma,Gamma,mass,n,mu,dt):
    i=0
    gammaloop = gamma*dt
    sigmaloop = np.sqrt(Gamma)/mass
    variancevel = np.zeros(n)
    variancepos = np.zeros(n)
    velonerun = np.zeros(n)
    posonerun = np.zeros(n)
    Dt=np.sqrt(dt)
    for j in range(n):
        if j in prdens_time:
            probdensvel[i,:] = (vel[0,:])
            probdenspos[i,:] = (postn[0,:])
            i+=1
        tempwall = (postn[0,:] > wall1) + (postn[0,:] < wall2)
        temp = np.multiply(vel[0,:],-1*(tempwall))
        vel[0,:] = np.multiply((vel[0,:] -(vel[0,:]-mu)*gammaloop +sigmaloop*np.random.normal(0,Dt,size=(1,ntrials))),~tempwall)
        vel[0,:] += temp
        postn[0,:] = postn[0,:]+ vel[0,:]*dt
        variancevel[j] = np.var(np.reshape(vel,-1))
        variancepos[j] = np.var(postn[0,:])
        velonerun[j] = vel[0,0]
        posonerun[j] = postn[0,0]
        

    return velonerun,posonerun,variancevel,variancepos,probdensvel,probdenspos
    
velonerun,posonerun,variancevel,variancepos,probdensvel,probdenspos = loop(vel,postn,probdensvel,probdenspos,gamma,Gamma,mass,n,mu,dt)
time0=time.time()-time0
print("Calculations done in ", time0, "seconds. Now starting plotting...")
# ---------Simulation Ended-------------------------#
time0=time.time()

# ----------- block related to plotting ----------- #
t = np.linspace(0., T, n)      # array of time for plotting x axis
fig, axs = plt.subplots(2,2 , figsize=(20,10))

axs[0,0].set_title('Brownian Motion(velocity)')
axs[0,0].grid(b=True, which='major', color='grey', linestyle='-')
axs[0,0].grid(b=True, which='minor', color='grey', linestyle='--')
axs[0,0].minorticks_on()
axs[0,1].set_title('Brownian Motion(position)')
axs[0,1].grid(b=True, which='major', color='grey', linestyle='-')
axs[0,1].grid(b=True, which='minor', color='grey', linestyle='--')
axs[0,1].minorticks_on()
axs[1,0].set_title('Velocity\'s probability density at different times')
axs[1,1].set_title('Position\'s probability density at different times')
styledict = ('-','--','-.','-','-.')
axs[0,0]
#--------------------------------------------------#

#----- plotting position and velocity wrt time ------#

axs[0,0].plot(t, np.zeros(n), c='black', lw=0.5) #refrence line
axs[0,1].plot(t,posonerun, lw=0.5)
axs[0,0].plot(t,velonerun, lw=0.5, c='purple', label=f"Velocity")
axs[0,0].plot(t,variancevel[:n], c='red', lw=0.5, label=f"Calculated Variance")
axs[0,0].plot(t,(mu**2)*(1-np.exp(-2*gamma*t)),'-.',c='g',lw=0.5,label=f"Theoretical Variance")
axs[0,0].legend()
axs[0,0].xaxis.set_major_locator(plt.MaxNLocator(10))
axs[0,0].yaxis.set_major_locator(plt.MaxNLocator(10))
axs[0,0].set_xlabel("t (seconds)")
axs[0,0].set_ylabel("Velocity", rotation=90)


#---- plotting prob density of velocity and position-----#
bins = np.linspace(-1, 4, 100)
i=0
for j in prdens_time: #plotting prob density for velocity at times in prdens_time
    hist, _ = np.histogram(np.reshape(probdensvel[i,:],-1), bins=bins, density =True)
    axs[1,0].plot((bins[1:] + bins[:-1]) / 2, hist,dict(zip(prdens_time,styledict))[j],label=f"t={j * dt:.2f}")
    axs[1,0].legend()
    i +=1

bins = np.linspace(-2, 2, 100)
i=0
for j in prdens_time: #only for these times create histogram
    hist, _ = np.histogram(np.reshape(probdenspos[i,:],-1), bins=bins, density =True)
    axs[1,1].plot((bins[1:] + bins[:-1]) / 2, hist,dict(zip(prdens_time,styledict))[j],label=f"t={j * dt:.2f}")
    axs[1,1].legend()
    i+=1
plt.title(f'Brownian Motion for {T} seconds.')
plt.savefig('test.png',dpi=600)
print("Time spent in plotting = ", time.time()-time0, "seconds.")

