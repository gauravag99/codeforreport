import matplotlib.pyplot as plt
import numpy as np


samples=int(1) # number of motions to simulate
time_limit=int(10000) # number of steps each brownian motion is evolved for
variance=1 #the variance of the normal distribution from which the increments are taken
           # since the increments ~ N(0,t-s); here t-s is 1 unit
initial_x, initial_y = [0],[0]


SAMPLEX = np.empty((samples,time_limit+1)) #store the position of the particle
SAMPLEY = np.empty((samples,time_limit+1))

for i in range(0,samples):
    
    x_inc = np.random.normal(0,variance,time_limit)  #get increments from a normal distribution
    y_inc = np.random.normal(0,variance,time_limit)

    SAMPLEX[i] = np.append(initial_x, np.cumsum(x_inc)) #set starting point to zero, then append to it 
    SAMPLEY[i] = np.append(initial_y, np.cumsum(y_inc)) #final position of particle at each point

    plt.plot(SAMPLEX[i],SAMPLEY[i], c='blue', linewidth=0.3)

plt.title('Random Walk')
plt.xlabel("X")
plt.ylabel("Y")
plt.savefig('randomwalk.png',dpi=600)