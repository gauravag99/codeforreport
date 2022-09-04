import numpy as np
import matplotlib.pyplot as plt
from numba import jit
import time
import gc

num_sims = 10000  # Number of runs to evaluate
length = (10**4)    #number of time iterations to process in one chunk
t_init = 0
t_end  =1000
dt     = 0.001
N      = int((t_end - t_init)/dt)  #this many grid points will be calculated
y_init = 0
print(f"{N} iterations for {num_sims} trials will being processed.")
print(f"{int(N/length)} chunks will be processed. Each chunk has {length} timesteps.")
ts = np.arange(t_init, t_end, dt) #timestep array

if (N%length != 0):
        raise Exception("Length not compatible. Change either N or length such that N is divisible by length.")
def mainfunction(dt,N,num_sims,y_init,length):
    if (N%length != 0):
        raise Exception("Length not compatible. Change either N or length such that N is divisible by length.")
    tempprob, tempposmaxvalue = np.zeros(length),np.zeros((length,num_sims))
    np.savetxt('test.txt', [])
    probability=np.array([])
    y = y_init
    with open('test.txt', 'a') as posmaxvaluefile:
        for i in range(int(N/length)):
            tempprob, tempposmaxvalue,y = loop(dt,length,num_sims,y,i)
            np.savetxt(posmaxvaluefile, tempposmaxvalue, delimiter=",")
            probability = np.hstack((probability,tempprob))
            del(tempprob,tempposmaxvalue)
            gc.collect()
            print(f"{(i+1)*100/(int(N/length))}% done.")
    return probability


@jit(nopython=True ,fastmath=True, parallel=True)
def loop(dt,N,num_sims,y_init,i):
    posmaxvalue = np.zeros((N,num_sims))
    probability = np.zeros(N)
    y=np.zeros(num_sims) + y_init
    Dt = np.sqrt(dt)
    k=2.719+i*N*dt
    for j in range(N):
        y += np.random.normal(0,Dt,num_sims)
        temp = np.absolute(posmaxvalue[j-1,:])>np.absolute(y)
        posmaxvalue[j,:] = np.absolute(np.multiply(~temp,y) + np.multiply(temp,posmaxvalue[j-1,:]))
        k=k+dt
        probability[j] = np.sum(np.absolute(y)>np.sqrt(2*k*np.log(np.log(k))))/num_sims

    return probability,posmaxvalue,y

time0 = time.time()
probability = mainfunction(dt,N,num_sims,y_init,length)
print("Simulation done in: ",time.time()-time0, ". Proceeding with plotting.")


#----------- PLOT KHINCHIN's PROB w/ TIME -------#
plt.xlabel("Time")
plt.ylabel("Probability", rotation='vertical')
plt.title(f"Probability of Brownian motion crossing \n the Khinchin's law boundary - {num_sims} samples", name='CMU Sans Serif')
plt.plot(ts[:N],probability,lw=0.5)
plt.savefig('stdbrownitlog', dpi=600)


#--------- PLOT LIM SUP OF POSITION ------------#

cols=50
plt.clf()
gc.collect()
for i in range(int(num_sims/cols)):
    j=[i*cols+k for k in range(cols)]
    temp = np.loadtxt('test.txt', usecols=j, delimiter=',')
    for k in range(cols):
        plt.plot(ts[:N], temp[:N,k], lw=0.5)
    del(temp)
    gc.collect()
    print(f"{(i+1)*cols} / {num_sims} done.", '\r')
iteratedlog2 = np.sqrt(np.multiply( 2*(ts+2.719) , np.log(np.log(ts+2.719)) ))
plt.plot(ts, iteratedlog2, 'k--', lw=0.9, label =r'$\pm \sqrt{2n\log{\log{t}}}$')
plt.xlabel("Time (sec)")
plt.ylabel("Position")
plt.title("Brownian Motion - 200 samples \n lim sup of absolute position")
plt.savefig('iteratedlogruns', dpi=600)
gc.collect()