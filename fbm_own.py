import numpy as np
import matplotlib.pyplot as plt

def autcovar(x,H,C=1):
    """ 
    Defines the covariance of fBm.
    Input -> x : index , H : hurst parameter, C : scale (default=1).
    Ouput -> covariance
    """
    autocovariance = C**2 /2 * ( (abs(x-1))**(2*H) + (abs(x+1))**(2*H) -2*(abs(x))**(2*H) )
    return autocovariance

def fGn_hosking(N,H,init_pos=0):
    """
    Generates fractional gaussian noise using hosking's recursion method.
    Input -> H: hurst index, N : number of samples to generate,init_pos : initial sample of noise.
    Output -> (N+1)x1 array of fGn
    """
    # Initialisation of all parameters
    mu = autcovar(1,H) * init_pos
    var = 1-autcovar(1,H) **2
  #  var = 1
    d = np.zeros(N)
    d[0] = autcovar(1,H)
    x = np.zeros(N+1)
    x[0] = init_pos
    x[1] = np.random.normal(mu,np.sqrt(var))
    c = [autcovar(i,H) for i in range(1,N)]

    for i in range(2,N+1):
        tau = np.dot(c[0:i-2],np.flip(d[0:i-2]))
        phi = (autcovar(i,H) - tau)/var
        var = (var - phi**2 * var)
        d[0:i-2] = d[0:i-2] - phi* np.flip(d[0:i-2])
        d[0:i-1] = phi
        mu = np.dot(d[0:i-1],np.flip(x[0:i-1]))
        x[i] = np.random.normal(mu,np.sqrt(var))

    return x

def fGn_cholesky(N,H):
    """
    Generates fractional gaussian noise using cholesky's decomposition method.
    Input -> H: hurst index, N : number of samples to generate,init_pos : initial sample of noise.
    Output -> (N+1)x1 array of fGn
    """
    Gamma = np.zeros((N,N))
   # Gamma = [ [autcovar(j-i,H) for i in range(j+1)] for j in range(N)]
    for i in range(N):
        for j in range(i+1): Gamma[i,j] = autcovar(i-j,H)
    cov = np.linalg.cholesky(Gamma)
    x = np.dot(cov, np.array(np.random.normal(0,1,N)).transpose())
    x = np.squeeze(x)
    return x


dt=0.001
N = int(1/dt)

fig,axs = plt.subplots(3, figsize=(15,10))


y=np.cumsum(fGn_hosking(N,0.2))
t = np.linspace(0,1,y.size)
axs[0].plot(t,y,label='H=0.2',lw=1,c='green')
axs[0].legend()

y=np.cumsum(fGn_hosking(N,0.5))
t = np.linspace(0,1,y.size)
axs[1].plot(t,y,label='H=0.5',lw=1,c='black')
axs[1].legend()

y=np.cumsum(fGn_hosking(N,0.8))
t = np.linspace(0,1,y.size)
axs[2].plot(t,y,label='H=0.8',lw=1,c='red')
axs[2].legend()

plt.savefig('fbmhosking.png',dpi=600)

plt.clf()

fig,axs = plt.subplots(3, figsize=(15,10))


y=np.cumsum(fGn_cholesky(N,0.2))
t = np.linspace(0,1,y.size)
axs[0].plot(t,y,label='H=0.2',lw=1,c='green')
axs[0].legend()

y=np.cumsum(fGn_cholesky(N,0.5))
t = np.linspace(0,1,y.size)
axs[1].plot(t,y,label='H=0.5',lw=1,c='black')
axs[1].legend()

y=np.cumsum(fGn_cholesky(N,0.8))
t = np.linspace(0,1,y.size)
axs[2].plot(t,y,label='H=0.8',lw=1,c='red')
axs[2].legend()

plt.savefig('fbmcholesky.png',dpi=600)