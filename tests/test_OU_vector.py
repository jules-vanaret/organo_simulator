import numpy as np
import napari
import matplotlib.pyplot as plt
from tqdm import tqdm
import numba
import time
from scipy.signal import fftconvolve


@numba.jit(nopython=True)
def integrate(Ttot, dt, exp_dt_tau, sqrt_term):

    vals = np.zeros(int(Ttot/dt))
    times = np.zeros(int(Ttot/dt))
    time=0

    vec = 0
    
    for i in range(int(Ttot/dt)):
        # sigma = np.sqrt(2*std**2*dt/tp)
        # ou_diff = - vec/tp * dt + np.random.normal(loc=0, scale=sigma, size=(d))
        # vec = vec + ou_diff

        vec = vec * exp_dt_tau + sqrt_term * np.random.normal(loc=0, scale=1)

        vals[i] = vec
        time+=dt
        times[i] = time

    return vals, times

# @numba.jit(nopython=True, parallel=True)
def main(tp):

    # numba.set_num_threads(2)

    
    std = 17
    dt = 0.1

    Ttot = 100000



    exp_dt_tau = np.exp(-dt/tp)
    sqrt_term = np.sqrt( std**2 * (1-np.exp(-2*dt/tp)) )

    acors = np.zeros((100, int(Ttot/dt/2)-1))

    # for i in numba.prange(100):
    for i in range(100):


    
        vals, times = integrate(Ttot, dt, exp_dt_tau, sqrt_term)


        yunbiased = vals-np.mean(vals)
        ynorm = np.sum(yunbiased**2)


        acor =fftconvolve(yunbiased, np.flip(yunbiased))
    

        res = acor[int((1+len(acor))*2/4):int((1+len(acor))*3/4-1)]/ynorm
        # res = np.arange(int(Ttot/dt)-1)
        # time.sleep(1)
        

        acors[i]=res

    return acors, times

tp = 50

acors, times = main(tp)




    

plt.figure()
plt.plot(times[:int(len(times)/2)-1],np.log(np.abs(np.mean(acors,axis=0))))
# plt.plot(times[:int(len(times)/2)],np.abs(np.mean(acors,axis=0)),'.')

times = np.array(times[:int(len(times)/2)-1])
plt.plot(times,np.log(np.exp(-times/tp)),'k--')

# plt.figure()

# plt.plot(times[:int(len(times)/2+1)],1-acor)


plt.show()

# viewer = napari.Viewer()

# viewer.add_vectors(
#     data=np.array(napari_vecs),
#     ndim=3,
# )


# napari.run()