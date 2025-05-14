import numpy as np
from tqdm import tqdm
# from organo_simulator.simulator_LJ_force import FastOverdampedSimulator
from organo_simulator.simulator_LJ_force import FastOverdampedSimulator

import napari

np.random.seed(1)

###
#! THIS SIMULATION IS AT THE BRINK OF INSTABILITY WRT TO THE ENERGY POTENTIAL
###


dt = 1
d = 2 # dimension of simulation
N_part = 300 # number of particles
nuclei_sizes = 1

# persistence times (in physical units) for the OU process
persistence_times = np.array([400]*(N_part))
viscosities = np.array([1]*(N_part))
Ds = np.array([1e-5]*(N_part)) # diffusion coefficients

Nx = 1000 # length of the box in pixels for the rendering 

simulator = FastOverdampedSimulator(
    L=None, # will be initializesd automatically
    Nx=Nx, # length of the box in pixels for the rendering 
    d=d, # dimensions (2 or 3)
    N_part=N_part, # number of particles
    nuclei_sizes=nuclei_sizes, # radius of particles
    viscosity=viscosities, # viscosities of particles
    D=Ds, # diffusion coefficients
    persistence_time=persistence_times, # how long the temporal autocorrelation of random movement goes
    energy_potential=2e-3,
    parallel=True
)

skip = 100 # return coordinates every s
total_steps = 10000 # total number of simulation steps
Nt = int(total_steps/skip)

# Initializing arrays to collect data
data = np.empty((N_part*Nt,d+1))
# data_points=[]

for i in tqdm(range(total_steps)):
    simulator.update_dynamics(dt=dt)

    if i%skip==0:
        positions = simulator.dump_coordinates()
        time_pos = np.hstack((i/skip * np.ones((N_part,1)), positions))

        data[int(i/skip)*len(time_pos):(int(i/skip)+1)*len(time_pos),:] = time_pos
        # data_points.append(positions)

import numpy as np
tids = np.tile(np.arange(N_part), Nt)
tracks = np.empty((N_part*Nt, 2+d))
tracks[:,0] = tids
tracks[:,1:] = data

# otherwise napari screws up the colors
tracks = sorted(tracks, key=lambda l: l[0])

viewer = napari.Viewer()
viewer.add_tracks(tracks)
napari.run()