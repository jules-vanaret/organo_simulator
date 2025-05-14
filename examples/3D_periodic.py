import numpy as np
from tqdm import tqdm
from organo_simulator.utils import display_particles_in_napari
from organo_simulator.simulator_LJ_force import FastOverdampedSimulator

np.random.seed(1)


dt = 1
d = 3 # dimension of simulation
N_part = 1000 # number of particles
nuclei_sizes = np.random.uniform(0.5,1,N_part)


# persistence times (in physical units) for the OU process
persistence_times = np.random.randint(10, 2000, N_part)
viscosities = np.array([1]*(N_part))
Ds = np.array([1e-5]*(N_part)) # diffusion coefficients

Nx = 1000 # length of the box in pixels for the rendering 
L=10

simulator = FastOverdampedSimulator(
    L=L, # will be initializesd automatically
    Nx=Nx, # length of the box in pixels for the rendering 
    d=d, # dimensions (2 or 3)
    N_part=N_part, # number of particles
    nuclei_sizes=nuclei_sizes, # radius of particles
    viscosity=viscosities, # viscosities of particles
    D=Ds, # diffusion coefficients
    persistence_time=persistence_times, # how long the temporal autocorrelation of random movement goes
    potential_energy=1e-8,
    parallel=True,
    periodic=True,
)

skip_time = 10 # return coordinates every ... s
skip = int(skip_time/dt)
total_time = 1000 # total number of simulation steps
total_steps = int(total_time/dt) 
Nt = int(total_steps/skip)

# Initializing arrays to collect data
data = np.empty((N_part*Nt,d+1))

for i in tqdm(range(total_steps)):
    simulator.update_dynamics(dt=dt)

    if i%skip==0:
        positions = simulator.dump_coordinates()
        time_pos = np.hstack((i/skip * np.ones((N_part,1)), positions))

        data[int(i/skip)*len(time_pos):(int(i/skip)+1)*len(time_pos),:] = time_pos


tids = np.tile(np.arange(N_part), Nt)
tracks = np.empty((N_part*Nt, 2+d))
tracks[:,0] = tids
tracks[:,1:] = data

# otherwise napari screws up the colors
tracks = sorted(tracks, key=lambda l: l[0])
tracks = np.array(tracks)

display_particles_in_napari(tracks, nuclei_sizes, Nt, L)