import numpy as np
from tqdm import tqdm
from organo_simulator.simulator_LJ_force import FastOverdampedSimulator
import napari

np.random.seed(1)

dt = 0.1
d=2 # dimension of simulation
N_part = 20 # number of particles
nuclei_sizes = 1

# persistence times (in physical units) for the OU process
persistence_times = np.array([10]*(N_part))
viscosities = np.array([1]*(N_part))
# Ds = np.array([1e-7]*(N_part)) # diffusion coefficients
Ds = 0

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
    energy_potential=0,#1e-6,
    parallel=True,
    flow_field=0.1
)

skip = 10 # return coordinates every s
total_steps = 10000 # total number of simulation steps
Nt = int(total_steps/skip)

# Initializing arrays to collect data
data = np.empty((N_part*Nt,d+1))
data_points=[]

fs = []


for i in tqdm(range(total_steps)):
    simulator.update_dynamics(dt=dt)

    if i%skip==0:
        positions = simulator.dump_coordinates()
        time_pos = np.hstack((i/skip * np.ones((N_part,1)), positions))

        #data = np.vstack((data, time_pos))
        data[int(i/skip)*len(time_pos):(int(i/skip)+1)*len(time_pos),:] = time_pos
        data_points.append(positions)

        fs.append(simulator.f)

import numpy as np
tids = np.tile(np.arange(N_part), Nt)
tracks = np.empty((N_part*Nt, 4))
tracks[:,0] = tids
tracks[:,1:] = data

tracks = sorted(tracks, key=lambda l: l[0])


points = np.ones((2,2)) * simulator.L
points[0,1] -= simulator.L/15
points[1,1] += simulator.L/15


viewer = napari.Viewer()
viewer.add_tracks(tracks)
viewer.add_points(points, size=3)

vectors = np.zeros((len(fs),N_part, 2,d+1))
for i, f in enumerate(fs):
    vectors[i,:,0,1:] = data_points[i]
    vectors[i,:,0,0] = i
    vectors[i,:,1,1:] = f

vectors = np.concatenate(vectors, axis=0)


viewer.add_vectors(vectors, edge_width=0.1)
napari.run()