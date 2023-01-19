import numpy as np
from tqdm import tqdm
from organo_simulator.simulator_yalla_force import FastOverdampedSimulator
from organo_simulator.renderer import Renderer
from organo_simulator.utils import make_bounding_box
import napari


np.random.seed(2022)

d=2 # dimension of simulation
N_part = 7 # number of particles
average_nuclei_size=8 # in physical units

skip=100
total_steps = 2000 # total number of simulation steps
Nt = int(total_steps/skip)

render = False # wether or not to add render (takes a looong time)
Nx = 300 # length of the box in pixels for the rendering 
n_rays = 32 # number of stardist rays used for rendering


# normal distribution of size, clipped
# nuclei_sizes = np.clip(
#                     np.random.normal(
#                         loc=average_nuclei_size,
#                         scale=0.1*average_nuclei_size,
#                         size=N_part
#                     ),
#                     0.9*average_nuclei_size,
#                     1.1*average_nuclei_size
#                 )
nuclei_sizes = average_nuclei_size

N_fast = int(0.05*N_part)
# persistence times (in physical units) for the OU process
persistence_times   = np.array([100 ]*(N_part-N_fast) + [100]*N_fast, dtype=float)
viscosities         = np.array([1000]*(N_part-N_fast) + [1000]*N_fast, dtype=float)
#Ds                  = np.array([0.01]*(N_part-N_fast) + [0.01]*N_fast, dtype=float) # diffusion coefficients
Ds                  = np.array([0]*(N_part-N_fast) + [0]*N_fast, dtype=float)


simulator = FastOverdampedSimulator(
    L=None, # will be initialized automatically
    Nx=Nx,
    d=d,
    N_part=N_part,
    nuclei_sizes=nuclei_sizes,
    viscosity=viscosities,
    D=Ds,
    persistence_time=persistence_times,
    energy_potential=1,
    max_distance_factor=2/0.8,#/0.71, # times nuclei size
    wiggle_room_factor=0.0, # times nuclei size
    parallel=True
)


# Initializing arrays to collect data
data = np.empty((N_part*Nt,d+1))
data_points=np.empty((Nt,N_part,d))


for i in tqdm(range(total_steps)):
    simulator.update_dynamics(dt=0.1)

    if i%skip==0:
        positions = simulator.dump_coordinates()
        time_pos = np.hstack((i/skip * np.ones((N_part,1)), positions))

        #data = np.vstack((data, time_pos))
        data[int(i/skip)*len(time_pos):(int(i/skip)+1)*len(time_pos),:] = time_pos
        data_points[int(i/skip)] = positions

        # if i>3*skip:

        #     path2save = '/home/jvanaret/data/data_trackability_study/simulations/test_long/coords'
        #     savename = f'positions_{str(i).zfill(int(np.log10(total_steps))+1)}.csv'

        #     np.savetxt(
        #         f'{path2save}/{savename}',
        #         positions,
        #         delimiter=','
        #     )

from scipy.spatial import KDTree
import matplotlib.pyplot as plt

tree = KDTree(data_points[0])
dists = tree.query(data_points[0],k=2)[0][:,1]
plt.hist(dists,bins=50,alpha=0.5)

tree = KDTree(data_points[-1])
dists = tree.query(data_points[-1],k=2)[0][:,1]
plt.hist(dists,bins=50,alpha=0.5)
plt.show()


# Show fast and slow cells in different colors
data_slow = np.zeros(
    shape=((N_part-N_fast)*(int(max(data[:,0]))+1),d+1)
)
data_fast = np.zeros(
    shape=((N_fast)*(int(max(data[:,0]))+1),d+1)
)
for t in tqdm(range(int(max(data[:,0]))+1)):
    data_slow[t*(N_part-N_fast):(t+1)*(N_part-N_fast),:] = \
        data[t*N_part:t*N_part + N_part-N_fast,:]
    
    data_fast[t*N_fast:(t+1)*N_fast,:] = \
        data[t*N_part + N_part-N_fast:(t+1)*N_part,:]


viewer = napari.Viewer(ndisplay=d)


viewer.add_points(data=data,      size=3, scale=(1,)+(Nx/(2*simulator.L),)*d, visible=False)

viewer.add_points(data=data_slow, size=4,face_color='blue', scale=(1,)+(Nx/(2*simulator.L),)*d, visible=not render)
viewer.add_points(data=data_fast, size=4,face_color='red', scale=(1,)+(Nx/(2*simulator.L),)*d, visible=not render)


bb = make_bounding_box(bb_shape=(Nx,)*d)
viewer.add_image(bb, blending='additive',opacity=0.25)


viewer.reset_view()
napari.run()
