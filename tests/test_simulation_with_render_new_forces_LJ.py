import numpy as np
from tqdm import tqdm
from organo_simulator.simulator_LJ_force import FastOverdampedSimulator
from organo_simulator.renderer import Renderer
from organo_simulator.utils import make_bounding_box
from organo_simulator.simulator_LJ_force import LJ_force_numba
import napari
from scipy.spatial import KDTree
import matplotlib.pyplot as plt



np.random.seed(2022)

d=3 # dimension of simulation
N_part = 1000 # number of particles
average_nuclei_size=8 # in physical units

skip=100
dt = 0.1
total_steps = 10000 # total number of simulation steps
Nt = int(total_steps/skip)

render = False # wether or not to add render (takes a looong time)
Nx = 300 # length of the box in pixels for the rendering 


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

N_fast = int(0.0*N_part)
# persistence times (in physical units) for the OU process
persistence_times   = np.array([10]*(N_part-N_fast) + [10]*N_fast, dtype=float)
viscosities         = np.array([1.0]*(N_part-N_fast) + [1.0]*N_fast, dtype=float)
# Ds                  = np.array([0.01]*(N_part-N_fast) + [0.01]*N_fast, dtype=float) # diffusion coefficients
Ds                  = np.array([0.001]*(N_part-N_fast) + [0.001]*N_fast, dtype=float) # diffusion coefficients


simulator = FastOverdampedSimulator(
    L=None, # will be initialized automatically
    Nx=Nx,
    d=d,
    N_part=N_part,
    nuclei_sizes=nuclei_sizes,
    viscosity=viscosities,
    D=Ds,
    persistence_time=persistence_times,
    energy_potential=0.001,
    parallel=True
)

# viewer = napari.Viewer()
# viewer.add_points(simulator.positions, size=5)

# napari.run()


# simulator.positions = np.zeros((N_part, d))
# simulator.positions[0,0] = -10.0
# simulator.positions[1,0] = 10.0
# simulator.positions[2,0] = 0
# simulator.positions[2,1] = 0.1
# simulator.positions = simulator.positions + simulator.L

# Initializing arrays to collect data
data = np.empty((N_part*Nt,d+1))
data_points=np.empty((Nt,N_part,d))

drag_norm = []
det_norm = []
rand_norm = []

drag_mean = []
det_mean = []
rand_mean = []

dists_list = []


for i in tqdm(range(total_steps)):

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

    simulator.update_dynamics(dt=dt)

    # drag_velocities, determisitic_velocities, random_velocities = simulator.dump_velocities()

    # drag_norm.append(np.linalg.norm(drag_velocities, axis=1).mean())
    # det_norm.append(np.linalg.norm(determisitic_velocities, axis=1).mean())
    # rand_norm.append(np.linalg.norm(random_velocities, axis=1).mean())

    # drag_mean.append(np.linalg.norm(drag_velocities.mean(axis=0)))
    # det_mean.append(np.linalg.norm(determisitic_velocities.mean(axis=0)))
    # rand_mean.append(np.linalg.norm(random_velocities.mean(axis=0)))

    # tree = KDTree(positions)
    # dists = tree.query(positions,k=3)[0][:,2]

    # dists_list.append(dists)





# tree = KDTree(data_points[0])
# dists = tree.query(data_points[0],k=2)[0][:,1]
# plt.hist(dists,bins=50,alpha=0.5)

# tree = KDTree(data_points[-1])
# dists = tree.query(data_points[-1],k=2)[0][:,1]
# plt.hist(dists,bins=50,alpha=0.5)

# plt.figure()
# plt.title('Norm')
# plt.plot(drag_norm, 'o')
# plt.plot(det_norm, 'o')
# plt.plot(rand_norm, 'o')

# plt.figure()
# plt.title('Mean')
# plt.plot(drag_mean, 'o')
# plt.plot(det_mean, 'o')
# plt.plot(rand_mean, 'o')

# # plt.figure()
# # plt.plot([elem[0] for elem in dists_list])
# # plt.plot([elem[1] for elem in dists_list])
# # plt.plot([elem[2] for elem in dists_list])


# plt.show()




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

    
