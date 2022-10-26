import numpy as np
from tqdm import tqdm
from organo_simulator.simulator import FastOverdampedSimulator
from organo_simulator.renderer import Renderer
from organo_simulator.utils import make_bounding_box
import napari


np.random.seed(2022)

d=3 # dimension of simulation
N_part = 200 # number of particles
average_nuclei_size=8 # in physical units

# normal distribution of size, clipped
nuclei_sizes = np.clip(
                    np.random.normal(
                        loc=average_nuclei_size,
                        scale=0.1*average_nuclei_size,
                        size=N_part
                    ),
                    0.8*average_nuclei_size,
                    1.2*average_nuclei_size
                )
# number of "fast" cells 
# (usually bigger persistence times 
# and/or lower viscosities
# and/or bigger diffusion coefficients)
N_fast = int(0.1*N_part)
# persistence times (in physical units) for the OU process
persistence_times = np.array([10]*(N_part-N_fast) + [100]*N_fast)
viscosities = np.array([1000]*(N_part-N_fast) + [1000]*N_fast)
Ds = np.array([1.0]*(N_part-N_fast) + [1.0]*N_fast) # diffusion coefficients


render = False # wether or not to add render (takes a looong time)
Nx = 200 # length of the box in pixels for the rendering 
n_rays=32 # number of stardist rays used for rendering



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
    max_distance_factor=2/0.7, # times nuclei size
    wiggle_room_factor=0.2, # times nuclei size
    parallel=True
)


skip=100
total_steps = 10000 # total number of simulation steps
Nt = int(total_steps/skip)


# Initializing arrays to collect data
data = np.empty((N_part*Nt,d+1))
data_points=[]


for i in tqdm(range(total_steps)):
    simulator.update_dynamics(dt=1)

    if i%skip==0:
        positions = simulator.dump_coordinates()
        time_pos = np.hstack((i/skip * np.ones((N_part,1)), positions))

        #data = np.vstack((data, time_pos))
        data[int(i/skip)*len(time_pos):(int(i/skip)+1)*len(time_pos),:] = time_pos
        data_points.append(positions)



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



if render:

    labels=np.empty(shape=((Nt,)+(Nx,)*d), dtype=int)
    voronoi_labels = np.empty(shape=((Nt,)+(Nx,)*d), dtype=int)
    # data = np.empty(shape=((Nt,)+(Nx,)*d), dtype=float)

    renderer = Renderer(
        nuclei_sizes=nuclei_sizes,
        N_part=N_part,
        n_rays=n_rays,
        Nx=Nx,
        d=d,
        L=simulator.L,
        gaussian_blur_sigma=1,
        gaussian_noise_mean=0.04,
        gaussian_noise_sigma=0.04
    )

    render_data = np.empty(shape=((Nt,)+(Nx,)*d), dtype=float)

    for ind_t,points in enumerate(tqdm(data_points)):

        labels_t = renderer.make_labels_from_points(points=points)

        labels[ind_t] = labels_t
        render_data[ind_t] = renderer.make_realistic_data_from_labels(labels_t)









viewer = napari.Viewer(ndisplay=d)


viewer.add_points(data=data,      size=3, scale=(1,)+(Nx/(2*simulator.L),)*d)

viewer.add_points(data=data_slow, size=4,face_color='blue', scale=(1,)+(Nx/(2*simulator.L),)*d)
viewer.add_points(data=data_fast, size=4,face_color='red', scale=(1,)+(Nx/(2*simulator.L),)*d)

if render:
    viewer.add_labels(labels)
    viewer.add_image(render_data)
    viewer.add_labels(voronoi_labels, visible=False)


bb = make_bounding_box(bb_shape=(Nx,)*d)
viewer.add_image(bb, blending='additive',opacity=0.25)


viewer.reset_view()
napari.run()
