import numpy as np
import freud
from scipy.spatial import KDTree
from tqdm import tqdm
# from organo_simulator.simulator_LJ_force import FastOverdampedSimulator
from organo_simulator.simulator_LJ_force_periodic import FastOverdampedSimulator
from lab_utils.utils import sort_track_array_by_id_then_time
import napari
from collections import defaultdict
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt

np.random.seed(1)


dt = 1
d = 2 # dimension of simulation
N_part = 100 # number of particles
# nuclei_sizes = 0.7
# nuclei_sizes = np.random.uniform(0.3,1,N_part)
nuclei_sizes = np.random.uniform(0.5,1,N_part)


# persistence times (in physical units) for the OU process
# persistence_times = np.array([300]*(N_part))
persistence_times = np.random.randint(10, 2000, N_part)
viscosities = np.array([1]*(N_part))
Ds = np.array([1e-5]*(N_part)) # diffusion coefficients
# Ds = np.array([0]*(N_part)) # diffusion coefficients

Nx = 1000 # length of the box in pixels for the rendering 
L=10

load = True

if load:
    tracks = np.load('/home/jvanaret/data/tracks.npy')
    Nt = len(np.unique(tracks[:,1]))+1
    
else:
    simulator = FastOverdampedSimulator(
        L=L, # will be initializesd automatically
        Nx=Nx, # length of the box in pixels for the rendering 
        d=d, # dimensions (2 or 3)
        N_part=N_part, # number of particles
        nuclei_sizes=nuclei_sizes, # radius of particles
        viscosity=viscosities, # viscosities of particles
        D=Ds, # diffusion coefficients
        persistence_time=persistence_times, # how long the temporal autocorrelation of random movement goes
        energy_potential=1e-6,
        parallel=True
    )

    skip_time = 100 # return coordinates every ... s
    skip = int(skip_time/dt)
    total_time = 10000 # total number of simulation steps
    total_steps = int(total_time/dt) 
    Nt = int(total_steps/skip)

    # Initializing arrays to collect data
    data = np.empty((N_part*(Nt-1),d+1))
    # data_points=[]

    for i in tqdm(range(total_steps)):
        simulator.update_dynamics(dt=dt)

        if i%skip==0 and i > 0:
            positions = simulator.dump_coordinates()
            time_pos = np.hstack((i/skip * np.ones((N_part,1)), positions))

            data[int(i/skip-1)*len(time_pos):(int(i/skip-1)+1)*len(time_pos),:] = time_pos
            # data_points.append(positions)


    tids = np.tile(np.arange(N_part), Nt-1)
    tracks = np.empty((N_part*(Nt-1), 2+d))
    tracks[:,0] = tids
    tracks[:,1:] = data

    # otherwise napari screws up the colors
    # tracks = sorted(tracks, key=lambda l: l[0])
    tracks = sort_track_array_by_id_then_time(tracks)

    np.save('/home/jvanaret/data/tracks.npy', tracks)

# viewer = napari.Viewer()
# tracks_layer = viewer.add_tracks(tracks)
# indices_tp = np.where(
#     np.max(
#         np.abs(np.diff(tracks_layer.data[:,2:], axis=0)),
#         axis=1
#     ) > L/2
# )
# tracks_layer._manager._track_connex[indices_tp] = False
# tracks_layer.events.rebuild_tracks()

# points =  tracks_layer.data[:,1:]
# sizes = np.repeat(nuclei_sizes, Nt-1, axis=0)
# face_colors_at_t = plt.cm.viridis(nuclei_sizes/np.max(nuclei_sizes))
# face_colors = np.repeat(face_colors_at_t, Nt-1, axis=0)

# viewer.add_points(points, size=sizes, face_color=face_colors, edge_color='black', name='particles')

# napari.run()



list_t = np.unique(tracks[:,1])

tracks_by_time = [tracks[tracks[:,1]==t] for t in list_t]

def ddr(tracks_by_time, L):
    displacement_distnn_ratios = []

    for i in range(len(tracks_by_time)-1):
        t1 = tracks_by_time[i]
        t2 = tracks_by_time[i+1]

        kdt = KDTree(t1[:,2:], boxsize=L)
        nn_dists = kdt.query(t1[:,2:], k=2)[0][:,1]

        tracks_ids_t_1 = t1[:,0].astype(int)
        tracks_ids_t = t2[:,0].astype(int)

        matched_ids = np.intersect1d(tracks_ids_t_1, tracks_ids_t)

        if len(matched_ids) == 0:
            continue

        for id_ in matched_ids:
            pos_t_1 = t1[t1[:,0]==id_,2:]
            pos_t = t2[t2[:,0]==id_,2:]

            dist_vector_ij = pos_t - pos_t_1
            
            dist = np.linalg.norm(
                (dist_vector_ij + L/2) % L - L/2 
            )

            nn_dist = nn_dists[t1[:,0]==id_]

            displacement_distnn_ratios.append(dist/nn_dist)

    displacement_distnn_ratios = np.array(displacement_distnn_ratios)

    return displacement_distnn_ratios



def contact_times(tracks_by_time, L):
    contact_dicts = []
    d = tracks_by_time[0].shape[1] - 2

    for i in tqdm(range(len(tracks_by_time))):
        tracks_at_t = tracks_by_time[i]
        index_to_tid_dict = {index: tid for index, tid in enumerate(tracks_at_t[:,0])}


        box = freud.box.Box.cube(L) if d == 3 else freud.box.Box.square(L)
        voronoi = freud.locality.Voronoi()
        positions = tracks_at_t[:,2:]
        #if d==2 then we need to add a z coordinate at 0
        if d == 2:
            positions = np.hstack((positions, np.zeros((positions.shape[0], 1))))
        voronoi.compute((box, positions))

        nlist = voronoi.nlist

        contact_dict = defaultdict(list)
        for i, j in nlist:
            tid_i, tid_j = index_to_tid_dict[i], index_to_tid_dict[j]
            if tid_i < tid_j: # avoid double counting
                contact_dict[tid_i].append(tid_j)

        contact_dicts.append(contact_dict)

    # count contact times
    contact_times = defaultdict(int)
    for i in tqdm(range(len(contact_dicts)-1)):
        for tid, contacts in contact_dicts[i].items():
            for contact in contacts:
                if contact in contact_dicts[i+1][tid]:
                    contact_times[(tid, contact)] += 1

    contact_times = np.array(list(contact_times.values()))
    return contact_times
    





c_times = contact_times(tracks_by_time, L)
t_otsu = threshold_otsu(c_times)

plt.hist(c_times, bins=30)
plt.axvline(t_otsu, color='r')


ddrs = ddr(tracks_by_time, L)
t_otsu = threshold_otsu(ddrs)

plt.figure()
plt.hist(ddrs, bins=30)
plt.axvline(t_otsu, color='r')

plt.show()

        


    

