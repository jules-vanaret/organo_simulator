import numpy as np
from tqdm import tqdm
from organo_simulator.simulator import FastOverdampedSimulator
from packajules.utils import make_bounding_box
import napari
from pyclesperanto_prototype import voronoi_labeling
from skimage.measure import regionprops
from skimage.morphology import binary_erosion
from scipy.spatial import Voronoi
from scipy.spatial  import Delaunay
from scipy.ndimage.morphology import distance_transform_edt
from scipy.spatial import KDTree as scipy_KDTree
import numba

np.random.seed(2022)

N_part = 200
d=2
L = 100 
Nx = 200


voronoi = False
delaunay = False
sphere = False

N_fast = int(0.5*N_part)
persistence_times = np.array([10]*(N_part-N_fast) + [500]*N_fast)
viscosities = np.array([1000]*(N_part-N_fast) + [500]*N_fast)
Ds = np.array([1]*(N_part-N_fast) + [1]*N_fast)

simulator = FastOverdampedSimulator(
    L=L,
    Nx=Nx,
    d=d,
    N_part=N_part,
    nuclei_size=5,
    viscosity=1000,
    D=Ds,
    persistence_time=persistence_times,
    energy_potential=10,
    max_distance_factor=2/0.7,
    wiggle_room_factor=0.1,
    clip_force=None,
    debug_dist=False
)

data = np.empty((0, d+1))
data_vor = []

skip=100
total_steps = 500

Nt = int(total_steps/skip)

data = np.empty((N_part*Nt,d+1))

data_delaunay=[]


for i in tqdm(range(total_steps)):
    positions = simulator.update_dynamics(dt=1)

    if i%skip==0:
        time_pos = np.hstack((i/skip * np.ones((N_part,1)), positions))

        #data = np.vstack((data, time_pos))
        data[int(i/skip)*len(time_pos):(int(i/skip)+1)*len(time_pos),:] = time_pos
        data_delaunay.append(positions)


if delaunay:

    data_shape = []

    @numba.njit
    def np_apply_along_axis(func1d, axis, arr):
        assert arr.ndim == 2
        assert axis in [0, 1]
        if axis == 0:
            result = np.empty(arr.shape[1])
            for i in range(len(result)):
                result[i] = func1d(arr[:, i])
        else:
            result = np.empty(arr.shape[0])
            for i in range(len(result)):
                result[i] = func1d(arr[i, :])
        return result

    @numba.njit
    def np_mean(array, axis):
        return np_apply_along_axis(np.mean, axis, array)

    @numba.jit(nopython=True, nogil=True)
    def fast_pca(positions, point, d):
        # calculate the mean of each column
        # M = np_mean(positions, axis=0)
        # # center columns by subtracting column means
        # C = positions - M

        C = positions - point

        # calculate covariance matrix of centered matrix
        V = np.cov(C.T)
        # eigendecomposition of covariance matrix
        values, vectors = np.linalg.eig(V)
        return d*values/20, vectors

    for j,points in enumerate(data_delaunay):


        delaunay = Delaunay(points)

    # import matplotlib.pyplot as plt
    # fig = plt.figure()
    # ax = fig.add_subplot(111)

        for i,point in enumerate(points):

            indptr, indices = delaunay.vertex_neighbor_vertices
            inds_neighbors = indices[indptr[i]:indptr[i+1]]
            vertices = points[inds_neighbors]



            dims, vecs = fast_pca(vertices, point, 2)

            angle = np.arctan(vecs[0][0]/vecs[0][1]) 
            
            # ellipse = Ellipse(
            #     xy=point,width=dims[1], height=dims[0],angle=angle,color='r'
            # )
            corners = np.array([
                [-dims[1],-dims[0]],
                [-dims[1],dims[0]],
                [dims[1],dims[0]],
                [dims[1],-dims[0]]
            ])

            rotation_matrix = np.array([
                [np.cos(angle), np.sin(angle)],
                [np.sin(angle), np.cos(angle)]
            ])

            corners = (rotation_matrix @ corners.T).T + point
            dat = np.hstack((np.ones((4,1))*j, corners))

            data_shape.append(dat)

if voronoi:

    labels=[]

    for elem in tqdm(data_vor):
        point_labels = np.zeros((Nx,)*d, dtype=bool)
        coords = tuple((elem/(2*L)*Nx).T.astype(int))
        point_labels[coords] = True

        raw_labels = np.array(voronoi_labeling(point_labels))

        uniques = []

        unique1 = raw_labels[[0,-1]]
        unique2 = raw_labels[:,[0,-1]]
        uniques.append(unique1.ravel())
        uniques.append(unique2.ravel())

        if d==3:
            unique3 = raw_labels[:,:,[0,-1]]
            uniques.append(unique3.ravel())

        labs = np.unique(uniques)
        all_labs = np.unique(raw_labels)

        props = regionprops(raw_labels)

        new_labels = np.zeros_like(raw_labels)

        for prop in props:
            roi_slice = prop.slice
            roi_data = raw_labels[roi_slice]

            if prop.label in labs:
                raw_labels[roi_slice][roi_data==prop.label] = 0
            else:

                img = prop.image

                # for _ in range(4):
                #     img = binary_erosion(img)

                #new_labels[roi_slice][img] = prop.label

                dist_arr = np.zeros(shape=[elem+2 for elem in img.shape], dtype=int)
                dist_arr[(slice(1,-1),)*d] = img

                dist = distance_transform_edt(dist_arr)
                dist = dist[(slice(1,-1),)*d]
                new_labels[roi_slice][dist>3.1] =  prop.label


        labels.append(new_labels)

    labels = np.array(labels)

if sphere:
    data_shape = []
    for j,points in enumerate(data_delaunay):
        tree = scipy_KDTree(positions)
        for i,point in enumerate(points):
            radius = tree.query(point,2)[0][1]/2
            radius = np.min((radius, simulator.typical_size))

            corners = np.array([
                [-radius,-radius],
                [-radius,radius],
                [radius,radius],
                [radius,-radius]
            ])

            dat = np.hstack((np.ones((4,1))*j, corners+point))

            data_shape.append(dat)


viewer= napari.Viewer(ndisplay=d)
bb = make_bounding_box(bb_shape=(Nx,)*d)

data_slow = np.zeros(
    shape=((N_part-N_fast)*(int(max(data[:,0]))+1),d+1)
)
data_fast = np.zeros(
    shape=((N_fast)*(int(max(data[:,0]))+1),d+1)
)
for t in range(int(max(data[:,0]))+1):
    data_slow[t*(N_part-N_fast):(t+1)*(N_part-N_fast),:] = \
        data[t*N_part:t*N_part + N_part-N_fast,:]
    
    data_fast[t*N_fast:(t+1)*N_fast,:] = \
        data[t*N_part + N_part-N_fast:(t+1)*N_part,:]


viewer.add_points(data=data, size=3)
viewer.add_points(data=data_slow, size=2,face_color='blue')
viewer.add_points(data=data_fast, size=2,face_color='red')
#viewer.add_shapes(data_shape,shape_type='ellipse')
viewer.add_image(bb, blending='additive',opacity=0.25, scale=(2*L/Nx,)*d)

#if voronoi: viewer.add_labels(labels, scale=(2*L/Nx,)*d)
if delaunay or sphere: viewer.add_shapes(data_shape, shape_type='ellipse',scale=(2*L/Nx,)*d)

viewer.reset_view()
napari.run()


    