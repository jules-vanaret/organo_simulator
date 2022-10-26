import stardist
import numpy as np
from packajules.utils import simulate_ellipsis_labels, make_bounding_box
import napari
from skimage.measure import regionprops
from scipy.ndimage import gaussian_filter
from scipy.spatial  import Delaunay, KDTree

# rays = stardist.Rays_GoldenSpiral(n=3)
np.random.seed(2022)


#points = np.load('/home/jvanaret/Desktop/points.npy')*2
points = 2*(50+100*np.random.rand(10,2))

typical_size=10
n_rays=16


N_points = len(points)

points_int = points.astype(int)

# viewer.add_labels(labels)


# dist = stardist.geometry.geom2d.star_dist(labels,n_rays=10)

# points = np.array(tuple(np.array(r.centroid).astype(int) for r in regionprops(lbl)))



# dist = np.random.randint(4,10,size=(N_points,32))
# dist = np.array([gaussian_filter(elem, sigma=2,mode='wrap') for elem in dist])

distance_vectors = points[:,None] - points
angles = np.pi + np.arctan2(distance_vectors[:,:,0], distance_vectors[:,:,1])
angles_stardist = np.linspace(0,2*np.pi,n_rays,endpoint=False)
angle_stardist = angles_stardist[1]
tan_angle_stardist = np.abs(np.tan(angle_stardist))
inds_inf_angle = (angles // angle_stardist).astype(int)



delaunay = Delaunay(points)
tree = KDTree(points)
dist_matrix = tree.sparse_distance_matrix(
                        tree,
                        max_distance=1e9
                    ).toarray()

target_surface = np.pi*(typical_size)**2

dist = []

for i,point in enumerate(points):
    indptr, indices = delaunay.vertex_neighbor_vertices
    inds_neighbors = indices[indptr[i]:indptr[i+1]]

    dists_neighbors = dist_matrix[i, inds_neighbors]
    inds_neighbors_rays = inds_inf_angle[i, inds_neighbors]
    
    inds_problem_rays = inds_neighbors_rays[dists_neighbors<typical_size*2]

    if len(inds_problem_rays)==0:
        rays = np.ones(n_rays)*typical_size
    else:
        dist_problem_rays = dists_neighbors[dists_neighbors<typical_size*2]

        rays = np.zeros(n_rays)
        rays[inds_problem_rays] = dist_problem_rays/2
        wrap = [int(elem%n_rays) for elem in inds_problem_rays]
        rays[wrap] = dist_problem_rays/2

        num_terms = np.sum(rays==0)

        rays[rays==0] = np.sqrt((2*target_surface/tan_angle_stardist - np.sum(rays**2))/num_terms)

    dist.append(rays)

dist = np.array(dist)

# dist = np.array([np.arange(1+5,5+n_rays+1)])
# points_int = np.array([[200,200]])
# points = points_int

print('target', target_surface)
for elem in dist:
    print(np.sum(elem**2)*tan_angle_stardist)

lbls = stardist.geometry.geom2d.polygons_to_label(dist, points_int, shape=(400,400))

viewer= napari.Viewer()
viewer.add_labels(lbls)
from pyclesperanto_prototype import voronoi_labeling
point_labels = np.zeros((400,)*2, dtype=bool)
coords = tuple((points).T.astype(int))
point_labels[coords] = True

raw_labels = np.array(voronoi_labeling(point_labels))

viewer.add_points(points,size=3)
bb = make_bounding_box(bb_shape=(400,400))
viewer.add_image(bb, blending='additive')
viewer.add_labels(raw_labels)
napari.run()