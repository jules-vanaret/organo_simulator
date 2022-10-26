import stardist
import numpy as np
from packajules.utils import simulate_ellipsis_labels, make_bounding_box
import napari
from skimage.measure import regionprops
from scipy.ndimage import gaussian_filter
from scipy.spatial  import Delaunay, KDTree, Voronoi
from skimage.draw import polygon2mask

# rays = stardist.Rays_GoldenSpiral(n=3)
np.random.seed(2022)


points = np.load('/home/jvanaret/Desktop/points.npy')*2
#points = 2*(50+100*np.random.rand(10,2))
foo = 400
points = np.vstack((
    points,
    np.array([[0,0],[0,foo],[foo,0],[foo,foo]])
))

typical_size=14
n_rays=64


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



vor = Voronoi(points)

target_surface = np.pi*(typical_size)**2

distances = []

vors = np.zeros(shape=(foo,foo), dtype=int)

for i,point in enumerate(points_int):
    indices = vor.regions[vor.point_region[i]]

    if -1 in indices or len(indices)==0: # some regions can be opened
        rays = np.ones(n_rays)*typical_size/2
    else:
        vertices = vor.vertices[indices]

        vor_cell = polygon2mask(image_shape=(vors.shape), polygon=vertices)
        vors = vors + vor_cell*(i+1)
        dist_sd = stardist.geometry.geom2d.star_dist(vor_cell, n_rays=n_rays)
        
        dist_sd = dist_sd[tuple(point)]
        dist = dist_sd.copy()

        dists_probs = dist[dist<typical_size]
        dists_ok = dist[dist>=typical_size]

        if len(dists_probs)==0:
            rays = np.ones(n_rays)*typical_size
        elif len(dists_probs)==n_rays:
            rays=dist
        else:

            num = (2*target_surface/tan_angle_stardist) - np.sum(dists_probs**2)
            denom = (len(dists_ok)*typical_size**2)
            factor = np.sqrt(
                num/denom
            )
        
            dist[dist>=typical_size] = factor*typical_size#dist[dist>=typical_size]#* factor
            dist = np.clip(dist, 0, dist_sd)
            rays = dist



    distances.append(rays)

distances = np.array(distances)

# dist = np.array([np.arange(1+5,5+n_rays+1)])
# points_int = np.array([[200,200]])
# points = points_int

print('target', target_surface)
for elem in distances:
    print(np.sum(elem**2)*tan_angle_stardist/2)

lbls = stardist.geometry.geom2d.polygons_to_label(distances, points_int, shape=(foo,foo))

viewer= napari.Viewer()
viewer.add_labels(lbls)


viewer.add_points(points,size=3)
bb = make_bounding_box(bb_shape=(foo,foo))
viewer.add_image(bb, blending='additive')
viewer.add_labels(vors.astype(int))
napari.run()