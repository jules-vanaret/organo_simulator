import numba
import numpy as np
from scipy.spatial import Voronoi
from scipy.spatial  import Delaunay
from matplotlib.patches import Ellipse

np.random.seed(2022)

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
    return d*values, vectors




points = np.random.rand(30,2)

voronoi = Voronoi(points)
delaunay = Delaunay(points)

import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)



#voronoi_plot_2d(voronoi,ax=ax)

# for i,point in enumerate(points):

#     regions = voronoi.regions[
#                 voronoi.point_region[i]
#             ]

#     if (not (-1 in regions)) and (len(regions)>3) : 

#         vertices = voronoi.vertices[
#                         regions
#                     ]

#         dims, vecs = fast_pca(vertices, point, 2)

#         angle = np.arctan(vecs[0][0]/vecs[0][1]) * 180/np.pi
        
#         ellipse = Ellipse(
#             xy=point,width=dims[1], height=dims[0],angle=angle 
#         )
#         # print(fast_pca(vertices, 2)[0])
#         ax.add_patch(ellipse)

for i,point in enumerate(points):

    indptr, indices = delaunay.vertex_neighbor_vertices
    inds_neighbors = indices[indptr[i]:indptr[i+1]]
    vertices = points[inds_neighbors]



    dims, vecs = fast_pca(vertices, point, 2)

    angle = np.arctan(vecs[0][0]/vecs[0][1]) * 180/np.pi
    
    ellipse = Ellipse(
        xy=point,width=dims[1], height=dims[0],angle=angle,color='r'
    )
    # print(fast_pca(vertices, 2)[0])
    ax.add_patch(ellipse)

ax.set_xlim(0,1)
ax.set_ylim(0,1)

plt.triplot(points[:,0], points[:,1], delaunay.simplices)
plt.scatter(points[:,0], points[:,1],c='k')
plt.show()

