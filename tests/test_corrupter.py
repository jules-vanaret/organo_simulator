import numpy as np
import organo_simulator.utils as simulator_utils
from organo_simulator.corrupter import SimulationCorrupter

np.random.seed(2022)

d=3
N_part = 30


radiuses = 30 * np.power(np.random.uniform(0,1,size=(N_part,1)),1/d)
if d==2:
    coords = radiuses * simulator_utils.random_2d_unit_vectors(N_part)
elif d==3:
    coords = radiuses * simulator_utils.random_3d_unit_vectors(N_part)

corrupter = SimulationCorrupter()
coords_fp, dict_fp = corrupter.add_fp_to_coords(coords, fp_rate=0.2, return_dict=True)
coords_fn, dict_fn = corrupter.remove_fn_from_coords(coords, fn_rate=0.2, return_dict=True)
coords_merge, dict_merge = corrupter.add_merge_to_coords(coords, merge_rate=0.2, return_dict=True, max_distance=10)
coords_split, dict_split = corrupter.add_split_to_coords(coords, split_rate=0.2, return_dict=True, nuclei_sizes=5)


import napari
viewer = napari.Viewer(ndisplay=d)
viewer.add_points(coords, size=4, face_color='green')

viewer.add_points(coords_fp,size=3, face_color='red')
viewer.add_points(coords_fn,size=3, face_color='red')
viewer.add_points(coords_merge,size=3, face_color='red')
viewer.add_points(coords_split,size=3, face_color='red')


napari.run()
