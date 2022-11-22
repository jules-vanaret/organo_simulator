import numpy as np
import tifffile
from packajules.segmentation_classes import Tracks

"""
CONCLUSION: Field 'A' from track_objects is the label index when labels are fed to utrack3D !
"""


make_input = False
read_track = True

path2arrays = '/home/jvanaret/data/data_trackability_study/simulations/test_delme'

if make_input:
    for i_frame in range(1, 5):
        arr = np.zeros(shape=(100,100,100), dtype='uint16')

        arr[[10+4*i_frame],[10,40],[10,10]] = np.array([127+i_frame, 140+i_frame*10])

        tifffile.imwrite(
            f'{path2arrays}/frame_{i_frame}.tif',
            arr
        )

if read_track:
    path = '/home/jvanaret/data/data_trackability_study/utrack/simulations/test_delme/delme'
    utracks = Tracks()
    property_trackability = utracks.load_utrack3d_json(
        path_to_json=f'{path}/tracks.json',
        path_to_trackability_json=f'{path}/trackability.json'
    )