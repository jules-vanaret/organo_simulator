import tifffile
import numpy as np
import napari
import os

from packajules.segmentation_classes import Tracks
from packajules.utils import folder_tifs_to_single_tif
from organo_simulator.utils import load_csv_coords
from skimage.measure import regionprops


scale = (1,)*4 # min/frame, um/pix, um/pix, um/pix

path2coords = '/home/jvanaret/data/data_trackability_study/simulations/test_long/coords'

files = sorted(os.listdir(path2coords))


gt_tracks = []

for indt,filename in enumerate(files):
    coords = load_csv_coords(
        path_to_csv=f'{path2coords}/{filename}'
    )

    for ind_track, coord in enumerate(coords, start=1):


        rescaled_coords = [c*300/(2*130) for c in coord]

        gt_tracks.append(
            [int(ind_track),int(indt), *rescaled_coords]
        )

gt_tracks = sorted(gt_tracks, key=lambda l: l[0])

# 3D Render
viewer = napari.Viewer() 

viewer.add_tracks(
    np.array(gt_tracks),
    name='gt_tracks',
    color_by='track_id',
    properties={}
)


path2data = '/home/jvanaret/data/data_trackability_study/simulations/test_long/'

merge_rates_str = [str(elem).zfill(3) for elem in np.arange(0,50,5)]

for merge_rate in merge_rates_str:

    name = f'coords_merge_{merge_rate}'

    # labels = folder_tifs_to_single_tif(
    #     f'{path2data}/{name}'
    # )

    # coords = [prop.centroid for prop in regionprops(labels[0])]

    # coords = np.array(coords)


    utracks = Tracks()
    property_trackability = utracks.load_utrack3d_json(
        path_to_json=f'/home/jvanaret/data/data_trackability_study/utrack/simulations/test_long/{name}/tracks.json',
        path_to_trackability_json=f'/home/jvanaret/data/data_trackability_study/utrack/simulations/test_long/{name}/trackability.json',
        scale=scale, # min/frame, um/pix, um/pix, um/pix
    )

    ppties = {
        'trackability': property_trackability
    }

    utracks_napari = utracks.dump_tracks_to_napari()

    viewer.add_tracks(
        utracks_napari,
        name=f'tracks_{name}',
        properties=ppties
    )
# path2data = '/home/jvanaret/data/data_trackability_study/simulations/test_long/fake_array_merge_005'


# labels = folder_tifs_to_single_tif(
#     path2data
# )


# utracks = Tracks()
# property_trackability = utracks.load_utrack3d_json(
#     path_to_json='/home/jvanaret/data/data_trackability_study/utrack/simulations/test_long/fake_array_merge_005/tracks.json',
#     path_to_trackability_json='/home/jvanaret/data/data_trackability_study/utrack/simulations/test_long/fake_array_merge_005/trackability.json',
#     scale=scale, # min/frame, um/pix, um/pix, um/pix
# )

# ppties2 = {
#     'trackability': property_trackability
# }


# utracks_napari2 = utracks.dump_tracks_to_napari()







# viewer.add_tracks(
#     utracks_napari2,
#     name='tracks2',
#     properties=ppties2
# )

# viewer.add_labels(
#     labels,
#     name='labels',
#     opacity=1
# )

viewer.dims.ndisplay = 3
viewer.grid.shape = (1,len(viewer.layers))

for layer in viewer.layers: layer.scale = scale


napari.run()