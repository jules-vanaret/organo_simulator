import numpy as np
import napari



path_to_save = '/data1/data_teflon/laura_simu'

viewer = napari.Viewer()

for index_simu in range(10):

    tracks = np.load(f'{path_to_save}/tracks_{index_simu}.npy')

    tracks_layer = viewer.add_tracks(
        tracks
    )

    tracks_layer.tail_width = 3
    tracks_layer.tail_length = 50


napari.run()