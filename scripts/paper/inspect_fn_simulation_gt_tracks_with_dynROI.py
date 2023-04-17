import tifffile
import numpy as np
import napari
import os
from skimage.measure import regionprops

from packajules.segmentation_classes import Tracks
from packajules.dynROI import DynROI
from packajules.render import Renderer
from packajules.utils import folder_tifs_to_single_tif

from organo_simulator.utils import load_csv_coords
import csv


scale = (1,)*4 # min/frame, um/pix, um/pix, um/pix

path2coords = '/home/jvanaret/data/data_trackability_study/simulations/test_long/coords'

files = sorted(os.listdir(path2coords))


gt_tracks = []

for indt,filename in enumerate(files):
    coords = load_csv_coords(
        path_to_csv=f'{path2coords}/{filename}'
    )

    for ind_track, coord in enumerate(coords, start=1):
        

        gt_tracks.append(
            [int(ind_track),int(indt), *coord]
        )

gt_tracks = sorted(gt_tracks, key=lambda l: l[0])

# 3D Render
# viewer = napari.Viewer() 

# viewer.add_tracks(
#     np.array(gt_tracks),
#     name='gt_tracks',
#     color_by='track_id',
#     properties={}
# )

total_time = len(np.unique(np.array(gt_tracks)[:,1]))
roi_fringes=(15,15,15)
list_of_center_inds = [150,150,150]

utracks = Tracks().load_napari_tracks(napari_tracks=gt_tracks)

dynROI = DynROI().from_coords(
    list_of_tinds=range(total_time),
    list_of_center_inds=list_of_center_inds,
    world_zyx_shape=(300,)*3,
    roi_fringes=roi_fringes
)


renderer = Renderer()


renderer.orthorender_dynROI(
    dynROI=dynROI,
    tracks=utracks,
    scale4d=scale
)

path2data = '/home/jvanaret/data/data_trackability_study/simulations/test_long/'

fn_rates_str = [str(elem).zfill(3) for elem in np.arange(5,10,5)]

for fn_rate in fn_rates_str:

    name = f'coords_fn_{fn_rate}'

    all_points = []
    all_facecolors = []
    path2coords = f'{path2data}/coords_fn_{fn_rate}'
    path2dicts = f'{path2data}/dicts_fn_{fn_rate}'

    files = sorted(os.listdir(path2coords))
    
    for ind_t,file in enumerate(files):
        coords = load_csv_coords(
            path_to_csv=f'{path2coords}/{file}'
        )
        raw_dict = csv.DictReader(
            open(f'{path2dicts}/{file.replace("positions", "dict_positions")}'),
            fieldnames=['new', 'old']
        )
        list_new_to_old = [[int(d['new']), d['old']] for d in raw_dict]



        for mapping in list_new_to_old:
            
            all_points.append([ind_t,*coords[mapping[0]]])

            try:
                _ = int(mapping[1])
                # all_facecolors.append('green')
                all_facecolors.append([0,1,0,1])
            except ValueError:
                # all_facecolors.append('red')
                all_facecolors.append([1,0,0,1])
                



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

    # viewer.add_tracks(
    #     utracks_napari,
    #     name=f'tracks_{name}',
    #     properties=ppties
    # )

    total_time = len(np.unique(np.array(utracks_napari)[:,1]))

    roi_fringes=(15,15,15)
    list_of_center_inds = [150,150,150]


    dynROI = DynROI().from_coords(
        list_of_tinds=range(total_time),
        list_of_center_inds=list_of_center_inds,
        world_zyx_shape=(300,)*3,
        roi_fringes=roi_fringes
    )

    # dynROI = track.get_dynROI(
    #         world_zyx_shape=labels.shape[1:],
    #         roi_fringes=roi_fringes
    #     )


    renderer = Renderer()


    viewerROI = renderer.orthorender_dynROI(
        dynROI=dynROI,
        tracks=utracks,
        scale4d=scale,
        points=all_points,
        points_face_color=all_facecolors
    )

    viewer_render3D = renderer.render_dynROI(
        dynROI=dynROI,
        tracks=utracks,
        points=all_points,
        points_face_color=all_facecolors,
        scale4d=scale,
        view3D=True
    )



# viewer.dims.ndisplay = 3
# viewer.grid.shape = (1, len(viewer.layers))

# for layer in viewer.layers: layer.scale = scale


napari.run()