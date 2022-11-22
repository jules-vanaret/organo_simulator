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


gt_coords = []

for indt,filename in enumerate(files):
    coords = load_csv_coords(
        path_to_csv=f'{path2coords}/{filename}'
    )

    for ind_track, coord in enumerate(coords, start=1):

        gt_coords.append(
            [int(indt), *coord]
        )

gt_coords = sorted(gt_coords, key=lambda l: l[0])




path2data = '/home/jvanaret/data/data_trackability_study/simulations/test_long/'

split_rates_str = [str(elem).zfill(3) for elem in np.arange(5,10,5)]

for split_rate in split_rates_str:

    name = f'coords_split_{split_rate}'

    all_points = []
    all_facecolors = []
    path2coords = f'{path2data}/coords_split_{split_rate}'
    path2dicts = f'{path2data}/dicts_split_{split_rate}'

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

        for coord, mapping in zip(coords, list_new_to_old):
            
            all_points.append([ind_t,*coord])

            try:
                _ = int(mapping[1])
                # all_facecolors.append('green')
                all_facecolors.append([0,1,0,1])
            except ValueError:
                # all_facecolors.append('red')
                all_facecolors.append([1,0,0,1])

viewer = napari.Viewer(ndisplay=3)

viewer.add_points(gt_coords, size=1.5)

viewer.add_points(all_points, face_color=all_facecolors, size=1.5)
                

napari.run()

