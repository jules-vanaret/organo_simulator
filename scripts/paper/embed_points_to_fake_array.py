import numpy as np
from organo_simulator.utils import load_csv_coords
import os
from skimage.measure import label
import tifffile


path2data = '/home/jvanaret/data/data_trackability_study/simulations/test_long'

merge_rates_str = [str(elem).zfill(3) for elem in np.arange(5,50,5)]

for merge_rate_str in merge_rates_str:
    path2coords = f'{path2data}/coords_merge_{merge_rate_str}'
    path2arrays = f'{path2data}/fake_array_merge_{merge_rate_str}'

    files = os.listdir(path2coords)

    Nx = 300

    for filename in files:

        coords = load_csv_coords(
            path_to_csv=f'{path2coords}/{filename}'
        )

        L = np.mean(coords)
        N_part, d = coords.shape

        labels = np.zeros(
            shape=(Nx,)*d,
            dtype='uint16'
        )

        coords_ind = (coords/(2*L)*Nx).astype(int)

        labels[tuple(coords_ind.T)] = np.arange(1, N_part+1)

        # labels = label(array)
        try:
            tifffile.imwrite(
                f'{path2arrays}/{filename.replace("csv","tif")}',
                labels.astype('uint16') # /!\ IMPORTANT FOR UTRACK !
            )
        except FileNotFoundError:
            os.mkdir(f'{path2arrays}')
            tifffile.imwrite(
                f'{path2arrays}/{filename.replace("csv","tif")}',
                labels.astype('uint16') # /!\ IMPORTANT FOR UTRACK !
            )
            print('failed')








