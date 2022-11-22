import numpy as np
from organo_simulator.utils import load_csv_coords
import os
from organo_simulator.corrupter import SimulationCorrupter
import tifffile
import csv



np.random.seed(2022)

path2data = '/home/jvanaret/data/data_trackability_study/simulations/test_long'

path2coords = f'{path2data}/coords'

files = os.listdir(path2coords)

Nx = 300

corrupter = SimulationCorrupter()

# merge_rates = np.linspace(0,0.2,5)
merge_rates = np.arange(0.05, 0.5, 0.05)

for merge_rate in merge_rates:

    merge_rate_str = f'{merge_rate:.2f}'

    path2arrays = f'{path2data}/fake_array_merge_{merge_rate_str.replace(".","")}'
    path2corrupted_coords = f'{path2data}/coords_merge_{merge_rate_str.replace(".","")}'
    path2dicts = f'{path2data}/dicts_merge_{merge_rate_str.replace(".","")}'
    
    os.mkdir(path2arrays)
    os.mkdir(path2corrupted_coords)
    os.mkdir(path2dicts)

    for filename in files:

        coords = load_csv_coords(
            path_to_csv=f'{path2coords}/{filename}'
        )

        coords, dict_new_old = corrupter.add_merge_to_coords(
            coords, 
            merge_rate=merge_rate,
            max_distance=1.1*(2*8),
            return_dict=True
        )


        np.savetxt(
            f'{path2corrupted_coords}/{filename}',
            coords,
            delimiter=','
        )


        L = np.mean(coords)
        N_part, d = coords.shape

        labels = np.zeros(
            shape=(Nx,)*d,
            dtype='uint16'
        )

        coords_ind = (coords/(2*L)*Nx).astype(int)

        labels[tuple(coords_ind.T)] = np.arange(1, len(coords)+1)

        # labels = label(array)

        tifffile.imwrite(
            f'{path2arrays}/{filename.replace("csv","tif")}',
            labels.astype('uint16') # /!\ IMPORTANT FOR UTRACK !
        )


        w = csv.writer(open(f'{path2dicts}/dict_{filename}', "w"))

        # loop over dictionary keys and values
        for key, val in dict_new_old.items():

            # write every key and value to file
            w.writerow([key, val])









