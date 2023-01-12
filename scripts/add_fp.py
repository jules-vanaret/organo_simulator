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

corrupter = SimulationCorrupter()

# fp_rates = np.linspace(0,0.2,5)
fp_rates = np.arange(0.0, 0.5, 0.05)

for fp_rate in fp_rates:

    fp_rate_str = f'{fp_rate:.2f}'

    path2corrupted_coords = f'{path2data}/coords_fp_{fp_rate_str.replace(".","")}'
    path2dicts = f'{path2data}/dicts_fp_{fp_rate_str.replace(".","")}'
    

    os.mkdir(path2corrupted_coords)
    os.mkdir(path2dicts)

    for filename in files:

        coords = load_csv_coords(
            path_to_csv=f'{path2coords}/{filename}'
        )

        coords, dict_new_old = corrupter.add_fp_to_coords(
            coords, 
            fp_rate=fp_rate,
            return_dict=True
        )


        np.savetxt(
            f'{path2corrupted_coords}/{filename}',
            coords,
            delimiter=','
        )

        w = csv.writer(open(f'{path2dicts}/dict_{filename}', "w"))

        # loop over dictionary keys and values
        for key, val in dict_new_old.items():

            # write every key and value to file
            w.writerow([key, val])









