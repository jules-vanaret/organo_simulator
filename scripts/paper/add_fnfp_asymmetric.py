import numpy as np
from organo_simulator.utils import load_csv_coords
import os
from organo_simulator.corrupter import SimulationCorrupter
import tifffile
import csv
from tqdm import tqdm


np.random.seed(2022)

path2data = '/home/jvanaret/data/data_trackability_study/simulations/fnfp_asymmetric'

path2coords = f'{path2data}/coords'

files = os.listdir(path2coords)

corrupter = SimulationCorrupter()

# fnfp_rates = np.linspace(0,0.2,5)
fn_rates = np.arange(0.0, 0.5, 0.05)
fp_rates = np.arange(0.0, 0.5, 0.05)

for ind_fn, fn_rate in enumerate(tqdm(fn_rates)):
    for ind_fp, fp_rate in enumerate(fp_rates):

        fnfp_rate_str = f'fn{ind_fn}_fp{ind_fp}'

        path2corrupted_coords = f'{path2data}/coords_{fnfp_rate_str}'
        path2dicts = f'{path2data}/dicts_{fnfp_rate_str}'
        

        os.mkdir(path2corrupted_coords)
        os.mkdir(path2dicts)

        for filename in files:

            coords = load_csv_coords(
                path_to_csv=f'{path2coords}/{filename}'
            )

            coords = corrupter.remove_fn_from_coords(
                coords, 
                fn_rate=fn_rate
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









