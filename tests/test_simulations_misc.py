import numpy as np
from organo_simulator.utils import load_csv_coords
import os
from skimage.measure import label
import tifffile
import matplotlib.pyplot as plt


path2data = '/home/jvanaret/data/data_trackability_study/simulations/test_long'


path2coords = f'{path2data}/coords_merge_000'

files = os.listdir(path2coords)

all_coords = []


for filename in files:

    coords = load_csv_coords(
        path_to_csv=f'{path2coords}/{filename}'
    )

    all_coords.append(coords)


all_coords = np.array(all_coords)

full_displacements = np.diff(all_coords, axis=0)
displacements = np.linalg.norm(full_displacements, axis=2)



fig = plt.figure()

ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

for elem in displacements:
    dat,_,_ = ax1.hist(elem, bins=np.linspace(0.01,10,1000))
    ax1.clear()
    ax2.plot(np.linspace(0.01,10,999), dat,'.')

ax1.set_xscale('log')

ax2.set_xscale('log')

plt.show()



