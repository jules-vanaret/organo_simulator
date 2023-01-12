import tifffile
from tqdm import tqdm
from organo_simulator.renderer import Renderer
import napari
import numpy as np

scale4d = (1,1.6,1,1)

input_image = tifffile.imread(
    "/home/jvanaret/data/mosaic/mosaic.tif"
)[:,:-1,152:152+192, 166:166+192]
segmentation_gt = tifffile.imread(
    '/home/jvanaret/data/data_trackability_study/raw_data/square_annotations_gt.tif'
)


renderer = Renderer(
    nuclei_sizes=1,
    N_part=1,
    n_rays=10,
    d=3,
    L=1,
    Nx=1,
    cell_to_nuc_ratio=1,
    gaussian_blur_sigma=1,
    gaussian_noise_mean=0.01,
    gaussian_noise_sigma=0.015,
    scale=scale4d[1:]
)

renders = np.array(
    [renderer.make_realistic_data_from_labels(lab, debug=True) for lab in tqdm(segmentation_gt[:2])]
)

viewer = napari.Viewer()
viewer.add_image(input_image, scale=scale4d)
viewer.add_labels(segmentation_gt, scale=scale4d)
viewer.add_image(renders, scale=scale4d)

napari.run()

