from organo_simulator.simulator import Renderer
import tifffile
import napari

labels = tifffile.imread('/home/jvanaret/Desktop/test_realistic.tif')
labels = labels[4]


renderer = Renderer(
    gaussian_blur_sigma=1,
    gaussian_noise_mean=0.01,
    gaussian_noise_sigma=0.1
)

data = renderer.make_realistic_data_from_labels(labels)


viewer = napari.Viewer()
viewer.add_labels(labels)
viewer.add_image(data)

napari.run()