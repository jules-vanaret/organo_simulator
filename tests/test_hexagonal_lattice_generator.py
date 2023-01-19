import numpy as np
import napari




positions = []


for i in range(5):
    for j in range(5):
        for k in range(10):

            positions.append(
                [
                    2*i + (j+k)%2,
                    np.sqrt(3) * (j+(k%2)/3),
                    2*np.sqrt(6)/3*k
                ]
            )


positions = np.array(positions)


viewer = napari.Viewer(ndisplay=3)

viewer.add_points(positions)

napari.run()