import numpy as np
import napari

napari_vecs = []
vec0 = np.array([0,0])
vec = np.array([10,0])

tp = 300
sigma = np.sqrt(200/tp)

for i in range(1000):
    vec = vec - vec/tp + np.random.normal(loc=0, scale=sigma, size=(2))

    napari_vecs.append(
        [[i,*vec0], [i, *vec]]
    )


viewer = napari.Viewer()

viewer.add_vectors(
    data=np.array(napari_vecs),
    ndim=3,
)


napari.run()