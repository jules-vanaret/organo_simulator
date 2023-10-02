import napari
import numpy as np

from time import sleep

from magicgui import magicgui, widgets
from napari.qt.threading import thread_worker



viewer = napari.Viewer(ndisplay=2)


def flow_function(positions, matrix):

    return (matrix @ positions.T).T


vectors_layer = viewer.add_vectors(name='vectors', edge_width=0.1, length=0.1)
points_layer = viewer.add_points(np.ones((1,2)),size=1, name='points')


@points_layer.mouse_double_click_callbacks.append
def on_click(layer, event):
    layer.data = np.concatenate((layer.data, np.array([event.position])))



def on_erase_clicked():
    global points_layer
    points_layer.data = np.zeros((0,2))



lin = np.linspace(-10, 10, 21)
points_y, points_x = np.meshgrid(lin, lin, indexing='ij')
points = np.array([points_x.flatten(), points_y.flatten()]).T

# flow_vectors = flow_function(np.array(points)) 
# napari_vectors = np.zeros((len(points), 2, 2))
# napari_vectors[:,0,:] = points
# napari_vectors[:,1,:] = flow_vectors

erase_points = widgets.PushButton()
erase_points.text = 'Erase points'
erase_points.clicked.connect(on_erase_clicked)

worker = None

@thread_worker
def _dynamics():
    mat = None
    matrix = None
    while True:
        sleep(0.01)
        matrix = yield # receive matrix sent
        if not (matrix is None):
            mat = matrix.copy()

        if not(mat is None):
            flow = flow_function(viewer.layers['points'].data, mat)
            viewer.layers['points'].data += 0.005*flow
        



worker = _dynamics()
worker.start()


@magicgui(auto_call=True,
          m_11={'min': -3, 'max': 3, 'step': 0.3},
          m_12={'min': -3, 'max': 3, 'step': 0.3},
          m_21={'min': -3, 'max': 3, 'step': 0.3},
          m_22={'min': -3, 'max': 3, 'step': 0.3},
          call_button=False
          )
def widget(viewer:napari.Viewer, 
           m_11: float=0, m_12: float=1, m_21: float=-1, m_22: float=0,
        ):

    global points
    
    matrix = np.array([[m_11, m_12], [m_21, m_22]])

    flow_points = flow_function(points, matrix)
    napari_vectors = np.zeros((len(points), 2, 2))
    napari_vectors[:,0,:] = points
    napari_vectors[:,1,:] = flow_points

    viewer.layers['vectors'].data = napari_vectors
    worker.send(matrix)







viewer.window.add_dock_widget(widget)
viewer.window.add_dock_widget(erase_points)

viewer.reset_view()

napari.run()
worker.quit()