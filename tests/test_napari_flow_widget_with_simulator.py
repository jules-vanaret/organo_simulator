import napari
import numpy as np

from time import sleep, time

from magicgui import magicgui, widgets
from napari.qt.threading import thread_worker

from organo_simulator.simulator_LJ_force import FastOverdampedSimulator







dt = 0.5
d = 2 # dimension of simulation
N_part = 100 # number of particles
nuclei_sizes = 1 + 0.15*(np.random.randn(N_part)-1/2)

# persistence times (in physical units) for the OU process
persistence_times = np.array([200]*(N_part))
viscosities = np.array([1]*(N_part))
Ds = np.array([1e-4]*(N_part)) # diffusion coefficients

Nx = 1000 # length of the box in pixels for the rendering 

simulator = FastOverdampedSimulator(
    L=None, # will be initializesd automatically
    Nx=Nx, # length of the box in pixels for the rendering 
    d=d, # dimensions (2 or 3)
    N_part=N_part, # number of particles
    nuclei_sizes=nuclei_sizes, # radius of particles
    viscosity=viscosities, # viscosities of particles
    D=Ds, # diffusion coefficients
    persistence_time=persistence_times, # how long the temporal autocorrelation of random movement goes
    energy_potential=1e-6,
    parallel=True,
    flow_field=True
)





















viewer = napari.Viewer(ndisplay=2)


vectors_layer = viewer.add_vectors(name='vectors', edge_width=0.1, length=0.1)
points_layer = viewer.add_points(simulator.dump_coordinates() ,size=simulator.nuclei_sizes, name='points')

def flow_function(positions, matrix):
    return (matrix @ (positions-simulator.L).T).T


@points_layer.mouse_double_click_callbacks.append
def on_click(layer, event):
    if not simulator.is_computing:
        worker.pause()
        simulator.add_particles(np.array(event.position).reshape((1,2)))
        worker.resume()
    else:
        print('not adding particles, already computing')



def on_erase_clicked():
    global points_layer
    points_layer.data = np.zeros((0,2))
    simulator.remove_all_particles()



lin = np.linspace(0, 2*simulator.L, 30)
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

@thread_worker()
def _dynamics():
    mat = None
    matrix = None
    counter = 0
    while True:
        # sleep(0.1)
        matrix = yield # receive matrix sent
        if not (matrix is None):
            mat = matrix.copy()

        if not(mat is None):
            simulator.matrix = mat
            simulator.update_dynamics(dt=dt)

            counter += 1

            if counter%20==0:
                viewer.layers['points'].data = simulator.dump_coordinates()
                counter=0
        



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