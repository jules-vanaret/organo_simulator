from organo_simulator.simulator_brownian import FastOverdampedSimulator
import napari
import numpy as np
from tqdm import tqdm

L = 10
viscosity = 1
N_part = 100
D=1



initial_positions = np.random.rand(N_part, 2)*L

simulator = FastOverdampedSimulator(
    L=L,
    d=2,
    N_part=N_part,
    viscosity=viscosity,
    D=D,
    initial_positions=initial_positions
)


dt = 0.1
N_steps = 1000

all_positions = []

for i in tqdm(range(N_steps)):

    simulator.update_dynamics(dt=dt)

    all_positions.append(simulator.positions)


napari_tracks = np.zeros((N_part*N_steps, 4))

for i, positions in enumerate(all_positions):
    napari_tracks[i*N_part:(i+1)*N_part, 0] = np.arange(N_part)
    napari_tracks[i*N_part:(i+1)*N_part, 1] = i

    napari_tracks[i*N_part:(i+1)*N_part, 2:] = positions

viewer = napari.Viewer()

viewer.add_tracks(napari_tracks)

napari.run()


