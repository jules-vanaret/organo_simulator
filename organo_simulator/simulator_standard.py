import numpy as np
from scipy.spatial import KDTree as scipy_KDTree
import organo_simulator.utils as simulator_utils
from itertools import product


class FastOverdampedSimulator:
    def __init__(self, parameters,
                random_forces_function, drag_velocities_function, interaction_forces_function,
                initialization=None):

        self.t = 0
        self.parameters = parameters

        self.random_forces_function = random_forces_function
        self.drag_velocities_function = drag_velocities_function
        self.interaction_forces_function = interaction_forces_function

        self.random_forces = self.__initialize_random_noise(self.parameters)
        self.drag_velocities = None
        self.interaction_forces = None

        self.positions = self.__initialize_positions(self.parameters, initialization)
        self.velocities = self.__initialize_velocities(self.parameters, initialization)

        # foo = 'bar'


    def update_dynamics(self, dt):
        """
        Random forces are simulated by a vector Ornstein-Uhlenbeck process
        (see Gillespie, PRE 1995 for numerical integration)

        For integration of the equation of motion, we use Heun's method for
        SDEs (see Garcia-Alvarez, 2011)
        """

        self.t += dt

        ### Compute distance matrix
        sparse_distance_matrix = self.__compute_distance_matrix(self.parameters, self.positions)
        ###

        ### Compute random forces
        self.random_forces = self.random_forces_function(self.parameters, sparse_distance_matrix,
                                                         self.positions, self.velocities, self.t, dt, 
                                                         self.random_forces)
        ###

        ### First step of Heun's method
        drag_velocities = self.drag_velocities_function(self.parameters, sparse_distance_matrix,
                                                        self.positions, self.velocities,
                                                        self.drag_velocities)
        interaction_forces = self.interaction_forces_function(self.parameters, sparse_distance_matrix,
                                                              self.positions, self.velocities, self.t, dt,
                                                              self.interaction_forces)

        forces = interaction_forces + self.random_force 
        
        dummy_velocities = drag_velocities + forces/self.parameters.viscosity
        dummy_positions = self.positions + dt * dummy_velocities
        ###

        # self.positions = dummy_positions
        # self.velocities = dummy_velocities

        ### Second step of Heun's method
        self.drag_velocities = self.drag_velocities_function(self.parameters, sparse_distance_matrix,
                                                             dummy_positions, dummy_velocities, self.t, dt,
                                                             self.drag_velocities)
        self.interaction_forces = self.interaction_forces_function(self.parameters, sparse_distance_matrix,
                                                                   dummy_positions, dummy_velocities, self.t, dt,
                                                                   self.interaction_forces)

        forces = self.interaction_forces + self.random_force 
        
        velocities = self.drag_velocities + forces/self.parameters.viscosity
        positions = self.positions + dt * 0.5 * (dummy_velocities + velocities)
        ###

        self.positions = positions
        self.velocities = velocities


    def dump_positions(self):
        return self.positions


    def dump_velocities(self):
        return self.velocities


    def dump_forces(self):
        return self.drag_velocities*self.parameters.viscosity, self.interaction_forces, self.random_forces


    def __compute_distance_matrix(self, parameters, positions):
        tree = scipy_KDTree(positions)

        sparse_distance_matrix = tree.sparse_distance_matrix(
                            tree,
                            max_distance=parameters.max_overall_distance,
                            output_type='coo_matrix'
                        ).tocsr()

        # remove zeros on the diagonal
        sparse_distance_matrix.eliminate_zeros()

        return sparse_distance_matrix
    
    def __initialize_positions_sphere(self, L, N_part, equilibrium_radius, d, initialization):
        radius = equilibrium_radius / 2
        radiuses = radius * np.power(np.random.uniform(0,1,size=(N_part,1)),1/d)
        if d==2:
            positions = L + radiuses * simulator_utils.random_2d_unit_vectors(N_part)
        elif d==3:
            if initialization == 'sausage':
                positions = self.__initialize_as_sausage(N_part)
            else:
                positions = L + radiuses * simulator_utils.random_3d_unit_vectors(N_part)
        
        return positions

    def __initialize_positions_cubic(self, L, N_part, equilibrium_radius, d, initialization):
        side = int(np.ceil((2**(d-1)*N_part)**(1/d)))

        foo = np.max(d-1,2)

        prod = product(*( (range(int(side/2)),)*foo+(range(side),)*(d-foo) ))
        prod = [elem for elem in prod][:N_part]
        np.random.shuffle(prod)

        positions = np.zeros((N_part,d))

        for i,inds in zip(range(N_part), prod):
            positions[i] = np.array(inds)

        noise = 0.25 * (np.random.rand(*positions.shape)-0.5)

        positions = (positions + noise) * 14

        return positions - np.mean(positions, axis=0) + L

    def __initialize_positions(self, parameters, initialization):
        """
        Initialize particles positions like a cylinder with 
        hexagonal packing + a bit of noise
        """
        side = int(np.ceil((2**(parameters.d-1)*parameters.N_part)**(1/parameters.d)))

        positions = []

        if parameters.d==2:
            for i in range(side):
                for j in range(int(side/2)):

                    positions.append(
                        [
                            2*i + j%2,
                            np.sqrt(3) * j
                        ]
                    )

        elif parameters.d==3:
            for i in range(side):
                for j in range(int(side/2)):
                    for k in range(int(side/2)):

                        positions.append(
                            [
                                2*i + (j+k)%2,
                                np.sqrt(3) * (j+(k%2)/3),
                                2*np.sqrt(6)/3*k
                            ]
                        )
        
        positions = np.array(positions).astype(float)[:parameters.N_part]

        noise = 0.25 * (np.random.rand(*positions.shape)-0.5)

        positions = (positions + noise) * 5

        return positions - np.mean(positions, axis=0) + parameters.L

    def __initialize_velocities(self, parameters, initialization):
        """
        Initialize velocities from interaction and random forces only
        """

        return np.zeros((self.N_part,self.d))

    def __initialize_random_noise(self, sigma, N_part, d):
        return self.__random_noise(sigma=sigma, N_part=N_part, d=d)

    def __initialize_as_sausage(self, N_part):
        
        mean_nuclei_size = np.mean(self.nuclei_sizes)
        l = (N_part * 4 * 64 * mean_nuclei_size**3 / 0.740) ** (1/self.d) / np.sqrt(2)
        self.L = l/1.5
        
        y_positions = np.random.uniform(0,l,size=(N_part))
        radiuses = l/8 * np.power(np.random.uniform(0,1,size=(N_part,1)),1/2)
        zx_positions = radiuses * simulator_utils.random_2d_unit_vectors(N_part)

        positions = np.array([zx_positions[:,0], y_positions, zx_positions[:,1]]).T

        return positions

    def __random_noise(self, sigma, N_part, d):
        return sigma * np.random.normal(size=(N_part, d))

    def __center_and_clip_positions(self, positions, L):
        average_positions = np.mean(positions, axis=0)
        positions = np.clip(L+(positions-average_positions), 0, (2-1e-3)*L)

        return positions
