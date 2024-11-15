import numpy as np
import numba
from scipy.spatial import KDTree as scipy_KDTree
import organo_simulator.utils as simulator_utils

@numba.jit(nopython=True)
def drag_velocity_from_neighbors(velocities, 
                            dist_indices,dist_indptr):
    N_part = velocities.shape[0]    
    
    drag_velocities = np.zeros(velocities.shape)

    # i iterates on each rows (each particle)        
    for i in range(N_part):
        num_neighbors = dist_indptr[i+1]-dist_indptr[i]
        
        if num_neighbors > 0:

            indiv_velocities_i = np.zeros(velocities.shape[1])
            
            for dataIdx in range(dist_indptr[i],dist_indptr[i+1]):
                # j is the index of a neighboring cell
                j = dist_indices[dataIdx]
                indiv_velocities_i += velocities[j]

            drag_velocities[i] = indiv_velocities_i / num_neighbors    

    return drag_velocities


@numba.jit(nopython=True)
def yalla_force_numba_corrected(r, 
    nuclei_size, wiggle_room, 
    neighbor_size, neighbor_room, 
    max_distance, eps):
        """
        See 'ya||a: GPU-Powered Spheroid Models for Mesenchyme
        and Epithelium, Sharpe et al (2019)'
        """

        if r>max_distance:
            return 0

        nuc_nuc_equilibrium_distance = nuclei_size+neighbor_size
        sum_wiggle_rooms = wiggle_room + neighbor_room

        repulsion = np.maximum(nuc_nuc_equilibrium_distance - r,0)
        attraction = np.maximum(r - (nuc_nuc_equilibrium_distance + sum_wiggle_rooms),0)

        # divide by r so that (positions[j]-positions[i]) can be 
        # directly multiplied by the magnitude without having to
        # compute the distance again 
        return eps * (attraction - 2 * repulsion)/r

@numba.jit(nopython=True)
def yalla_force_numba(r, nuclei_size, wiggle_room, max_distance, eps):
        """
        See 'ya||a: GPU-Powered Spheroid Models for Mesenchyme
        and Epithelium, Sharpe et al (2019)'
        """

        if r>max_distance:
            return 0

        nuclei_diameter = 2 * nuclei_size

        repulsion = np.maximum(nuclei_diameter - r,0)
        attraction = np.maximum(r - (nuclei_diameter + 2*wiggle_room),0)

        # divide by r so that (positions[j]-positions[i]) can be 
        # directly multiplied by the magnitude without having to
        # compute the distance again 
        return eps * (attraction - 4 * repulsion)/r

def forces_numba_setup(parallel):
    
    @numba.jit(nopython=True,parallel=parallel)
    def individual_forces_from_scratch(positions, 
                        dist_data,dist_indices,dist_indptr,
                        nuclei_sizes, eps, wiggle_rooms, max_distances):

        N_part = positions.shape[0]    
        individual_forces = np.zeros(positions.shape)

        # i iterates on each rows (each particle)        
        for i in numba.prange(N_part):
            indiv_forces_i = np.zeros(positions.shape[1])
            
            nuclei_size = nuclei_sizes[i,0]
            wiggle_room = wiggle_rooms[i,0]
            max_distance = max_distances[i,0]
            
            for dataIdx in range(dist_indptr[i],dist_indptr[i+1]):
                # j is the index of a neighboring cell
                j = dist_indices[dataIdx]

                # force_magnitude = yalla_force_numba(
                #     dist_data[dataIdx],
                #     nuclei_size,
                #     wiggle_room,
                #     max_distance,
                #     eps
                # )
                force_magnitude = yalla_force_numba_corrected(
                    r=dist_data[dataIdx],
                    nuclei_size=nuclei_size,
                    wiggle_room=wiggle_room,
                    neighbor_size=nuclei_sizes[j,0],
                    neighbor_room=wiggle_rooms[j,0],
                    max_distance=max_distance,
                    eps=eps
                )
                dist_vector_ij = (positions[j]-positions[i])

                indiv_forces_i += force_magnitude*dist_vector_ij

            individual_forces[i] = indiv_forces_i              

        return individual_forces
    
    return individual_forces_from_scratch




class FastOverdampedSimulator:
    def __init__(self, L, Nx, d, N_part, nuclei_sizes, viscosity, 
                 D, persistence_time, energy_potential, 
                 max_distance_factor, wiggle_room_factor,
                 initialisation=None,
                 parallel=False):

        self.Nx = Nx
        self.d = d
        self.N_part = N_part

        if isinstance(nuclei_sizes, int):
            nuclei_sizes = nuclei_sizes * np.ones(shape=(N_part,1),dtype=float)
        elif isinstance(nuclei_sizes, np.ndarray):
            if nuclei_sizes.ndim == 1:
                nuclei_sizes = nuclei_sizes[:,None]  
        self.nuclei_sizes = nuclei_sizes
        self.max_nuclei_size = np.max(nuclei_sizes)
        self.max_overall_distance = self.max_nuclei_size * max_distance_factor

        rho = 0.906 if d==2 else 0.740
        self.equilibrium_radius = np.mean(nuclei_sizes) * np.power(N_part/rho,1/d)
        if L is None:
            # Use hexagonal lattice packing fraction to infer
            # the total radius at equlibrium
            L = 2 * self.equilibrium_radius
        self.L = L


        if isinstance(viscosity, np.ndarray):
            if viscosity.ndim == 1:
                viscosity = viscosity[:,None]
        self.viscosity = viscosity   

        if isinstance(D, np.ndarray):
            if D.ndim == 1:
                D = D[:,None]
        self.D = D
        self.sigma_langevin = np.sqrt(2*D)

        if isinstance(persistence_time, np.ndarray):
            if persistence_time.ndim == 1:
                persistence_time = persistence_time[:,None]
        self.persistence_time = persistence_time
        
        self.energy_potential = energy_potential
        self.max_distance_factor = max_distance_factor
        self.max_distances = max_distance_factor * self.nuclei_sizes
        self.wiggle_rooms = wiggle_room_factor * self.nuclei_sizes
        self.wiggle_room_factor = wiggle_room_factor

        self.positions = self.__initialize_positions(L, N_part, self.equilibrium_radius, d, initialisation)
        self.langevin_force = self.__initialize_langevin_noise(self.sigma_langevin, N_part, d)

        self.individual_forces_from_scratch = forces_numba_setup(parallel)

        self.t = 0

        if parallel:
            numba.set_num_threads(2)
        # foo = 'bar'

    def update_dynamics(self, dt):

        self.t += dt

        ornstein_uhlenbeck_process = (0 - self.langevin_force)/self.persistence_time \
                                   + self.__langevin_noise(
                                        sigma=self.sigma_langevin, 
                                        N_part=self.N_part, 
                                        d=self.d
                                    )
        self.langevin_force = self.langevin_force + dt * ornstein_uhlenbeck_process
        
        deterministic_forces,sparse_distance_norms = self.__attraction_repulsion_force(
            positions=self.positions,
            nuclei_sizes=self.nuclei_sizes,
            max_overall_distance=self.max_overall_distance,
            max_distances=self.max_distances,
            wiggle_rooms=self.wiggle_rooms,
            energy_potential=self.energy_potential
        )
        F = deterministic_forces + self.langevin_force 
        velocities = F/self.viscosity
        velocities = velocities + drag_velocity_from_neighbors(
            velocities,
            sparse_distance_norms.indices,
            sparse_distance_norms.indptr
        )
        
        # Heun's method
        dummy_positions = self.positions + dt * velocities
        deterministic_forces_dummy,sparse_distance_norms = self.__attraction_repulsion_force(
            positions=dummy_positions,
            nuclei_sizes=self.nuclei_sizes,
            max_overall_distance=self.max_overall_distance,
            max_distances=self.max_distances,
            wiggle_rooms=self.wiggle_rooms,
            energy_potential=self.energy_potential
        )
        deterministic_forces  = (deterministic_forces + deterministic_forces_dummy)/2

        
        F = deterministic_forces + self.langevin_force 
        velocities = F/self.viscosity
        velocities = velocities + drag_velocity_from_neighbors(
            velocities,
            sparse_distance_norms.indices,
            sparse_distance_norms.indptr
        )

        self.positions = self.positions + dt * velocities

        self.positions = self.__center_and_clip_positions(
            self.positions,
            self.L
        )


    def dump_array(self):
        pass

    def dump_coordinates(self):
        return self.positions
    
    def __initialize_positions(self, L, N_part, equilibrium_radius, d, initialisation):
        radius = equilibrium_radius / np.sqrt(2)
        radiuses = radius * np.power(np.random.uniform(0,1,size=(N_part,1)),1/d)
        if d==2:
            positions = L + radiuses * simulator_utils.random_2d_unit_vectors(N_part)
        elif d==3:
            if initialisation == 'sausage':
                positions = self.__initialize_as_sausage(N_part)
            else:
                positions = L + radiuses * simulator_utils.random_3d_unit_vectors(N_part)
        
        return positions

    def __initialize_langevin_noise(self, sigma, N_part, d):
        return self.__langevin_noise(sigma=sigma, N_part=N_part, d=d)

    def __initialize_as_sausage(self, N_part):
        
        mean_nuclei_size = np.mean(self.nuclei_sizes)
        l = (N_part * 4 * 64 * mean_nuclei_size**3 / 0.740) ** (1/self.d) / np.sqrt(2)
        self.L = l/1.5
        
        y_positions = np.random.uniform(0,l,size=(N_part))
        radiuses = l/8 * np.power(np.random.uniform(0,1,size=(N_part,1)),1/2)
        zx_positions = radiuses * simulator_utils.random_2d_unit_vectors(N_part)

        positions = np.array([zx_positions[:,0], y_positions, zx_positions[:,1]]).T

        return positions

    def __attraction_repulsion_force(self, positions, nuclei_sizes, 
                                    max_overall_distance, max_distances, wiggle_rooms,
                                     energy_potential):
        
        tree = scipy_KDTree(positions)

        sparse_distance_norms = tree.sparse_distance_matrix(
                            tree,
                            max_distance=max_overall_distance,
                            output_type='coo_matrix'
                        ).tocsr()

        sparse_distance_norms.eliminate_zeros()

        F = self.individual_forces_from_scratch(
            positions=positions,
            dist_data=sparse_distance_norms.data,
            dist_indices=sparse_distance_norms.indices,
            dist_indptr=sparse_distance_norms.indptr,
            nuclei_sizes=nuclei_sizes,
            eps=energy_potential,
            wiggle_rooms=wiggle_rooms,
            max_distances=max_distances
        )

        return F, sparse_distance_norms

    def __potential_force(self, r, nuclei_size, eps, r_bar=2):

        nuclei_diameter = 2 * nuclei_size

        r_condition1 = r < nuclei_diameter * 2**(1/6)
        r_condition2 = r > nuclei_diameter * r_bar

        R = r - (r_bar - 2**(1/6)) * nuclei_diameter

        value_1 = 4 * eps * (-12 * (nuclei_diameter**12/np.power(r,13)) + 6 * (nuclei_diameter**6/np.power(r,7)))
        value_2 = 4 * eps * (-12 * (nuclei_diameter**12/np.power(R,13)) + 6 * (nuclei_diameter**6/np.power(R,7)))

        return value_1 * r_condition1 + value_2 * r_condition2

    def __langevin_noise(self, sigma, N_part, d):
        return sigma * np.random.normal(size=(N_part, d))

    def __center_and_clip_positions(self, positions, L):
        average_positions = np.mean(positions, axis=0)
        positions = np.clip(L+(positions-average_positions), 0, (2-1e-3)*L)

        return positions