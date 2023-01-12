import numpy as np
import numba
from scipy.spatial import KDTree as scipy_KDTree
import organo_simulator.utils as simulator_utils


"""
TODO:
    - implement parallel drag velocities
    - test if force symmetrization is faster
    - change 'langevin' to 'random'

"""




@numba.jit(nopython=True)
def drag_velocities_from_neighbors(velocities, 
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
def yalla_force_numba(r, nuclei_size, wiggle_room, 
                        neighbor_size, neighbor_room, 
                        max_distance, eps):
        """
        See 'ya||a: GPU-Powered Spheroid Models for Mesenchyme
        and Epithelium, Sharpe et al (2019)'
        """

        if r>max_distance:
            return 0

        nuc_nuc_equilibrium_distance = nuclei_size + neighbor_size
        sum_wiggle_rooms = wiggle_room + neighbor_room

        repulsion = np.maximum(nuc_nuc_equilibrium_distance - r,0)
        attraction = np.maximum(r - (nuc_nuc_equilibrium_distance + sum_wiggle_rooms),0)

        # divide by r so that (positions[j]-positions[i]) can be 
        # directly multiplied by the magnitude without having to
        # compute the distance again 
        return eps * (attraction - 2 * repulsion)/r


@numba.jit(nopython=True)
def pinheiro_force_numba(r, nuclei_size, 
                    neighbor_size, max_distance, eps):
        """
        See 'Morphogen gradient orchestrates pattern-
        preserving tissue morphogenesis via
        motility-driven unjamming, Pinheiro et al (2022)'
        """

        sum_size = nuclei_size+neighbor_size
        sigma = sum_size/2**(1/6)

        r_sup_rcut = r > 1.7*sigma
  
        if r_sup_rcut:
            return 0

        r_sup_sum_size = r > sum_size

        if not r_sup_sum_size:
            r_minus_1 = np.power(r, -1)
            r_minus_7 = np.power(r_minus_1, 7)

            repulsion = 12*r*r_minus_7*r_minus_7
            attraction = 6*r_minus_7

            # divide by r so that (positions[j]-positions[i]) can be 
            # directly multiplied by the magnitude without having to
            # compute the distance again 
            return 4 * eps * (attraction - repulsion) * r_minus_1

        r_sup_rbar = r > 1.33*sigma

        if r_sup_rbar:
            sigma_bar = 1.33*sigma - sum_size
            r_minus_1_bar = np.power(r-sigma_bar, -1)
            r_minus_7_bar = np.power(r_minus_1_bar, 7)

            repulsion = 12*(r-sigma_bar)*r_minus_7_bar*r_minus_7_bar
            attraction = 6*r_minus_7_bar

            # divide by r so that (positions[j]-positions[i]) can be 
            # directly multiplied by the magnitude without having to
            # compute the distance again 
            return 4 * eps * (attraction - repulsion) / r

        return 
    

@numba.jit(nopython=True)
def pinheiro_force_numba(r, nuclei_size, 
                    neighbor_size, eps):
    """
    See 'Morphogen gradient orchestrates pattern-
    preserving tissue morphogenesis via
    motility-driven unjamming, Pinheiro et al (2022)'
    """

    sum_size = nuclei_size+neighbor_size
    sigma = sum_size/2**(1/6)

    r_sup_rcut = r > 1.7*sigma
    r_sup_sum_size = r > sum_size
    r_sup_rbar = r > 1.33*sigma

    r_minus_1 = np.power(r, -1)
    r_minus_7 = np.power(r_minus_1, 7)

    sigma_bar = 1.33*sigma - sum_size
    r_minus_1_bar = np.power(r-sigma_bar, -1)
    r_minus_7_bar = np.power(r_minus_1_bar, 7)

    term1 = (not r_sup_sum_size) * (6*r_minus_7 - 12*r*r_minus_7*r_minus_7)
    term2 = ((not r_sup_rcut) and (r_sup_rbar)) * (6*r_minus_7_bar - 12*(r-sigma_bar)*r_minus_7_bar*r_minus_7_bar)

    return (term1 + term2) * r_minus_1
        


def forces_numba_setup(parallel):
    
    @numba.jit(nopython=True,parallel=parallel)
    def individual_att_rep_forces(positions, 
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

                force_magnitude = yalla_force_numba(
                    r=dist_data[dataIdx],
                    nuclei_size=nuclei_size,
                    wiggle_room=wiggle_room,
                    neighbor_size=nuclei_sizes[j,0],
                    neighbor_room=wiggle_rooms[j,0],
                    max_distance=max_distance,
                    eps=eps
                )
                dist_vector_ij = positions[j]-positions[i]

                indiv_forces_i += force_magnitude*dist_vector_ij

            individual_forces[i] = indiv_forces_i              

        return individual_forces
    
    return individual_att_rep_forces



class FastOverdampedSimulator:
    def __init__(self, L, Nx, d, N_part, nuclei_sizes, viscosity, 
                 D, persistence_time, energy_potential, 
                 max_distance_factor, wiggle_room_factor,
                 initialisation=None,
                 parallel=False):

        self.t = 0
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

        packing_fraction = 0.906 if d==2 else 0.740 # hexagonal lattice packing fraction
        self.equilibrium_radius = np.mean(nuclei_sizes) * np.power(N_part/packing_fraction,1/d)
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

        self.individual_att_rep_forces = forces_numba_setup(parallel)

        if parallel:
            numba.set_num_threads(2)
            
        self.positions = self.__initialize_positions(L, N_part, self.equilibrium_radius, d, initialisation)
        self.langevin_force = self.__initialize_langevin_noise(self.sigma_langevin, N_part, d)
        self.velocities = self.__initialize_velocities(
            self.positions,
            self.nuclei_sizes,
            self.max_overall_distance,
            self.max_distances,
            self.wiggle_rooms,
            self.energy_potential,
            self.langevin_force
        )

        # foo = 'bar'

    def update_dynamics(self, dt):
        """
        Random forces are simulated by a vector Ornstein-Uhlenbeck process
        (see Gillespie, PRE 1995 for numerical integration)

        For integration of the equation of motion, we use Heun's method for
        SDEs (see Garcia-Alvarez, 2011)
        """

        self.t += dt

        ### Compute random forces
        langevin_noise = self.__langevin_noise(
            sigma=self.sigma_langevin, 
            N_part=self.N_part, 
            d=self.d
        )
        ornstein_uhlenbeck_differential = (0 - self.langevin_force)/self.persistence_time * dt \
                                        + langevin_noise * np.sqrt(dt) # this part is multiplied by sqrt(dt) as it is the differential
                                                    # of a Wiener process whose variance is proportional to dt
                                                    # (Gillespie, PRE 95)
        self.langevin_force = self.langevin_force + ornstein_uhlenbeck_differential
        ###

        ### First step of Heun's method
        deterministic_forces, sparse_distance_norms = self.__attraction_repulsion_force(
            positions=self.positions,
            nuclei_sizes=self.nuclei_sizes,
            max_overall_distance=self.max_overall_distance,
            max_distances=self.max_distances,
            wiggle_rooms=self.wiggle_rooms,
            energy_potential=self.energy_potential
        )
        F = deterministic_forces + self.langevin_force 
        dummy_velocities = F/self.viscosity
        dummy_velocities = velocities + drag_velocities_from_neighbors(
            self.velocities, # use velocities at previous time steps (see Ya||a)
            sparse_distance_norms.indices,
            sparse_distance_norms.indptr
        )
        self.velocities = dummy_velocities # should we update velocities at this point ???
        ###
        
        ### Second step of Heun's method
        dummy_positions = self.positions + dt * dummy_velocities
        deterministic_forces, sparse_distance_norms = self.__attraction_repulsion_force(
            positions=dummy_positions,
            nuclei_sizes=self.nuclei_sizes,
            max_overall_distance=self.max_overall_distance,
            max_distances=self.max_distances,
            wiggle_rooms=self.wiggle_rooms,
            energy_potential=self.energy_potential
        )
        
        F = deterministic_forces + self.langevin_force 
        velocities = F/self.viscosity
        velocities = velocities + drag_velocities_from_neighbors(
            self.velocities,
            sparse_distance_norms.indices,
            sparse_distance_norms.indptr
        )
        ###

        self.positions = self.positions + dt * 0.5 * (dummy_velocities + velocities)

        self.positions = self.__center_and_clip_positions(
            self.positions,
            self.L
        )
    
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

    def __initialize_velocities(self, positions, nuclei_sizes, max_overall_distance, max_distances, 
                                wiggle_rooms, energy_potential, langevin_force):
        """
        Initialize velocities from interaction and random forces only
        """
        
        deterministic_forces, sparse_distance_norms = self.__attraction_repulsion_force(
            positions=positions,
            nuclei_sizes=nuclei_sizes,
            max_overall_distance=max_overall_distance,
            max_distances=max_distances,
            wiggle_rooms=wiggle_rooms,
            energy_potential=energy_potential
        )
        F = deterministic_forces + langevin_force 
        velocities = F/self.viscosity
        # velocities = velocities + drag_velocities_from_neighbors(
        #     velocities,
        #     sparse_distance_norms.indices,
        #     sparse_distance_norms.indptr
        # )
        
        return velocities

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

        # remove zeros on the diagonal
        sparse_distance_norms.eliminate_zeros()

        F = self.individual_att_rep_forces(
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

    def __langevin_noise(self, sigma, N_part, d):
        return sigma * np.random.normal(size=(N_part, d))

    def __center_and_clip_positions(self, positions, L):
        average_positions = np.mean(positions, axis=0)
        positions = np.clip(L+(positions-average_positions), 0, (2-1e-3)*L)

        return positions

    def dump_array(self):
        raise NotImplementedError

    def dump_coordinates(self):
        return self.positions




