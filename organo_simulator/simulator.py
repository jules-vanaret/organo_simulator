import numpy as np
import numba
from scipy.ndimage import gaussian_filter as scipy_gaussian
from scipy.spatial import KDTree as scipy_KDTree

@numba.jit(nopython=True)
def drag_velocity_from_neighbors_with_confinement(velocities, positions,
                            dist_indices,dist_indptr,
                            k, L, R_eq):
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
        else:
            ones = np.ones(velocities.shape[1])
            
            position = positions[i]
            vector_from_center = position - L * ones
            distance_to_center = np.linalg.norm(vector_from_center) 

            drag_velocities[i] = - k * (distance_to_center - R_eq) * vector_from_center/distance_to_center    

    return drag_velocities

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
def yalla_force_numba_corrected(r, nuclei_size, wiggle_room, max_distance, eps):
        """
        See 'ya||a: GPU-Powered Spheroid Models for Mesenchyme
        and Epithelium, Sharpe et al (2019)'
        """

        if r>max_distance:
            return 0

        nuclei_diameter = nuclei_size+neighbor_size

        repulsion = np.maximum(nuclei_diameter - r,0)
        attraction = np.maximum(r - (nuclei_diameter + 2*wiggle_room),0)

        # divide by r so that (positions[j]-positions[i]) can be 
        # directly multiplied by the magnitude without having to
        # compute the distance again 
        return eps * (attraction - 4 * repulsion)/r

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
        # drag_forces = np.zeros(positions.shape)

        # i iterates on each rows (each particle)        
        for i in numba.prange(N_part):
            indiv_forces_i = np.zeros(positions.shape[1])
            
            # num_neighbors = dist_indptr[i+1]-dist_indptr[i]
            
            for dataIdx in range(dist_indptr[i],dist_indptr[i+1]):
                # j is the index of a neighboring cell
                j = dist_indices[dataIdx]

                force_magnitude = yalla_force_numba(
                    dist_data[dataIdx],
                    nuclei_sizes[j,0],
                    wiggle_rooms[j,0],
                    max_distances[j,0],
                    eps
                )
                dist_vector_ij = (positions[j]-positions[i])

                indiv_forces_i += force_magnitude*dist_vector_ij
                # drag_forces[j] -= (force_magnitude/num_neighbors)*dist_vector_ij

            individual_forces[i] = indiv_forces_i              

        return individual_forces# + drag_forces
    
    return individual_forces_from_scratch


@numba.jit(nopython=True)#,nogil=True)#,parallel=True)
def individual_forces_from_positions(positions, 
                      forces_data,forces_indices,forces_indptr): 

    numRows_forces = positions.shape[0]    
    individual_forces = np.zeros(positions.shape)

    for i in range(numRows_forces):#numba.prange(numRows_forces):
        indiv_forces_i = np.zeros(positions.shape[1])       
        for dataIdx in range(forces_indptr[i],forces_indptr[i+1]):

            j = forces_indices[dataIdx]
            indiv_forces_i += forces_data[dataIdx]*(positions[j]-positions[i])

        individual_forces[i] = indiv_forces_i            

    return individual_forces

@numba.njit#('(int32[:],int32[:],float64[:,:],int32[:])')
def positions_to_vectors(rows, cols, positions, shape):
   
    vectors = np.zeros(shape)
    for i,j in zip(rows,cols):
        vectors[i,j] = positions[i] - positions[j]
    return vectors

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
            nuclei_sizes = nuclei_sizes * np.ones(shape=(N_part,1))
        elif isinstance(nuclei_sizes, np.ndarray):
            if nuclei_sizes.ndim == 1:
                nuclei_sizes = nuclei_sizes[:,None]  
        self.nuclei_sizes = nuclei_sizes
        self.max_nuclei_size = np.max(nuclei_sizes)

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
        self.wiggle_room_factor = wiggle_room_factor

        self.positions = self.__initialize_positions(L, N_part, self.equilibrium_radius, d, initialisation)
        self.langevin_force = self.__initialize_langevin_noise(self.sigma_langevin, N_part, d)

        self.individual_forces_from_scratch = forces_numba_setup(parallel)

        if parallel:
            numba.set_num_threads(2)
        # foo = 'bar'

    def update_dynamics(self, dt):

        deterministic_forces,sparse_distance_norms = self.__attraction_repulsion_force(
            positions=self.positions,
            nuclei_sizes=self.nuclei_sizes,
            max_distance_factor=self.max_distance_factor,
            wiggle_room_factor=self.wiggle_room_factor,
            energy_potential=self.energy_potential
        )
        # Heun's method
        dummy_positions = self.positions + dt * deterministic_forces/self.viscosity
        deterministic_forces_dummy,_ = self.__attraction_repulsion_force(
            positions=dummy_positions,
            nuclei_sizes=self.nuclei_sizes,
            max_distance_factor=self.max_distance_factor,
            wiggle_room_factor=self.wiggle_room_factor,
            energy_potential=self.energy_potential
        )
        deterministic_forces  = (deterministic_forces + deterministic_forces_dummy)/2

        ornstein_uhlenbeck_process = (0 - self.langevin_force)/self.persistence_time \
                                   + self.__langevin_noise(sigma=self.sigma_langevin, N_part=self.N_part, d=self.d)
        self.langevin_force = self.langevin_force + dt * ornstein_uhlenbeck_process
        
        F = deterministic_forces + self.langevin_force 
        velocities = F/self.viscosity
        velocities = velocities + drag_velocity_from_neighbors(
            velocities,
            sparse_distance_norms.indices,
            sparse_distance_norms.indptr
        )
        # velocities = velocities + drag_velocity_from_neighbors_with_confinement(
        #     velocities, 
        #     self.positions,
        #     sparse_distance_norms.indices,
        #     sparse_distance_norms.indptr,
        #     k=0.1,
        #     L=self.L,
        #     R_eq=self.equilibrium_radius
        # )
        self.positions = self.positions + dt * velocities

        self.positions = self.center_and_clip_positions(
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
            positions = L + radiuses * self.__random_2d_unit_vectors(N_part)
        elif d==3:
            if initialisation == 'sausage':
                positions = self.__initiate_as_sausage(N_part)
            else:
                positions = L + radiuses * self.__random_3d_unit_vectors(N_part)
        
        return positions

    def __initialize_langevin_noise(self, sigma, N_part, d):
        return self.__langevin_noise(sigma=sigma, N_part=N_part, d=d)

    def __random_2d_unit_vectors(self, N_part):
        phi = np.random.uniform(0,2*np.pi,size=N_part)
        x = np.cos( phi )
        y = np.sin( phi )

        return np.array([x,y]).T

    def __random_3d_unit_vectors(self, N_part):
        phi = np.random.uniform(0,2*np.pi,size=N_part)
        costheta = np.random.uniform(-1,1,size=N_part)

        theta = np.arccos( costheta )
        x = np.sin( theta ) * np.cos( phi )
        y = np.sin( theta ) * np.sin( phi )
        z = np.cos( theta )

        return np.array([x,y,z]).T

    def __initiate_as_sausage(self, N_part):
        
        mean_nuclei_size = np.mean(self.nuclei_sizes)
        l = (N_part * 4 * 64 * mean_nuclei_size**3 / 0.740) ** (1/self.d) / np.sqrt(2)
        self.L = l/1.5
        
        y_positions = np.random.uniform(0,l,size=(N_part))
        radiuses = l/8 * np.power(np.random.uniform(0,1,size=(N_part,1)),1/2)
        zx_positions = radiuses * self.__random_2d_unit_vectors(N_part)

        positions = np.array([zx_positions[:,0], y_positions, zx_positions[:,1]]).T

        return positions

    def __attraction_repulsion_force(self, positions, nuclei_sizes, 
                                    max_distance_factor, wiggle_room_factor,
                                     energy_potential):
        
        tree = scipy_KDTree(positions)

        sparse_distance_norms = tree.sparse_distance_matrix(
                            tree,
                            max_distance=max_distance_factor*self.max_nuclei_size,
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
            wiggle_rooms=wiggle_room_factor*nuclei_sizes,
            max_distances=max_distance_factor*nuclei_sizes
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

    def center_and_clip_positions(self, positions, L):
        average_positions = np.mean(positions, axis=0)
        positions = np.clip(L+(positions-average_positions), 0, (2-1e-3)*L)

        return positions



if False:
    # def __naive_potential_force_sparse(self, r, nuclei_size, eps, wiggle_room):
    #     """
    #     See 'ya||a: GPU-Powered Spheroid Models for Mesenchyme
    #     and Epithelium, Sharpe et al (2019)'
    #     """

    #     nuclei_diameter = 2 * nuclei_size

    #     matrix_of_1s = r.power(0)

    #     value_1 = (matrix_of_1s*nuclei_diameter - r).maximum(0)
    #     value_2 = (r - matrix_of_1s * (nuclei_diameter + 2*wiggle_room)).maximum(0)

    #     return - eps * (4 * value_1 - value_2)

    # def __naive_potential_force(self, r, nuclei_size, eps, wiggle_room):
    #     """
    #     See 'ya||a: GPU-Powered Spheroid Models for Mesenchyme
    #     and Epithelium, Sharpe et al (2019)'
    #     """

    #     nuclei_diameter = 2 * nuclei_size

    #     value_1 = np.maximum(nuclei_diameter - r, 0)
    #     value_2 = np.maximum(r - nuclei_diameter - 2*wiggle_room, 0)

    #     return - eps * (4 * value_1 - value_2)



    # def __attraction_repulsion_force_old3(self, positions, nuclei_size, 
    #                                 max_distance_factor, wiggle_room_factor,
    #                                  energy_potential, clip_force):
        
    #     tree = scipy_KDTree(positions)

    #     sparse_distance_norms = tree.sparse_distance_matrix(
    #                         tree,
    #                         max_distance=max_distance_factor*nuclei_size,
    #                         output_type='coo_matrix'
    #                     ).tocsr()

    #     sparse_distance_norms.eliminate_zeros()

    #     forces = self.__naive_potential_force_sparse(
    #         r=sparse_distance_norms,
    #         nuclei_size=nuclei_size,
    #         eps=energy_potential,
    #         wiggle_room=wiggle_room_factor*nuclei_size
    #     )

    #     # Artificially account for the unit vectors normalization
    #     forces = forces.multiply(sparse_distance_norms.power(-1))

    #     F = individual_forces_from_positions(
    #         positions=positions,
    #         forces_data=forces.data,
    #         forces_indices=forces.indices,
    #         forces_indptr=forces.indptr
    #     )

    #     if not(clip_force is None):
    #         total_force_norms = np.linalg.norm(F, axis=1)
    #         mask = total_force_norms!=0
            
    #         F[mask] = F[mask]/total_force_norms[mask][:,None]

    #         total_force_norms = np.clip(total_force_norms,-clip_force, clip_force)
    #         F = F * total_force_norms[:,None]

    #     return F


    # def __attraction_repulsion_force_old2(self, positions, nuclei_size, 
    #                                 max_distance_factor, wiggle_room_factor,
    #                                  energy_potential, clip_force):
        
    #     tree = scipy_KDTree(positions)

    #     sparse_distance_norms = tree.sparse_distance_matrix(
    #                         tree,
    #                         max_distance=max_distance_factor*nuclei_size,
    #                         output_type='coo_matrix'
    #                     ).tocsr()

    #     sparse_distance_norms.eliminate_zeros()

    #     forces = self.__naive_potential_force_sparse(
    #         r=sparse_distance_norms,
    #         nuclei_size=nuclei_size,
    #         eps=energy_potential,
    #         wiggle_room=wiggle_room_factor*nuclei_size
    #     )

    #     # Artificially account for the unit vectors normalization
    #     forces = forces.multiply(sparse_distance_norms.power(-1))

    #     non_zero_elems = sparse_distance_norms.nonzero()
    #     unit_vectors = positions_to_vectors(
    #         rows=non_zero_elems[0],
    #         cols=non_zero_elems[1],
    #         positions=positions,
    #         shape=(len(positions),len(positions),self.d)
    #     )

    #     F = np.sum(forces.toarray()[:,:,None] * unit_vectors, axis=0)

    #     if not(clip_force is None):
    #         total_force_norms = np.linalg.norm(F, axis=1)
    #         mask = total_force_norms!=0
            
    #         F[mask] = F[mask]/total_force_norms[mask][:,None]

    #         total_force_norms = np.clip(total_force_norms,-clip_force, clip_force)
    #         F = F * total_force_norms[:,None]

    #     return F

    # def __attraction_repulsion_force_old(self, positions, nuclei_size, 
    #                                 max_distance_factor, wiggle_room_factor,
    #                                  energy_potential, clip_force):

    #     distance_vectors = positions[:,None] - positions
        

    #     tree = scipy_KDTree(positions)

    #     foo = tree.sparse_distance_matrix(
    #                         tree,
    #                         max_distance=max_distance_factor*nuclei_size
    #                     )

    #     distance_norms = tree.sparse_distance_matrix(
    #                         tree,
    #                         max_distance=max_distance_factor*nuclei_size
    #                     ).toarray()
        
    #     non_zero_elems = np.nonzero(distance_norms)

    #     #lower_non_zero_elems = tuple(np.array([[i,j] for i,j in zip(*non_zero_elems) if i>j]).T)
    #     # upper_non_zero_elems = lower_non_zero_elems[::-1]

    #     unit_vectors = np.zeros(distance_vectors.shape)
    #     unit_vectors[non_zero_elems] = distance_vectors[non_zero_elems] / distance_norms[non_zero_elems][:,None]


    #     F_norm = np.zeros(shape=distance_norms.shape)

    #     forces = self.__naive_potential_force(
    #         r=distance_norms[non_zero_elems],
    #         nuclei_size=nuclei_size,
    #         eps=energy_potential,
    #         wiggle_room=wiggle_room_factor*nuclei_size
    #     )

    #     F_norm[non_zero_elems] = forces

    #     # forces = self.__naive_potential_force(
    #     #     r=distance_norms[upper_non_zero_elems],
    #     #     nuclei_size=nuclei_size,
    #     #     eps=energy_potential,
    #     #     wiggle_room=0*nuclei_size
    #     # )

    #     # F_norm[lower_non_zero_elems] = forces
    #     # F_norm[upper_non_zero_elems] = forces

    #     F = np.sum(F_norm[:,:,None] * unit_vectors, axis=0)

    #     if not(clip_force is None):
    #         total_force_norms = np.linalg.norm(F, axis=1)
    #         mask = total_force_norms!=0
            
    #         F[mask] = F[mask]/total_force_norms[mask][:,None]

    #         total_force_norms = np.clip(total_force_norms,-clip_force, clip_force)
    #         F = F * total_force_norms[:,None]

    #     return F


    # def __attraction_repulsion_force_bak(self, positions, nuclei_size, max_distance_factor,
    #                                  energy_potential, clip_force):

    #     distance_vectors = positions[:,None] - positions

    #     tree = scipy_KDTree(positions)

    #     distance_norms = tree.sparse_distance_matrix(
    #                         tree,
    #                         max_distance=nuclei_size * max_distance_factor + self.debug_dist * 1000
    #                     ).toarray()
        
    #     non_zero_elems = np.nonzero(distance_norms)

    #     lower_non_zero_elems = list(np.array([[i,j] for i,j in zip(*non_zero_elems) if i>j]).T)
    #     upper_non_zero_elems = lower_non_zero_elems[::-1]

    #     unit_vectors = np.zeros(distance_vectors.shape)
    #     unit_vectors[non_zero_elems] = distance_vectors[non_zero_elems] / distance_norms[non_zero_elems][:,None]

    #     F_norm = np.zeros(shape=distance_norms.shape)
        
    #     # forces = self.__potential_force(
    #     #     distance_norms[upper_non_zero_elems],
    #     #     nuclei_size=nuclei_size,
    #     #     eps=energy_potential
    #     # )

    #     forces = self.__naive_potential_force(
    #         r=distance_norms[upper_non_zero_elems],
    #         nuclei_size=nuclei_size,
    #         eps=energy_potential,
    #         wiggle_room=0*nuclei_size
    #     )

    #     F_norm[lower_non_zero_elems] = forces
    #     F_norm[upper_non_zero_elems] = forces

    #     F = np.sum(F_norm[:,:,None] * unit_vectors, axis=0)

    #     if not(clip_force is None):
    #         total_force_norms = np.linalg.norm(F, axis=1)
            
    #         F[total_force_norms!=0] = F[total_force_norms!=0]/total_force_norms[total_force_norms!=0][:,None]

    #         total_force_norms = np.clip(total_force_norms,-clip_force, clip_force)
    #         F = F*total_force_norms[:,None]

    #     return F
    pass

