import numpy as np
import numba
from scipy.spatial import KDTree as scipy_KDTree
import organo_simulator.utils as simulator_utils
from itertools import product



@numba.jit(nopython=True, parallel=True)
def drag_velocities_from_neighbors(velocities, 
                            dist_indices,dist_indptr):
    N_part = velocities.shape[0]    
    
    drag_velocities = np.zeros(velocities.shape)

    #i iterates on each rows (each particle)        
    for i in numba.prange(N_part):
        num_neighbors = dist_indptr[i+1]-dist_indptr[i]
        
        if num_neighbors > 0:

            indiv_velocities_i = np.zeros(velocities.shape[1])
            
            for dataIdx in range(dist_indptr[i], dist_indptr[i+1]):
                # j is the index of a neighboring cell
                j = dist_indices[dataIdx]
                indiv_velocities_i = indiv_velocities_i + velocities[j]

            drag_velocities[i] = indiv_velocities_i / num_neighbors    

    return drag_velocities


@numba.jit(nopython=True)
def LJ_force_numba(r, nuclei_size, neighbor_size, eps):
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

        sum_size_6 = np.power(sigma, 6)
        sum_size_12 = np.power(sum_size_6, 2)
        
        # if not r_sup_sum_size:
        r_minus_1 = 1/r
        r_minus_7 = np.power(r_minus_1, 7)


        repulsion = 2*r*r_minus_7*r_minus_7 * sum_size_12
        attraction = r_minus_7 * sum_size_6

        # divide by r so that (positions[j]-positions[i]) can be 
        # directly multiplied by the magnitude without having to
        # compute the distance again 
        return 24 * eps * (attraction - repulsion) * r_minus_1

        # r_sup_rbar = r > 1.33*sigma

        # if r_sup_rbar:
        #     sigma_bar = 1.33*sigma - sum_size
        #     r_minus_1_bar = 1/(r-sigma_bar)
        #     r_minus_7_bar = np.power(r_minus_1_bar, 7)

        #     repulsion = 2*(r-sigma_bar)*r_minus_7_bar*r_minus_7_bar * sum_size_12
        #     attraction = r_minus_7_bar * sum_size_6

        #     # divide by r so that (positions[j]-positions[i]) can be 
        #     # directly multiplied by the magnitude without having to
        #     # compute the distance again 
        #     return 24 * eps * (attraction - repulsion) / r

        # return 0
    


def forces_numba_setup(parallel):
    
    @numba.jit(nopython=True, parallel=parallel)
    def individual_att_rep_forces(positions, 
                        dist_data,dist_indices,dist_indptr,
                        nuclei_sizes, eps, L, periodic):
        """
        Symmetrizing forces (setting Fij and Fji at the same
        time) is NOT faster !
        """

        N_part = positions.shape[0]    
        individual_forces = np.zeros(positions.shape)

        # i iterates on each rows (each particle)        
        for i in numba.prange(N_part):
            indiv_forces_i = np.zeros(positions.shape[1])
            
            nuclei_size = nuclei_sizes[i,0]
            
            for dataIdx in range(dist_indptr[i],dist_indptr[i+1]):
                # j is the index of a neighboring cell
                j = dist_indices[dataIdx]

                force_magnitude = LJ_force_numba(
                    r=dist_data[dataIdx],
                    nuclei_size=nuclei_size,
                    neighbor_size=nuclei_sizes[j,0],
                    eps=eps
                )
                dist_vector_ij = positions[j]-positions[i]
                if periodic:
                    dist_vector_ij = (dist_vector_ij + L/2) % L - L/2

                indiv_forces_i = indiv_forces_i + force_magnitude*dist_vector_ij

            individual_forces[i] = indiv_forces_i              

        return individual_forces
    
    return individual_att_rep_forces



class FastOverdampedSimulator:
    def __init__(self, L, Nx, d, N_part, nuclei_sizes, viscosity, 
                 D, persistence_time, potential_energy,
                 parallel=False, flow_field_function=None, periodic=True):

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
        self.max_overall_distance = self.max_nuclei_size * 1.7/2**(1/6) * 2

        print(f'Max overall distance: {self.max_overall_distance:.2f}')
        self.L = L
        print(f'Box size: {self.L:.2f}')


        if isinstance(viscosity, np.ndarray):
            if viscosity.ndim == 1:
                viscosity = viscosity[:,None]
        self.viscosity = viscosity   

        if isinstance(D, np.ndarray):
            if D.ndim == 1:
                D = D[:,None]
        self.D = D
        self.sigma_random = np.sqrt(2*D)

        if isinstance(persistence_time, np.ndarray):
            if persistence_time.ndim == 1:
                persistence_time = persistence_time[:,None]
        self.persistence_time = persistence_time
        
        self.potential_energy = potential_energy

        self.individual_att_rep_forces = forces_numba_setup(parallel)

        if parallel:
            numba.set_num_threads(2)
            
        self.positions = self.__initialize_positions(L, N_part, d)
        self.random_force = self.__initialize_random_noise(self.sigma_random, N_part, d)
        self.velocities = self.__initialize_velocities(
            self.positions,
            self.nuclei_sizes,
            self.max_overall_distance,
            self.potential_energy,
            self.random_force
        )

        self.flow_field_function = flow_field_function

        self.periodic = periodic

    def update_dynamics(self, dt):
        """
        Random forces are simulated by a vector Ornstein-Uhlenbeck process
        (see Gillespie, PRE 1995 for numerical integration)

        For integration of the equation of motion, we use Heun's method for
        SDEs (see Garcia-Alvarez, 2011)
        """

        self.is_computing = True 

        self.t += dt

        ### Compute random forces
        random_noise = self.__random_noise(
            sigma=np.sqrt(2*self.D),
            N_part=self.N_part, 
            d=self.d
        )

        # see Gillespie, PRE 95
        # "Exact numerical simulation of the Ornstein-Uhlenbeck process and its integral"
        deterministic_ou_term = np.exp(-dt/self.persistence_time)
        random_ou_term = random_noise * np.sqrt(1-np.exp(-2*dt/self.persistence_time))

        self.random_force = self.random_force * deterministic_ou_term + random_ou_term
        ###

        ### First step of Heun's method
        deterministic_forces, sparse_distance_norms = self.__attraction_repulsion_force(
            positions=self.positions,
            nuclei_sizes=self.nuclei_sizes,
            max_overall_distance=self.max_overall_distance,
            potential_energy=self.potential_energy,
            L=self.L,
            periodic=self.periodic
        )

        F = deterministic_forces #+ self.random_force #! removed random force from 1st step
        dummy_velocities = F/self.viscosity
        drag_velocities = drag_velocities_from_neighbors(
            self.velocities, # use velocities at previous time steps (see Ya||a)
            sparse_distance_norms.indices,
            sparse_distance_norms.indptr
        ) * self.viscosity/(1+self.viscosity) 
        dummy_velocities = dummy_velocities + drag_velocities
        if self.flow_field_function:
            dummy_velocities = dummy_velocities + self.flow_field_function(self.positions)
        dummy_positions = self.positions + dt * dummy_velocities

        if self.periodic:
            dummy_positions = self.__wrap_positions(
                dummy_positions,
                self.L
            )
        ###

        
        # ### Second step of Heun's method
        deterministic_forces, sparse_distance_norms = self.__attraction_repulsion_force(
            positions=dummy_positions,
            nuclei_sizes=self.nuclei_sizes,
            max_overall_distance=self.max_overall_distance,
            potential_energy=self.potential_energy,
            L=self.L,
            periodic=self.periodic
        )

        F = deterministic_forces + self.random_force 
        velocities = F/self.viscosity
        drag_velocities = drag_velocities_from_neighbors(
            dummy_velocities, # use velocities from first Heun's step
            sparse_distance_norms.indices,
            sparse_distance_norms.indptr
        ) * self.viscosity/(1+self.viscosity)
        velocities = velocities + drag_velocities

        if self.flow_field_function:
            velocities = velocities + self.flow_field_function(dummy_positions)

        self.velocities = velocities
        self.positions = self.positions + dt * 0.5 * (dummy_velocities + velocities)
        ###

        self.determisitic_velocities = deterministic_forces / self.viscosity
        self.drag_velocities = drag_velocities
        self.random_velocities = self.random_force / self.viscosity

        if self.periodic:        
            self.positions = self.__wrap_positions(
                self.positions,
                self.L
            )
    
    def __wrap_positions(self, positions, L):
        return np.mod(positions, L)

    def __initialize_positions(self, L, N_part, d):
        """
        Initialize particles positions like a cylinder with 
        hexagonal packing + a bit of noise
        """
        box_size = int(np.ceil(N_part**(1/d)))

        positions = np.meshgrid(*[np.linspace(0.05*L,0.95*L,box_size)]*d)
        positions = np.array([positions[i].flatten() for i in range(d)]).T

        positions = positions[:N_part]

        dx = 0.9 * L / box_size

        noise = 0.1 * dx * (np.random.rand(*positions.shape)-0.5)

        positions = (positions + noise)

        return positions

    def __initialize_velocities(self, positions, nuclei_sizes, max_overall_distance, 
                                potential_energy, random_force):
        """
        Initialize velocities from interaction and random forces only
        """

        return np.zeros((self.N_part,self.d))

    def __initialize_random_noise(self, sigma, N_part, d):
        return self.__random_noise(sigma=sigma, N_part=N_part, d=d)

    def __attraction_repulsion_force(self, positions, nuclei_sizes, 
                                    max_overall_distance,
                                     potential_energy, L, periodic):
        
        if periodic:
            tree = scipy_KDTree(positions, boxsize=L)
        else:
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
            eps=potential_energy,
            L=L,
            periodic=periodic
        )

        return F, sparse_distance_norms

    def __random_noise(self, sigma, N_part, d):
        return sigma * np.random.normal(size=(N_part, d))

    def __center_and_clip_positions(self, positions, L):
        average_positions = np.mean(positions, axis=0)
        positions = np.clip(L+(positions-average_positions), 0, (2-1e-3)*L)

        return positions

    def dump_coordinates(self):
        return self.positions

    def dump_velocities(self):
        return self.drag_velocities, self.determisitic_velocities, self.random_velocities

    def OLDflow_field_function(self, positions):

        
        # positions has shape (N_part, d)
        p = positions - self.L

        f = np.zeros_like(p)
        f[:,1] = p[:,1]
        # f[:,0] = p[:,0]

        x1 = p[:,1] - self.L/15
        x2 = p[:,1] + self.L/15
        y = p[:,0] / self.L/2
        y2 = y**2

        n1 = np.sqrt(x1**2 + y2).reshape(-1,1)
        n2 = np.sqrt(x2**2 + y2).reshape(-1,1)

        f1 = np.zeros_like(p)
        f2 = np.zeros_like(p)

        f1[:,1] = -y
        f1[:,0] = x1

        f2[:,1] = y
        f2[:,0] = -x2

        f_tot = (f1/n1 + f2/n2) #- f * 0.1
        # f_tot = - f * 0.1

        f_tot = f_tot / np.linalg.norm(f_tot, axis=1).reshape(-1,1)
        self.f = f_tot + f
        # print(f)
        return f_tot * self.flow_field_function
    

    def flow_field_function(self, positions):
        return (self.matrix @ (positions-self.L).T).T



