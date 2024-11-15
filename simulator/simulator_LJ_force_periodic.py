import numpy as np
import numba
from scipy.spatial import KDTree as scipy_KDTree
import simulator.utils as simulator_utils
from itertools import product


"""
TODO:
    - test if force symmetrization is faster
""" 




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

        sum_size_6 = np.power(sum_size, 6)
        sum_size_12 = np.power(sum_size_6, 2)
        
        if not r_sup_sum_size:
            r_minus_1 = 1/r
            r_minus_7 = np.power(r_minus_1, 7)


            repulsion = 12*r*r_minus_7*r_minus_7 * sum_size_12
            attraction = 6*r_minus_7 * sum_size_6

            # divide by r so that (positions[j]-positions[i]) can be 
            # directly multiplied by the magnitude without having to
            # compute the distance again 
            return 4 * eps * (attraction - repulsion) * r_minus_1

        r_sup_rbar = r > 1.33*sigma

        if r_sup_rbar:
            sigma_bar = 1.33*sigma - sum_size
            r_minus_1_bar = 1/(r-sigma_bar)
            r_minus_7_bar = np.power(r_minus_1_bar, 7)

            repulsion = 12*(r-sigma_bar)*r_minus_7_bar*r_minus_7_bar * sum_size_12
            attraction = 6*r_minus_7_bar * sum_size_6

            # divide by r so that (positions[j]-positions[i]) can be 
            # directly multiplied by the magnitude without having to
            # compute the distance again 
            return 4 * eps * (attraction - repulsion) / r

        return 0
    


def forces_numba_setup(parallel):
    
    @numba.jit(nopython=True,parallel=parallel)
    def individual_att_rep_forces(positions, 
                        dist_data,dist_indices,dist_indptr,
                        nuclei_sizes, eps, L):
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
                dist_vector_ij = positions[j]-positions[i] # needs to be wrapped
                dist_vector_ij = (dist_vector_ij + L/2) % L - L/2 

                indiv_forces_i = indiv_forces_i + force_magnitude*dist_vector_ij

            individual_forces[i] = indiv_forces_i              

        return individual_forces
    
    return individual_att_rep_forces



class FastOverdampedSimulator:
    def __init__(self, L, Nx, d, N_part, nuclei_sizes, viscosity, 
                 D, persistence_time, energy_potential,
                 initialization=None,
                 parallel=False, flow_field=None):

        self.t = 0
        self.Nx = Nx
        self.d = d
        self.N_part = N_part

        if isinstance(nuclei_sizes, int | float):
            nuclei_sizes = nuclei_sizes * np.ones(shape=(N_part,1),dtype=float)
        elif isinstance(nuclei_sizes, np.ndarray):
            if nuclei_sizes.ndim == 1:
                nuclei_sizes = nuclei_sizes[:,None]  
        self.nuclei_sizes = nuclei_sizes
        self.max_nuclei_size = np.max(nuclei_sizes)
        self.max_overall_distance = self.max_nuclei_size * 1.7/2**(1/6) * 2

        print(f'Max overall distance: {self.max_overall_distance:.2f}')

        packing_fraction = 0.906 if d==2 else 0.740 # hexagonal lattice packing fraction
        self.equilibrium_radius = np.mean(nuclei_sizes) * np.power(N_part/packing_fraction,1/d)
        if L is None:
            # Use hexagonal lattice packing fraction to infer
            # the total radius at equlibrium
            L = 2 * self.equilibrium_radius
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
        
        self.energy_potential = energy_potential

        self.individual_att_rep_forces = forces_numba_setup(parallel)

        if parallel:
            numba.set_num_threads(2)
            
        self.positions = self.__initialize_positions(L, N_part, self.equilibrium_radius, d, initialization)
        self.random_force = self.__initialize_random_noise(self.sigma_random, N_part, d)
        self.velocities = self.__initialize_velocities(
            self.positions,
            self.nuclei_sizes,
            self.max_overall_distance,
            self.energy_potential,
            self.random_force
        )

        self.flow_field = flow_field
        self.matrix = np.eye(d) * -1

        # foo = 'bar'

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
            energy_potential=self.energy_potential
        )
        # deterministic_forces = np.clip(deterministic_forces, -1000*self.energy_potential, 1000*self.energy_potential)
        
        F = deterministic_forces #+ self.random_force #! removed random force from 1st step
        F_norm = np.linalg.norm(F, axis=1).reshape(-1,1)
        F = np.divide(F, F_norm, out=np.zeros_like(F), where=F_norm!=0) * np.clip(F_norm, 0, 0.01)
        dummy_velocities = F/self.viscosity
        # drag_velocities = drag_velocities_from_neighbors(
        #     self.velocities, # use velocities at previous time steps (see Ya||a)
        #     sparse_distance_norms.indices,
        #     sparse_distance_norms.indptr
        # ) *0.9 
        # dummy_velocities = dummy_velocities + drag_velocities
        # if self.flow_field:
        #     dummy_velocities = dummy_velocities + self.__flow_field(self.positions)
        dummy_positions = self.positions + dt * dummy_velocities

        dummy_positions = self.__wrap_positions(
            dummy_positions,
            self.L
        )
        ###

        
        # ### Second step of Heun's method
        # self.positions = dummy_positions
        # self.velocities = dummy_velocities
        deterministic_forces, sparse_distance_norms = self.__attraction_repulsion_force(
            positions=dummy_positions,
            nuclei_sizes=self.nuclei_sizes,
            max_overall_distance=self.max_overall_distance,
            energy_potential=self.energy_potential
        )
        # deterministic_forces = np.clip(deterministic_forces, -1000*self.energy_potential, 1000*self.energy_potential)

        F = deterministic_forces + self.random_force 
        F_norm = np.linalg.norm(F, axis=1).reshape(-1,1)
        F = np.divide(F, F_norm, out=np.zeros_like(F), where=F_norm!=0) * np.clip(F_norm, 0, 0.01)
        velocities = F/self.viscosity
        # drag_velocities = drag_velocities_from_neighbors(
        #     dummy_velocities, # use velocities from first Heun's step
        #     sparse_distance_norms.indices,
        #     sparse_distance_norms.indptr
        # ) *0.9 
        # velocities = velocities + drag_velocities

        if self.flow_field:
            velocities = velocities + 0.01*self.__flow_field(dummy_positions)

        self.velocities = velocities
        self.positions = self.positions + dt * 0.5 * (dummy_velocities + velocities)
        ###

        self.determisitic_velocities = deterministic_forces / self.viscosity
        # self.drag_velocities = drag_velocities
        self.random_velocities = self.random_force / self.viscosity
        #/!\ DONT CENTER !
        # self.positions = self.__center_and_clip_positions(
        #     self.positions,
        #     self.L
        # )

        self.positions = self.__wrap_positions(
            self.positions,
            self.L
        )

        self.is_computing = False
    
    def __initialize_positions(self, L, N_part, equilibrium_radius, d, initialization):
        """
        Initialize particles positions on a uniform square grid
        """

        box_size = int(np.ceil(N_part**(1/d)))

        positions = np.meshgrid(*[np.linspace(0.05*L,0.95*L,box_size)]*d)
        positions = np.array([positions[i].flatten() for i in range(d)]).T

        positions = positions[:N_part]

        dx = 0.9 * L / box_size

        # import napari

        # viewer = napari.Viewer()
        # viewer.add_points(positions, size=1)

        # napari.run()

        noise = 0.1 * dx * (np.random.rand(*positions.shape)-0.5)

        positions = (positions + noise)

        return positions


    def __initialize_velocities(self, positions, nuclei_sizes, max_overall_distance, 
                                energy_potential, random_force):
        """
        Initialize velocities from interaction and random forces only
        """

        return np.zeros((self.N_part,self.d))

    def __initialize_random_noise(self, sigma, N_part, d):
        return self.__random_noise(sigma=sigma, N_part=N_part, d=d)


    def __attraction_repulsion_force(self, positions, nuclei_sizes, 
                                    max_overall_distance,
                                     energy_potential):
        
        tree = scipy_KDTree(positions, boxsize=self.L)

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
            L=self.L
        )

        return F, sparse_distance_norms

    def __random_noise(self, sigma, N_part, d):
        return sigma * np.random.normal(size=(N_part, d))

    def __wrap_positions(self, positions, L):
        return np.mod(positions, L)

    def __center_and_clip_positions(self, positions, L):
        average_positions = np.mean(positions, axis=0)
        positions = np.clip(L+(positions-average_positions), 0, (2-1e-3)*L)

        return positions

    def dump_array(self):
        raise NotImplementedError

    def dump_coordinates(self):
        return self.positions

    def dump_velocities(self):
        return self.drag_velocities, self.determisitic_velocities, self.random_velocities

    def OLD__flow_field(self, positions):

        
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
        return f_tot * self.flow_field
    

    def __flow_field(self, positions):

        return (self.matrix @ (positions-self.L).T).T
        
        
    def remove_all_particles(self):
        self.positions = np.zeros((0,self.d))
        self.velocities = np.zeros((0,self.d))
        self.nuclei_sizes = np.zeros((0,1))
        self.viscosity = np.zeros((0,1))
        self.D = np.zeros((0,1))
        self.persistence_time = np.zeros((0,1))
        self.random_force = np.zeros((0,self.d))

    def add_particles(self, positions, velocities=None, 
                      nuclei_sizes=None, viscosity=None, D=None,
                      persistence_time=None):

        N_new_part = positions.shape[0]
        
        if velocities is None:
            velocities = np.zeros_like(positions)
        if nuclei_sizes is None and self.nuclei_sizes.shape[0] > 0:
            nuclei_sizes = np.ones((N_new_part,1)) * np.mean(self.nuclei_sizes)
        else:
            nuclei_sizes = np.ones((N_new_part,1))
        if viscosity is None and self.viscosity.shape[0] > 0:
            viscosity = np.ones((N_new_part,1)) * np.mean(self.viscosity)
        else:
            viscosity = np.ones((N_new_part,1))
        if D is None and self.D.shape[0] > 0:
            D = np.ones((N_new_part,1)) * np.mean(self.D)
        else:
            D = np.ones((N_new_part,1))
        if persistence_time is None and self.persistence_time.shape[0] > 0:
            persistence_time = np.ones((N_new_part,1)) * np.mean(self.persistence_time)
        else:
            persistence_time = np.ones((N_new_part,1))
        
        self.positions = np.concatenate((self.positions, positions))
        self.velocities = np.concatenate((self.velocities, velocities))
        self.nuclei_sizes = np.concatenate((self.nuclei_sizes, nuclei_sizes))
        self.viscosity = np.concatenate((self.viscosity, viscosity))
        self.D = np.concatenate((self.D, D))
        self.persistence_time = np.concatenate((self.persistence_time, persistence_time))
        self.random_force = np.concatenate((self.random_force, self.__initialize_random_noise(
            sigma=np.sqrt(2*D),
            N_part=positions.shape[0], 
            d=self.d
        )))
        self.N_part = self.positions.shape[0]

        print('Now positions has shape', self.positions.shape)




