import numpy as np
import numba


@numba.jit(nopython=True, parallel=True)
def neighbor_velocities(velocities,
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
    

def drag_velocities_function(parameters, sparse_distance_matrix,
                            positions, velocities,
                            drag_velocities_old):

    neigh_velocities = parameters.viscosity_regularization_coeff * neighbor_velocities(
        velocities, 
        sparse_distance_matrix.indices,
        sparse_distance_matrix.indptr
    )

    return neigh_velocities

    
    


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
    



@numba.jit(nopython=True,parallel=True)
def _LJ_interaction_forces_function(parameters, sparse_distance_matrix,
                                    positions, velocities, t, dt,
                                    random_forces):
    """
    Symmetrizing forces (setting Fij and Fji at the same
    time) is NOT faster !
    """

    N_part = positions.shape[0]    
    interaction_forces = np.zeros(positions.shape)

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

            indiv_forces_i = indiv_forces_i + force_magnitude*dist_vector_ij

        interaction_forces[i] = indiv_forces_i              

    return interaction_forces
    


def OU_random_forces_function(parameters, sparse_distance_matrix,
                            positions, velocities, t, dt,
                            random_forces_old):
    """
    Ornstein Uhlenbeck process. see Gillespie, PRE 95
    "Exact numerical simulation of the Ornstein-Uhlenbeck process and its integral"
    """

    random_noise = random_noise(
        sigma=np.sqrt(2*parameters.D),
        N_part=parameters.N_part, 
        d=parameters.d
    )

    deterministic_ou_term = np.exp(-dt/parameters.persistence_time)
    random_ou_term = random_noise * np.sqrt(1-np.exp(-2*dt/parameters.persistence_time))

    random_forces = random_forces_old * deterministic_ou_term + random_ou_term

    return random_forces



def random_noise(sigma, N_part, d):
    return sigma * np.random.normal(size=(N_part, d))