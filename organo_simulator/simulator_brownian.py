import numpy as np


class FastOverdampedSimulator:
    def __init__(self, L, d, N_part, viscosity, D,
                 parallel=False, 
                 out_of_view_rescue_strategy='clip',
                 lifetime=None,average_velocity_phi=0):

        self.d = d
        self.N_part = N_part

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

        self.positions = self.__initialize_positions(L, N_part, d)
        self.out_of_view_rescue_strategy = out_of_view_rescue_strategy

        self.particle_ids = np.arange(1, N_part+1)
        self.particle_lifetimes = np.random.randint(0, lifetime-4, N_part)

        self.langevin_force = self.__initialize_langevin_noise(self.sigma_langevin, N_part, d)

        self.average_velocity_phi = average_velocity_phi
        self.particle_velocities = self.__initialize_random_velocities(N_part, d, average_velocity_phi)
        


        self.t = 0

        self.lifetime = lifetime
        # foo = 'bar'

    def update_dynamics(self, dt):

        self.t += dt

        self.langevin_force = self.__langevin_noise(
                                sigma=self.sigma_langevin, 
                                N_part=self.N_part, 
                                d=self.d
                            ) * np.sqrt(dt)
        
  
        F = self.langevin_force

        velocities = F/self.viscosity + self.particle_velocities

        positions = self.positions + dt * velocities
        particle_ids = self.particle_ids.copy()
        particle_lifetimes = self.particle_lifetimes.copy()
        particle_velocities = self.particle_velocities.copy()

        if self.lifetime is not None:
            positions, particle_ids, particle_lifetimes, particle_velocities = self.__kill_old_particles(
                positions,
                particle_ids,
                particle_lifetimes,
                self.lifetime,
                particle_velocities
            )
        
        if self.out_of_view_rescue_strategy == 'wrap':
            positions, particle_ids = self.__wrap_positions(
                positions,
                self.L,
                particle_ids,
            )
    
        elif self.out_of_view_rescue_strategy == 'clip':
            positions = self.__clip_positions(
                positions,
                self.L,
            )

        elif self.out_of_view_rescue_strategy == 'kill':
            positions, particle_ids, particle_lifetimes, particle_velocities = self.__kill_outsiders(
                positions,
                self.L,
                particle_ids,
                particle_lifetimes,
                particle_velocities
            )

        elif self.out_of_view_rescue_strategy == None:
            pass

        else:
            raise NotImplementedError
        
        self.positions = positions
        self.particle_ids = particle_ids
        self.particle_lifetimes = particle_lifetimes + dt
        self.particle_velocities = particle_velocities


    def __initialize_positions(self, L, N_part, d):
        return np.random.rand(N_part, d)*L * 0.9 + L * 0.05

    def __initialize_langevin_noise(self, sigma, N_part, d):
        return self.__langevin_noise(sigma=sigma, N_part=N_part, d=d)
    
    def __initialize_random_velocities(self, N_part, d, average_velocity_phi):
        return average_velocity_phi * (np.random.rand(N_part, d) - 0.5)


    def __langevin_noise(self, sigma, N_part, d):
        return sigma * np.random.normal(size=(N_part, d))
    
    def __clip_positions(self, positions, L):
        positions = np.clip(positions, 0, (1-1e-3)*L)

        return positions
    
    def __wrap_positions(self, positions, L, previous_particle_ids):
        previous_particle_ids = previous_particle_ids.copy()

        indices_quadrant, wrapped_positions = np.divmod(positions, L)

        quadrant_changed = np.any(indices_quadrant != 0, axis=1)
        max_id = np.max(previous_particle_ids)

        # update particle ids
        n_new_particles = np.sum(quadrant_changed)
        previous_particle_ids[quadrant_changed] = np.arange(max_id+1, max_id+1+n_new_particles)

        # update velocities



        return wrapped_positions, previous_particle_ids
    
    def __kill_outsiders(self, positions, L, previous_particle_ids, 
                         previous_lifetimes, previous_velocities):

        previous_particle_ids = previous_particle_ids.copy()

        is_out_mask = np.any(positions < 0, axis=1) | np.any(positions > L, axis=1)

        n_out = np.sum(is_out_mask)

        if n_out > 0:
            new_positions = self.__initialize_positions(L, n_out, self.d)

            positions[is_out_mask] = new_positions

            max_id = np.max(previous_particle_ids)

            previous_particle_ids[is_out_mask] = np.arange(max_id+1, max_id+1+n_out)

            previous_lifetimes[is_out_mask] = 0

            previous_velocities[is_out_mask] = self.__initialize_random_velocities(n_out, self.d, self.average_velocity_phi)

        return positions, previous_particle_ids, previous_lifetimes, previous_velocities
    
    def __kill_old_particles(self, positions, particle_ids, particle_lifetimes, 
                             lifetime, particle_velocities):
        is_dead = particle_lifetimes >= lifetime
        n_dead = np.sum(is_dead)

        positions[is_dead] = self.__initialize_positions(self.L, n_dead, self.d)

        max_id = np.max(particle_ids)
        particle_ids[is_dead] = np.arange(max_id+1, max_id+1+n_dead)

        particle_lifetimes[is_dead] = 0

        particle_velocities[is_dead] = self.__initialize_random_velocities(n_dead, self.d, self.average_velocity_phi)

        return positions, particle_ids, particle_lifetimes, particle_velocities



        

        