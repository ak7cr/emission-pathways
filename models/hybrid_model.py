"""
Hybrid Dispersion Model
Lagrangian particles near source + Eulerian grid for far field
Best of both worlds: accuracy near source, efficiency far away
"""
import numpy as np
from .base_model import BaseDispersionModel

class HybridModel(BaseDispersionModel):
    """Hybrid model combining Lagrangian and Eulerian approaches"""
    
    def __init__(self, config):
        super().__init__(config)
        # Transition distance (particles -> grid)
        self.transition_distance = config.get('transition_distance', 10.0)  # km
    
    def initialize(self, hotspots, npph):
        """Initialize both particles and concentration field"""
        from particle_physics import initialize_particles
        import simulation_state
        
        # Lagrangian particles
        particles, particle_active = initialize_particles(hotspots, npph)
        
        # Eulerian grid
        C = np.zeros((simulation_state.ny, simulation_state.nx))
        
        self.initialized = True
        return {
            'particles': particles,
            'particle_active': particle_active,
            'concentration': C,
            'hotspots': hotspots,
            'model_type': 'hybrid'
        }
    
    def step(self, state, t, dt, wind_U, wind_V):
        """
        1. Advect particles (Lagrangian) near sources
        2. Transfer far particles to grid
        3. Advect grid (Eulerian)
        """
        from particle_physics import advect
        from wind_field import get_wind_at_particles
        import simulation_state
        
        particles = state['particles']
        particle_active = state['particle_active']
        C = state['concentration']
        hotspots = state['hotspots']
        
        # Identify near-field and far-field particles
        distances = self._calculate_distances_from_sources(particles, hotspots)
        near_field = distances < self.transition_distance
        far_field = (distances >= self.transition_distance) & particle_active
        
        # Count transfers
        n_transferred = far_field.sum()
        
        # 1. Transfer far-field particles to grid
        if n_transferred > 0:
            C = self._transfer_particles_to_grid(
                particles[far_field],
                C,
                simulation_state
            )
            # Deactivate transferred particles
            particle_active[far_field] = False
        
        # 2. Advect near-field particles (Lagrangian)
        if (particle_active & near_field).sum() > 0:
            sigma_turb = self.config.get('sigma_turb', 2.5)
            boundary_type = self.config.get('boundary_type', 'absorbing')
            wind_interpolators = self.config.get('wind_interpolators', {
                'U_interp': None, 'V_interp': None
            })
            
            particles = advect(
                particles,
                particle_active,
                t, dt, sigma_turb,
                'synthetic',
                wind_interpolators,
                boundary_type,
                self.config.get('enable_deposition', True),
                self.config.get('deposition_velocity', 0.001),
                self.config.get('mixing_height', 1000.0),
                self.config.get('enable_decay', False),
                self.config.get('decay_rate', 0.0),
                simulation_state.x, simulation_state.y,
                simulation_state.nx, simulation_state.ny,
                simulation_state.xmin, simulation_state.xmax,
                simulation_state.ymin, simulation_state.ymax,
                {},
                get_wind_at_particles,
                simulation_state.KM_TO_M,
                True, 1.0
            )
        
        # 3. Advect concentration field (Eulerian)
        C = self._eulerian_step(C, wind_U, wind_V, dt, simulation_state)
        
        state['particles'] = particles
        state['particle_active'] = particle_active
        state['concentration'] = C
        self.time = t
        
        return state
    
    def _calculate_distances_from_sources(self, particles, hotspots):
        """Calculate minimum distance of each particle from any source"""
        distances = np.full(len(particles), np.inf)
        
        for hotspot in hotspots:
            hx, hy = hotspot
            dx = particles[:, 0] - hx
            dy = particles[:, 1] - hy
            dist = np.sqrt(dx**2 + dy**2)
            distances = np.minimum(distances, dist)
        
        return distances
    
    def _transfer_particles_to_grid(self, particles, C, sim_state):
        """Deposit particle mass onto Eulerian grid"""
        # Grid cell size
        dx = (sim_state.xmax - sim_state.xmin) / sim_state.nx
        dy = (sim_state.ymax - sim_state.ymin) / sim_state.ny
        
        # Convert particle positions to grid indices
        i_indices = ((particles[:, 0] - sim_state.xmin) / dx).astype(int)
        j_indices = ((particles[:, 1] - sim_state.ymin) / dy).astype(int)
        
        # Clip to grid bounds
        i_indices = np.clip(i_indices, 0, sim_state.nx - 1)
        j_indices = np.clip(j_indices, 0, sim_state.ny - 1)
        
        # Add particle mass to grid cells (weighted by mass fraction)
        for k in range(len(particles)):
            C[j_indices[k], i_indices[k]] += particles[k, 2]  # mass fraction
        
        return C
    
    def _eulerian_step(self, C, U, V, dt, sim_state):
        """Eulerian advection-diffusion step"""
        dx = (sim_state.xmax - sim_state.xmin) / sim_state.nx * 1000  # to m
        dy = (sim_state.ymax - sim_state.ymin) / sim_state.ny * 1000
        
        # Upwind advection
        U_pos = np.maximum(U, 0)
        U_neg = np.minimum(U, 0)
        V_pos = np.maximum(V, 0)
        V_neg = np.minimum(V, 0)
        
        dC_dx_fwd = (C - np.roll(C, 1, axis=1)) / dx
        dC_dx_bwd = (np.roll(C, -1, axis=1) - C) / dx
        dC_dy_fwd = (C - np.roll(C, 1, axis=0)) / dy
        dC_dy_bwd = (np.roll(C, -1, axis=0) - C) / dy
        
        C_new = C - dt * (U_pos * dC_dx_fwd + U_neg * dC_dx_bwd +
                          V_pos * dC_dy_fwd + V_neg * dC_dy_bwd)
        
        # Diffusion
        K = self.config.get('sigma_turb', 2.5)**2
        laplacian = (
            (np.roll(C, -1, axis=1) - 2*C + np.roll(C, 1, axis=1)) / dx**2 +
            (np.roll(C, -1, axis=0) - 2*C + np.roll(C, 1, axis=0)) / dy**2
        )
        C_new = C_new + dt * K * laplacian
        
        return np.maximum(C_new, 0)
    
    def get_concentration(self, state):
        """Combine particle and grid concentrations"""
        from particle_physics import concentration_field
        import simulation_state
        
        # Concentration from particles (Lagrangian)
        mass_per_particle = self.config.get('mass_per_particle', 1.0)
        cell_area = self.config.get('cell_area', 2777777.78)
        mixing_height = self.config.get('mixing_height', 1000.0)
        
        C_particles_norm, C_particles_phys = concentration_field(
            state['particles'],
            state['particle_active'],
            mass_per_particle,
            cell_area,
            mixing_height,
            simulation_state.nx, simulation_state.ny,
            simulation_state.xmin, simulation_state.xmax,
            simulation_state.ymin, simulation_state.ymax
        )
        
        # Concentration from grid (Eulerian)
        C_grid = state['concentration']
        npph = self.config.get('npph', 2500)
        total_mass = mass_per_particle * npph * len(state['hotspots'])
        C_grid_phys = C_grid * total_mass / (cell_area * mixing_height)
        
        # Combine (sum both contributions)
        C_total_phys = C_particles_phys + C_grid_phys
        C_total_norm = C_total_phys / (C_total_phys.max() + 1e-9)
        
        return C_total_norm, C_total_phys
    
    def emit_particles(self, state, hotspots, npph):
        """Emit new particles (only Lagrangian component)"""
        from particle_physics import emit_new_particles
        
        state['particles'], state['particle_active'] = emit_new_particles(
            state['particles'],
            state['particle_active'],
            hotspots,
            npph
        )
        state['hotspots'] = hotspots
        return state
    
    def get_info(self):
        """Get hybrid model statistics"""
        info = super().get_info()
        info.update({
            'description': 'Hybrid Lagrangian-Eulerian model',
            'advantages': 'Accuracy near source, efficiency far away, best of both',
            'computational_cost': 'Medium',
            'transition_distance_km': self.transition_distance
        })
        return info
