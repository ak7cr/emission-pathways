"""
Lagrangian Particle Dispersion Model
Track individual particles through space and time
"""
import numpy as np
from .base_model import BaseDispersionModel

class LagrangianModel(BaseDispersionModel):
    """Lagrangian particle tracking model"""
    
    def initialize(self, hotspots, npph):
        """Initialize particles around hotspots"""
        from particle_physics import initialize_particles
        particles, particle_active = initialize_particles(hotspots, npph)
        
        self.initialized = True
        return {
            'particles': particles,
            'particle_active': particle_active,
            'model_type': 'lagrangian'
        }
    
    def step(self, state, t, dt, wind_U, wind_V):
        """Advect particles with turbulent diffusion"""
        from particle_physics import advect
        from wind_field import get_wind_at_particles
        import simulation_state
        
        # Extract configuration
        sigma_turb = self.config.get('sigma_turb', 2.5)
        boundary_type = self.config.get('boundary_type', 'absorbing')
        enable_deposition = self.config.get('enable_deposition', True)
        deposition_velocity = self.config.get('deposition_velocity', 0.001)
        mixing_height = self.config.get('mixing_height', 1000.0)
        enable_decay = self.config.get('enable_decay', False)
        decay_rate = self.config.get('decay_rate', 0.0)
        use_bilinear_interp = self.config.get('use_bilinear_interp', True)
        wind_speed_multiplier = self.config.get('wind_speed_multiplier', 1.0)
        
        # Wind interpolators cache
        wind_interpolators = self.config.get('wind_interpolators', {
            'U_interp': None,
            'V_interp': None
        })
        
        # Advect particles
        state['particles'] = advect(
            state['particles'],
            state['particle_active'],
            t,
            dt,
            sigma_turb,
            'synthetic',  # Wind type handled externally
            wind_interpolators,
            boundary_type,
            enable_deposition,
            deposition_velocity,
            mixing_height,
            enable_decay,
            decay_rate,
            simulation_state.x,
            simulation_state.y,
            simulation_state.nx,
            simulation_state.ny,
            simulation_state.xmin,
            simulation_state.xmax,
            simulation_state.ymin,
            simulation_state.ymax,
            {},  # wind_data_cache
            get_wind_at_particles,
            simulation_state.KM_TO_M,
            use_bilinear_interp,
            wind_speed_multiplier
        )
        
        self.time = t
        return state
    
    def get_concentration(self, state):
        """Calculate concentration from particle distribution"""
        from particle_physics import concentration_field
        import simulation_state
        
        mass_per_particle = self.config.get('mass_per_particle', 1.0)
        cell_area = self.config.get('cell_area', 2777777.78)
        mixing_height = self.config.get('mixing_height', 1000.0)
        
        return concentration_field(
            state['particles'],
            state['particle_active'],
            mass_per_particle,
            cell_area,
            mixing_height,
            simulation_state.nx,
            simulation_state.ny,
            simulation_state.xmin,
            simulation_state.xmax,
            simulation_state.ymin,
            simulation_state.ymax
        )
    
    def emit_particles(self, state, hotspots, npph):
        """Emit new particles from hotspots"""
        from particle_physics import emit_new_particles
        
        state['particles'], state['particle_active'] = emit_new_particles(
            state['particles'],
            state['particle_active'],
            hotspots,
            npph
        )
        return state
    
    def get_info(self):
        """Get Lagrangian model statistics"""
        info = super().get_info()
        info.update({
            'description': 'Lagrangian particle tracking',
            'advantages': 'Excellent for point sources, mass conservative',
            'computational_cost': 'Medium-High'
        })
        return info
