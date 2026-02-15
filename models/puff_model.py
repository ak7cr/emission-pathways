"""
Puff Dispersion Model
Track discrete expanding Gaussian puffs
"""
import numpy as np
from .base_model import BaseDispersionModel

class PuffModel(BaseDispersionModel):
    """Puff model - discrete Gaussian puffs that advect and grow"""
    
    def initialize(self, hotspots, npph):
        """Initialize with empty puff list"""
        self.initialized = True
        return {
            'puffs': [],  # List of puffs
            'next_puff_id': 0,
            'model_type': 'puff'
        }
    
    def step(self, state, t, dt, wind_U, wind_V):
        """Advect and grow all puffs"""
        import simulation_state
        
        puffs = state['puffs']
        sigma_turb = self.config.get('sigma_turb', 2.5)
        
        # Update each puff
        for puff in puffs[:]:  # Copy list to allow removal
            # Interpolate wind at puff location
            u_p = self._interpolate_wind(wind_U, puff['x'], puff['y'], simulation_state)
            v_p = self._interpolate_wind(wind_V, puff['x'], puff['y'], simulation_state)
            
            # Advect puff center
            puff['x'] += (u_p * dt) / 1000.0  # m/s to km
            puff['y'] += (v_p * dt) / 1000.0
            
            # Grow puff (diffusion)
            # σ² = σ₀² + 2Kt, where K = σ_turb²
            puff['age'] += dt
            K = sigma_turb**2  # m²/s
            puff['sigma'] = np.sqrt(puff['sigma_0']**2 + 2 * K * puff['age']) / 1000.0  # to km
            
            # Apply decay
            if self.config.get('enable_decay', False):
                decay_rate = self.config.get('decay_rate', 0.0)
                puff['mass'] *= np.exp(-decay_rate * dt)
            
            # Remove very small or old puffs
            if puff['mass'] < 1e-6 or puff['age'] > 7200:  # 2 hours max
                puffs.remove(puff)
            
            # Remove puffs that left domain
            if (puff['x'] < simulation_state.xmin or puff['x'] > simulation_state.xmax or
                puff['y'] < simulation_state.ymin or puff['y'] > simulation_state.ymax):
                puffs.remove(puff)
        
        state['puffs'] = puffs
        self.time = t
        return state
    
    def _interpolate_wind(self, wind_field, x, y, sim_state):
        """Bilinear interpolation of wind at point (x, y)"""
        # Find grid indices
        i = np.clip(int((x - sim_state.xmin) / (sim_state.xmax - sim_state.xmin) * sim_state.nx), 0, sim_state.nx-1)
        j = np.clip(int((y - sim_state.ymin) / (sim_state.ymax - sim_state.ymin) * sim_state.ny), 0, sim_state.ny-1)
        
        return wind_field[j, i]
    
    def get_concentration(self, state):
        """Calculate concentration from all puffs"""
        import simulation_state
        
        # Grid
        X, Y = np.meshgrid(simulation_state.x, simulation_state.y)
        C = np.zeros_like(X)
        
        # Add contribution from each puff
        for puff in state['puffs']:
            # Gaussian distribution
            r_squared = (X - puff['x'])**2 + (Y - puff['y'])**2
            sigma_sq = puff['sigma']**2
            
            # 2D Gaussian
            gaussian = np.exp(-r_squared / (2 * sigma_sq))
            
            # Normalize and scale by mass
            normalization = 2 * np.pi * sigma_sq
            C += (puff['mass'] / normalization) * gaussian
        
        # Convert to physical concentration
        cell_area = self.config.get('cell_area', 2777777.78)
        mixing_height = self.config.get('mixing_height', 1000.0)
        
        C_physical = C / (cell_area * mixing_height) * 1e6  # to µg/m³
        C_normalized = C / (C.max() + 1e-9)
        
        return C_normalized, C_physical
    
    def emit_particles(self, state, hotspots, npph):
        """Emit new puffs from hotspots"""
        mass_per_particle = self.config.get('mass_per_particle', 1.0)
        
        for hotspot in hotspots:
            # Create new puff
            puff = {
                'id': state['next_puff_id'],
                'x': hotspot[0] + np.random.normal(0, 0.5),  # Small initial spread
                'y': hotspot[1] + np.random.normal(0, 0.5),
                'mass': mass_per_particle * npph,  # Total mass in puff
                'sigma_0': 500.0,  # Initial spread (m)
                'sigma': 500.0 / 1000.0,  # Current spread (km)
                'age': 0.0,  # seconds
                'birth_time': self.time
            }
            
            state['puffs'].append(puff)
            state['next_puff_id'] += 1
        
        return state
    
    def get_info(self):
        """Get puff model statistics"""
        info = super().get_info()
        info.update({
            'description': 'Discrete Gaussian puff model',
            'advantages': 'Time-varying winds, intermittent sources',
            'computational_cost': 'Medium'
        })
        return info
