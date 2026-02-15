"""
Gaussian Plume Dispersion Model
Analytical steady-state solution for continuous point sources
"""
import numpy as np
from .base_model import BaseDispersionModel

class GaussianPlumeModel(BaseDispersionModel):
    """Gaussian plume analytical model (Pasquill-Gifford)"""
    
    # Pasquill-Gifford dispersion parameters
    # Stability classes: A (very unstable) to F (very stable)
    STABILITY_PARAMS = {
        'A': {'sigma_y': (0.22, 0.894), 'sigma_z': (0.20, 0.894)},  # Very unstable
        'B': {'sigma_y': (0.16, 0.894), 'sigma_z': (0.12, 0.894)},  # Unstable
        'C': {'sigma_y': (0.11, 0.894), 'sigma_z': (0.08, 0.894)},  # Slightly unstable
        'D': {'sigma_y': (0.08, 0.894), 'sigma_z': (0.06, 0.894)},  # Neutral
        'E': {'sigma_y': (0.06, 0.894), 'sigma_z': (0.03, 0.894)},  # Slightly stable
        'F': {'sigma_y': (0.04, 0.894), 'sigma_z': (0.016, 0.894)}, # Stable
    }
    
    def initialize(self, hotspots, npph):
        """Initialize plume sources"""
        self.initialized = True
        return {
            'sources': hotspots,
            'emission_rate': npph,  # Use npph as emission rate parameter
            'model_type': 'gaussian_plume'
        }
    
    def step(self, state, t, dt, wind_U, wind_V):
        """Plume is steady-state, just update time"""
        self.time = t
        return state
    
    def get_concentration(self, state):
        """Calculate Gaussian plume concentration field"""
        import simulation_state
        
        # Grid
        X, Y = np.meshgrid(simulation_state.x, simulation_state.y)
        C = np.zeros_like(X)
        
        # Average wind speed
        wind_speed = self.config.get('wind_speed_multiplier', 1.0) * 10.0  # m/s
        wind_speed = max(wind_speed, 0.5)  # Minimum wind speed
        
        # Stability class
        stability = self.config.get('stability_class', 'D')
        
        # Emission parameters
        mass_per_particle = self.config.get('mass_per_particle', 1.0)
        npph = state.get('emission_rate', 2500)
        Q = mass_per_particle * npph / 1e6  # g/s (converted from µg)
        
        # Source height
        H = self.config.get('source_height', 10.0)  # meters
        
        # Calculate concentration from each source
        for source in state['sources']:
            x0, y0 = source
            
            # Transform to plume coordinates (downwind from source)
            # Simplified: assume wind in +x direction
            dx = (X - x0) * 1000  # km to m
            dy = (Y - y0) * 1000
            
            # Only calculate for downwind locations (x > x0)
            downwind = dx > 0
            
            if downwind.any():
                # Distance downwind
                x_down = dx[downwind]
                y_cross = dy[downwind]
                
                # Calculate dispersion parameters
                sigma_y = self._calculate_sigma(x_down, 'sigma_y', stability)
                sigma_z = self._calculate_sigma(x_down, 'sigma_z', stability)
                
                # Gaussian plume formula (ground-level concentration)
                # C = (Q / (π * u * σy * σz)) * exp(-y²/(2σy²)) * exp(-H²/(2σz²))
                
                C_plume = (Q / (np.pi * wind_speed * sigma_y * sigma_z)) * \
                          np.exp(-y_cross**2 / (2 * sigma_y**2)) * \
                          np.exp(-H**2 / (2 * sigma_z**2))
                
                # Add to concentration field
                C_temp = np.zeros_like(X)
                C_temp[downwind] = C_plume
                C += C_temp
        
        # Convert to µg/m³
        C_physical = C * 1e6  # g/m³ to µg/m³
        
        # Normalized
        C_normalized = C_physical / (C_physical.max() + 1e-9)
        
        return C_normalized, C_physical
    
    def _calculate_sigma(self, x, param_type, stability):
        """
        Calculate dispersion parameter using Pasquill-Gifford curves
        
        Parameters:
        -----------
        x : ndarray
            Downwind distance (meters)
        param_type : str
            'sigma_y' or 'sigma_z'
        stability : str
            Stability class (A-F)
        
        Returns:
        --------
        sigma : ndarray
            Dispersion parameter (meters)
        """
        if stability not in self.STABILITY_PARAMS:
            stability = 'D'  # Default to neutral
        
        a, b = self.STABILITY_PARAMS[stability][param_type]
        
        # Power law: σ = a * x^b (x in km, σ in meters)
        x_km = x / 1000.0
        sigma = a * (x_km ** b) * 1000.0  # Convert back to meters
        
        # Minimum value to avoid division by zero
        sigma = np.maximum(sigma, 1.0)
        
        return sigma
    
    def emit_particles(self, state, hotspots, npph):
        """Update source locations and emission rate"""
        state['sources'] = hotspots
        state['emission_rate'] = npph
        return state
    
    def get_info(self):
        """Get Gaussian plume model statistics"""
        info = super().get_info()
        info.update({
            'description': 'Gaussian plume analytical solution',
            'advantages': 'Extremely fast, no numerical errors, regulatory standard',
            'computational_cost': 'Very Low',
            'limitations': 'Steady-state only, uniform wind, flat terrain'
        })
        return info
