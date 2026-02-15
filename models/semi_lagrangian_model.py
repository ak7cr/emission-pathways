"""
Semi-Lagrangian Dispersion Model
Backward trajectory on grid - combines benefits of Lagrangian and Eulerian
"""
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from .base_model import BaseDispersionModel

class SemiLagrangianModel(BaseDispersionModel):
    """Semi-Lagrangian advection with Eulerian diffusion"""
    
    def initialize(self, hotspots, npph):
        """Initialize concentration field"""
        import simulation_state
        
        # Create concentration field
        C = np.zeros((simulation_state.ny, simulation_state.nx))
        
        # Initialize at hotspots
        dx = (simulation_state.xmax - simulation_state.xmin) / simulation_state.nx
        X, Y = np.meshgrid(simulation_state.x, simulation_state.y)
        
        for hotspot in hotspots:
            hx, hy = hotspot
            sigma_init = 2.0 * dx
            gaussian = np.exp(-((X - hx)**2 + (Y - hy)**2) / (2 * sigma_init**2))
            C += gaussian
        
        if C.sum() > 0:
            C = C / C.sum()
        
        self.initialized = True
        return {
            'concentration': C,
            'model_type': 'semi_lagrangian'
        }
    
    def step(self, state, t, dt, wind_U, wind_V):
        """Semi-Lagrangian step: backward trajectories + diffusion"""
        import simulation_state
        
        C = state['concentration']
        
        # Grid spacing
        dx = (simulation_state.xmax - simulation_state.xmin) / simulation_state.nx
        dy = (simulation_state.ymax - simulation_state.ymin) / simulation_state.ny
        
        # 1. Semi-Lagrangian advection (backward trajectories)
        C = self._semi_lagrangian_advect(C, wind_U, wind_V, dt, simulation_state)
        
        # 2. Diffusion step (Eulerian)
        sigma_turb = self.config.get('sigma_turb', 2.5)
        K = sigma_turb**2  # Diffusion coefficient
        
        dx_m = dx * 1000  # km to m
        dy_m = dy * 1000
        
        laplacian = (
            (np.roll(C, -1, axis=1) - 2*C + np.roll(C, 1, axis=1)) / dx_m**2 +
            (np.roll(C, -1, axis=0) - 2*C + np.roll(C, 1, axis=0)) / dy_m**2
        )
        C = C + dt * K * laplacian
        C = np.maximum(C, 0)
        
        # 3. Apply losses
        if self.config.get('enable_deposition', True):
            mixing_height = self.config.get('mixing_height', 1000.0)
            deposition_velocity = self.config.get('deposition_velocity', 0.001)
            lambda_dep = deposition_velocity / mixing_height
            C = C * np.exp(-lambda_dep * dt)
        
        state['concentration'] = C
        self.time = t
        return state
    
    def _semi_lagrangian_advect(self, C, U, V, dt, sim_state):
        """Semi-Lagrangian advection using backward trajectories"""
        # Create interpolator for current concentration
        interp = RegularGridInterpolator(
            (sim_state.y, sim_state.x),
            C,
            method='linear',
            bounds_error=False,
            fill_value=0.0
        )
        
        # Grid points
        X, Y = np.meshgrid(sim_state.x, sim_state.y)
        
        # Backward trajectories (departure points)
        # x_dep = x_arrival - u * dt
        X_dep = X - (U * dt) / 1000.0  # m/s to km
        Y_dep = Y - (V * dt) / 1000.0
        
        # Interpolate concentration at departure points
        points = np.column_stack([Y_dep.ravel(), X_dep.ravel()])
        C_new = interp(points).reshape(C.shape)
        
        return C_new
    
    def get_concentration(self, state):
        """Get concentration field"""
        C = state['concentration']
        
        # Physical concentration
        mass_per_particle = self.config.get('mass_per_particle', 1.0)
        npph = self.config.get('npph', 2500)
        hotspots = self.config.get('hotspots', [[10, 30]])
        total_mass = mass_per_particle * npph * len(hotspots)
        
        cell_area = self.config.get('cell_area', 2777777.78)
        mixing_height = self.config.get('mixing_height', 1000.0)
        
        C_physical = C * total_mass / (cell_area * mixing_height)
        C_normalized = C / (C.max() + 1e-9)
        
        return C_normalized, C_physical
    
    def emit_particles(self, state, hotspots, npph):
        """Add mass at hotspots"""
        import simulation_state
        
        C = state['concentration']
        dx = (simulation_state.xmax - simulation_state.xmin) / simulation_state.nx
        X, Y = np.meshgrid(simulation_state.x, simulation_state.y)
        
        for hotspot in hotspots:
            hx, hy = hotspot
            sigma_emit = 1.0 * dx
            gaussian = np.exp(-((X - hx)**2 + (Y - hy)**2) / (2 * sigma_emit**2))
            C += gaussian * 0.01
        
        state['concentration'] = C
        return state
    
    def get_info(self):
        """Get Semi-Lagrangian model statistics"""
        info = super().get_info()
        info.update({
            'description': 'Semi-Lagrangian backward trajectory',
            'advantages': 'Reduced numerical diffusion, stable with large timesteps',
            'computational_cost': 'Medium'
        })
        return info
