"""
Eulerian Grid-Based Dispersion Model
Solve advection-diffusion PDE on fixed grid
"""
import numpy as np
from scipy import ndimage
from .base_model import BaseDispersionModel

class EulerianModel(BaseDispersionModel):
    """Eulerian finite difference model for advection-diffusion"""
    
    def initialize(self, hotspots, npph):
        """Initialize concentration field on grid"""
        import simulation_state
        
        # Create concentration field
        C = np.zeros((simulation_state.ny, simulation_state.nx))
        
        # Initialize with Gaussian distributions at hotspots
        dx = (simulation_state.xmax - simulation_state.xmin) / simulation_state.nx
        dy = (simulation_state.ymax - simulation_state.ymin) / simulation_state.ny
        
        X, Y = np.meshgrid(simulation_state.x, simulation_state.y)
        
        # Add Gaussian blob at each hotspot
        for hotspot in hotspots:
            hx, hy = hotspot
            # Initial spread (2 grid cells)
            sigma_init = 2.0 * dx
            gaussian = np.exp(-((X - hx)**2 + (Y - hy)**2) / (2 * sigma_init**2))
            C += gaussian
        
        # Normalize to total mass
        if C.sum() > 0:
            C = C / C.sum()
        
        self.initialized = True
        return {
            'concentration': C,
            'model_type': 'eulerian'
        }
    
    def step(self, state, t, dt, wind_U, wind_V):
        """Solve advection-diffusion equation"""
        import simulation_state
        
        C = state['concentration']
        
        # Grid spacing (in meters)
        dx = (simulation_state.xmax - simulation_state.xmin) / simulation_state.nx * 1000  # km to m
        dy = (simulation_state.ymax - simulation_state.ymin) / simulation_state.ny * 1000
        
        # Diffusion coefficient from turbulence
        sigma_turb = self.config.get('sigma_turb', 2.5)  # m/s
        K = sigma_turb**2  # m²/s (simplified diffusion coefficient)
        
        # Stability check (CFL condition)
        max_u = np.abs(wind_U).max()
        max_v = np.abs(wind_V).max()
        cfl_advect = max(max_u * dt / dx, max_v * dt / dy)
        cfl_diffusion = 2 * K * dt / min(dx, dy)**2
        
        # Adaptive sub-stepping if needed
        n_substeps = max(1, int(np.ceil(max(cfl_advect, cfl_diffusion) / 0.5)))
        dt_sub = dt / n_substeps
        
        for _ in range(n_substeps):
            # 1. Advection (upwind scheme)
            C = self._advection_step(C, wind_U, wind_V, dt_sub, dx, dy)
            
            # 2. Diffusion (central differences)
            C = self._diffusion_step(C, K, dt_sub, dx, dy)
            
            # 3. Apply loss processes
            C = self._apply_losses(C, dt_sub)
        
        # Apply boundary conditions
        C = self._apply_boundaries(C)
        
        state['concentration'] = C
        self.time = t
        return state
    
    def _advection_step(self, C, U, V, dt, dx, dy):
        """Upwind advection scheme"""
        C_new = C.copy()
        
        # U advection (x-direction)
        U_pos = np.maximum(U, 0)
        U_neg = np.minimum(U, 0)
        
        # Forward difference for positive velocity
        dC_dx_fwd = (C - np.roll(C, 1, axis=1)) / dx
        # Backward difference for negative velocity
        dC_dx_bwd = (np.roll(C, -1, axis=1) - C) / dx
        
        C_new -= dt * (U_pos * dC_dx_fwd + U_neg * dC_dx_bwd)
        
        # V advection (y-direction)
        V_pos = np.maximum(V, 0)
        V_neg = np.minimum(V, 0)
        
        # Forward difference for positive velocity
        dC_dy_fwd = (C - np.roll(C, 1, axis=0)) / dy
        # Backward difference for negative velocity
        dC_dy_bwd = (np.roll(C, -1, axis=0) - C) / dy
        
        C_new -= dt * (V_pos * dC_dy_fwd + V_neg * dC_dy_bwd)
        
        return np.maximum(C_new, 0)  # Ensure non-negative
    
    def _diffusion_step(self, C, K, dt, dx, dy):
        """Central difference diffusion"""
        # Laplacian operator
        laplacian = (
            (np.roll(C, -1, axis=1) - 2*C + np.roll(C, 1, axis=1)) / dx**2 +
            (np.roll(C, -1, axis=0) - 2*C + np.roll(C, 1, axis=0)) / dy**2
        )
        
        C_new = C + dt * K * laplacian
        return np.maximum(C_new, 0)
    
    def _apply_losses(self, C, dt):
        """Apply deposition and decay"""
        if self.config.get('enable_deposition', True):
            mixing_height = self.config.get('mixing_height', 1000.0)
            deposition_velocity = self.config.get('deposition_velocity', 0.001)
            lambda_dep = deposition_velocity / mixing_height
            C = C * np.exp(-lambda_dep * dt)
        
        if self.config.get('enable_decay', False):
            decay_rate = self.config.get('decay_rate', 0.0)
            C = C * np.exp(-decay_rate * dt)
        
        return C
    
    def _apply_boundaries(self, C):
        """Apply boundary conditions"""
        boundary_type = self.config.get('boundary_type', 'absorbing')
        
        if boundary_type == 'absorbing':
            # Zero at boundaries
            C[0, :] = 0
            C[-1, :] = 0
            C[:, 0] = 0
            C[:, -1] = 0
        elif boundary_type == 'reflecting':
            # Mirror at boundaries
            C[0, :] = C[1, :]
            C[-1, :] = C[-2, :]
            C[:, 0] = C[:, 1]
            C[:, -1] = C[:, -2]
        # periodic is handled automatically by np.roll
        
        return C
    
    def get_concentration(self, state):
        """Get concentration field"""
        C = state['concentration']
        
        # Physical concentration
        mass_per_particle = self.config.get('mass_per_particle', 1.0)
        cell_area = self.config.get('cell_area', 2777777.78)
        mixing_height = self.config.get('mixing_height', 1000.0)
        
        # Total mass in system (sum of normalized concentration)
        total_mass = mass_per_particle * self.config.get('npph', 2500) * len(self.config.get('hotspots', [[10, 30]]))
        
        C_physical = C * total_mass / (cell_area * mixing_height)
        C_normalized = C / (C.max() + 1e-9)
        
        return C_normalized, C_physical
    
    def emit_particles(self, state, hotspots, npph):
        """Add mass at hotspot locations"""
        import simulation_state
        
        C = state['concentration']
        
        # Add Gaussian at each hotspot
        dx = (simulation_state.xmax - simulation_state.xmin) / simulation_state.nx
        X, Y = np.meshgrid(simulation_state.x, simulation_state.y)
        
        for hotspot in hotspots:
            hx, hy = hotspot
            sigma_emit = 1.0 * dx  # Emission spread
            gaussian = np.exp(-((X - hx)**2 + (Y - hy)**2) / (2 * sigma_emit**2))
            C += gaussian * 0.01  # Add small amount
        
        state['concentration'] = C
        return state
    
    def get_info(self):
        """Get Eulerian model statistics"""
        info = super().get_info()
        info.update({
            'description': 'Eulerian finite difference PDE solver',
            'advantages': 'Fast for large domains, no statistical noise',
            'computational_cost': 'Low-Medium'
        })
        return info
