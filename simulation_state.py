"""
Simulation state management and configuration
"""
import numpy as np

# Domain constants (in km for display, converted to meters for physics)
xmin, xmax, ymin, ymax = 0.0, 200.0, 0.0, 200.0
nx, ny = 120, 120
x = np.linspace(xmin, xmax, nx)
y = np.linspace(ymin, ymax, ny)

# Unit conversion factor
KM_TO_M = 1000.0  # meters per kilometer

def calculate_cell_area():
    """Calculate grid cell area in m²"""
    dx_km = (xmax - xmin) / nx  # km
    dy_km = (ymax - ymin) / ny  # km
    dx_m = dx_km * KM_TO_M  # m
    dy_m = dy_km * KM_TO_M  # m
    return dx_m * dy_m  # m²

# Global simulation state
sim_state = {
    'hotspots': [[40.0, 120.0], [60.0, 130.0], [30.0, 90.0]],
    'sigma_turb': 2.5,  # Increased turbulence for more visible diffusion
    'npph': 2500,
    'dt': 30.0,  # Increased timestep from 0.2 to 30 seconds for faster simulation
    'current_frame': 0,
    'particles': None,
    'wind_type': 'synthetic',  # or 'real' when ERA5/GFS data is available
    'is_playing': False,
    'wind_data': None,  # Cache for loaded wind data
    'show_wind_vectors': True,  # Show wind vectors by default
    # Physical parameters for concentration calculation
    'mixing_height': 1000.0,  # Mixing layer height in meters
    'cell_area': calculate_cell_area(),
    'mass_per_particle': 1.0,  # Mass per particle in µg (will be calculated)
    # Boundary conditions and loss processes
    'boundary_type': 'absorbing',  # 'absorbing', 'reflecting', or 'periodic'
    'enable_deposition': True,  # Enable dry deposition
    'deposition_velocity': 0.001,  # Dry deposition velocity in m/s (typical for PM2.5)
    'enable_decay': False,  # Enable chemical decay/loss
    'decay_rate': 0.0,  # Decay rate constant (1/s), e.g., 1/(24*3600) for 24h lifetime
    'particle_active': None,  # Boolean array tracking active particles
    # Performance optimization
    'super_particle_ratio': 1,  # Each computational particle represents this many physical particles
    'use_bilinear_interp': True,  # Use bilinear interpolation for wind (more accurate)
}

# Wind field interpolators (cached for performance)
wind_interpolators = {
    'U_interp': None,
    'V_interp': None,
    'last_update_time': -1
}

# Emission metadata for each hotspot
emission_data = {
    'hotspot_emissions': [],  # List of emission info for each hotspot
    'pollutant_type': 'PM2.5',  # Default pollutant
    'emission_factor': 1.5,  # g/kg fuel burned (example for PM2.5)
    'total_mass_per_hotspot': 1000.0,  # grams per hotspot emission event
}

# Wind data cache
wind_data_cache = {
    'loaded': False,
    'time_array': None,
    'u_data': None,
    'v_data': None,
    'interpolator_u': None,
    'interpolator_v': None,
    'x_wind': None,
    'y_wind': None
}

def update_mass_per_particle():
    """
    Update mass per particle based on total emission and particle count
    
    When using super-particles, each computational particle represents
    multiple physical particles to improve performance.
    """
    total_computational_particles = sim_state['npph'] * len(sim_state['hotspots'])
    super_ratio = sim_state.get('super_particle_ratio', 1)
    
    if total_computational_particles > 0:
        # Convert total mass from grams to micrograms
        total_mass_ug = emission_data['total_mass_per_hotspot'] * len(sim_state['hotspots']) * 1e6
        
        # Each computational particle represents super_ratio physical particles
        sim_state['mass_per_particle'] = (total_mass_ug / total_computational_particles) * super_ratio
    return sim_state['mass_per_particle']
