from flask import Flask, render_template, jsonify, request, send_file
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
import io
import base64
import json
from scipy.interpolate import RegularGridInterpolator
from datetime import datetime, timedelta
import os
from wind_api import WindDataFetcher
from emission_utils import (frp_to_emission, get_emission_factor, 
                            list_pollutants, list_vegetation_types, 
                            EMISSION_FACTORS)

# Optional: Uncomment for significant performance boost with large particle counts
# try:
#     from numba import jit
#     HAS_NUMBA = True
# except ImportError:
#     HAS_NUMBA = False
#     # Define dummy decorator if numba not available
#     def jit(*args, **kwargs):
#         def decorator(func):
#             return func
#         return decorator if args and callable(args[0]) else decorator

app = Flask(__name__)
wind_fetcher = WindDataFetcher()

# Global simulation state
sim_state = {
    'hotspots': [[40.0, 120.0], [60.0, 130.0], [30.0, 90.0]],
    'sigma_turb': 0.9,
    'npph': 2500,
    'dt': 0.2,
    'current_frame': 0,
    'particles': None,
    'wind_type': 'synthetic',  # or 'real' when ERA5/GFS data is available
    'is_playing': False,
    'wind_data': None,  # Cache for loaded wind data
    'show_wind_vectors': True,  # Show wind vectors by default
    # Physical parameters for concentration calculation
    'mixing_height': 1000.0,  # Mixing layer height in meters
    'cell_area': None,  # Will be calculated based on domain
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

# Domain constants (MUST be defined before calculate_cell_area)
# Domain in km for display, but internally converted to meters for physics
xmin, xmax, ymin, ymax = 0.0, 200.0, 0.0, 200.0  # km
nx, ny = 120, 120
x = np.linspace(xmin, xmax, nx)  # km
y = np.linspace(ymin, ymax, ny)  # km

# Unit conversion factor
KM_TO_M = 1000.0  # meters per kilometer

# Calculate cell area (assuming square cells in km, convert to m²)
def calculate_cell_area():
    """Calculate grid cell area in m²"""
    dx_km = (xmax - xmin) / nx  # km
    dy_km = (ymax - ymin) / ny  # km
    dx_m = dx_km * KM_TO_M  # m
    dy_m = dy_km * KM_TO_M  # m
    return dx_m * dy_m  # m²

sim_state['cell_area'] = calculate_cell_area()

# Calculate mass per particle based on total emissions and super-particle ratio
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

def synthetic_wind_field(t):
    """
    Generate synthetic wind field for testing
    
    Returns:
    --------
    U, V : ndarray, shape (ny, nx)
        Wind components in m/s
    """
    uu = 3.0 + 1.0 * np.sin(2*np.pi*(y/200.0 + 0.06*t))
    vv = 0.3 * np.cos(2*np.pi*(x/200.0 - 0.03*t))
    U = np.tile(uu, (nx,1)).T  # m/s
    V = np.tile(vv, (ny,1))  # m/s
    return U, V

def get_wind_at_particles(U, V, particle_x, particle_y):
    """
    Interpolate wind field at particle positions using bilinear interpolation
    
    Parameters:
    -----------
    U, V : ndarray, shape (ny, nx)
        Wind field components in m/s
    particle_x, particle_y : ndarray, shape (n_particles,)
        Particle positions in km
    
    Returns:
    --------
    u_p, v_p : ndarray, shape (n_particles,)
        Wind components at particle positions in m/s
    """
    if not sim_state.get('use_bilinear_interp', True):
        # Old method: simple averaging (faster but less accurate)
        u_p = np.interp(particle_x, x, U.mean(axis=0))
        v_p = np.interp(particle_y, y, V.mean(axis=1))
        return u_p, v_p
    
    # Bilinear interpolation using RegularGridInterpolator
    # Create interpolators if not cached or if wind field changed
    if (wind_interpolators['U_interp'] is None or 
        wind_interpolators['V_interp'] is None):
        # Note: RegularGridInterpolator expects (y, x) ordering for 2D grids
        wind_interpolators['U_interp'] = RegularGridInterpolator(
            (y, x), U, method='linear', bounds_error=False, fill_value=0.0
        )
        wind_interpolators['V_interp'] = RegularGridInterpolator(
            (y, x), V, method='linear', bounds_error=False, fill_value=0.0
        )
    else:
        # Update values (faster than recreating interpolator)
        wind_interpolators['U_interp'].values = U
        wind_interpolators['V_interp'].values = V
    
    # Prepare points for interpolation: (y, x) pairs
    points = np.column_stack([particle_y, particle_x])
    
    # Interpolate wind at particle positions
    u_p = wind_interpolators['U_interp'](points)
    v_p = wind_interpolators['V_interp'](points)
    
    return u_p, v_p

def load_wind_data(filepath=None):
    """
    Load wind data from NetCDF file (ERA5/GFS format)
    
    Expected file structure:
    - Variables: 'u10', 'v10' (or 'u', 'v' for upper level winds)
    - Dimensions: time, latitude, longitude
    - Can be NetCDF (.nc) or numpy archive (.npz)
    """
    if filepath is None:
        filepath = 'wind_data.npz'  # Default file
    
    if not os.path.exists(filepath):
        print(f"Wind data file not found: {filepath}")
        return False
    
    try:
        if filepath.endswith('.npz'):
            # Load from numpy archive
            data = np.load(filepath)
            wind_data_cache['time_array'] = data['time']  # Should be in hours or seconds
            wind_data_cache['u_data'] = data['u']  # Shape: (time, lat, lon)
            wind_data_cache['v_data'] = data['v']
            wind_data_cache['x_wind'] = data['lon']  # Longitude values
            wind_data_cache['y_wind'] = data['lat']  # Latitude values
        elif filepath.endswith('.nc'):
            # Load from NetCDF (requires netCDF4 or xarray)
            try:
                import netCDF4 as nc
                dataset = nc.Dataset(filepath)
                wind_data_cache['time_array'] = dataset.variables['time'][:]
                wind_data_cache['u_data'] = dataset.variables['u10'][:]  # or 'u'
                wind_data_cache['v_data'] = dataset.variables['v10'][:]  # or 'v'
                wind_data_cache['x_wind'] = dataset.variables['longitude'][:]
                wind_data_cache['y_wind'] = dataset.variables['latitude'][:]
                dataset.close()
            except ImportError:
                print("netCDF4 not installed. Install with: pip install netCDF4")
                return False
        else:
            print(f"Unsupported file format: {filepath}")
            return False
        
        # Create interpolators
        nt = len(wind_data_cache['time_array'])
        ny_wind = len(wind_data_cache['y_wind'])
        nx_wind = len(wind_data_cache['x_wind'])
        
        wind_data_cache['interpolator_u'] = RegularGridInterpolator(
            (wind_data_cache['time_array'], 
             wind_data_cache['y_wind'], 
             wind_data_cache['x_wind']),
            wind_data_cache['u_data'],
            bounds_error=False,
            fill_value=0.0
        )
        
        wind_data_cache['interpolator_v'] = RegularGridInterpolator(
            (wind_data_cache['time_array'], 
             wind_data_cache['y_wind'], 
             wind_data_cache['x_wind']),
            wind_data_cache['v_data'],
            bounds_error=False,
            fill_value=0.0
        )
        
        wind_data_cache['loaded'] = True
        print(f"Wind data loaded successfully: {nt} time steps, {ny_wind}x{nx_wind} grid")
        return True
        
    except Exception as e:
        print(f"Error loading wind data: {e}")
        return False

def real_wind_field(t):
    """
    Get interpolated wind field from ERA5/GFS data at time t
    
    If wind data is not loaded, falls back to synthetic wind
    """
    if not wind_data_cache['loaded']:
        # Attempt to load wind data
        if not load_wind_data():
            # Fallback to synthetic wind
            print("Using synthetic wind (wind data not available)")
            return synthetic_wind_field(t)
    
    # Create mesh grid for interpolation
    X, Y = np.meshgrid(x, y)
    
    # Create points for interpolation (time, y, x)
    t_interp = np.clip(t, 
                       wind_data_cache['time_array'].min(), 
                       wind_data_cache['time_array'].max())
    
    points = np.column_stack([
        np.full(X.size, t_interp),
        Y.ravel(),
        X.ravel()
    ])
    
    # Interpolate wind components
    U_flat = wind_data_cache['interpolator_u'](points)
    V_flat = wind_data_cache['interpolator_v'](points)
    
    # Reshape to grid
    U = U_flat.reshape(ny, nx)
    V = V_flat.reshape(ny, nx)
    
    return U, V

def create_sample_wind_data():
    """
    Create a sample wind data file for testing
    This generates synthetic wind data in the expected format
    """
    # Time array (0 to 100 hours)
    time = np.linspace(0, 100, 50)
    
    # Spatial grid (matching domain approximately)
    lon = np.linspace(0, 200, 40)
    lat = np.linspace(0, 200, 40)
    
    # Create synthetic wind data with temporal and spatial variation
    nt, nlat, nlon = len(time), len(lat), len(lon)
    u_data = np.zeros((nt, nlat, nlon))
    v_data = np.zeros((nt, nlat, nlon))
    
    for i, t in enumerate(time):
        for j, la in enumerate(lat):
            for k, lo in enumerate(lon):
                # Varying wind pattern
                u_data[i, j, k] = 5.0 + 2.0 * np.sin(2*np.pi*(la/200.0 + t/100.0)) + \
                                  1.0 * np.cos(2*np.pi*lo/200.0)
                v_data[i, j, k] = 1.0 + 1.5 * np.cos(2*np.pi*(lo/200.0 - t/100.0)) + \
                                  0.5 * np.sin(2*np.pi*la/200.0)
    
    # Save to npz file
    np.savez('wind_data.npz', 
             time=time, 
             lat=lat, 
             lon=lon, 
             u=u_data, 
             v=v_data)
    
    print("Sample wind data created: wind_data.npz")

def initialize_particles():
    hotspots = np.array(sim_state['hotspots'])
    npph = sim_state['npph']
    n_particles = npph * len(hotspots)
    
    particles = np.zeros((n_particles, 3))  # x, y, mass_fraction
    for i, h in enumerate(hotspots):
        a = i * npph
        b = (i + 1) * npph
        particles[a:b, 0] = h[0] + np.random.normal(scale=2.0, size=npph)
        particles[a:b, 1] = h[1] + np.random.normal(scale=2.0, size=npph)
        particles[a:b, 2] = 1.0  # Initial mass fraction = 1.0 (full mass)
    
    # Initialize particle active status (all active initially)
    sim_state['particle_active'] = np.ones(n_particles, dtype=bool)
    
    return particles

def advect(p, t, dt, sigma_turb):
    """
    Advect particles with turbulent diffusion and apply loss processes
    
    All units are consistent:
    - Positions (x, y) in km
    - Wind velocities (u, v) in m/s
    - Time (t, dt) in seconds
    - Diffusion coefficient (sigma_turb) in m/s
    
    Parameters:
    -----------
    p : ndarray, shape (n_particles, 3)
        Particle positions and mass: [x(km), y(km), mass_fraction]
    t : float
        Current time in seconds
    dt : float
        Time step in seconds
    sigma_turb : float
        Turbulent diffusion coefficient in m/s
    
    Returns:
    --------
    p : ndarray
        Updated particle array
    """
    # Get active particles only
    active = sim_state['particle_active']
    n_active = active.sum()
    
    if n_active == 0:
        return p
    
    # Get wind field (m/s)
    if sim_state['wind_type'] == 'real':
        U, V = real_wind_field(t)
    else:
        U, V = synthetic_wind_field(t)
    
    # Interpolate wind at particle positions (bilinear interpolation)
    # Particle positions are in km, wind is in m/s
    u_p, v_p = get_wind_at_particles(U, V, p[active, 0], p[active, 1])
    
    # Advection: convert wind (m/s) to displacement (km)
    # dx = u * dt (m) = u * dt / 1000 (km)
    dx_km = (u_p * dt) / KM_TO_M  # km
    dy_km = (v_p * dt) / KM_TO_M  # km
    
    # Turbulent diffusion: sigma_turb is in m/s
    # Convert to km for consistency with domain units
    diffusion_std_km = np.sqrt(2 * sigma_turb * dt) / KM_TO_M  # km
    
    # Update positions (vectorized)
    p[active, 0] += dx_km + diffusion_std_km * np.random.randn(n_active)
    p[active, 1] += dy_km + diffusion_std_km * np.random.randn(n_active)
    
    # Apply boundary conditions
    boundary_type = sim_state.get('boundary_type', 'absorbing')
    
    if boundary_type == 'absorbing':
        # Remove particles that leave the domain
        outside = (p[active, 0] < xmin) | (p[active, 0] > xmax) | \
                  (p[active, 1] < ymin) | (p[active, 1] > ymax)
        
        # Deactivate particles outside domain
        active_indices = np.where(active)[0]
        sim_state['particle_active'][active_indices[outside]] = False
        
    elif boundary_type == 'reflecting':
        # Reflect particles at boundaries
        # X boundaries
        outside_xmin = p[active, 0] < xmin
        outside_xmax = p[active, 0] > xmax
        p[active & outside_xmin, 0] = 2 * xmin - p[active & outside_xmin, 0]
        p[active & outside_xmax, 0] = 2 * xmax - p[active & outside_xmax, 0]
        
        # Y boundaries
        outside_ymin = p[active, 1] < ymin
        outside_ymax = p[active, 1] > ymax
        p[active & outside_ymin, 1] = 2 * ymin - p[active & outside_ymin, 1]
        p[active & outside_ymax, 1] = 2 * ymax - p[active & outside_ymax, 1]
        
    elif boundary_type == 'periodic':
        # Periodic boundaries (wrap around)
        p[active, 0] = np.mod(p[active, 0] - xmin, xmax - xmin) + xmin
        p[active, 1] = np.mod(p[active, 1] - ymin, ymax - ymin) + ymin
    
    # Apply deposition (dry deposition)
    if sim_state.get('enable_deposition', True):
        v_dep = sim_state.get('deposition_velocity', 0.001)  # m/s
        H_mix = sim_state['mixing_height']  # m
        
        # Deposition loss rate: λ_dep = v_dep / H_mix (1/s)
        lambda_dep = v_dep / H_mix
        
        # Apply exponential decay for deposition
        deposition_factor = np.exp(-lambda_dep * dt)
        p[active, 2] *= deposition_factor
    
    # Apply chemical decay
    if sim_state.get('enable_decay', False):
        lambda_decay = sim_state.get('decay_rate', 0.0)  # 1/s
        
        if lambda_decay > 0:
            # Apply exponential decay
            decay_factor = np.exp(-lambda_decay * dt)
            p[active, 2] *= decay_factor
    
    # Remove particles with very small mass (efficiency)
    mass_threshold = 1e-6  # Fraction of original mass
    very_small = p[active, 2] < mass_threshold
    active_indices = np.where(active)[0]
    sim_state['particle_active'][active_indices[very_small]] = False
    
    return p

def concentration_field(p):
    """
    Calculate both normalized and physical concentration fields
    
    Parameters:
    -----------
    p : ndarray, shape (n_particles, 3)
        Particle array [x, y, mass_fraction]
    
    Returns:
    --------
    H_normalized : ndarray
        Normalized concentration (0-1)
    H_physical : ndarray
        Physical concentration in µg/m³
    """
    # Only use active particles
    active = sim_state['particle_active']
    p_active = p[active]
    
    if len(p_active) == 0:
        # No active particles
        H_normalized = np.zeros((ny, nx))
        H_physical = np.zeros((ny, nx))
        return H_normalized, H_physical
    
    # Create 2D histogram weighted by mass fraction
    H, xe, ye = np.histogram2d(
        p_active[:, 0], 
        p_active[:, 1], 
        bins=[nx, ny], 
        range=[[xmin, xmax], [ymin, ymax]],
        weights=p_active[:, 2]  # Weight by mass fraction
    )
    H = H.T
    
    # Calculate physical concentration (µg/m³)
    # C = (N_particles * mass_per_particle * mass_fraction) / (cell_area * mixing_height)
    mass_per_particle = sim_state['mass_per_particle']  # µg
    cell_area = sim_state['cell_area']  # m²
    mixing_height = sim_state['mixing_height']  # m
    
    # Physical concentration in µg/m³
    H_physical = (H * mass_per_particle) / (cell_area * mixing_height)
    
    # Normalized concentration (0-1)
    H_normalized = H / (H.max() + 1e-9)
    
    return H_normalized, H_physical

def generate_frame(particles, t):
    """Generate a single frame and return as base64 encoded PNG"""
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Get both normalized and physical concentrations
    H_normalized, H_physical = concentration_field(particles)
    
    # Get max physical concentration for display
    max_conc = H_physical.max()
    
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_title(f"Lagrangian Transport - t = {t:.1f} s | Max: {max_conc:.2f} µg/m³", fontsize=14)
    ax.set_xlabel("X (km)")
    ax.set_ylabel("Y (km)")
    
    # Use normalized concentration for visualization
    im = ax.imshow(H_normalized, origin='lower', extent=(xmin, xmax, ymin, ymax), 
                   alpha=0.7, vmin=0, vmax=1, cmap=cm.inferno)
    
    # Plot wind vectors if enabled
    if sim_state.get('show_wind_vectors', True):
        # Get wind field at current time
        if sim_state['wind_type'] == 'real':
            U, V = real_wind_field(t)
        else:
            U, V = synthetic_wind_field(t)
        
        # Create wind vector field (subsample for clarity)
        skip = 8  # Show every 8th arrow
        X_grid, Y_grid = np.meshgrid(x[::skip], y[::skip])
        U_sub = U[::skip, ::skip]
        V_sub = V[::skip, ::skip]
        
        # Plot wind vectors
        quiver = ax.quiver(X_grid, Y_grid, U_sub, V_sub,
                           color='white', alpha=0.6, scale=50, width=0.003,
                           headwidth=4, headlength=5, linewidths=0.5)
        
        # Add quiver key (scale reference)
        ax.quiverkey(quiver, 0.9, 0.95, 5, '5 m/s', labelpos='E',
                     coordinates='axes', color='white', labelcolor='white',
                     fontproperties={'size': 10})
    
    # Sample particles for display (only active particles)
    active = sim_state['particle_active']
    particles_active = particles[active]
    
    if len(particles_active) > 0:
        n_display = min(1500, len(particles_active))
        sample_idx = np.random.choice(len(particles_active), size=n_display, replace=False)
        sample = particles_active[sample_idx]
        ax.scatter(sample[:, 0], sample[:, 1], s=1, alpha=0.5, c='white')
    
    # Plot hotspots
    for hx, hy in sim_state['hotspots']:
        ax.plot(hx, hy, 'wo', markersize=10, markeredgecolor='k', markeredgewidth=2)
    
    plt.colorbar(im, ax=ax, label='Normalized Concentration')
    
    # Save to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    
    # Encode as base64
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    return img_base64

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/state', methods=['GET'])
def get_state():
    """Get current simulation state"""
    return jsonify({
        'hotspots': sim_state['hotspots'],
        'sigma_turb': sim_state['sigma_turb'],
        'npph': sim_state['npph'],
        'dt': sim_state['dt'],
        'current_frame': sim_state['current_frame'],
        'wind_type': sim_state['wind_type'],
        'is_playing': sim_state['is_playing'],
        'show_wind_vectors': sim_state.get('show_wind_vectors', True),
        'boundary_type': sim_state.get('boundary_type', 'absorbing'),
        'enable_deposition': sim_state.get('enable_deposition', True),
        'deposition_velocity': sim_state.get('deposition_velocity', 0.001),
        'enable_decay': sim_state.get('enable_decay', False),
        'decay_rate': sim_state.get('decay_rate', 0.0),
        'active_particles': int(sim_state['particle_active'].sum()) if sim_state['particle_active'] is not None else 0,
        'total_particles': len(sim_state['particle_active']) if sim_state['particle_active'] is not None else 0,
        'super_particle_ratio': sim_state.get('super_particle_ratio', 1),
        'use_bilinear_interp': sim_state.get('use_bilinear_interp', True)
    })

@app.route('/api/physics', methods=['GET', 'POST'])
def manage_physics():
    """Get or update physical process parameters"""
    if request.method == 'GET':
        return jsonify({
            'success': True,
            'physics': {
                'boundary_type': sim_state.get('boundary_type', 'absorbing'),
                'enable_deposition': sim_state.get('enable_deposition', True),
                'deposition_velocity': sim_state.get('deposition_velocity', 0.001),
                'enable_decay': sim_state.get('enable_decay', False),
                'decay_rate': sim_state.get('decay_rate', 0.0),
                'mixing_height': sim_state['mixing_height']
            }
        })
    
    elif request.method == 'POST':
        data = request.json
        
        if 'boundary_type' in data:
            if data['boundary_type'] in ['absorbing', 'reflecting', 'periodic']:
                sim_state['boundary_type'] = data['boundary_type']
        
        if 'enable_deposition' in data:
            sim_state['enable_deposition'] = bool(data['enable_deposition'])
        
        if 'deposition_velocity' in data:
            sim_state['deposition_velocity'] = float(data['deposition_velocity'])
        
        if 'enable_decay' in data:
            sim_state['enable_decay'] = bool(data['enable_decay'])
        
        if 'decay_rate' in data:
            sim_state['decay_rate'] = float(data['decay_rate'])
        
        return jsonify({
            'success': True,
            'message': 'Physics parameters updated'
        })

@app.route('/api/performance', methods=['GET'])
def get_performance_stats():
    """Get performance and scaling statistics"""
    n_total = len(sim_state['particle_active']) if sim_state['particle_active'] is not None else 0
    n_active = int(sim_state['particle_active'].sum()) if sim_state['particle_active'] is not None else 0
    super_ratio = sim_state.get('super_particle_ratio', 1)
    
    # Calculate effective physical particle count
    n_physical = n_total * super_ratio
    n_physical_active = n_active * super_ratio
    
    return jsonify({
        'success': True,
        'performance': {
            'computational_particles': {
                'total': n_total,
                'active': n_active,
                'inactive': n_total - n_active
            },
            'physical_particles': {
                'total': n_physical,
                'active': n_physical_active,
                'inactive': n_physical - n_physical_active
            },
            'super_particle_ratio': super_ratio,
            'use_bilinear_interp': sim_state.get('use_bilinear_interp', True),
            'grid_resolution': f'{nx}x{ny}',
            'domain_size_km': f'{xmax-xmin}x{ymax-ymin}',
            'cell_size_m': f'{np.sqrt(sim_state["cell_area"]):.1f}'
        }
    })

@app.route('/api/hotspots', methods=['POST'])
def update_hotspots():
    """Add, remove, or move hotspots"""
    data = request.json
    sim_state['hotspots'] = data['hotspots']
    sim_state['particles'] = None  # Reset particles
    sim_state['current_frame'] = 0
    return jsonify({'success': True, 'hotspots': sim_state['hotspots']})

@app.route('/api/params', methods=['POST'])
def update_params():
    """Update simulation parameters"""
    data = request.json
    reset_needed = False
    
    if 'sigma_turb' in data:
        sim_state['sigma_turb'] = float(data['sigma_turb'])
        reset_needed = True
        
    if 'npph' in data:
        sim_state['npph'] = int(data['npph'])
        reset_needed = True
        
    if 'dt' in data:
        sim_state['dt'] = float(data['dt'])
        reset_needed = True
        
    if 'wind_type' in data:
        sim_state['wind_type'] = data['wind_type']
        # Attempt to load wind data if switching to real
        if data['wind_type'] == 'real' and not wind_data_cache['loaded']:
            load_wind_data()
        reset_needed = True
        
    if 'show_wind_vectors' in data:
        sim_state['show_wind_vectors'] = bool(data['show_wind_vectors'])
        # Don't reset particles for this change
        return jsonify({'success': True})
    
    if 'super_particle_ratio' in data:
        ratio = int(data['super_particle_ratio'])
        if ratio >= 1:
            sim_state['super_particle_ratio'] = ratio
            update_mass_per_particle()
            reset_needed = True
    
    if 'use_bilinear_interp' in data:
        sim_state['use_bilinear_interp'] = bool(data['use_bilinear_interp'])
        # Clear cached interpolators
        wind_interpolators['U_interp'] = None
        wind_interpolators['V_interp'] = None
    
    if reset_needed:
        sim_state['particles'] = None
        sim_state['current_frame'] = 0
    
    return jsonify({'success': True})

@app.route('/api/reset', methods=['POST'])
def reset_simulation():
    """Reset simulation to initial state"""
    sim_state['particles'] = None
    sim_state['current_frame'] = 0
    sim_state['is_playing'] = False
    return jsonify({'success': True})

@app.route('/api/emissions', methods=['GET', 'POST'])
def manage_emissions():
    """Get or update emission parameters"""
    if request.method == 'GET':
        # Return current emission data and calculated values
        update_mass_per_particle()
        return jsonify({
            'success': True,
            'emissions': {
                'pollutant_type': emission_data['pollutant_type'],
                'emission_factor': emission_data['emission_factor'],
                'total_mass_per_hotspot': emission_data['total_mass_per_hotspot'],
                'mass_per_particle': sim_state['mass_per_particle'],
                'mixing_height': sim_state['mixing_height'],
                'cell_area': sim_state['cell_area'],
                'num_hotspots': len(sim_state['hotspots']),
                'total_particles': sim_state['npph'] * len(sim_state['hotspots'])
            }
        })
    
    elif request.method == 'POST':
        # Update emission parameters
        data = request.json
        
        if 'pollutant_type' in data:
            emission_data['pollutant_type'] = data['pollutant_type']
        
        if 'emission_factor' in data:
            emission_data['emission_factor'] = float(data['emission_factor'])
        
        if 'total_mass_per_hotspot' in data:
            emission_data['total_mass_per_hotspot'] = float(data['total_mass_per_hotspot'])
        
        if 'mixing_height' in data:
            sim_state['mixing_height'] = float(data['mixing_height'])
        
        # Recalculate mass per particle
        update_mass_per_particle()
        
        # Reset simulation with new parameters
        sim_state['particles'] = None
        sim_state['current_frame'] = 0
        
        return jsonify({
            'success': True,
            'mass_per_particle': sim_state['mass_per_particle']
        })

@app.route('/api/emissions/from-frp', methods=['POST'])
def calculate_emission_from_frp():
    """Calculate emission from FRP (Fire Radiative Power)"""
    data = request.json
    
    frp_mw = data.get('frp_mw', 100.0)  # MW
    duration_hours = data.get('duration_hours', 1.0)
    pollutant = data.get('pollutant', 'PM2.5')
    vegetation_type = data.get('vegetation_type', 'tropical_forest')
    combustion_efficiency = data.get('combustion_efficiency', 0.5)
    
    # Calculate emission
    emission_g = frp_to_emission(frp_mw, pollutant, vegetation_type, 
                                 duration_hours, combustion_efficiency)
    
    # Update emission data
    emission_data['total_mass_per_hotspot'] = emission_g
    emission_data['pollutant_type'] = pollutant
    
    # Get emission factor used
    ef = get_emission_factor(pollutant, vegetation_type)
    
    return jsonify({
        'success': True,
        'emission_g': emission_g,
        'emission_factor': ef,
        'pollutant': pollutant,
        'vegetation_type': vegetation_type
    })

@app.route('/api/emissions/factors', methods=['GET'])
def get_emission_factors():
    """Get available emission factors"""
    return jsonify({
        'success': True,
        'pollutants': list_pollutants(),
        'vegetation_types': list_vegetation_types(),
        'emission_factors': EMISSION_FACTORS
    })

@app.route('/api/play', methods=['POST'])
def toggle_play():
    """Toggle play/pause state"""
    sim_state['is_playing'] = not sim_state['is_playing']
    return jsonify({'success': True, 'is_playing': sim_state['is_playing']})

@app.route('/api/wind/upload', methods=['POST'])
def upload_wind_data():
    """Upload wind data file"""
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file provided'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No file selected'})
    
    # Save uploaded file
    filepath = os.path.join('uploads', file.filename)
    os.makedirs('uploads', exist_ok=True)
    file.save(filepath)
    
    # Try to load the wind data
    success = load_wind_data(filepath)
    
    if success:
        return jsonify({
            'success': True, 
            'message': 'Wind data loaded successfully',
            'loaded': True
        })
    else:
        return jsonify({
            'success': False, 
            'error': 'Failed to load wind data file'
        })

@app.route('/api/wind/status', methods=['GET'])
def wind_status():
    """Get wind data loading status"""
    return jsonify({
        'loaded': wind_data_cache['loaded'],
        'has_data': wind_data_cache['u_data'] is not None,
        'time_steps': len(wind_data_cache['time_array']) if wind_data_cache['time_array'] is not None else 0
    })

@app.route('/api/wind/sample', methods=['POST'])
def create_sample():
    """Create sample wind data file"""
    create_sample_wind_data()
    success = load_wind_data('wind_data.npz')
    
    return jsonify({
        'success': success,
        'message': 'Sample wind data created and loaded' if success else 'Failed to create sample data'
    })

@app.route('/api/wind/fetch-era5', methods=['POST'])
def fetch_era5():
    """Fetch wind data from ERA5 API"""
    data = request.json
    
    # Get parameters
    bbox = data.get('bbox', [45, -10, 35, 10])  # [N, W, S, E]
    start_date_str = data.get('start_date')
    end_date_str = data.get('end_date')
    
    try:
        start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
        end_date = datetime.strptime(end_date_str, '%Y-%m-%d')
    except:
        start_date = datetime.now() - timedelta(days=2)
        end_date = datetime.now() - timedelta(days=1)
    
    try:
        output_file = 'era5_wind.npz'
        wind_data = wind_fetcher.fetch_era5_data(bbox, start_date, end_date, output_file)
        
        # Load the fetched data
        success = load_wind_data(output_file)
        
        return jsonify({
            'success': success,
            'message': f'ERA5 data fetched and loaded successfully',
            'file': output_file
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/api/wind/fetch-gfs', methods=['POST'])
def fetch_gfs():
    """Fetch wind data from GFS API"""
    data = request.json
    
    # Get parameters
    bbox = data.get('bbox', [45, -10, 35, 10])  # [N, W, S, E]
    forecast_hour = data.get('forecast_hour', 0)
    
    try:
        output_file = 'gfs_wind.npz'
        wind_data = wind_fetcher.fetch_gfs_data(bbox, forecast_hour, output_file)
        
        # Load the fetched data
        success = load_wind_data(output_file)
        
        return jsonify({
            'success': success,
            'message': f'GFS data fetched and loaded successfully',
            'file': output_file
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/api/wind/setup-creds', methods=['POST'])
def setup_credentials():
    """Setup CDS API credentials"""
    data = request.json
    api_key = data.get('api_key')
    
    if not api_key:
        return jsonify({'success': False, 'error': 'No API key provided'})
    
    try:
        from wind_api import setup_cds_credentials
        setup_cds_credentials(api_key)
        
        # Also set environment variable for current session
        os.environ['CDS_API_KEY'] = api_key
        
        return jsonify({
            'success': True,
            'message': 'CDS credentials configured successfully'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/api/step', methods=['POST'])
def step_simulation():
    """Advance simulation by one step and return frame"""
    if sim_state['particles'] is None:
        sim_state['particles'] = initialize_particles()
        sim_state['current_frame'] = 0
        update_mass_per_particle()  # Ensure mass per particle is updated
    
    t = sim_state['current_frame'] * sim_state['dt']
    sim_state['particles'] = advect(sim_state['particles'], t, 
                                     sim_state['dt'], sim_state['sigma_turb'])
    
    # Calculate concentration statistics
    H_normalized, H_physical = concentration_field(sim_state['particles'])
    max_conc = float(H_physical.max())
    mean_conc = float(H_physical.mean())
    total_mass = float(H_physical.sum() * sim_state['cell_area'] * sim_state['mixing_height'] / 1e6)  # grams
    
    # Count active particles
    n_active = int(sim_state['particle_active'].sum())
    n_total = len(sim_state['particle_active'])
    
    img_base64 = generate_frame(sim_state['particles'], t)
    sim_state['current_frame'] += 1
    
    return jsonify({
        'success': True,
        'frame': sim_state['current_frame'],
        'time': t,
        'image': img_base64,
        'concentration': {
            'max_ugm3': max_conc,
            'mean_ugm3': mean_conc,
            'total_mass_g': total_mass
        },
        'particles': {
            'active': n_active,
            'total': n_total,
            'fraction_remaining': n_active / n_total if n_total > 0 else 0
        }
    })

@app.route('/api/run', methods=['POST'])
def run_simulation():
    """Run simulation for multiple steps"""
    data = request.json
    n_steps = data.get('n_steps', 10)
    
    if sim_state['particles'] is None:
        sim_state['particles'] = initialize_particles()
        sim_state['current_frame'] = 0
    
    images = []
    for _ in range(n_steps):
        t = sim_state['current_frame'] * sim_state['dt']
        sim_state['particles'] = advect(sim_state['particles'], t, 
                                         sim_state['dt'], sim_state['sigma_turb'])
        img_base64 = generate_frame(sim_state['particles'], t)
        images.append(img_base64)
        sim_state['current_frame'] += 1
    
    return jsonify({
        'success': True,
        'frame': sim_state['current_frame'],
        'images': images
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
