"""
Particle initialization and advection physics
"""
import numpy as np

def initialize_particles(hotspots, npph):
    """
    Initialize particles around hotspots
    
    Parameters:
    -----------
    hotspots : list of [x, y]
        Hotspot positions in km
    npph : int
        Number of particles per hotspot
    
    Returns:
    --------
    particles : ndarray, shape (n_particles, 3)
        Particle array [x(km), y(km), mass_fraction]
    particle_active : ndarray, shape (n_particles,)
        Boolean array tracking active particles
    """
    hotspots = np.array(hotspots)
    n_particles = npph * len(hotspots)
    
    particles = np.zeros((n_particles, 3))  # x, y, mass_fraction
    for i, h in enumerate(hotspots):
        a = i * npph
        b = (i + 1) * npph
        particles[a:b, 0] = h[0] + np.random.normal(scale=2.0, size=npph)
        particles[a:b, 1] = h[1] + np.random.normal(scale=2.0, size=npph)
        particles[a:b, 2] = 1.0  # Initial mass fraction = 1.0 (full mass)
    
    # Initialize particle active status (all active initially)
    particle_active = np.ones(n_particles, dtype=bool)
    
    return particles, particle_active

def apply_boundary_conditions(p, particle_active, boundary_type, xmin, xmax, ymin, ymax):
    """
    Apply boundary conditions to particles
    
    Parameters:
    -----------
    p : ndarray, shape (n_particles, 3)
        Particle array
    particle_active : ndarray, shape (n_particles,)
        Boolean array of active particles
    boundary_type : str
        'absorbing', 'reflecting', or 'periodic'
    xmin, xmax, ymin, ymax : float
        Domain boundaries in km
    """
    active = particle_active
    
    if boundary_type == 'absorbing':
        # Remove particles that leave the domain
        outside = (p[active, 0] < xmin) | (p[active, 0] > xmax) | \
                  (p[active, 1] < ymin) | (p[active, 1] > ymax)
        
        # Deactivate particles outside domain
        active_indices = np.where(active)[0]
        particle_active[active_indices[outside]] = False
        
    elif boundary_type == 'reflecting':
        # Reflect particles at boundaries
        # Get active indices
        active_indices = np.where(active)[0]
        
        # Check boundaries for active particles
        outside_xmin = p[active, 0] < xmin
        outside_xmax = p[active, 0] > xmax
        outside_ymin = p[active, 1] < ymin
        outside_ymax = p[active, 1] > ymax
        
        # Reflect X boundaries
        p[active_indices[outside_xmin], 0] = 2 * xmin - p[active_indices[outside_xmin], 0]
        p[active_indices[outside_xmax], 0] = 2 * xmax - p[active_indices[outside_xmax], 0]
        
        # Reflect Y boundaries
        p[active_indices[outside_ymin], 1] = 2 * ymin - p[active_indices[outside_ymin], 1]
        p[active_indices[outside_ymax], 1] = 2 * ymax - p[active_indices[outside_ymax], 1]
        
    elif boundary_type == 'periodic':
        # Periodic boundaries (wrap around)
        p[active, 0] = np.mod(p[active, 0] - xmin, xmax - xmin) + xmin
        p[active, 1] = np.mod(p[active, 1] - ymin, ymax - ymin) + ymin

def apply_loss_processes(p, particle_active, dt, enable_deposition, deposition_velocity, 
                         mixing_height, enable_decay, decay_rate):
    """
    Apply deposition and decay loss processes
    
    Parameters:
    -----------
    p : ndarray, shape (n_particles, 3)
        Particle array
    particle_active : ndarray, shape (n_particles,)
        Boolean array of active particles
    dt : float
        Time step in seconds
    enable_deposition : bool
        Enable dry deposition
    deposition_velocity : float
        Deposition velocity in m/s
    mixing_height : float
        Mixing layer height in m
    enable_decay : bool
        Enable chemical decay
    decay_rate : float
        Decay rate constant in 1/s
    """
    active = particle_active
    
    # Apply deposition (dry deposition)
    if enable_deposition:
        # Deposition loss rate: λ_dep = v_dep / H_mix (1/s)
        lambda_dep = deposition_velocity / mixing_height
        
        # Apply exponential decay for deposition
        deposition_factor = np.exp(-lambda_dep * dt)
        p[active, 2] *= deposition_factor
    
    # Apply chemical decay
    if enable_decay and decay_rate > 0:
        # Apply exponential decay
        decay_factor = np.exp(-decay_rate * dt)
        p[active, 2] *= decay_factor
    
    # Remove particles with very small mass (efficiency)
    mass_threshold = 1e-6  # Fraction of original mass
    very_small = p[active, 2] < mass_threshold
    active_indices = np.where(active)[0]
    particle_active[active_indices[very_small]] = False

def advect(p, particle_active, t, dt, sigma_turb, wind_type, wind_interpolators,
           boundary_type, enable_deposition, deposition_velocity, mixing_height,
           enable_decay, decay_rate, x, y, nx, ny, xmin, xmax, ymin, ymax,
           wind_data_cache, get_wind_func, KM_TO_M, use_bilinear_interp):
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
    particle_active : ndarray, shape (n_particles,)
        Boolean array tracking active particles
    t : float
        Current time in seconds
    dt : float
        Time step in seconds
    sigma_turb : float
        Turbulent diffusion coefficient in m/s
    wind_type : str
        'synthetic' or 'real'
    wind_interpolators : dict
        Cache for wind interpolators
    boundary_type : str
        'absorbing', 'reflecting', or 'periodic'
    enable_deposition : bool
        Enable dry deposition
    deposition_velocity : float
        Deposition velocity in m/s
    mixing_height : float
        Mixing layer height in m
    enable_decay : bool
        Enable chemical decay
    decay_rate : float
        Decay rate constant in 1/s
    x, y : ndarray
        Grid coordinates in km
    nx, ny : int
        Grid dimensions
    xmin, xmax, ymin, ymax : float
        Domain boundaries in km
    wind_data_cache : dict
        Wind data cache
    get_wind_func : function
        Function to get wind at particles
    KM_TO_M : float
        Conversion factor from km to m
    use_bilinear_interp : bool
        Use bilinear interpolation
    
    Returns:
    --------
    p : ndarray
        Updated particle array
    """
    # Get active particles only
    active = particle_active
    n_active = active.sum()
    
    if n_active == 0:
        return p
    
    # Get wind field (m/s)
    if wind_type == 'real':
        from wind_field import real_wind_field
        U, V = real_wind_field(t, x, y, nx, ny, wind_data_cache)
    else:
        from wind_field import synthetic_wind_field
        U, V = synthetic_wind_field(t, x, y, nx, ny)
    
    # Interpolate wind at particle positions (bilinear interpolation)
    u_p, v_p = get_wind_func(U, V, p[active, 0], p[active, 1], x, y, 
                             wind_interpolators, use_bilinear_interp)
    
    # Advection: convert wind (m/s) to displacement (km)
    dx_km = (u_p * dt) / KM_TO_M  # km
    dy_km = (v_p * dt) / KM_TO_M  # km
    
    # Turbulent diffusion: sigma_turb is in m/s
    # Convert to km for consistency with domain units
    diffusion_std_km = np.sqrt(2 * sigma_turb * dt) / KM_TO_M  # km
    
    # Update positions (vectorized)
    p[active, 0] += dx_km + diffusion_std_km * np.random.randn(n_active)
    p[active, 1] += dy_km + diffusion_std_km * np.random.randn(n_active)
    
    # Apply boundary conditions
    apply_boundary_conditions(p, particle_active, boundary_type, xmin, xmax, ymin, ymax)
    
    # Apply loss processes
    apply_loss_processes(p, particle_active, dt, enable_deposition, deposition_velocity,
                        mixing_height, enable_decay, decay_rate)
    
    return p

def concentration_field(p, particle_active, mass_per_particle, cell_area, mixing_height,
                       nx, ny, xmin, xmax, ymin, ymax):
    """
    Calculate both normalized and physical concentration fields
    
    Parameters:
    -----------
    p : ndarray, shape (n_particles, 3)
        Particle array [x, y, mass_fraction]
    particle_active : ndarray, shape (n_particles,)
        Boolean array of active particles
    mass_per_particle : float
        Mass per particle in µg
    cell_area : float
        Grid cell area in m²
    mixing_height : float
        Mixing layer height in m
    nx, ny : int
        Grid dimensions
    xmin, xmax, ymin, ymax : float
        Domain boundaries in km
    
    Returns:
    --------
    H_normalized : ndarray
        Normalized concentration (0-1)
    H_physical : ndarray
        Physical concentration in µg/m³
    """
    # Only use active particles
    p_active = p[particle_active]
    
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
    
    # Physical concentration in µg/m³
    H_physical = (H * mass_per_particle) / (cell_area * mixing_height)
    
    # Normalized concentration (0-1)
    H_normalized = H / (H.max() + 1e-9)
    
    return H_normalized, H_physical
