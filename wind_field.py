"""
Wind field generation and interpolation
"""
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import os

def synthetic_wind_field(t, x, y, nx, ny):
    """
    Generate synthetic wind field for testing
    
    Parameters:
    -----------
    t : float
        Current time in seconds
    x, y : ndarray
        Grid coordinates in km
    nx, ny : int
        Grid dimensions
    
    Returns:
    --------
    U, V : ndarray, shape (ny, nx)
        Wind components in m/s
    """
    # Create wind components that vary in space and time
    # Increased wind speeds for more visible particle transport
    # uu varies with y (north-south wind component) - dominant eastward wind
    # vv varies with x (east-west wind component) - weaker northward component
    uu = 12.0 + 4.0 * np.sin(2*np.pi*(y/200.0 + 0.06*t))  # Shape: (ny,) - Increased from 3-4 to 8-16 m/s
    vv = 3.0 + 2.0 * np.cos(2*np.pi*(x/200.0 - 0.03*t))   # Shape: (nx,) - Increased from ~0.3 to 1-5 m/s
    
    # Tile to create 2D fields
    # U should have shape (ny, nx) - each row has the same u value (depends on y)
    # V should have shape (ny, nx) - each column has the same v value (depends on x)
    U = np.tile(uu.reshape(-1, 1), (1, nx))  # m/s, shape (ny, nx)
    V = np.tile(vv.reshape(1, -1), (ny, 1))  # m/s, shape (ny, nx)
    return U, V

def get_wind_at_particles(U, V, particle_x, particle_y, x, y, wind_interpolators, use_bilinear=True):
    """
    Interpolate wind field at particle positions using bilinear interpolation
    
    Parameters:
    -----------
    U, V : ndarray, shape (ny, nx)
        Wind field components in m/s
    particle_x, particle_y : ndarray, shape (n_particles,)
        Particle positions in km
    x, y : ndarray
        Grid coordinates in km
    wind_interpolators : dict
        Cache for interpolators
    use_bilinear : bool
        Use bilinear interpolation (more accurate) vs simple averaging
    
    Returns:
    --------
    u_p, v_p : ndarray, shape (n_particles,)
        Wind components at particle positions in m/s
    """
    if not use_bilinear:
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

def load_wind_data(filepath, wind_data_cache):
    """
    Load wind data from NetCDF file (ERA5/GFS format)
    
    Expected file structure:
    - Variables: 'u10', 'v10' (or 'u', 'v' for upper level winds)
    - Dimensions: time, latitude, longitude
    - Can be NetCDF (.nc) or numpy archive (.npz)
    """
    if not os.path.exists(filepath):
        print(f"Wind data file not found: {filepath}")
        return False
    
    try:
        if filepath.endswith('.npz'):
            # Load from numpy archive
            data = np.load(filepath)
            wind_data_cache['time_array'] = data['time']
            wind_data_cache['u_data'] = data['u']
            wind_data_cache['v_data'] = data['v']
            wind_data_cache['x_wind'] = data['lon']
            wind_data_cache['y_wind'] = data['lat']
        elif filepath.endswith('.nc'):
            # Load from NetCDF (requires netCDF4 or xarray)
            try:
                import netCDF4 as nc
                dataset = nc.Dataset(filepath)
                wind_data_cache['time_array'] = dataset.variables['time'][:]
                wind_data_cache['u_data'] = dataset.variables['u10'][:]
                wind_data_cache['v_data'] = dataset.variables['v10'][:]
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

def real_wind_field(t, x, y, nx, ny, wind_data_cache):
    """
    Get interpolated wind field from ERA5/GFS data at time t
    
    If wind data is not loaded, falls back to synthetic wind
    """
    if not wind_data_cache['loaded']:
        # Attempt to load wind data
        if not load_wind_data('wind_data.npz', wind_data_cache):
            # Fallback to synthetic wind
            print("Using synthetic wind (wind data not available)")
            return synthetic_wind_field(t, x, y, nx, ny)
    
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
    Updated for 50km x 50km domain (city-scale)
    """
    # Time array (0 to 100 hours)
    time = np.linspace(0, 100, 50)
    
    # Spatial grid (matching new 50km domain)
    lon = np.linspace(0, 50, 40)
    lat = np.linspace(0, 50, 40)
    
    # Create synthetic wind data with temporal and spatial variation
    nt, nlat, nlon = len(time), len(lat), len(lon)
    u_data = np.zeros((nt, nlat, nlon))
    v_data = np.zeros((nt, nlat, nlon))
    
    for i, t in enumerate(time):
        for j, la in enumerate(lat):
            for k, lo in enumerate(lon):
                # Varying wind pattern (adjusted for 50km domain)
                u_data[i, j, k] = 5.0 + 2.0 * np.sin(2*np.pi*(la/50.0 + t/100.0)) + \
                                  1.0 * np.cos(2*np.pi*lo/50.0)
                v_data[i, j, k] = 1.0 + 1.5 * np.cos(2*np.pi*(lo/50.0 - t/100.0)) + \
                                  0.5 * np.sin(2*np.pi*la/50.0)
    
    # Save to npz file
    np.savez('wind_data.npz', 
             time=time, 
             lat=lat, 
             lon=lon, 
             u=u_data, 
             v=v_data)
    
    print("Sample wind data created: wind_data.npz (50km x 50km domain)")

