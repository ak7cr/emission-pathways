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
    'show_wind_vectors': True  # Show wind vectors by default
}

# Domain constants
xmin, xmax, ymin, ymax = 0.0, 200.0, 0.0, 200.0
nx, ny = 120, 120
x = np.linspace(xmin, xmax, nx)
y = np.linspace(ymin, ymax, ny)

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
    uu = 3.0 + 1.0 * np.sin(2*np.pi*(y/200.0 + 0.06*t))
    vv = 0.3 * np.cos(2*np.pi*(x/200.0 - 0.03*t))
    U = np.tile(uu, (nx,1)).T
    V = np.tile(vv, (ny,1))
    return U, V

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
    
    particles = np.zeros((n_particles, 2))
    for i, h in enumerate(hotspots):
        a = i * npph
        b = (i + 1) * npph
        particles[a:b, 0] = h[0] + np.random.normal(scale=2.0, size=npph)
        particles[a:b, 1] = h[1] + np.random.normal(scale=2.0, size=npph)
    
    return particles

def advect(p, t, dt, sigma_turb):
    if sim_state['wind_type'] == 'real':
        U, V = real_wind_field(t)
    else:
        U, V = synthetic_wind_field(t)
    
    u = np.interp(p[:, 0], x, U.mean(axis=0))
    v = np.interp(p[:, 1], y, V.mean(axis=1))
    p[:, 0] += u * dt + np.sqrt(2 * sigma_turb * dt) * np.random.randn(len(p))
    p[:, 1] += v * dt + np.sqrt(2 * sigma_turb * dt) * np.random.randn(len(p))
    p[:, 0] = np.clip(p[:, 0], xmin, xmax)
    p[:, 1] = np.clip(p[:, 1], ymin, ymax)
    return p

def concentration_field(p):
    H, xe, ye = np.histogram2d(p[:, 0], p[:, 1], bins=[nx, ny], 
                               range=[[xmin, xmax], [ymin, ymax]])
    H = H.T
    H = H / (H.max() + 1e-9)
    return H

def generate_frame(particles, t):
    """Generate a single frame and return as base64 encoded PNG"""
    fig, ax = plt.subplots(figsize=(8, 8))
    
    H = concentration_field(particles)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_title(f"Lagrangian Transport - t = {t:.1f} s", fontsize=14)
    ax.set_xlabel("X (km)")
    ax.set_ylabel("Y (km)")
    
    im = ax.imshow(H, origin='lower', extent=(xmin, xmax, ymin, ymax), 
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
    
    # Sample particles for display
    n_display = min(1500, len(particles))
    sample = particles[np.random.choice(len(particles), size=n_display, replace=False)]
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
        'show_wind_vectors': sim_state.get('show_wind_vectors', True)
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
    if 'sigma_turb' in data:
        sim_state['sigma_turb'] = float(data['sigma_turb'])
    if 'npph' in data:
        sim_state['npph'] = int(data['npph'])
    if 'dt' in data:
        sim_state['dt'] = float(data['dt'])
    if 'wind_type' in data:
        sim_state['wind_type'] = data['wind_type']
        # Attempt to load wind data if switching to real
        if data['wind_type'] == 'real' and not wind_data_cache['loaded']:
            load_wind_data()
    if 'show_wind_vectors' in data:
        sim_state['show_wind_vectors'] = bool(data['show_wind_vectors'])
        # Don't reset particles for this change
        return jsonify({'success': True})
    
    sim_state['particles'] = None  # Reset particles
    sim_state['current_frame'] = 0
    return jsonify({'success': True})

@app.route('/api/reset', methods=['POST'])
def reset_simulation():
    """Reset simulation to initial state"""
    sim_state['particles'] = None
    sim_state['current_frame'] = 0
    sim_state['is_playing'] = False
    return jsonify({'success': True})

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
    
    t = sim_state['current_frame'] * sim_state['dt']
    sim_state['particles'] = advect(sim_state['particles'], t, 
                                     sim_state['dt'], sim_state['sigma_turb'])
    
    img_base64 = generate_frame(sim_state['particles'], t)
    sim_state['current_frame'] += 1
    
    return jsonify({
        'success': True,
        'frame': sim_state['current_frame'],
        'time': t,
        'image': img_base64
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
