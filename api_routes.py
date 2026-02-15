"""
Flask API routes for simulation control
"""
from flask import jsonify, request
import os
from datetime import datetime, timedelta
import numpy as np

def register_routes(app, sim_state, emission_data, wind_data_cache, 
                   wind_fetcher, update_mass_per_particle,
                   initialize_particles_func, emit_new_particles_func, advect_func, concentration_field_func,
                   generate_frame_func, load_wind_data_func, create_sample_wind_data_func):
    """Register all API routes"""
    
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
            'use_bilinear_interp': sim_state.get('use_bilinear_interp', True),
            'emission_mode': sim_state.get('emission_mode', 'continuous'),
            'emission_interval': sim_state.get('emission_interval', 30.0)
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
            reset_needed = False
            
            if 'boundary_type' in data:
                if data['boundary_type'] in ['absorbing', 'reflecting', 'periodic']:
                    sim_state['boundary_type'] = data['boundary_type']
                    # Reset simulation to apply new boundary conditions
                    reset_needed = True
            
            if 'enable_deposition' in data:
                sim_state['enable_deposition'] = bool(data['enable_deposition'])
                # Deposition affects particles over time, no immediate reset needed
            
            if 'deposition_velocity' in data:
                sim_state['deposition_velocity'] = float(data['deposition_velocity'])
                # Deposition affects particles over time, no immediate reset needed
            
            if 'enable_decay' in data:
                sim_state['enable_decay'] = bool(data['enable_decay'])
                # Decay affects particles over time, no immediate reset needed
            
            if 'decay_rate' in data:
                sim_state['decay_rate'] = float(data['decay_rate'])
                # Decay affects particles over time, no immediate reset needed
            
            if 'mixing_height' in data:
                sim_state['mixing_height'] = float(data['mixing_height'])
                # Recalculate mass per particle and reset
                update_mass_per_particle()
                reset_needed = True
            
            if reset_needed:
                sim_state['particles'] = None
                sim_state['current_frame'] = 0
            
            return jsonify({
                'success': True,
                'message': 'Physics parameters updated',
                'reset': reset_needed
            })

    @app.route('/api/performance', methods=['GET'])
    def get_performance_stats():
        """Get performance and scaling statistics"""
        from simulation_state import nx, ny, xmin, xmax, ymin, ymax
        
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
                'cell_size_m': f'{(sim_state["cell_area"])**0.5:.1f}'
            }
        })

    @app.route('/api/hotspots', methods=['POST'])
    def update_hotspots():
        """Add, remove, or move hotspots"""
        data = request.json
        old_hotspot_count = len(sim_state['hotspots'])
        sim_state['hotspots'] = data['hotspots']
        new_hotspot_count = len(sim_state['hotspots'])
        
        # Reset particles
        sim_state['particles'] = None
        sim_state['current_frame'] = 0
        
        # If hotspot count changed, recalculate mass per particle
        if old_hotspot_count != new_hotspot_count:
            update_mass_per_particle()
        
        return jsonify({
            'success': True, 
            'hotspots': sim_state['hotspots'],
            'message': f'Hotspots updated: {new_hotspot_count} hotspots'
        })

    @app.route('/api/params', methods=['POST'])
    def update_params():
        """Update simulation parameters"""
        from simulation_state import wind_interpolators
        
        data = request.json
        reset_needed = False
        updated_params = []
        
        if 'sigma_turb' in data:
            old_val = sim_state['sigma_turb']
            sim_state['sigma_turb'] = float(data['sigma_turb'])
            updated_params.append(f'sigma_turb: {old_val} → {sim_state["sigma_turb"]}')
            reset_needed = True
            
        if 'npph' in data:
            old_val = sim_state['npph']
            sim_state['npph'] = int(data['npph'])
            updated_params.append(f'npph: {old_val} → {sim_state["npph"]}')
            # Recalculate mass per particle when particle count changes
            update_mass_per_particle()
            reset_needed = True
            
        if 'dt' in data:
            old_val = sim_state['dt']
            sim_state['dt'] = float(data['dt'])
            updated_params.append(f'dt: {old_val} → {sim_state["dt"]}')
            reset_needed = True
            
        if 'wind_type' in data:
            old_val = sim_state['wind_type']
            sim_state['wind_type'] = data['wind_type']
            updated_params.append(f'wind_type: {old_val} → {sim_state["wind_type"]}')
            # Attempt to load wind data if switching to real
            if data['wind_type'] == 'real' and not wind_data_cache['loaded']:
                load_wind_data_func('wind_data.npz', wind_data_cache)
            reset_needed = True
            
        if 'show_wind_vectors' in data:
            sim_state['show_wind_vectors'] = bool(data['show_wind_vectors'])
            updated_params.append(f'show_wind_vectors: {sim_state["show_wind_vectors"]}')
            # Don't reset particles for this change (visual only)
            return jsonify({
                'success': True,
                'updated': updated_params,
                'reset': False
            })
        
        if 'super_particle_ratio' in data:
            ratio = int(data['super_particle_ratio'])
            if ratio >= 1:
                old_val = sim_state.get('super_particle_ratio', 1)
                sim_state['super_particle_ratio'] = ratio
                updated_params.append(f'super_particle_ratio: {old_val} → {ratio}')
                update_mass_per_particle()
                reset_needed = True
        
        if 'use_bilinear_interp' in data:
            old_val = sim_state.get('use_bilinear_interp', True)
            sim_state['use_bilinear_interp'] = bool(data['use_bilinear_interp'])
            updated_params.append(f'use_bilinear_interp: {old_val} → {sim_state["use_bilinear_interp"]}')
            # Clear cached interpolators to force recalculation
            wind_interpolators['U_interp'] = None
            wind_interpolators['V_interp'] = None
            # Don't reset particles, but interpolation method will change
        
        if 'wind_speed_multiplier' in data:
            old_val = sim_state.get('wind_speed_multiplier', 1.0)
            sim_state['wind_speed_multiplier'] = float(data['wind_speed_multiplier'])
            updated_params.append(f'wind_speed_multiplier: {old_val} → {sim_state["wind_speed_multiplier"]}')
            # Don't reset particles - wind speed change is applied on-the-fly
        
        if reset_needed:
            sim_state['particles'] = None
            sim_state['current_frame'] = 0
        
        return jsonify({
            'success': True,
            'updated': updated_params,
            'reset': reset_needed,
            'message': f'Updated {len(updated_params)} parameter(s)' + (' and reset simulation' if reset_needed else '')
        })

    @app.route('/api/domain-scale', methods=['POST'])
    def update_domain_scale():
        """Update domain scale dynamically"""
        import simulation_state as simstate
        
        data = request.json
        scale_name = data.get('scale', 'city')
        domain_size = data.get('domain_size', 50)
        hotspots = data.get('hotspots', [[10.0, 30.0], [15.0, 32.5], [7.5, 22.5]])
        
        # Update domain bounds
        simstate.xmin = 0.0
        simstate.xmax = float(domain_size)
        simstate.ymin = 0.0
        simstate.ymax = float(domain_size)
        
        # Recreate grid
        simstate.x = np.linspace(simstate.xmin, simstate.xmax, simstate.nx)
        simstate.y = np.linspace(simstate.ymin, simstate.ymax, simstate.ny)
        
        # Update hotspots
        sim_state['hotspots'] = hotspots
        
        # Scale sigma_turb based on domain size
        # Base value: 2.5 m/s for 50km domain (city scale)
        # Scale proportionally: smaller domains need less diffusion, larger need more
        base_domain = 50.0  # km (reference city scale)
        base_sigma = 2.5    # m/s
        scale_factor = domain_size / base_domain
        new_sigma_turb = base_sigma * np.sqrt(scale_factor)  # sqrt scaling for diffusion
        sim_state['sigma_turb'] = round(new_sigma_turb, 2)
        
        # Recalculate cell area
        sim_state['cell_area'] = simstate.calculate_cell_area()
        
        # Recalculate mass per particle
        update_mass_per_particle()
        
        # Reset simulation
        sim_state['particles'] = None
        sim_state['particle_active'] = None
        sim_state['current_frame'] = 0
        
        # Clear wind cache
        wind_data_cache['loaded'] = False
        
        resolution_m = (domain_size * 1000) / simstate.nx
        
        return jsonify({
            'success': True,
            'scale': scale_name,
            'domain_size': domain_size,
            'resolution_m': resolution_m,
            'hotspots': hotspots,
            'sigma_turb': sim_state['sigma_turb'],
            'message': f'Domain scaled to {scale_name}: {domain_size}×{domain_size} km (~{resolution_m:.0f}m resolution)'
        })

    @app.route('/api/reset', methods=['POST'])
    def reset_simulation():
        """Reset simulation to initial state"""
        sim_state['particles'] = None
        sim_state['particle_active'] = None
        sim_state['model_state'] = None  # Clear model-specific state
        sim_state['current_frame'] = 0
        sim_state['is_playing'] = False
        sim_state['last_emission_time'] = 0.0
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
        from emission_utils import frp_to_emission, get_emission_factor
        
        data = request.json
        
        frp_mw = data.get('frp_mw', 100.0)
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
        from emission_utils import list_pollutants, list_vegetation_types, EMISSION_FACTORS
        
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
        success = load_wind_data_func(filepath, wind_data_cache)
        
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
        create_sample_wind_data_func()
        success = load_wind_data_func('wind_data.npz', wind_data_cache)
        
        return jsonify({
            'success': success,
            'message': 'Sample wind data created and loaded' if success else 'Failed to create sample data'
        })

    @app.route('/api/wind/fetch-era5', methods=['POST'])
    def fetch_era5():
        """Fetch wind data from ERA5 API"""
        data = request.json
        
        # Get parameters
        bbox = data.get('bbox', [45, -10, 35, 10])
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
            success = load_wind_data_func(output_file, wind_data_cache)
            
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
        bbox = data.get('bbox', [45, -10, 35, 10])
        forecast_hour = data.get('forecast_hour', 0)
        
        try:
            output_file = 'gfs_wind.npz'
            wind_data = wind_fetcher.fetch_gfs_data(bbox, forecast_hour, output_file)
            
            # Load the fetched data
            success = load_wind_data_func(output_file, wind_data_cache)
            
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
    @app.route('/api/emission', methods=['GET', 'POST'])
    @app.route('/api/emission', methods=['GET', 'POST'])
    def manage_emission():
        """Get or update emission parameters"""
        if request.method == 'GET':
            return jsonify({
                'success': True,
                'emission': {
                    'mode': sim_state.get('emission_mode', 'continuous'),
                    'interval': sim_state.get('emission_interval', 30.0),
                    'last_emission_time': sim_state.get('last_emission_time', 0.0)
                }
            })
        
        elif request.method == 'POST':
            data = request.json
            reset_needed = False
            
            if 'emission_mode' in data:
                if data['emission_mode'] in ['continuous', 'single']:
                    old_mode = sim_state.get('emission_mode', 'continuous')
                    sim_state['emission_mode'] = data['emission_mode']
                    
                    # Reset if switching modes
                    if old_mode != data['emission_mode']:
                        reset_needed = True
            
            if 'emission_interval' in data:
                sim_state['emission_interval'] = float(data['emission_interval'])
            
            if reset_needed:
                sim_state['particles'] = None
                sim_state['current_frame'] = 0
                sim_state['last_emission_time'] = 0.0
            
            return jsonify({
                'success': True,
                'message': 'Emission parameters updated',
                'reset': reset_needed
            })

    @app.route('/api/step', methods=['POST'])
    def step_simulation():
        """Advance simulation by one step and return frame"""
        from universal_step import universal_step, universal_emit
        
        # Initialize model state if needed
        if sim_state.get('model_state') is None:
            # Will be initialized in universal_step
            sim_state['particles'] = None
            sim_state['particle_active'] = None
            sim_state['current_frame'] = 0
            sim_state['last_emission_time'] = 0.0
            update_mass_per_particle()
        
        t = sim_state['current_frame'] * sim_state['dt']
        
        # Check if we need to emit new particles (continuous mode)
        emission_mode = sim_state.get('emission_mode', 'continuous')
        emission_interval = sim_state.get('emission_interval', 30.0)
        last_emission_time = sim_state.get('last_emission_time', 0.0)
        
        if emission_mode == 'continuous' and (t - last_emission_time) >= emission_interval:
            # Emit new particles/mass
            sim_state = universal_emit(sim_state)
            sim_state['last_emission_time'] = t
        
        # Step simulation with current model
        sim_state, H_normalized, H_physical = universal_step(sim_state, t, sim_state['dt'])
        
        # Calculate concentration statistics
        max_conc = float(H_physical.max())
        mean_conc = float(H_physical.mean())
        total_mass = float(H_physical.sum() * sim_state['cell_area'] * sim_state['mixing_height'] / 1e6)
        
        # Count active particles (for particle-based models)
        model_type = sim_state.get('model_type', 'lagrangian')
        if model_type in ['lagrangian', 'hybrid'] and sim_state['particle_active'] is not None:
            n_active = int(sim_state['particle_active'].sum())
            n_total = len(sim_state['particle_active'])
        else:
            # For grid-based models, use concentration as proxy
            n_active = int((H_physical > 0.1).sum())  # Cells with significant concentration
            n_total = H_physical.size
        
        # Generate frame with pre-calculated concentration
        img_base64 = generate_frame_func(sim_state, t, (H_normalized, H_physical))
        sim_state['current_frame'] += 1
        
        return jsonify({
            'success': True,
            'frame': sim_state['current_frame'],
            'time': t,
            'image': img_base64,
            'model_type': model_type,
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
        from universal_step import universal_step
        
        data = request.json
        n_steps = data.get('n_steps', 10)
        
        # Initialize if needed
        if sim_state.get('model_state') is None:
            sim_state['particles'] = None
            sim_state['particle_active'] = None
            sim_state['current_frame'] = 0
            update_mass_per_particle()
        
        images = []
        for _ in range(n_steps):
            t = sim_state['current_frame'] * sim_state['dt']
            sim_state, H_normalized, H_physical = universal_step(sim_state, t, sim_state['dt'])
            img_base64 = generate_frame_func(sim_state, t, (H_normalized, H_physical))
            images.append(img_base64)
            sim_state['current_frame'] += 1
        
        return jsonify({
            'success': True,
            'frame': sim_state['current_frame'],
            'images': images
        })
    
    @app.route('/api/validate', methods=['POST'])
    def validate_simulation():
        """
        Validate model against ground observations
        
        Request body:
        {
            "source": "openaq" | "cpcb" | "synthetic",
            "lat_min": float, "lat_max": float,
            "lon_min": float, "lon_max": float,
            "parameter": "pm25" | "pm10",
            "threshold": float (optional, for exceedance metrics),
            "n_synthetic_stations": int (for synthetic data)
        }
        """
        from validation import ValidationPipeline
        from simulation_state import x, y, xmin, xmax, ymin, ymax
        
        data = request.json
        source = data.get('source', 'synthetic')
        
        # Initialize validation pipeline
        validator = ValidationPipeline()
        
        # Fetch observations based on source
        observations = []
        if source == 'openaq':
            observations = validator.fetch_openaq_data(
                lat_min=data.get('lat_min', 0),
                lat_max=data.get('lat_max', 50),
                lon_min=data.get('lon_min', 0),
                lon_max=data.get('lon_max', 50),
                parameter=data.get('parameter', 'pm25'),
                limit=data.get('limit', 1000)
            )
        elif source == 'cpcb':
            observations = validator.fetch_cpcb_data(
                state=data.get('state', 'Delhi'),
                city=data.get('city', 'Delhi'),
                parameter=data.get('parameter', 'pm2.5'),
                api_key=data.get('api_key')
            )
        elif source == 'synthetic':
            # Create synthetic observations for testing
            observations = validator.create_synthetic_observations(
                n_stations=data.get('n_synthetic_stations', 10),
                lat_range=(ymin, ymax),
                lon_range=(xmin, xmax),
                concentration_range=(10, 100)
            )
        
        if len(observations) == 0:
            return jsonify({
                'success': False,
                'error': 'No observations fetched',
                'message': 'Try using source="synthetic" for testing'
            }), 400
        
        # Get current concentration field
        if sim_state['particles'] is None:
            return jsonify({
                'success': False,
                'error': 'No simulation data available. Run simulation first.'
            }), 400
        
        # concentration_field_func returns (H_normalized, H_physical)
        # We need the physical concentration for validation
        _, conc_field_physical = concentration_field_func(sim_state)
        
        # Simple lat/lon to grid coordinate conversion (identity for now)
        # In real application, you'd use proper geographic projection
        def lon_to_x(lon):
            return lon
        def lat_to_y(lat):
            return lat
        
        # Import simulation_state to get current grid values
        import simulation_state
        
        # Extract modeled values at station locations
        modeled, observed = validator.extract_model_values_at_stations(
            conc_field_physical, simulation_state.x, simulation_state.y, observations, lat_to_y, lon_to_x
        )
        
        # Compute validation metrics
        threshold = data.get('threshold', None)  # e.g., 35 µg/m³ for PM2.5
        metrics = validator.compute_metrics(modeled, observed, threshold)
        
        # Generate report
        report = validator.generate_validation_report(metrics, modeled, observed)
        
        return jsonify({
            'success': True,
            'n_stations': len(observations),
            'metrics': metrics,
            'report': report,
            'stations': [{
                'id': obs['station_id'],
                'name': obs['station_name'],
                'lat': obs['latitude'],
                'lon': obs['longitude'],
                'observed': obs['value'],
                'modeled': float(mod)
            } for obs, mod in zip(observations[:20], modeled[:20])]  # First 20 stations
        })
    
    @app.route('/api/ensemble/generate', methods=['POST'])
    def generate_ensemble():
        """
        Generate ensemble configurations with perturbed parameters
        
        Request body:
        {
            "n_members": int,
            "perturbations": {
                "wind_speed_factor": [min, max],
                "wind_direction_offset": [min_deg, max_deg],
                "mixing_height_factor": [min, max],
                "emission_factor": [min, max],
                "turbulence_factor": [min, max]
            },
            "seed": int (optional)
        }
        """
        from ensemble import EnsembleSimulation
        
        data = request.json
        n_members = data.get('n_members', 10)
        perturbations = data.get('perturbations', {
            'wind_speed_factor': (0.8, 1.2),
            'mixing_height_factor': (0.7, 1.3),
            'emission_factor': (0.5, 1.5),
            'turbulence_factor': (0.8, 1.2)
        })
        seed = data.get('seed', 42)
        
        # Initialize ensemble
        ensemble = EnsembleSimulation(sim_state)
        
        # Generate ensemble configurations
        configs = ensemble.generate_ensemble_configs(
            n_members=n_members,
            perturbations=perturbations,
            seed=seed
        )
        
        # Store ensemble in session (in production, use proper session management)
        sim_state['ensemble'] = ensemble
        sim_state['ensemble_configs'] = configs
        
        return jsonify({
            'success': True,
            'n_members': n_members,
            'perturbations': perturbations,
            'configs': [{
                'member': i,
                'wind_speed_pert': cfg.get('wind_speed_perturbation', 1.0),
                'wind_dir_offset': cfg.get('wind_direction_offset', 0.0),
                'mixing_height': cfg.get('mixing_height', sim_state['mixing_height']),
                'emission_scaling': cfg.get('emission_scaling', 1.0),
                'sigma_turb': cfg.get('sigma_turb', sim_state['sigma_turb'])
            } for i, cfg in enumerate(configs)]
        })
    
    @app.route('/api/ensemble/run', methods=['POST'])
    def run_ensemble():
        """
        Run ensemble simulation and compute statistics
        
        Request body:
        {
            "n_steps": int,
            "compute_arrival_stats": bool,
            "target_location": [x, y] (if computing arrival stats)
        }
        """
        from ensemble import EnsembleSimulation
        import numpy as np
        
        data = request.json
        n_steps = data.get('n_steps', 50)
        
        if 'ensemble' not in sim_state or 'ensemble_configs' not in sim_state:
            return jsonify({
                'success': False,
                'error': 'Ensemble not initialized. Call /api/ensemble/generate first.'
            }), 400
        
        ensemble = sim_state['ensemble']
        configs = sim_state['ensemble_configs']
        
        # Run each ensemble member
        concentration_fields = []
        particle_trajectories = []
        
        print(f"\nRunning ensemble: {len(configs)} members × {n_steps} steps...")
        
        for i, config in enumerate(configs):
            print(f"  Member {i+1}/{len(configs)}...", end='', flush=True)
            
            # Create temporary sim state for this member
            member_state = sim_state.copy()
            member_state.update(config)
            
            # Initialize particles
            particles, particle_active = initialize_particles_func(
                member_state['hotspots'],
                member_state['npph']
            )
            member_state['particles'] = particles
            member_state['particle_active'] = particle_active
            
            # Run simulation for n_steps
            for step in range(n_steps):
                t = step * member_state['dt']
                member_state['particles'] = advect_func(member_state, t)
            
            # Compute final concentration field
            # concentration_field_func returns (H_normalized, H_physical) - use physical
            _, conc_field_physical = concentration_field_func(member_state)
            concentration_fields.append(conc_field_physical)
            particle_trajectories.append(member_state['particles'])
            
            print(f" done!")
        
        print(f"Ensemble complete! Computing statistics...")
        
        # Compute ensemble statistics
        statistics = ensemble.compute_ensemble_statistics(concentration_fields)
        
        # Compute arrival time statistics if requested
        arrival_stats = None
        if data.get('compute_arrival_stats', False):
            target_loc = data.get('target_location', [100, 100])
            arrival_stats = ensemble.compute_arrival_time_statistics(
                particle_trajectories,
                tuple(target_loc),
                threshold_distance=data.get('threshold_distance', 5.0)
            )
        
        # Generate report
        report = ensemble.generate_ensemble_report(statistics, arrival_stats)
        
        # Store results
        sim_state['ensemble_results'] = {
            'statistics': {k: v.tolist() if isinstance(v, np.ndarray) else v 
                          for k, v in statistics.items()},
            'arrival_stats': arrival_stats,
            'n_members': len(configs)
        }
        
        return jsonify({
            'success': True,
            'n_members': len(configs),
            'n_steps': n_steps,
            'statistics': {
                'mean_max': float(np.max(statistics['mean'])),
                'std_max': float(np.max(statistics['std'])),
                'spread': float(np.max(statistics['max']) - np.min(statistics['min']))
            },
            'arrival_stats': arrival_stats,
            'report': report
        })
    
    @app.route('/api/ensemble/statistics', methods=['GET'])
    def get_ensemble_statistics():
        """Get ensemble statistics from last run"""
        if 'ensemble_results' not in sim_state:
            return jsonify({
                'success': False,
                'error': 'No ensemble results available. Run ensemble first.'
            }), 400
        
        return jsonify({
            'success': True,
            'results': sim_state['ensemble_results']
        })

