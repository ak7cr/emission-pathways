"""
Flask API routes for simulation control
"""
from flask import jsonify, request
import os
from datetime import datetime, timedelta

def register_routes(app, sim_state, emission_data, wind_data_cache, 
                   wind_fetcher, update_mass_per_particle,
                   initialize_particles_func, advect_func, concentration_field_func,
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
        
        if reset_needed:
            sim_state['particles'] = None
            sim_state['current_frame'] = 0
        
        return jsonify({
            'success': True,
            'updated': updated_params,
            'reset': reset_needed,
            'message': f'Updated {len(updated_params)} parameter(s)' + (' and reset simulation' if reset_needed else '')
        })

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

    @app.route('/api/step', methods=['POST'])
    def step_simulation():
        """Advance simulation by one step and return frame"""
        if sim_state['particles'] is None:
            particles, particle_active = initialize_particles_func(
                sim_state['hotspots'], 
                sim_state['npph']
            )
            sim_state['particles'] = particles
            sim_state['particle_active'] = particle_active
            sim_state['current_frame'] = 0
            update_mass_per_particle()
        
        t = sim_state['current_frame'] * sim_state['dt']
        sim_state['particles'] = advect_func(sim_state, t)
        
        # Calculate concentration statistics
        H_normalized, H_physical = concentration_field_func(sim_state)
        max_conc = float(H_physical.max())
        mean_conc = float(H_physical.mean())
        total_mass = float(H_physical.sum() * sim_state['cell_area'] * sim_state['mixing_height'] / 1e6)
        
        # Count active particles
        n_active = int(sim_state['particle_active'].sum())
        n_total = len(sim_state['particle_active'])
        
        img_base64 = generate_frame_func(sim_state, t)
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
            particles, particle_active = initialize_particles_func(
                sim_state['hotspots'], 
                sim_state['npph']
            )
            sim_state['particles'] = particles
            sim_state['particle_active'] = particle_active
            sim_state['current_frame'] = 0
        
        images = []
        for _ in range(n_steps):
            t = sim_state['current_frame'] * sim_state['dt']
            sim_state['particles'] = advect_func(sim_state, t)
            img_base64 = generate_frame_func(sim_state, t)
            images.append(img_base64)
            sim_state['current_frame'] += 1
        
        return jsonify({
            'success': True,
            'frame': sim_state['current_frame'],
            'images': images
        })
