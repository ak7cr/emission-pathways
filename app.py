from flask import Flask, render_template
from wind_api import WindDataFetcher

# Import modular components
import simulation_state
from simulation_state import (
    sim_state, emission_data, wind_data_cache, wind_interpolators,
    update_mass_per_particle, KM_TO_M
)
from particle_physics import initialize_particles, advect, concentration_field
from wind_field import get_wind_at_particles, load_wind_data, create_sample_wind_data
from visualization import generate_frame
from api_routes import register_routes

app = Flask(__name__)
wind_fetcher = WindDataFetcher()

# Wrapper functions to integrate with API routes
def advect_wrapper(sim_state_dict, t):
    """Wrapper for advect function to work with sim_state dictionary"""
    # Access domain values dynamically from simulation_state module
    return advect(
        sim_state_dict['particles'],
        sim_state_dict['particle_active'],
        t,
        sim_state_dict['dt'],
        sim_state_dict['sigma_turb'],
        sim_state_dict['wind_type'],
        wind_interpolators,
        sim_state_dict.get('boundary_type', 'absorbing'),
        sim_state_dict.get('enable_deposition', True),
        sim_state_dict.get('deposition_velocity', 0.001),
        sim_state_dict['mixing_height'],
        sim_state_dict.get('enable_decay', False),
        sim_state_dict.get('decay_rate', 0.0),
        simulation_state.x, simulation_state.y, 
        simulation_state.nx, simulation_state.ny, 
        simulation_state.xmin, simulation_state.xmax, 
        simulation_state.ymin, simulation_state.ymax,
        wind_data_cache,
        get_wind_at_particles,
        KM_TO_M,
        sim_state_dict.get('use_bilinear_interp', True),
        sim_state_dict.get('wind_speed_multiplier', 1.0)
    )

def concentration_field_wrapper(sim_state_dict):
    """Wrapper for concentration_field function"""
    # Access domain values dynamically from simulation_state module
    return concentration_field(
        sim_state_dict['particles'],
        sim_state_dict['particle_active'],
        sim_state_dict['mass_per_particle'],
        sim_state_dict['cell_area'],
        sim_state_dict['mixing_height'],
        simulation_state.nx, simulation_state.ny, 
        simulation_state.xmin, simulation_state.xmax, 
        simulation_state.ymin, simulation_state.ymax
    )

def generate_frame_wrapper(sim_state_dict, t):
    """Wrapper for generate_frame function"""
    # Access domain values dynamically from simulation_state module
    return generate_frame(
        sim_state_dict['particles'],
        sim_state_dict['particle_active'],
        t,
        lambda: concentration_field_wrapper(sim_state_dict),
        sim_state_dict['wind_type'],
        wind_data_cache,
        sim_state_dict.get('show_wind_vectors', True),
        sim_state_dict['hotspots'],
        simulation_state.x, simulation_state.y, 
        simulation_state.nx, simulation_state.ny, 
        simulation_state.xmin, simulation_state.xmax, 
        simulation_state.ymin, simulation_state.ymax
    )

@app.route('/')
def index():
    return render_template('index.html')

# Register all API routes
register_routes(
    app, 
    sim_state, 
    emission_data, 
    wind_data_cache,
    wind_fetcher,
    update_mass_per_particle,
    initialize_particles,
    advect_wrapper,
    concentration_field_wrapper,
    generate_frame_wrapper,
    load_wind_data,
    create_sample_wind_data
)

if __name__ == '__main__':
    app.run(debug=True, port=5000)

