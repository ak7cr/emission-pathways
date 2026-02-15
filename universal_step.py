"""
Universal stepping function for all dispersion models
Handles model initialization, stepping, and state management
"""
from model_manager import model_manager
from wind_field import synthetic_wind_field
import simulation_state

def universal_step(sim_state, t, dt):
    """
    Execute one simulation step with the currently selected model
    
    Parameters:
    -----------
    sim_state : dict
        Global simulation state
    t : float
        Current time (seconds)
    dt : float
        Time step (seconds)
    
    Returns:
    --------
    sim_state : dict
        Updated simulation state
    C_normalized : ndarray
        Normalized concentration field (0-1)
    C_physical : ndarray
        Physical concentration (µg/m³)
    """
    model_type = sim_state.get('model_type', 'lagrangian')
    
    # Create model configuration
    config = {
        'sigma_turb': sim_state.get('sigma_turb', 2.5),
        'npph': sim_state.get('npph', 2500),
        'boundary_type': sim_state.get('boundary_type', 'absorbing'),
        'enable_deposition': sim_state.get('enable_deposition', True),
        'deposition_velocity': sim_state.get('deposition_velocity', 0.001),
        'mixing_height': sim_state.get('mixing_height', 1000.0),
        'enable_decay': sim_state.get('enable_decay', False),
        'decay_rate': sim_state.get('decay_rate', 0.0),
        'use_bilinear_interp': sim_state.get('use_bilinear_interp', True),
        'wind_speed_multiplier': sim_state.get('wind_speed_multiplier', 1.0),
        'mass_per_particle': sim_state.get('mass_per_particle', 1.0),
        'cell_area': sim_state.get('cell_area', 2777777.78),
        'hotspots': sim_state.get('hotspots', [[10.0, 30.0]]),
        'wind_interpolators': simulation_state.wind_interpolators,
        'transition_distance': sim_state.get('transition_distance', 10.0),  # For hybrid model
        'stability_class': sim_state.get('stability_class', 'D'),  # For Gaussian plume
        'source_height': sim_state.get('source_height', 10.0)  # For Gaussian plume
    }
    
    # Get or create model
    model = model_manager.create_model(model_type, config)
    
    # Initialize model state if needed
    model_state = sim_state.get('model_state')
    if model_state is None or model_state.get('model_type') != model_type:
        hotspots = sim_state.get('hotspots', [[10.0, 30.0]])
        npph = sim_state.get('npph', 2500)
        model_state = model.initialize(hotspots, npph)
        sim_state['model_state'] = model_state
    
    # Get wind field
    wind_U, wind_V = synthetic_wind_field(
        t,
        simulation_state.x,
        simulation_state.y,
        simulation_state.nx,
        simulation_state.ny
    )
    
    # Step model
    model_state = model.step(model_state, t, dt, wind_U, wind_V)
    sim_state['model_state'] = model_state
    
    # Get concentration
    C_normalized, C_physical = model.get_concentration(model_state)
    
    # Update legacy particle state for backward compatibility
    if model_type == 'lagrangian':
        sim_state['particles'] = model_state.get('particles')
        sim_state['particle_active'] = model_state.get('particle_active')
    elif model_type == 'hybrid':
        sim_state['particles'] = model_state.get('particles')
        sim_state['particle_active'] = model_state.get('particle_active')
    else:
        # For non-particle models, create dummy particle state
        sim_state['particles'] = None
        sim_state['particle_active'] = None
    
    return sim_state, C_normalized, C_physical


def universal_emit(sim_state):
    """
    Emit new particles/mass from sources
    
    Parameters:
    -----------
    sim_state : dict
        Global simulation state
    
    Returns:
    --------
    sim_state : dict
        Updated simulation state
    """
    model_type = sim_state.get('model_type', 'lagrangian')
    model_state = sim_state.get('model_state')
    
    if model_state is None:
        return sim_state
    
    # Create config and model
    config = {
        'sigma_turb': sim_state.get('sigma_turb', 2.5),
        'npph': sim_state.get('npph', 2500),
        'mass_per_particle': sim_state.get('mass_per_particle', 1.0),
        'cell_area': sim_state.get('cell_area', 2777777.78),
        'mixing_height': sim_state.get('mixing_height', 1000.0),
        'hotspots': sim_state.get('hotspots', [[10.0, 30.0]]),
    }
    
    model = model_manager.create_model(model_type, config)
    
    # Emit
    hotspots = sim_state.get('hotspots', [[10.0, 30.0]])
    npph = sim_state.get('npph', 2500)
    model_state = model.emit_particles(model_state, hotspots, npph)
    
    sim_state['model_state'] = model_state
    
    # Update legacy state
    if model_type in ['lagrangian', 'hybrid']:
        sim_state['particles'] = model_state.get('particles')
        sim_state['particle_active'] = model_state.get('particle_active')
    
    return sim_state
