"""
Test wind speed multiplier functionality
"""
import numpy as np
from simulation_state import sim_state
from particle_physics import advect
from wind_field import get_wind_at_particles
import simulation_state

print("="*70)
print("WIND SPEED MULTIPLIER TEST")
print("="*70)

# Create test particles
n_particles = 100
particles = np.zeros((n_particles, 3))
particles[:, 0] = 25.0  # Center of domain
particles[:, 1] = 25.0
particles[:, 2] = 100.0
particle_active = np.ones(n_particles, dtype=bool)

# Test parameters
dt = 30.0  # seconds
t = 0.0
sigma_turb = 0.0  # No diffusion - only wind transport

wind_interpolators = {'U_interp': None, 'V_interp': None, 'last_update_time': -1}
wind_data_cache = {'loaded': False}

# Test different wind speed multipliers
multipliers = [0.1, 0.5, 1.0, 2.0, 5.0]

print(f"\nInitial position: (25.0, 25.0) km")
print(f"Time step: {dt} seconds")
print(f"Turbulent diffusion: OFF (σ_turb = 0)")
print(f"\nTesting wind speed multiplier effect:\n")

initial_pos = particles.copy()

for mult in multipliers:
    # Reset particles
    test_particles = initial_pos.copy()
    test_active = np.ones(n_particles, dtype=bool)
    
    # Run one timestep with this multiplier
    test_particles = advect(
        test_particles, test_active, t, dt, sigma_turb,
        'synthetic', wind_interpolators,
        'absorbing', False, 0.001, 1000.0,  # Disable deposition for cleaner test
        False, 0.0,
        simulation_state.x, simulation_state.y,
        simulation_state.nx, simulation_state.ny,
        simulation_state.xmin, simulation_state.xmax,
        simulation_state.ymin, simulation_state.ymax,
        wind_data_cache, get_wind_at_particles,
        1000.0, True, mult  # Wind speed multiplier
    )
    
    # Calculate displacement
    dx = test_particles[0, 0] - initial_pos[0, 0]
    dy = test_particles[0, 1] - initial_pos[0, 1]
    distance = np.sqrt(dx**2 + dy**2)
    
    # Convert to meters for easier interpretation
    distance_m = distance * 1000
    
    print(f"Wind Speed Multiplier: {mult:3.1f}x")
    print(f"  Displacement: {distance:.4f} km = {distance_m:.1f} meters")
    print(f"  New position: ({test_particles[0, 0]:.4f}, {test_particles[0, 1]:.4f}) km")
    
    if mult > multipliers[0]:
        # Compare to first multiplier to verify linear scaling
        ratio = distance / distances[0] if 'distances' in locals() and distances[0] > 0 else 0
        expected_ratio = mult / multipliers[0]
        print(f"  Scaling ratio: {ratio:.2f} (expected: {expected_ratio:.2f})")
    
    print()
    
    # Store first distance for comparison
    if mult == multipliers[0]:
        distances = [distance]
    else:
        distances.append(distance)

print("="*70)
print("VERIFICATION")
print("="*70)

# Check if scaling is approximately linear
print("\nChecking if wind speed multiplier scales linearly:")
for i, mult in enumerate(multipliers[1:], 1):
    expected_distance = distances[0] * (mult / multipliers[0])
    actual_distance = distances[i]
    error = abs(actual_distance - expected_distance) / expected_distance * 100
    
    print(f"  {mult}x: Expected {expected_distance*1000:.1f}m, Got {actual_distance*1000:.1f}m (error: {error:.1f}%)")

print("\n✅ If errors are < 1%, wind speed multiplier is working correctly!")

print("\n" + "="*70)
print("SIMULATION STATE")
print("="*70)
print(f"\nCurrent wind_speed_multiplier: {sim_state.get('wind_speed_multiplier', 1.0)}x")
print(f"Domain size: {simulation_state.xmax - simulation_state.xmin} × {simulation_state.ymax - simulation_state.ymin} km")
print(f"Base wind speed (synthetic): ~12 m/s eastward, ~3 m/s northward")
print(f"\nWith multiplier = 2.0x:")
print(f"  Effective wind: ~24 m/s eastward, ~6 m/s northward")
print(f"\nWith multiplier = 0.1x:")
print(f"  Effective wind: ~1.2 m/s eastward, ~0.3 m/s northward (calm conditions)")

print("\n" + "="*70)
