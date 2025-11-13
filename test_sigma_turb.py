"""
Test to verify turbulent diffusion (sigma_turb) is working correctly
"""
import numpy as np
from simulation_state import sim_state
from particle_physics import initialize_particles, advect
from wind_field import get_wind_at_particles
import simulation_state

# Test 1: Verify sigma_turb scaling with domain size
print("="*70)
print("TEST 1: Sigma_turb Scaling with Domain Size")
print("="*70)

domain_sizes = [5, 10, 50, 100, 200, 500]  # km
base_domain = 50.0
base_sigma = 2.5

print(f"\nBase: {base_domain}km domain → σ_turb = {base_sigma} m/s")
print("\nScaled values (using sqrt scaling for diffusion):")

for size in domain_sizes:
    scale_factor = size / base_domain
    scaled_sigma = base_sigma * np.sqrt(scale_factor)
    print(f"  {size:3d}km domain → σ_turb = {scaled_sigma:.2f} m/s (scale factor: {scale_factor:.2f})")

# Test 2: Verify sigma_turb affects particle spreading
print("\n" + "="*70)
print("TEST 2: Verify Turbulent Diffusion is Active")
print("="*70)

# Create test particles at a single point
n_particles = 1000
particles = np.zeros((n_particles, 3))  # x, y, z (height)
particles[:, 0] = 25.0  # Center of 50km domain
particles[:, 1] = 25.0
particles[:, 2] = 100.0  # 100m above ground
particle_active = np.ones(n_particles, dtype=bool)

# Store initial positions
initial_positions = particles.copy()

# Test with different sigma_turb values
sigma_values = [0.0, 1.0, 2.5, 5.0, 10.0]
dt = 30.0  # seconds
t = 0.0

print(f"\nInitial: All {n_particles} particles at (25.0, 25.0) km")
print(f"Time step: {dt} seconds")
print(f"Wind type: synthetic (for consistent testing)")
print()

wind_interpolators = {'U_interp': None, 'V_interp': None, 'last_update_time': -1}
wind_data_cache = {'loaded': False}

for sigma in sigma_values:
    # Reset particles
    test_particles = initial_positions.copy()
    test_active = np.ones(n_particles, dtype=bool)
    
    # Run one time step with this sigma_turb
    test_particles = advect(
        test_particles, test_active, t, dt, sigma,
        'synthetic', wind_interpolators,
        'absorbing', True, 0.001, 1000.0,
        False, 0.0,
        simulation_state.x, simulation_state.y,
        simulation_state.nx, simulation_state.ny,
        simulation_state.xmin, simulation_state.xmax,
        simulation_state.ymin, simulation_state.ymax,
        wind_data_cache, get_wind_at_particles,
        1000.0, True
    )
    
    # Calculate spread
    std_x = np.std(test_particles[test_active, 0])
    std_y = np.std(test_particles[test_active, 1])
    total_spread = np.sqrt(std_x**2 + std_y**2)
    
    # Theoretical diffusion spread (in km)
    theoretical_spread_m = np.sqrt(2 * sigma * dt)
    theoretical_spread_km = theoretical_spread_m / 1000.0
    
    print(f"σ_turb = {sigma:4.1f} m/s:")
    print(f"  Observed spread:    {total_spread:.4f} km")
    print(f"  Theoretical spread: {theoretical_spread_km:.4f} km")
    if theoretical_spread_km > 0:
        ratio = total_spread/theoretical_spread_km
        print(f"  Ratio (obs/theory): {ratio:.2f}")
    else:
        print(f"  Ratio (obs/theory): N/A (no diffusion)")
    print()

# Test 3: Current simulation state
print("="*70)
print("TEST 3: Current Simulation State")
print("="*70)

print(f"\nCurrent sigma_turb: {sim_state['sigma_turb']} m/s")
print(f"Domain size: {simulation_state.xmax - simulation_state.xmin} × {simulation_state.ymax - simulation_state.ymin} km")
print(f"Time step (dt): {sim_state['dt']} seconds")

# Calculate expected diffusion per step
expected_diffusion_m = np.sqrt(2 * sim_state['sigma_turb'] * sim_state['dt'])
expected_diffusion_km = expected_diffusion_m / 1000.0

print(f"\nExpected particle diffusion per step:")
print(f"  {expected_diffusion_m:.2f} meters")
print(f"  {expected_diffusion_km:.4f} km")
print(f"  ~{expected_diffusion_km * 100:.2f}% of domain width per step")

# Verify it's reasonable
resolution_km = (simulation_state.xmax - simulation_state.xmin) / simulation_state.nx
print(f"\nGrid resolution: {resolution_km:.4f} km = {resolution_km * 1000:.1f} meters")
print(f"Diffusion/Resolution ratio: {expected_diffusion_m / (resolution_km * 1000):.2f}")

if expected_diffusion_m / (resolution_km * 1000) < 0.1:
    print("⚠️  WARNING: Diffusion is very small compared to grid resolution!")
elif expected_diffusion_m / (resolution_km * 1000) > 10:
    print("⚠️  WARNING: Diffusion is very large compared to grid resolution!")
else:
    print("✅ Diffusion is reasonable compared to grid resolution")

print("\n" + "="*70)
print("CONCLUSION")
print("="*70)
print("\nIf observed spread matches theoretical spread in Test 2,")
print("then turbulent diffusion is working correctly!")
print("\nThe spreading should increase with sigma_turb value.")
print("="*70)
