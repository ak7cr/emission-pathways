"""
Test script to verify modular components work correctly
"""
import numpy as np
from simulation_state import (
    sim_state, x, y, nx, ny, xmin, xmax, ymin, ymax, KM_TO_M,
    wind_interpolators
)
from particle_physics import initialize_particles, advect
from wind_field import get_wind_at_particles, synthetic_wind_field

# Initialize particles
print("Initializing particles...")
particles, particle_active = initialize_particles(
    sim_state['hotspots'],
    sim_state['npph']
)
print(f"Created {len(particles)} particles")
print(f"Initial position range: X=[{particles[:, 0].min():.2f}, {particles[:, 0].max():.2f}], Y=[{particles[:, 1].min():.2f}, {particles[:, 1].max():.2f}]")

# Generate wind field
print("\nGenerating wind field...")
t = 0
U, V = synthetic_wind_field(t, x, y, nx, ny)
print(f"Wind U range: [{U.min():.2f}, {U.max():.2f}] m/s")
print(f"Wind V range: [{V.min():.2f}, {V.max():.2f}] m/s")

# Test advection
print("\nAdvecting particles...")
sim_state['particles'] = particles
sim_state['particle_active'] = particle_active

particles_updated = advect(
    particles,
    particle_active,
    t=0,
    dt=sim_state['dt'],
    sigma_turb=sim_state['sigma_turb'],
    wind_type='synthetic',
    wind_interpolators=wind_interpolators,
    boundary_type=sim_state['boundary_type'],
    enable_deposition=sim_state['enable_deposition'],
    deposition_velocity=sim_state['deposition_velocity'],
    mixing_height=sim_state['mixing_height'],
    enable_decay=sim_state['enable_decay'],
    decay_rate=sim_state['decay_rate'],
    x=x, y=y, nx=nx, ny=ny,
    xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax,
    wind_data_cache={},
    get_wind_func=get_wind_at_particles,
    KM_TO_M=KM_TO_M,
    use_bilinear_interp=True
)

print(f"After advection position range: X=[{particles_updated[:, 0].min():.2f}, {particles_updated[:, 0].max():.2f}], Y=[{particles_updated[:, 1].min():.2f}, {particles_updated[:, 1].max():.2f}]")
print(f"Particle displacement: X={particles_updated[:, 0].mean() - particles[:, 0].mean():.4f} km, Y={particles_updated[:, 1].mean() - particles[:, 1].mean():.4f} km")
print(f"Active particles: {particle_active.sum()} / {len(particle_active)}")

# Try multiple steps
print("\nRunning 10 time steps...")
for i in range(10):
    particles_updated = advect(
        particles_updated,
        particle_active,
        t=i*sim_state['dt'],
        dt=sim_state['dt'],
        sigma_turb=sim_state['sigma_turb'],
        wind_type='synthetic',
        wind_interpolators=wind_interpolators,
        boundary_type=sim_state['boundary_type'],
        enable_deposition=sim_state['enable_deposition'],
        deposition_velocity=sim_state['deposition_velocity'],
        mixing_height=sim_state['mixing_height'],
        enable_decay=sim_state['enable_decay'],
        decay_rate=sim_state['decay_rate'],
        x=x, y=y, nx=nx, ny=ny,
        xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax,
        wind_data_cache={},
        get_wind_func=get_wind_at_particles,
        KM_TO_M=KM_TO_M,
        use_bilinear_interp=True
    )
    
print(f"\nAfter 10 steps position range: X=[{particles_updated[:, 0].min():.2f}, {particles_updated[:, 0].max():.2f}], Y=[{particles_updated[:, 1].min():.2f}, {particles_updated[:, 1].max():.2f}]")
print(f"Active particles: {particle_active.sum()} / {len(particle_active)}")
print("\n✅ Test completed successfully!")
