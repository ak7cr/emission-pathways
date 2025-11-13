# Extension to 3D - Implementation Guide

## Overview
This document provides guidance for extending the current 2D Lagrangian Transport Simulator to 3D, including vertical dispersion, plume rise, and 3D visualization.

## 1. Why 3D?

### Limitations of 2D
- Assumes instant vertical mixing within boundary layer
- Cannot capture vertical wind shear
- No plume rise from buoyancy
- Oversimplifies complex atmospheric structure

### Benefits of 3D
- Realistic vertical structure
- Buoyant plume rise from hot emissions
- Vertical wind shear effects
- Better long-range transport
- More accurate concentration predictions

## 2. Data Structure Changes

### Current 2D Particle State
```python
particles = {
    'lat': array of shape (N,),
    'lon': array of shape (N,),
    'source': array of shape (N,)  # hotspot index
}
```

### New 3D Particle State
```python
particles = {
    'lat': array of shape (N,),
    'lon': array of shape (N,),
    'z': array of shape (N,),      # HEIGHT above ground [m]
    'w': array of shape (N,),      # vertical velocity [m/s]
    'temp': array of shape (N,),   # temperature [K]
    'source': array of shape (N,)
}
```

### 3D Grid
```python
# Current 2D grid
H = np.zeros((nx, ny))  # concentration

# New 3D grid
H = np.zeros((nx, ny, nz))  # concentration
# where nz = number of vertical levels (e.g., 20)
```

## 3. Vertical Coordinate System

### Option A: Height Coordinates (Simpler)
```python
z_levels = np.array([
    0, 50, 100, 150, 250, 350, 450, 550, 700, 900, 
    1100, 1400, 1800, 2200, 2700, 3300, 4000, 5000, 6000, 8000
])  # meters above ground level (AGL)
```

**Advantages:**
- Intuitive
- Easy to implement
- Good for boundary layer modeling

**Disadvantages:**
- Doesn't follow fluid motion in upper atmosphere
- Terrain complications

### Option B: Pressure Coordinates (More realistic)
```python
p_levels = np.array([
    1000, 975, 950, 925, 900, 850, 800, 750, 700, 650,
    600, 550, 500, 450, 400, 350, 300, 250, 200, 150
])  # hPa (millibars)
```

**Advantages:**
- Standard in meteorology
- Matches ERA5 data format
- Better for large-scale transport

**Disadvantages:**
- More complex to implement
- Need pressure-to-height conversion

### Recommendation for Your Project
Start with **height coordinates** (Option A) for simplicity.

## 4. Vertical Wind Component

### Obtaining Vertical Wind (w)

#### Method 1: ERA5 Vertical Velocity
```python
# ERA5 provides 'w' in Pa/s (pressure vertical velocity)
# Convert to m/s using:
w_ms = -w_pas * (R * T) / (p * g)
# where:
#   R = 287 J/(kg·K)  - gas constant
#   T = temperature [K]
#   p = pressure [Pa]
#   g = 9.81 m/s²
```

#### Method 2: Mass Continuity Equation
For incompressible flow:
```python
∂u/∂x + ∂v/∂y + ∂w/∂z = 0

# Finite difference:
w[z+1] = w[z] - Δz * (∂u/∂x + ∂v/∂y)
```

**Implementation:**
```python
def compute_vertical_wind(u, v, z_levels):
    """
    u, v: 3D arrays (nx, ny, nz) [m/s]
    z_levels: 1D array (nz,) [m]
    Returns w: 3D array (nx, ny, nz) [m/s]
    """
    w = np.zeros_like(u)
    
    # Compute horizontal divergence
    dudx = np.gradient(u, axis=0) / dx
    dvdy = np.gradient(v, axis=1) / dy
    div_h = dudx + dvdy
    
    # Integrate upward from surface (w=0 at ground)
    for k in range(1, nz):
        dz = z_levels[k] - z_levels[k-1]
        w[:, :, k] = w[:, :, k-1] - dz * div_h[:, :, k-1]
    
    return w
```

#### Method 3: Prescribed Vertical Motion (Simplest)
For initial testing:
```python
# Uniform subsidence (typical in high-pressure systems)
w = -0.01  # m/s (1 cm/s downward)

# Or no vertical motion
w = 0.0
```

## 5. Plume Rise Physics

### Buoyancy-Driven Rise

For hot emissions (wildfires), particles initially rise due to buoyancy:

```python
def calculate_plume_rise(T_plume, T_ambient, z_initial, dt):
    """
    Simplified plume rise equation (Briggs, 1975)
    
    Parameters:
    T_plume: Plume temperature [K]
    T_ambient: Ambient temperature [K]
    z_initial: Initial height [m]
    dt: timestep [s]
    
    Returns:
    z_final: Final height [m]
    w_plume: Vertical velocity [m/s]
    """
    # Buoyancy parameter
    g = 9.81  # m/s²
    F_b = g * (T_plume - T_ambient) / T_ambient  # buoyancy flux
    
    # Plume rise velocity (Briggs)
    w_plume = (F_b * z_initial / 3.5)**(1/3)
    
    # Update height
    z_final = z_initial + w_plume * dt
    
    # Temperature decay (entrainment)
    T_plume_new = T_plume - 0.1 * (T_plume - T_ambient) * dt / 60
    
    return z_final, w_plume, T_plume_new
```

### Typical Fire Parameters
```python
# Small fire
T_fire = 400 K  # 127°C
T_ambient = 298 K  # 25°C

# Large fire
T_fire = 800 K  # 527°C
T_ambient = 298 K

# Injection height (final rise)
# Small: 500-1500 m
# Medium: 1500-3500 m
# Large: 3500-8000 m (pyrocumulonimbus)
```

## 6. 3D Particle Update Scheme

### Enhanced Update Algorithm

```python
def update_particles_3d(particles, u, v, w, T_ambient, dt, sigma_turb):
    """
    3D Lagrangian particle update with vertical motion
    """
    N = len(particles['lat'])
    
    # Interpolate wind at particle positions
    u_p = interpolate_3d(u, particles['lat'], particles['lon'], particles['z'])
    v_p = interpolate_3d(v, particles['lat'], particles['lon'], particles['z'])
    w_p = interpolate_3d(w, particles['lat'], particles['lon'], particles['z'])
    T_p = interpolate_3d(T_ambient, particles['lat'], particles['lon'], particles['z'])
    
    # Advection
    particles['lat'] += (v_p * dt / 111000)
    particles['lon'] += (u_p * dt / (111000 * np.cos(np.deg2rad(particles['lat']))))
    particles['z'] += w_p * dt
    
    # Buoyancy (for hot particles)
    hot_mask = particles['temp'] > T_p
    if np.any(hot_mask):
        z_new, w_new, T_new = calculate_plume_rise(
            particles['temp'][hot_mask],
            T_p[hot_mask],
            particles['z'][hot_mask],
            dt
        )
        particles['z'][hot_mask] = z_new
        particles['w'][hot_mask] = w_new
        particles['temp'][hot_mask] = T_new
    
    # Turbulent diffusion (3D)
    particles['lat'] += np.random.normal(0, sigma_turb, N) * dt / 111000
    particles['lon'] += np.random.normal(0, sigma_turb, N) * dt / (111000 * np.cos(np.deg2rad(particles['lat'])))
    particles['z'] += np.random.normal(0, sigma_turb, N) * dt
    
    # Boundary conditions
    # Vertical: reflection at ground and top
    particles['z'] = np.clip(particles['z'], 0, z_max)
    
    # Ground reflection (elastic)
    ground_hit = particles['z'] < 0
    particles['z'][ground_hit] = -particles['z'][ground_hit]
    particles['w'][ground_hit] = -particles['w'][ground_hit]
    
    return particles
```

## 7. 3D Concentration Field

### Grid Assignment

```python
def concentration_field_3d(particles, domain, grid_size, z_levels, mass_per_particle):
    """
    Compute 3D concentration field
    
    Returns:
    C: 3D array (nx, ny, nz) with concentration [µg/m³]
    """
    nx, ny, nz = grid_size[0], grid_size[1], len(z_levels)
    
    # Initialize
    H = np.zeros((nx, ny, nz))  # particle count
    
    # Grid cell volumes
    dx = (domain['lon_max'] - domain['lon_min']) / nx * 111000  # m
    dy = (domain['lat_max'] - domain['lat_min']) / ny * 111000  # m
    dz = np.diff(z_levels)  # m
    
    # Assign particles to grid
    lat_idx = np.floor((particles['lat'] - domain['lat_min']) / 
                       (domain['lat_max'] - domain['lat_min']) * nx).astype(int)
    lon_idx = np.floor((particles['lon'] - domain['lon_min']) / 
                       (domain['lon_max'] - domain['lon_min']) * ny).astype(int)
    z_idx = np.searchsorted(z_levels, particles['z']) - 1
    
    # Clip to domain
    valid = (lat_idx >= 0) & (lat_idx < nx) & \
            (lon_idx >= 0) & (lon_idx < ny) & \
            (z_idx >= 0) & (z_idx < nz)
    
    # Count particles per cell
    for i in range(len(particles['lat'])):
        if valid[i]:
            H[lat_idx[i], lon_idx[i], z_idx[i]] += 1
    
    # Convert to concentration [µg/m³]
    C = np.zeros_like(H)
    for k in range(nz):
        if k < nz - 1:
            dz_k = z_levels[k+1] - z_levels[k]
        else:
            dz_k = dz[-1]
        
        volume = dx * dy * dz_k  # m³
        C[:, :, k] = (H[:, :, k] * mass_per_particle) / volume
    
    return C
```

## 8. Vertical Integration for Comparison

### Column-Integrated Mass (for validation)

```python
def column_integrated_mass(C_3d, z_levels):
    """
    Integrate concentration vertically
    Similar to satellite column measurements
    
    Returns:
    column_mass: 2D array (nx, ny) [µg/m²]
    """
    dz = np.diff(z_levels)
    
    # Integrate C * dz over vertical
    column_mass = np.zeros((C_3d.shape[0], C_3d.shape[1]))
    
    for k in range(len(dz)):
        column_mass += C_3d[:, :, k] * dz[k]
    
    return column_mass
```

This is useful for:
- Comparing with satellite AOD (Aerosol Optical Depth)
- Validation against MODIS/VIIRS observations
- Mass balance checks

## 9. 3D Visualization

### Option A: Multiple Horizontal Slices
```python
import matplotlib.pyplot as plt

def plot_vertical_slices(C_3d, z_levels):
    """
    Plot concentration at different heights
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    levels_to_plot = [0, 3, 6, 9, 12, 15]  # indices
    
    for idx, ax in enumerate(axes.flat):
        k = levels_to_plot[idx]
        im = ax.contourf(C_3d[:, :, k], levels=20, cmap='YlOrRd')
        ax.set_title(f'z = {z_levels[k]:.0f} m')
        plt.colorbar(im, ax=ax, label='µg/m³')
    
    plt.tight_layout()
    return fig
```

### Option B: Vertical Cross-Section
```python
def plot_vertical_cross_section(C_3d, z_levels, lat_index):
    """
    Plot concentration in vertical plane (lon-z)
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Extract cross-section
    C_cross = C_3d[lat_index, :, :]  # (ny, nz)
    
    # Plot
    im = ax.contourf(C_cross.T, levels=20, cmap='YlOrRd', extend='max')
    ax.set_xlabel('Longitude index')
    ax.set_ylabel('Height [m]')
    ax.set_yticks(range(len(z_levels)))
    ax.set_yticklabels([f'{z:.0f}' for z in z_levels])
    
    plt.colorbar(im, label='µg/m³')
    plt.title('Vertical Cross-Section')
    
    return fig
```

### Option C: 3D Isosurface (Advanced)
```python
import plotly.graph_objects as go

def plot_3d_isosurface(C_3d, threshold=10):
    """
    3D isosurface visualization using Plotly
    Shows volume where C > threshold
    """
    fig = go.Figure(data=go.Isosurface(
        x=np.arange(C_3d.shape[0]),
        y=np.arange(C_3d.shape[1]),
        z=np.arange(C_3d.shape[2]),
        value=C_3d.flatten(),
        isomin=threshold,
        isomax=C_3d.max(),
        surface_count=5,
        colorscale='YlOrRd',
        caps=dict(x_show=False, y_show=False, z_show=False),
    ))
    
    fig.update_layout(
        title='3D Concentration Field',
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Height'
        )
    )
    
    return fig
```

## 10. Implementation Roadmap

### Phase 1: Basic 3D Structure (Week 1-2)
- [ ] Add z-coordinate to particle state
- [ ] Implement 10-level vertical grid (0-5000m)
- [ ] Add w=0 (no vertical motion initially)
- [ ] Test with 3D particle tracking
- [ ] Validate mass conservation

### Phase 2: Vertical Wind (Week 3)
- [ ] Fetch ERA5 w-velocity data
- [ ] Implement continuity equation w-calculation
- [ ] Add vertical advection to particle update
- [ ] Compare results: w=0 vs w≠0

### Phase 3: Plume Rise (Week 4)
- [ ] Implement buoyancy calculation
- [ ] Add temperature tracking for particles
- [ ] Test with different fire intensities
- [ ] Validate injection heights vs literature

### Phase 4: Visualization (Week 5)
- [ ] Horizontal slice plots
- [ ] Vertical cross-sections
- [ ] Column-integrated mass
- [ ] (Optional) 3D isosurface with Plotly

### Phase 5: Validation (Week 6)
- [ ] Compare 2D vs 3D for same scenario
- [ ] Validate vertical profiles vs observations
- [ ] Test with real fire case (FINN + ERA5)
- [ ] Document differences and improvements

## 11. Testing Strategy

### Test 1: Vertical Diffusion Only
```python
# Release particles at z=500m
# No horizontal wind (u=v=0)
# No vertical wind (w=0)
# Only vertical turbulence

# Expected: Gaussian profile in z-direction
# Check: spread rate matches σ_z = σ_turb * sqrt(t)
```

### Test 2: Vertical Shear
```python
# Wind profile:
u = u0 * (z / z_ref)^α  # power law
# where α = 0.2 (typical for neutral stability)

# Release at ground level
# Expected: Plume tilts with height
```

### Test 3: Plume Rise
```python
# Hot release (T=600K)
# Ambient T=300K
# Expected: particles rise 1000-2000m before leveling off
```

## 12. Computational Considerations

### Memory Scaling
```python
# 2D: nx × ny = 120 × 120 = 14,400 cells
# 3D: nx × ny × nz = 120 × 120 × 20 = 288,000 cells
# → 20× memory increase

# Typical memory usage:
# - Particle state: N × 6 × 8 bytes (lat,lon,z,w,temp,source)
# - Grid: 288,000 × 8 bytes ≈ 2.3 MB per field
# - Total for 5000 particles: ~5 MB (manageable)
```

### Speed Optimization
```python
# Bottlenecks in 3D:
1. 3D interpolation (use scipy.RegularGridInterpolator)
2. Particle-to-grid assignment (use numpy histogram3d)
3. Plotting (use downsampled grid for display)
```

### Optimization Tips
```python
# Use vectorized operations
from scipy.interpolate import RegularGridInterpolator

# Pre-create interpolator
u_interp = RegularGridInterpolator((lat_grid, lon_grid, z_levels), u)

# Interpolate all particles at once
points = np.column_stack([particles['lat'], particles['lon'], particles['z']])
u_p = u_interp(points)  # vectorized!
```

## 13. Extension Ideas (Beyond 3D)

### Deposition
```python
# Add to particle update:
# Dry deposition (gravitational settling)
v_settle = (ρ_p * d_p^2 * g) / (18 * μ_air)  # Stokes law

particles['z'] -= v_settle * dt
```

### Chemistry (Simple Decay)
```python
# Add decay rate
tau_decay = 24 * 3600  # 24-hour lifetime (e.g., for organic aerosol)

particles['mass'] *= np.exp(-dt / tau_decay)
```

### Wet Deposition (Rainfall)
```python
# Scavenging coefficient
Lambda = 1e-4  # s^-1 (typical for light rain)

# Particle loss
particles['mass'] *= np.exp(-Lambda * dt)
```

## 14. Resources & References

### Essential Reading
1. Stohl et al. (1998): "Computation of trajectories using ECMWF data"
2. Briggs (1975): "Plume Rise Predictions"
3. Freitas et al. (2007): "PREP-CHEM-SRC 1.0: Preprocessing of emissions"
4. HYSPLIT User Guide (NOAA ARL)

### Data Sources
- **ERA5 3D Fields**: u, v, w, T on pressure levels
- **CALIPSO**: Vertical aerosol profiles (validation)
- **CALIOP**: Lidar backscatter (plume height)
- **MISR**: Plume height retrievals

### Software Comparisons
- **HYSPLIT**: Industry-standard Lagrangian model
- **FLEXPART**: Advanced features, global coverage
- **STILT**: Receptor-oriented, used for source attribution

## Quick Start for 3D

```python
# Minimal changes to existing code:

# 1. Add z to particles
particles['z'] = np.full(N, 100.0)  # release at 100m

# 2. Add vertical levels
z_levels = np.array([0, 100, 250, 500, 1000, 2000, 5000])

# 3. Initialize 3D grid
H = np.zeros((nx, ny, len(z_levels)))

# 4. Update particles (add z-component)
particles['z'] += w_p * dt + np.random.normal(0, sigma_turb, N) * dt

# 5. Plot at specific height
plt.contourf(H[:, :, 2])  # level 2 (250m)
```

## Conclusion

The 3D extension is a natural progression that adds significant realism to your model. Start simple (just add the z-coordinate), then gradually add complexity (vertical wind, plume rise, 3D visualization). The physics is straightforward, and the implementation builds directly on your existing 2D code.

**Recommended timeline:** 4-6 weeks for full 3D implementation and validation.

Good luck! 🚀
