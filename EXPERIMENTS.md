# Recommended Experiments & Validation Tests

## Purpose
This document outlines validation experiments and test scenarios for your Lagrangian Transport Simulator final project.

## 1. Basic Validation Experiments

### Experiment 1.1: Point Source Plume Test
**Objective:** Verify basic dispersion physics

**Setup:**
- Single emission hotspot at domain center
- No wind (u=0, v=0)
- σ_turb = 0.5
- Total mass = 1000 g
- Run for 50 steps

**Expected Result:**
- Radially symmetric plume
- Gaussian-like concentration profile
- Maximum concentration at source decreases with time
- Total mass conserved

**Validation:**
```
Check: max_concentration × cell_volume ≈ total_mass
       (accounting for spreading)
```

### Experiment 1.2: Uniform Wind Transport
**Objective:** Validate advection

**Setup:**
- Single hotspot at (lat_min + 0.2, lon_min + 0.2)
- Uniform wind: u = 5 m/s, v = 0 m/s (eastward)
- σ_turb = 0.3
- Run for 100 steps

**Expected Result:**
- Plume moves eastward
- Plume center displacement ≈ u × timestep × n_steps
- Plume shape: elongated in wind direction
- Concentration peak decreases as plume spreads

**Validation:**
```python
# Calculate plume centroid movement
displacement_expected = 5 m/s × 600s × 100 = 300 km
# Compare with actual centroid position
```

### Experiment 1.3: Multiple Sources
**Objective:** Test superposition of plumes

**Setup:**
- 3 hotspots in triangle formation
- Each emitting 500 g
- Wind: u=2, v=2 m/s (northeast)
- σ_turb = 0.4

**Expected Result:**
- Three distinct plumes initially
- Plumes merge downstream
- Combined concentration = sum of individual plumes
- No spurious interaction

## 2. Physical Realism Tests

### Experiment 2.1: Diurnal Wind Variation
**Objective:** Test response to changing wind fields

**Procedure:**
1. Run with ERA5 data for specific date/time
2. Compare with synthetic wind field
3. Document differences in plume evolution

**Analysis:**
- How does realistic wind shear affect plume?
- Do eddies/convergence zones trap pollutants?
- Compare max concentration locations

### Experiment 2.2: Mixing Height Sensitivity
**Objective:** Understand vertical mixing impact

**Test Cases:**
```python
scenarios = [
    {"H_mix": 500, "description": "Stable morning"},
    {"H_mix": 1000, "description": "Neutral"},
    {"H_mix": 2000, "description": "Well-mixed afternoon"},
]
```

**Expected Results:**
- Lower H_mix → higher concentrations (C ∝ 1/H_mix)
- 500m mixing: ~2× concentration vs 1000m
- 2000m mixing: ~0.5× concentration vs 1000m

**Analysis:**
```python
# For same emission and area:
# C_500 / C_1000 should ≈ 2.0
# C_2000 / C_1000 should ≈ 0.5
```

### Experiment 2.3: Emission Factor Validation
**Objective:** Test FRP-to-emission conversion

**Reference Data:**
Use MODIS/VIIRS FRP observations:
- Small fire: FRP = 10-50 MW
- Medium fire: FRP = 50-200 MW
- Large fire: FRP = 200-1000 MW

**Test:**
```python
# Small savanna fire
frp_to_emission(
    frp_mw=30,
    duration_hours=2,
    pollutant="PM2.5",
    vegetation_type="savanna"
)
# Expected PM2.5: ~369 g

# Large tropical forest fire
frp_to_emission(
    frp_mw=500,
    duration_hours=6,
    pollutant="PM2.5",
    vegetation_type="tropical_forest"
)
# Expected PM2.5: ~74,239 g (74 kg)
```

**Validation:**
- Compare with literature values (Ichoku & Kaufman, 2005)
- Check if emissions match FINN inventory for known fires

## 3. Air Quality Standard Tests

### Experiment 3.1: WHO Guideline Exceedance
**Objective:** Assess health impact scenarios

**WHO PM2.5 Standards:**
- 24-hour mean: 15 µg/m³
- Annual mean: 5 µg/m³
- Interim target 1: 35 µg/m³

**Scenario:**
- Wildfire emitting 50 kg PM2.5 over 4 hours
- Wind speed: 3 m/s
- Distance to populated area: 50 km

**Questions:**
1. At what distance does concentration drop below 35 µg/m³?
2. How long does exceedance persist?
3. What wind speed would keep levels safe?

**Analysis:**
```python
# Run simulation and extract concentration time series
# at specific location (e.g., 50 km downwind)
distances = [10, 25, 50, 75, 100]  # km
for d in distances:
    C_at_distance = get_concentration(lat, lon)
    if C_at_distance < 35:
        print(f"Safe distance: {d} km")
        break
```

### Experiment 3.2: Evacuation Planning
**Objective:** Model smoke dispersion for emergency response

**Scenario:**
- Large peat fire (high PM2.5 emission factor: 16 g/kg)
- FRP = 300 MW, duration = 12 hours
- Variable wind conditions (use ERA5 data)

**Deliverables:**
1. Concentration heatmap at T = 6, 12, 24 hours
2. Time to reach threshold at 10, 25, 50 km
3. Area exceeding 100 µg/m³ (hazardous level)

## 4. Sensitivity Analysis

### Experiment 4.1: Turbulence Parameter
**Objective:** Understand σ_turb impact

**Test Range:**
```python
sigma_values = [0.1, 0.3, 0.5, 0.7, 1.0]
```

**Metrics to Compare:**
- Plume width at 50 km downwind
- Maximum concentration decay rate
- Time to background levels

**Expected Trend:**
- Higher σ_turb → faster spreading
- Lower peak concentration
- More diffuse plume

### Experiment 4.2: Particle Number
**Objective:** Convergence test

**Test Cases:**
```python
n_particles = [500, 1000, 2500, 5000, 10000]
```

**Validation:**
- Results should converge for N > 2500
- Max concentration should stabilize
- Check computational time vs accuracy trade-off

### Experiment 4.3: Grid Resolution
**Objective:** Spatial resolution impact

**Current:** 120 × 120 (~1.67 km cells)

**Test:**
- Coarse: 60 × 60 (~3.33 km cells)
- Fine: 180 × 180 (~1.11 km cells)

**Compare:**
- Peak concentration capture
- Plume structure detail
- Computation time

## 5. Real-World Case Studies

### Case Study 5.1: 2023 Canadian Wildfires
**Data Needed:**
- FINN inventory for June-August 2023
- ERA5 wind data for region
- Ground truth: EPA AirNow PM2.5 measurements

**Procedure:**
1. Extract emission hotspots from FINN
2. Calculate emissions using your FRP converter
3. Run simulation with ERA5 winds
4. Compare modeled PM2.5 with AirNow observations

**Success Metrics:**
- Correlation > 0.6 with observations
- Capture major transport events
- Identify model limitations

### Case Study 5.2: Agricultural Burning (India/Southeast Asia)
**Focus:** Crop residue burning season

**Data:**
- VIIRS/MODIS active fires (Oct-Nov)
- Crop residue emission factors
- ERA5 data

**Analysis:**
- Seasonal air quality impact
- Urban exposure levels (Delhi, Bangkok)
- Effect of meteorology on dispersion

### Case Study 5.3: Prescribed Burns (Controlled)
**Advantage:** Known emission timing and amount

**Data:**
- Prescribed burn reports (area, fuel type, duration)
- Pre/post air quality monitoring

**Validation:**
- Compare predicted vs measured PM2.5
- Test emission factor accuracy
- Refine dispersion parameters

## 6. Advanced Experiments (Future Work)

### Experiment 6.1: Deposition Effects
**Addition:** Dry deposition velocity (v_d)

**Implementation:**
```python
# In particle update:
vertical_loss = v_d / H_mix * dt  # fraction lost per timestep
particle_weight *= (1 - vertical_loss)
```

**Test:**
- v_d = 0.001 m/s for PM2.5
- Compare total mass evolution with/without deposition
- Expected: ~5-10% loss over 24 hours

### Experiment 6.2: Nighttime Stability
**Feature:** Reduced mixing height at night

**Scenario:**
```python
# Day: H_mix = 1500 m
# Night: H_mix = 300 m
# Transition at sunrise/sunset
```

**Expected:**
- Nighttime fumigation events
- Higher concentrations in stable conditions
- Morning mixing dilution

### Experiment 6.3: Complex Terrain (Future 3D)
**Challenge:** Orographic effects

**Implementation:**
- Add elevation data
- Modify wind field for terrain blocking
- Include valley channeling

## 7. Documentation Requirements

### For Final Report/Presentation

**Include:**
1. **Parameter Table**
   ```
   | Parameter | Value | Source/Justification |
   |-----------|-------|---------------------|
   | H_mix     | 1000m | Typical daytime BL  |
   | σ_turb    | 0.5   | Moderate turbulence |
   | dt        | 600s  | Numerical stability |
   | ...       | ...   | ...                 |
   ```

2. **Validation Plots**
   - Concentration vs distance (compare with Gaussian plume)
   - Mass conservation time series
   - Sensitivity analysis graphs

3. **Error Analysis**
   - Uncertainty in emission factors (±30-50%)
   - Wind field uncertainty
   - Grid resolution effects
   - Particle number convergence

4. **Limitations Discussion**
   - 2D vs 3D reality
   - No chemistry/deposition
   - Simplified mixing height
   - Boundary effects

5. **Comparison with Existing Tools**
   - HYSPLIT (NOAA)
   - CALPUFF
   - WRF-Chem
   - Advantages/disadvantages of your approach

## 8. Performance Benchmarks

### Computational Efficiency

**Target Performance:**
```python
# Benchmark configuration
N_particles = 2500
N_grid = 120 × 120
N_steps = 100

# Expected time: < 30 seconds for full simulation
# Memory usage: < 500 MB
```

**Optimization Ideas:**
- Vectorized numpy operations (already done)
- Numba JIT compilation for particle updates
- Sparse grid representation
- Parallel processing for multiple scenarios

## Quick Start: Run Your First Experiment

```bash
# 1. Start the app
python app.py

# 2. Open browser to http://localhost:5000

# 3. Set up Experiment 1.1 (Point Source)
- Add hotspot at center: (lat_min+1, lon_min+1)
- Set wind to (0, 0)
- Set σ_turb = 0.5
- Set emissions: 1000g PM2.5, savanna
- Click Play

# 4. Observe results
- Watch radial spreading
- Note max concentration decreasing
- Check mass conservation in console

# 5. Export data
- Save frame images
- Record concentration values
- Document in lab notebook
```

## References for Validation

1. Gaussian Plume Model (comparison)
2. Pasquill-Gifford Stability Classes
3. EPA AP-42 Emission Factors
4. FINN Fire Inventory (2023 data)
5. AirNow Air Quality Observations
6. MODIS/VIIRS Active Fire Data
7. ERA5 Reanalysis Documentation

## Success Criteria

Your implementation is successful if:

✅ Mass is conserved (within 5% over 100 steps)  
✅ Plume moves realistically with wind  
✅ Concentrations decrease with distance  
✅ Results are reproducible  
✅ Physical units are correct (µg/m³)  
✅ Emission calculations match literature  
✅ Can replicate published case studies  

Good luck with your experiments! 🔬🌍
