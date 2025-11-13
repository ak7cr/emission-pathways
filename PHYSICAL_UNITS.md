# Physical Units & Emission Estimation - Implementation Guide

## Overview
This document describes the physical units implementation and emission estimation capabilities added to the Lagrangian Transport Simulator.

## 1. Physical Concentration Calculation

### Formula Implementation

The particle counts are converted to physical concentrations (µg/m³) using:

```
C(i,j,t) = (N(i,j,t) × m_p) / (A_cell × H_mix)
```

Where:
- `C(i,j,t)` = Concentration at grid cell (i,j) at time t [µg/m³]
- `N(i,j,t)` = Number of particles in cell (i,j) at time t
- `m_p` = Mass per particle [µg]
- `A_cell` = Grid cell horizontal area [m²]
- `H_mix` = Mixing layer height [m]

### Mass Per Particle Calculation

```
m_p = M_total / N_tot
```

Where:
- `M_total` = Total emitted mass (sum over all hotspots) [µg]
- `N_tot` = Total number of particles

### Default Parameters

- **Mixing Height**: 1000 m (typical daytime boundary layer)
- **Total Mass per Hotspot**: 1000 g (1 kg)
- **Grid Cell Area**: Calculated from domain (200 km × 200 km / 120 × 120 grid)
  - Cell size: ~1.67 km × 1.67 km
  - Cell area: ~2.78 × 10⁶ m²

## 2. Emission Estimation

### A. From Fire Radiative Power (FRP)

#### FRP to Fuel Consumption

Based on Wooster et al. (2005):

```python
FRE = FRP × duration × 3600  # MJ
fuel_consumed = FRE / 0.368  # kg
```

Where:
- FRP = Fire Radiative Power [MW]
- duration = Fire duration [hours]
- FRE = Fire Radiative Energy [MJ]
- 0.368 = Wooster coefficient

#### Fuel to Emissions

```python
Emission = fuel_consumed × EF × CE
```

Where:
- EF = Emission Factor [g/kg] - pollutant and vegetation specific
- CE = Combustion Efficiency (typically 0.5)

### B. Emission Factors Database

Included emission factors (g/kg dry matter) for:

**Pollutants:**
- PM2.5 (Primary particulate matter < 2.5 µm)
- PM10 (Primary particulate matter < 10 µm)
- CO (Carbon monoxide)
- CO2 (Carbon dioxide)
- NOx (Nitrogen oxides)
- SO2 (Sulfur dioxide)

**Vegetation Types:**
- Savanna
- Tropical forest
- Extratropical forest
- Crop residue
- Peat

**Example Emission Factors (PM2.5):**
```python
{
    'savanna': 3.4 g/kg,
    'tropical_forest': 9.1 g/kg,
    'extratropical_forest': 13.0 g/kg,
    'crop_residue': 3.9 g/kg,
    'peat': 16.0 g/kg
}
```

## 3. API Endpoints

### GET /api/emissions
Get current emission parameters and calculated values.

**Response:**
```json
{
    "success": true,
    "emissions": {
        "pollutant_type": "PM2.5",
        "emission_factor": 1.5,
        "total_mass_per_hotspot": 1000.0,
        "mass_per_particle": 0.133,
        "mixing_height": 1000.0,
        "cell_area": 2777777.78,
        "num_hotspots": 3,
        "total_particles": 7500
    }
}
```

### POST /api/emissions
Update emission parameters.

**Request:**
```json
{
    "pollutant_type": "PM2.5",
    "total_mass_per_hotspot": 1500.0,
    "mixing_height": 1200.0,
    "emission_factor": 9.1
}
```

### POST /api/emissions/from-frp
Calculate emissions from FRP.

**Request:**
```json
{
    "frp_mw": 100.0,
    "duration_hours": 2.0,
    "pollutant": "PM2.5",
    "vegetation_type": "tropical_forest",
    "combustion_efficiency": 0.5
}
```

**Response:**
```json
{
    "success": true,
    "emission_g": 4947.83,
    "emission_factor": 9.1,
    "pollutant": "PM2.5",
    "vegetation_type": "tropical_forest"
}
```

### GET /api/emissions/factors
Get all available emission factors.

**Response:**
```json
{
    "success": true,
    "pollutants": ["PM2.5", "PM10", "CO", "CO2", "NOx", "SO2"],
    "vegetation_types": ["savanna", "tropical_forest", ...],
    "emission_factors": { ... }
}
```

### Enhanced POST /api/step
Now returns concentration statistics.

**Response (additional fields):**
```json
{
    "concentration": {
        "max_ugm3": 45.23,
        "mean_ugm3": 2.15,
        "total_mass_g": 2847.5
    }
}
```

## 4. Usage Examples

### Example 1: Using FRP from Satellite Data

```python
# Satellite detects FRP = 150 MW
# Fire burning for 3 hours in tropical forest
# Calculate PM2.5 emissions

POST /api/emissions/from-frp
{
    "frp_mw": 150,
    "duration_hours": 3,
    "pollutant": "PM2.5",
    "vegetation_type": "tropical_forest"
}

# This will:
# 1. Calculate fuel consumed: ~4421 kg
# 2. Calculate PM2.5 emission: ~20,120 g
# 3. Update simulation with this mass
```

### Example 2: Custom Emission Scenario

```python
# Set up custom emission for agricultural burning
POST /api/emissions
{
    "pollutant_type": "PM2.5",
    "total_mass_per_hotspot": 500,  # 500g per hotspot
    "emission_factor": 3.9,  # crop residue EF
    "mixing_height": 800  # shallow mixing
}

# Run simulation to see dispersion
```

### Example 3: Monitor Concentration Levels

```python
# Step through simulation and check concentrations
for step in range(100):
    response = POST /api/step
    max_conc = response["concentration"]["max_ugm3"]
    
    # Check if exceeding air quality standards
    if max_conc > 35:  # µg/m³ (WHO PM2.5 24h guideline)
        print(f"Warning: Exceeding air quality guideline!")
```

## 5. Integration with External Data

### FINN/GFED Data Integration (Future)

To integrate fire emissions inventories:

```python
# Example FINN data structure
finn_data = {
    "lat": 10.5,
    "lon": 120.3,
    "frp": 85.5,  # MW
    "area": 0.5,  # km²
    "landcover": "tropical_forest",
    "datetime": "2024-11-13T14:00:00"
}

# Convert to hotspot with emissions
emission = frp_to_emission(
    frp_mw=finn_data["frp"],
    pollutant="PM2.5",
    vegetation_type=finn_data["landcover"]
)

# Add to simulation
hotspot = [finn_data["lat"], finn_data["lon"]]
```

## 6. Validation and Calibration

### Recommended Checks

1. **Mass Conservation**
   ```python
   total_emitted = total_mass_per_hotspot * num_hotspots
   total_in_domain = sum(concentration × cell_volume)
   # Should be approximately equal (accounting for domain boundaries)
   ```

2. **Concentration Range Validation**
   - Near source: 10-1000 µg/m³ (varies with emissions)
   - Background: < 1 µg/m³
   - Compare with EPA/WHO air quality standards

3. **Dispersion Rate**
   - Check that plume spreads realistically with wind
   - Verify turbulent diffusion matches atmospheric conditions

## 7. Future Enhancements

### Planned Features

1. **3D Extension**
   - Add vertical dimension (z-axis)
   - Plume rise calculations
   - Vertical wind shear
   - Buoyancy effects

2. **Time-varying Emissions**
   - Support emission rate curves
   - Diurnal variation
   - Multi-event scenarios

3. **Chemical Transformation**
   - Simple decay/deposition
   - Gas-to-particle conversion
   - Photochemical reactions

4. **Deposition**
   - Dry deposition velocity
   - Wet deposition (rainfall)
   - Surface loss rates

5. **Health Impact Assessment**
   - Population exposure calculation
   - AQI computation
   - Dose-response relationships

## References

1. Wooster, M. J., et al. (2005). "Retrieval of biomass combustion rates and totals from fire radiative power observations"
2. Andreae, M. O., & Merlet, P. (2001). "Emission of trace gases and aerosols from biomass burning"
3. WHO Air Quality Guidelines (2021)
4. US EPA National Ambient Air Quality Standards

## File Structure

```
emission-pathways/
├── app.py                    # Main Flask application (enhanced)
├── emission_utils.py         # NEW: Emission calculations
├── wind_api.py              # Wind data fetching
├── templates/
│   └── index.html           # Web interface
├── README.md
└── PHYSICAL_UNITS.md        # This file
```
