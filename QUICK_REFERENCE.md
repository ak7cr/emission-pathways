# Quick Reference Card

## Essential Formulas

### Concentration Calculation
```
C(i,j,t) = (N(i,j,t) × m_p) / (A_cell × H_mix)
```
- C = Concentration [µg/m³]
- N = Particle count in cell
- m_p = Mass per particle [µg]
- A_cell = Cell area [m²]
- H_mix = Mixing height [m]

### Mass Per Particle
```
m_p = M_total / N_tot
```
- M_total = Total emissions [µg]
- N_tot = Total particle count

### Cell Area
```
A_cell = (domain_width / nx) × (domain_height / ny) × (111000)²
```
- Result in m²
- 111000 = km to m conversion

### FRP to Fuel Consumption
```
FRE = FRP × duration × 3600  [MJ]
fuel = FRE / 0.368  [kg]
```

### Emission Calculation
```
Emission = fuel × EF × CE  [g]
```
- EF = Emission Factor [g/kg]
- CE = Combustion Efficiency (0.5 typical)

## Default Parameters

| Parameter | Value | Unit |
|-----------|-------|------|
| Mixing Height (H_mix) | 1000 | m |
| Total Mass per Hotspot | 1000 | g |
| Grid Size | 120×120 | cells |
| Domain Size | 200×200 | km |
| Cell Size | ~1.67 | km |
| Cell Area | 2.78×10⁶ | m² |
| Timestep (dt) | 600 | s |
| σ_turb | 0.5 | m/s |
| Particles per Hotspot | 2500 | - |

## Emission Factors (PM2.5) [g/kg]

| Vegetation Type | PM2.5 | PM10 | CO |
|----------------|-------|------|-----|
| Savanna | 3.4 | 8.5 | 65 |
| Tropical Forest | 9.1 | 13.0 | 93 |
| Extratropical Forest | 13.0 | 17.6 | 107 |
| Crop Residue | 3.9 | 7.2 | 92 |
| Peat | 16.0 | 24.0 | 210 |

## Air Quality Standards (PM2.5) [µg/m³]

| Standard | Value | Duration |
|----------|-------|----------|
| WHO Guideline | 15 | 24-hour |
| WHO Annual | 5 | Annual |
| EPA NAAQS | 35 | 24-hour |
| Good AQI | 0-12 | - |
| Moderate | 12.1-35.4 | - |
| Unhealthy (Sensitive) | 35.5-55.4 | - |
| Unhealthy | 55.5-150.4 | - |
| Very Unhealthy | 150.5-250.4 | - |
| Hazardous | >250.5 | - |

## API Endpoints

### GET /api/emissions
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
    "cell_area": 2777777.78
  }
}
```

### POST /api/emissions
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
**Request:**
```json
{
  "frp_mw": 150,
  "duration_hours": 3,
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
  "pollutant": "PM2.5",
  "vegetation_type": "tropical_forest"
}
```

### GET /api/emissions/factors
**Response:**
```json
{
  "success": true,
  "pollutants": ["PM2.5", "PM10", "CO", "CO2", "NOx", "SO2"],
  "vegetation_types": ["savanna", "tropical_forest", ...],
  "emission_factors": { ... }
}
```

### POST /api/step
**Response (excerpt):**
```json
{
  "success": true,
  "frame": 42,
  "time_hours": 7.0,
  "concentration": {
    "max_ugm3": 45.23,
    "mean_ugm3": 2.15,
    "total_mass_g": 2847.5
  }
}
```

## Common Tasks

### Calculate Emissions from Satellite FRP
```python
import requests

# Satellite detected: FRP = 200 MW, burning 4 hours
response = requests.post('http://localhost:5000/api/emissions/from-frp', json={
    "frp_mw": 200,
    "duration_hours": 4,
    "pollutant": "PM2.5",
    "vegetation_type": "savanna"
})

emission = response.json()['emission_g']  # grams
```

### Update Emission Parameters
```python
# Set custom emission scenario
requests.post('http://localhost:5000/api/emissions', json={
    "total_mass_per_hotspot": 5000,  # 5 kg
    "mixing_height": 800,  # shallow mixing
    "pollutant_type": "PM10"
})
```

### Monitor Air Quality During Simulation
```python
for step in range(100):
    response = requests.post('http://localhost:5000/api/step').json()
    
    max_conc = response['concentration']['max_ugm3']
    
    if max_conc > 35:  # EPA 24h standard
        print(f"Step {step}: EXCEEDING EPA STANDARD ({max_conc:.1f} µg/m³)")
```

## Typical Fire Scenarios

### Small Savanna Fire
```
FRP: 10-50 MW
Duration: 1-2 hours
Vegetation: savanna
PM2.5 Emission: ~100-500 g
Injection Height: 500-1000 m
```

### Medium Forest Fire
```
FRP: 50-200 MW
Duration: 4-8 hours
Vegetation: tropical_forest
PM2.5 Emission: 5-40 kg
Injection Height: 1500-3000 m
```

### Large Wildfire
```
FRP: 200-1000 MW
Duration: 12-48 hours
Vegetation: extratropical_forest
PM2.5 Emission: 50-500 kg
Injection Height: 3500-8000 m
```

### Peat Fire
```
FRP: 50-300 MW (long duration)
Duration: 24-168 hours (days/weeks)
Vegetation: peat
PM2.5 Emission: 100-2000 kg
Injection Height: 500-2000 m
```

## Unit Conversions

| From | To | Multiply by |
|------|-----|-------------|
| km | m | 1000 |
| m | km | 0.001 |
| g | µg | 10⁶ |
| µg | g | 10⁻⁶ |
| kg | g | 1000 |
| MW | W | 10⁶ |
| hours | seconds | 3600 |
| degrees | radians | π/180 |

## Grid Calculations

```python
# Domain
lat_min, lat_max = 10.0, 11.8  # degrees
lon_min, lon_max = 120.0, 121.8
nx, ny = 120, 120

# Cell size
dlat = (lat_max - lat_min) / nx  # degrees
dlon = (lon_max - lon_min) / ny
dx = dlat * 111000  # meters
dy = dlon * 111000

# Cell area
A_cell = dx * dy  # m²

# Total domain area
A_total = (lat_max - lat_min) * (lon_max - lon_min) * (111000)**2  # m²
```

## Typical Wind Speeds

| Condition | Wind Speed | Description |
|-----------|------------|-------------|
| Calm | 0-2 m/s | Light/variable |
| Light breeze | 2-5 m/s | Typical daytime |
| Moderate wind | 5-10 m/s | Breezy |
| Fresh wind | 10-15 m/s | Strong dispersion |
| Strong wind | 15-20 m/s | Rapid transport |

## Mixing Height Guidance

| Time/Condition | H_mix | Reason |
|----------------|-------|--------|
| Nighttime | 200-500 m | Stable atmosphere |
| Morning | 500-1000 m | Growing boundary layer |
| Afternoon | 1000-2500 m | Convective mixing |
| Evening | 500-1000 m | Decaying mixing |
| Overcast/Rain | 300-800 m | Suppressed convection |

## Validation Checks

### Mass Conservation
```python
total_emitted = total_mass_per_hotspot * num_hotspots
total_in_domain = concentration.sum() * cell_volume
# Should be approximately equal (within 5%)
```

### Concentration Range
```python
# Near source
C_near = 100-1000 µg/m³  # typical for fires

# Background (far field)
C_background < 10 µg/m³

# Maximum ever
C_max < 10,000 µg/m³  # extreme scenarios
```

### Plume Spread
```python
# With wind speed u and time t
expected_distance = u * t

# Plume width (rough estimate)
width ≈ 0.1 * distance  # for σ_turb = 0.5
```

## Troubleshooting

| Issue | Possible Cause | Solution |
|-------|----------------|----------|
| Concentration too high | Low mixing height | Increase H_mix |
| No dispersion | σ_turb = 0 | Set σ_turb > 0.3 |
| Jagged concentration field | Too few particles | Increase npph > 2500 |
| Plume doesn't move | No wind | Check wind field |
| Mass not conserved | Particles leaving domain | Check boundary conditions |

## Keyboard Shortcuts (Browser)

| Key | Action |
|-----|--------|
| F5 | Refresh page |
| Ctrl+R | Reload simulation |
| F12 | Open developer console |
| Ctrl++ | Zoom in |
| Ctrl+- | Zoom out |

## File Locations

```
Project Root: d:\Minor\emission-pathways\

Key Files:
├── app.py                  Main application
├── emission_utils.py       Emission calculations
├── templates/index.html    Web interface
├── PHYSICAL_UNITS.md       This reference
├── EXPERIMENTS.md          Test scenarios
└── 3D_EXTENSION.md         Future roadmap
```

## Getting Help

1. Check documentation files
2. Review example scenarios in EXPERIMENTS.md
3. Inspect browser console (F12) for errors
4. Check Flask terminal output
5. Validate input parameters against typical ranges

## Quick Start Checklist

- [ ] Install dependencies: `pip install -r requirements.txt`
- [ ] Start Flask: `python app.py`
- [ ] Open browser: http://localhost:5000
- [ ] Toggle dark mode (optional)
- [ ] Add emission hotspots
- [ ] Set emission parameters
- [ ] Configure wind field (synthetic or real)
- [ ] Click Play or Step Forward
- [ ] Monitor concentration statistics
- [ ] Experiment with parameters

## Contact/Resources

- GitHub Issues: (your repository)
- ERA5 Documentation: https://confluence.ecmwf.int/display/CKB/ERA5
- GFS Documentation: https://www.ncei.noaa.gov/products/weather-climate-models/global-forecast
- WHO Air Quality: https://www.who.int/news-room/feature-stories/detail/what-are-the-who-air-quality-guidelines
