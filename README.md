# Lagrangian Transport Simulator with Physical Units

An advanced web-based interface for simulating and visualizing Lagrangian particle transport with turbulent dispersion, **now with physical concentration units (µg/m³)** and comprehensive emission estimation capabilities.

## 🌟 New Features (v2.0)

### Physical Units & Real-World Concentrations
- **Concentration in µg/m³**: Particle counts converted to physical concentrations
- **Emission Mass Tracking**: Total emitted mass per hotspot (configurable in grams)
- **Mass Per Particle**: Automatic calculation based on total emissions
- **Mixing Height**: Configurable boundary layer height (default: 1000m)
- **Air Quality Metrics**: Real-time max/mean concentration and total mass

### Emission Estimation
- **Emission Factors Database**: PM2.5, PM10, CO, CO2, NOx, SO2 for various vegetation types
  - Savanna, Tropical Forest, Extratropical Forest, Crop Residue, Peat
  - Based on literature values (Andreae & Merlet, 2001)
- **FRP to Emission Conversion**: Calculate emissions from Fire Radiative Power
  - Uses Wooster et al. (2005) methodology
  - Supports multiple pollutant types
  - Configurable combustion efficiency
- **API Endpoints**: Manage emissions programmatically

### Enhanced API
- `GET/POST /api/emissions` - Manage emission parameters
- `POST /api/emissions/from-frp` - Convert FRP to emissions
- `GET /api/emissions/factors` - Retrieve emission factor database
- `POST /api/step` - Now returns concentration statistics (max, mean, total mass)

### Dark Mode UI
- Toggle between light/dark themes
- Improved visibility and modern design
- Persistent theme preference (localStorage)

## Features

- **Interactive Parameter Control**
  - Adjust turbulent diffusion (σ_turb) in real-time
  - Change particles per hotspot (npph) from 500 to 10,000
  - Modify time step (dt) for simulation precision
  - **NEW**: Set total mass per hotspot and mixing height
  - **NEW**: Select pollutant type and emission factors

- **Dynamic Hotspot Management**
  - Add new emission sources with custom coordinates (vertical layout)
  - Remove existing hotspots
  - Move hotspots by editing coordinates
  - Visual representation on the simulation canvas

- **Wind Field Integration**
  - Synthetic wind fields for testing
  - **ERA5 Integration**: Fetch real wind data from Copernicus Climate Data Store
  - **GFS Integration**: NOAA Global Forecast System support
  - Vertical bounding box inputs for easier configuration

- **Simulation Controls**
  - Play/Pause animation (auto-stepping)
  - Step-by-step execution for detailed analysis
  - Batch runs (10 or 50 steps) for faster exploration
  - Reset functionality to restart with new parameters
  - Real-time frame, time, and concentration tracking

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. (Optional) For ERA5 data, configure your CDS API key:
   - Register at https://cds.climate.copernicus.eu
   - Create `~/.cdsapirc` with your credentials

## Usage

1. Start the Flask server:
```bash
python app.py
```

2. Open your browser and navigate to:
```
http://localhost:5000
```

3. Experiment with the controls:
   - **Dark Mode**: Click the moon/sun icon (top-left) to toggle theme
   - **Add/Remove Hotspots**: Use the Emission Hotspots section to manage sources
   - **Adjust σ_turb**: Control turbulent spreading (0-3)
   - **Change npph**: Modify particle count per hotspot (500-10,000)
   - **Set Emissions**: Configure pollutant type, total mass, and emission factors
   - **Play/Pause**: Auto-step through the simulation
   - **Fetch Wind Data**: Use ERA5 or GFS panels to get real wind fields

4. Use the API for advanced scenarios:
```python
# Example: Calculate emissions from satellite FRP
import requests

response = requests.post('http://localhost:5000/api/emissions/from-frp', json={
    "frp_mw": 150,  # Fire Radiative Power in MW
    "duration_hours": 3,
    "pollutant": "PM2.5",
    "vegetation_type": "tropical_forest",
    "combustion_efficiency": 0.5
})

emission_data = response.json()
print(f"PM2.5 emission: {emission_data['emission_g']:.2f} grams")
```

## Documentation

- **[PHYSICAL_UNITS.md](PHYSICAL_UNITS.md)** - Complete guide to physical units, emission estimation, and API usage
- **[EXPERIMENTS.md](EXPERIMENTS.md)** - Recommended validation experiments and test scenarios
- **[3D_EXTENSION.md](3D_EXTENSION.md)** - Roadmap for extending to 3D simulation
- **[wind_data_guide.md](wind_data_guide.md)** - Guide for ERA5/GFS wind data integration

## Project Structure

```
emission-pathways/
├── app.py                    # Main Flask application (enhanced with physical units)
├── emission_utils.py         # Emission factors and FRP conversion utilities
├── wind_api.py              # Wind data fetching (ERA5/GFS)
├── templates/
│   └── index.html           # Web interface (dark mode enabled)
├── requirements.txt
├── README.md                # This file
├── PHYSICAL_UNITS.md        # Physical units implementation guide
├── EXPERIMENTS.md           # Validation experiments
└── 3D_EXTENSION.md          # 3D extension roadmap
```

## Domain Configuration

- **Spatial**: 200 km × 200 km
- **Grid**: 120 × 120 cells (~1.67 km resolution)
- **Vertical**: Single layer with configurable mixing height (default: 1000m)
- **Temporal**: Configurable timestep (default: 600s = 10 minutes)

## Parameters Explained

### Core Physics
- **σ_turb**: Turbulent diffusion coefficient (m/s) - controls random spreading
- **npph**: Number of particles per hotspot (higher = smoother fields, slower computation)
- **dt**: Time step in seconds (affects simulation speed and numerical accuracy)

### Physical Units (NEW)
- **Total Mass per Hotspot**: Emitted mass in grams (default: 1000g = 1kg)
- **Mixing Height**: Boundary layer height in meters (default: 1000m)
- **Mass per Particle**: Automatically calculated from total mass and particle count
- **Cell Area**: Computed from domain size and grid resolution

### Emission Parameters (NEW)
- **Pollutant Type**: PM2.5, PM10, CO, CO2, NOx, or SO2
- **Emission Factor**: Grams of pollutant per kg of fuel burned
- **Vegetation Type**: Affects emission factor selection
- **FRP**: Fire Radiative Power in MW (for satellite-based estimation)

## Concentration Formula

```
C(i,j,t) = (N(i,j,t) × m_p) / (A_cell × H_mix)
```

Where:
- `C` = Concentration [µg/m³]
- `N` = Particle count in grid cell
- `m_p` = Mass per particle [µg]
- `A_cell` = Grid cell area [m²]
- `H_mix` = Mixing layer height [m]

## Air Quality Standards Reference

For PM2.5 (µg/m³):
- **WHO 24-hour guideline**: 15 µg/m³
- **WHO Annual guideline**: 5 µg/m³
- **EPA NAAQS 24-hour**: 35 µg/m³
- **Hazardous level**: > 250 µg/m³

## Tips

- Start with default parameters and make small adjustments
- Use "Step Forward" to observe detailed behavior
- Higher npph gives smoother results but slower computation
- Monitor concentration statistics to assess air quality impacts
- Use FRP conversion for real-world fire scenarios
- Compare synthetic vs. real wind fields (ERA5/GFS)
- Check EXPERIMENTS.md for validation scenarios
- Reset after changing hotspots or major parameter shifts

## Future Enhancements

- [ ] FINN/GFED emissions inventory integration
- [ ] User upload of emission metadata (time, mass, pollutant)
- [ ] Dry/wet deposition
- [ ] Chemical transformation (simple decay)
- [ ] 3D extension (vertical dispersion, plume rise)
- [ ] Population exposure calculation
- [ ] AQI computation
- [ ] Comparison with HYSPLIT/CALPUFF

## References

1. Wooster, M. J., et al. (2005). "Retrieval of biomass combustion rates and totals from fire radiative power observations"
2. Andreae, M. O., & Merlet, P. (2001). "Emission of trace gases and aerosols from biomass burning"
3. WHO Air Quality Guidelines (2021)
4. Stohl et al. (1998): "Computation of trajectories using ECMWF data"

## License

This project is for educational purposes (Minor Project).

## Links

- ERA5 Data: https://cds.climate.copernicus.eu
- GFS Data: https://www.ncei.noaa.gov/products/weather-climate-models/global-forecast
- EARTHKIT: https://earthkit.ecmwf.int/
