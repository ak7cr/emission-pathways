https://earthkit.ecmwf.int/
# Lagrangian Transport Simulator - Web UI

An interactive web-based interface for simulating and visualizing Lagrangian particle transport with turbulent dispersion.

## Features

- **Interactive Parameter Control**
  - Adjust turbulent diffusion (σ_turb) in real-time
  - Change particles per hotspot (npph) from 500 to 10,000
  - Modify time step (dt) for simulation precision
  - Toggle between synthetic and real wind fields (ERA5/GFS support planned)

- **Dynamic Hotspot Management**
  - Add new emission sources with custom coordinates
  - Remove existing hotspots
  - Move hotspots by editing coordinates
  - Visual representation on the simulation canvas

- **Simulation Controls**
  - Step-by-step execution for detailed analysis
  - Batch runs (10 or 50 steps) for faster exploration
  - Reset functionality to restart with new parameters
  - Real-time frame and time tracking

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

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
   - **Add/Remove Hotspots**: Use the Emission Hotspots section to manage sources
   - **Adjust σ_turb**: Control turbulent spreading (0-3)
   - **Change npph**: Modify particle count per hotspot (500-10,000)
   - **Run Simulation**: Click "Step Forward" for single steps or "Run X Steps" for batch execution

## Wind Field Integration (TODO)

The application is designed to support real wind data. To integrate ERA5 or GFS data:

1. Replace the `real_wind_field()` function in `app.py`
2. Implement interpolation from your wind data files
3. The function should return `(U, V)` wind components matching the domain grid

Example structure:
```python
def real_wind_field(t):
    # Load ERA5/GFS data
    # Interpolate to current time t and domain grid
    U = ...  # shape: (ny, nx)
    V = ...  # shape: (ny, nx)
    return U, V
```

## Domain Configuration

- X range: 0-200 km
- Y range: 0-200 km
- Grid resolution: 120×120 cells

## Parameters Explained

- **σ_turb**: Controls random turbulent spreading (higher = more diffusion)
- **npph**: Number of particles per hotspot (higher = smoother concentration field)
- **dt**: Time step in seconds (affects simulation speed and accuracy)

## Tips

- Start with default parameters and make small adjustments
- Use "Step Forward" to observe detailed behavior
- Higher npph gives smoother results but slower computation
- Reset after changing hotspots or major parameter shifts
