# Validation & Ensemble API Documentation

## Overview

This document describes the validation pipeline and ensemble simulation capabilities added to the Lagrangian Transport Simulator.

---

## 🔬 Validation Pipeline

### Purpose
Compare model output with ground observations from air quality monitoring networks to assess model performance.

### Supported Data Sources

1. **OpenAQ** - Global air quality data network
2. **CPCB** - Central Pollution Control Board (India)
3. **Synthetic** - Generated test data for development

---

## API Endpoints

### 1. `/api/validate` (POST)

Validate model output against ground observations.

**Request Body:**
```json
{
  "source": "openaq | cpcb | synthetic",
  "lat_min": 0.0,
  "lat_max": 50.0,
  "lon_min": 0.0,
  "lon_max": 50.0,
  "parameter": "pm25 | pm10 | no2 | so2 | o3 | co",
  "threshold": 35.0,
  "n_synthetic_stations": 15,
  "limit": 1000
}
```

**Parameters:**
- `source` - Data source (required)
- `lat_min/max, lon_min/max` - Geographic bounding box
- `parameter` - Pollutant type
- `threshold` - Concentration threshold for exceedance metrics (µg/m³)
- `n_synthetic_stations` - Number of synthetic stations (for testing)
- `limit` - Maximum observations to fetch

**Response:**
```json
{
  "success": true,
  "n_stations": 15,
  "metrics": {
    "n_pairs": 15,
    "correlation": 0.856,
    "r2": 0.733,
    "bias": -5.23,
    "rmse": 12.45,
    "mae": 9.87,
    "nmb": -0.15,
    "nme": 0.28,
    "threshold": 35.0,
    "hit_rate": 0.875,
    "false_alarm_ratio": 0.123,
    "critical_success_index": 0.782,
    "n_observed_exceedances": 8,
    "n_modeled_exceedances": 9
  },
  "report": "... formatted text report ...",
  "stations": [
    {
      "id": "STATION_001",
      "name": "Downtown Monitor",
      "lat": 28.6,
      "lon": 77.2,
      "observed": 45.3,
      "modeled": 42.1
    }
  ]
}
```

**Validation Metrics:**

| Metric | Description | Interpretation |
|--------|-------------|----------------|
| **correlation** | Pearson correlation coefficient | r > 0.8: excellent, r > 0.6: good |
| **r²** | Coefficient of determination | Fraction of variance explained |
| **bias** | Mean bias (model - obs) | Positive: overestimation |
| **rmse** | Root mean square error | Lower is better |
| **mae** | Mean absolute error | Average absolute difference |
| **nmb** | Normalized mean bias | Percentage bias |
| **nme** | Normalized mean error | Percentage error |
| **hit_rate** | Probability of detection | Fraction of exceedances detected |
| **false_alarm_ratio** | False alarms / total alarms | Lower is better |
| **csi** | Critical success index | Overall skill (0-1) |

**Example Usage:**

```python
import requests

# Validate with synthetic data
response = requests.post('http://localhost:5000/api/validate', json={
    "source": "synthetic",
    "n_synthetic_stations": 20,
    "threshold": 35.0,
    "lat_min": 0, "lat_max": 200,
    "lon_min": 0, "lon_max": 200
})

result = response.json()
print(f"Correlation: {result['metrics']['correlation']:.3f}")
print(f"RMSE: {result['metrics']['rmse']:.2f} µg/m³")
print(result['report'])
```

```python
# Validate with OpenAQ data
response = requests.post('http://localhost:5000/api/validate', json={
    "source": "openaq",
    "lat_min": 28.4, "lat_max": 28.8,  # Delhi
    "lon_min": 77.0, "lon_max": 77.4,
    "parameter": "pm25",
    "threshold": 60.0,
    "limit": 100
})
```

---

## 🎲 Ensemble Simulation

### Purpose
Quantify uncertainty by running multiple simulations with perturbed parameters.

### Perturbable Parameters

1. **Wind Speed** - Multiplicative factor (e.g., 0.8-1.2 for ±20%)
2. **Wind Direction** - Additive offset in degrees (e.g., ±15°)
3. **Mixing Height** - PBL height factor (e.g., 0.7-1.3 for ±30%)
4. **Emission Factor** - Emission scaling (e.g., 0.5-1.5)
5. **Turbulence** - Turbulent diffusion factor (e.g., 0.8-1.2)

---

### 2. `/api/ensemble/generate` (POST)

Generate ensemble member configurations.

**Request Body:**
```json
{
  "n_members": 20,
  "perturbations": {
    "wind_speed_factor": [0.8, 1.2],
    "wind_direction_offset": [-15, 15],
    "mixing_height_factor": [0.7, 1.3],
    "emission_factor": [0.5, 1.5],
    "turbulence_factor": [0.8, 1.2]
  },
  "seed": 42
}
```

**Parameters:**
- `n_members` - Number of ensemble members (default: 10)
- `perturbations` - Dictionary of parameter perturbation ranges
- `seed` - Random seed for reproducibility

**Response:**
```json
{
  "success": true,
  "n_members": 20,
  "perturbations": {...},
  "configs": [
    {
      "member": 0,
      "wind_speed_pert": 1.05,
      "wind_dir_offset": -8.3,
      "mixing_height": 850.0,
      "emission_scaling": 1.23,
      "sigma_turb": 2.15
    }
  ]
}
```

---

### 3. `/api/ensemble/run` (POST)

Run ensemble simulation and compute statistics.

**Request Body:**
```json
{
  "n_steps": 50,
  "compute_arrival_stats": true,
  "target_location": [150, 150],
  "threshold_distance": 10.0
}
```

**Parameters:**
- `n_steps` - Number of simulation steps per member
- `compute_arrival_stats` - Calculate arrival time statistics
- `target_location` - [x, y] coordinates for arrival calculation
- `threshold_distance` - Distance threshold (km) for arrival

**Response:**
```json
{
  "success": true,
  "n_members": 20,
  "n_steps": 50,
  "statistics": {
    "mean_max": 85.6,
    "std_max": 12.3,
    "spread": 45.7
  },
  "arrival_stats": {
    "arrival_probability": 0.85,
    "mean_arrival_time": 23.4,
    "std_arrival_time": 5.2,
    "min_arrival_time": 15,
    "max_arrival_time": 35,
    "n_arrived": 17,
    "n_members": 20
  },
  "report": "... formatted text report ..."
}
```

**Ensemble Statistics:**

The endpoint computes the following fields across all ensemble members:

- **mean** - Ensemble mean concentration field
- **std** - Standard deviation (uncertainty)
- **min/max** - Minimum and maximum values
- **median** - Median concentration
- **p10/p90** - 10th and 90th percentiles
- **coefficient_of_variation** - Std / Mean

---

### 4. `/api/ensemble/statistics` (GET)

Retrieve statistics from last ensemble run.

**Response:**
```json
{
  "success": true,
  "results": {
    "statistics": {
      "mean": [[...], [...]],
      "std": [[...], [...]],
      ...
    },
    "arrival_stats": {...},
    "n_members": 20
  }
}
```

---

## Usage Examples

### Complete Workflow

```python
import requests
import numpy as np
import matplotlib.pyplot as plt

BASE_URL = "http://localhost:5000"

# 1. Run base simulation
requests.post(f"{BASE_URL}/api/play")
for _ in range(10):
    requests.post(f"{BASE_URL}/api/step")

# 2. Validate against observations
validation = requests.post(f"{BASE_URL}/api/validate", json={
    "source": "synthetic",
    "n_synthetic_stations": 25,
    "threshold": 35.0
}).json()

print(f"Model Correlation: {validation['metrics']['correlation']:.3f}")
print(f"RMSE: {validation['metrics']['rmse']:.2f} µg/m³")

# 3. Generate ensemble
ensemble_config = requests.post(f"{BASE_URL}/api/ensemble/generate", json={
    "n_members": 30,
    "perturbations": {
        "wind_speed_factor": [0.8, 1.2],
        "mixing_height_factor": [0.7, 1.3],
        "emission_factor": [0.6, 1.4],
        "turbulence_factor": [0.9, 1.1]
    }
}).json()

print(f"Generated {ensemble_config['n_members']} ensemble members")

# 4. Run ensemble
ensemble_results = requests.post(f"{BASE_URL}/api/ensemble/run", json={
    "n_steps": 40,
    "compute_arrival_stats": True,
    "target_location": [120, 130]
}).json()

print(f"\nEnsemble Uncertainty:")
print(f"  Mean max concentration: {ensemble_results['statistics']['mean_max']:.2f} µg/m³")
print(f"  Maximum std: {ensemble_results['statistics']['std_max']:.2f} µg/m³")
print(f"  Arrival probability: {ensemble_results['arrival_stats']['arrival_probability']*100:.1f}%")

# 5. Get detailed statistics
stats = requests.get(f"{BASE_URL}/api/ensemble/statistics").json()

# Plot uncertainty bands
mean_field = np.array(stats['results']['statistics']['mean'])
std_field = np.array(stats['results']['statistics']['std'])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.imshow(mean_field, cmap='YlOrRd')
ax1.set_title('Ensemble Mean Concentration')
ax2.imshow(std_field, cmap='viridis')
ax2.set_title('Ensemble Uncertainty (Std Dev)')
plt.tight_layout()
plt.savefig('ensemble_results.png')
```

---

## Interpretation Guidelines

### Validation Metrics Benchmarks

**Excellent Performance:**
- Correlation > 0.8
- NMB < ±15%
- NME < 25%
- Hit Rate > 0.75 (for exceedances)

**Good Performance:**
- Correlation > 0.6
- NMB < ±30%
- NME < 35%
- Hit Rate > 0.60

**Needs Improvement:**
- Correlation < 0.5
- NMB > ±50%
- NME > 50%
- Hit Rate < 0.50

### Ensemble Uncertainty

**Low Uncertainty:**
- Coefficient of Variation < 0.2 (20%)
- Tight arrival time distribution (std < 20% of mean)

**Moderate Uncertainty:**
- CV = 0.2 - 0.5
- Moderate arrival spread

**High Uncertainty:**
- CV > 0.5
- Wide arrival time distribution
- Low arrival probability (< 0.6)

---

## Best Practices

1. **Validation:**
   - Use at least 10-15 stations for robust statistics
   - Match observation and model times as closely as possible
   - Apply appropriate thresholds for your pollutant and region

2. **Ensemble:**
   - Use 20-50 members for stable statistics
   - Choose perturbation ranges based on parameter uncertainty
   - Larger ensembles for critical applications

3. **Interpretation:**
   - High model-observation correlation doesn't guarantee unbiased predictions
   - Check both correlation AND bias metrics
   - Use ensemble spread to communicate forecast confidence

---

## Technical Notes

### Coordinate Systems
- Current implementation assumes grid coordinates match lat/lon
- For real applications, implement proper geographic projections
- Use libraries like `pyproj` for coordinate transformations

### Performance
- Ensemble simulations run sequentially (parallelization possible)
- Memory usage scales with n_members × grid_size
- Consider reducing grid resolution for large ensembles

### OpenAQ API
- Requires internet connection
- Rate-limited (check OpenAQ documentation)
- May return no data for remote regions
- Use synthetic data for testing

---

## Troubleshooting

**Error: "No simulation data available"**
- Solution: Run simulation first with `/api/play` and `/api/step`

**Error: "No observations fetched"**
- Solution: Try `"source": "synthetic"` for testing
- Check geographic bounds contain monitoring stations
- Verify internet connection for OpenAQ/CPCB

**Error: "Ensemble not initialized"**
- Solution: Call `/api/ensemble/generate` before `/api/ensemble/run`

**High RMSE values:**
- Check if simulation has reached steady state
- Verify emission rates are realistic
- Consider model spin-up time

---

## References

- OpenAQ API: https://docs.openaq.org/
- Validation metrics: EPA Model Performance Guidelines
- Ensemble methods: Monte Carlo uncertainty quantification

---

**Last Updated:** 2025-11-13
