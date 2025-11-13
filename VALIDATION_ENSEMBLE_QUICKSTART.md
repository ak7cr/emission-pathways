# Validation & Ensemble System - Quick Reference

## ⚡ Quick Start

### 1. Test the System
```bash
# Run automated tests
python test_validation_ensemble.py

# Run all examples
python examples_validation_ensemble.py
```

### 2. Basic Validation
```python
import requests

# Validate with synthetic data
response = requests.post('http://localhost:5000/api/validate', json={
    "source": "synthetic",
    "n_synthetic_stations": 20,
    "threshold": 35.0
})

print(response.json()['report'])
```

### 3. Basic Ensemble
```python
# Generate ensemble
requests.post('http://localhost:5000/api/ensemble/generate', json={
    "n_members": 20,
    "perturbations": {
        "wind_speed_factor": [0.8, 1.2],
        "emission_factor": [0.6, 1.4]
    }
})

# Run ensemble
result = requests.post('http://localhost:5000/api/ensemble/run', json={
    "n_steps": 40
}).json()

print(f"Uncertainty: {result['statistics']['std_max']:.2f} µg/m³")
```

## 📊 Key Metrics

### Validation Metrics
- **correlation**: Model-observation correlation (target: > 0.6)
- **rmse**: Root mean square error (lower is better)
- **bias**: Mean difference (model - obs)
- **hit_rate**: Fraction of exceedances detected (target: > 0.75)

### Ensemble Metrics
- **mean**: Ensemble average
- **std**: Uncertainty (standard deviation)
- **p10/p90**: 10th/90th percentiles
- **arrival_probability**: Chance of reaching target

## 🎯 Common Use Cases

### Use Case 1: Validate Model Performance
```python
def validate_model():
    # Run simulation
    for _ in range(10):
        requests.post('http://localhost:5000/api/step')
    
    # Validate
    result = requests.post('http://localhost:5000/api/validate', json={
        "source": "synthetic",
        "n_synthetic_stations": 25,
        "threshold": 35.0
    }).json()
    
    # Check performance
    if result['metrics']['correlation'] > 0.7:
        print("✓ Good model performance")
    else:
        print("✗ Model needs improvement")
```

### Use Case 2: Quantify Uncertainty
```python
def quantify_uncertainty():
    # Generate 30-member ensemble
    requests.post('http://localhost:5000/api/ensemble/generate', json={
        "n_members": 30,
        "perturbations": {
            "wind_speed_factor": [0.8, 1.2],
            "mixing_height_factor": [0.7, 1.3]
        }
    })
    
    # Run and get uncertainty
    result = requests.post('http://localhost:5000/api/ensemble/run', json={
        "n_steps": 50
    }).json()
    
    mean = result['statistics']['mean_max']
    std = result['statistics']['std_max']
    
    print(f"Concentration: {mean:.1f} ± {std:.1f} µg/m³")
    print(f"95% CI: [{mean-2*std:.1f}, {mean+2*std:.1f}]")
```

### Use Case 3: Sensitivity Analysis
```python
def sensitivity_to_emissions():
    import numpy as np
    
    emission_factors = [0.5, 0.75, 1.0, 1.25, 1.5]
    results = []
    
    for ef in emission_factors:
        # Test single emission factor
        requests.post('http://localhost:5000/api/ensemble/generate', json={
            "n_members": 1,
            "perturbations": {"emission_factor": [ef, ef]}
        })
        
        result = requests.post('http://localhost:5000/api/ensemble/run', json={
            "n_steps": 30
        }).json()
        
        results.append(result['statistics']['mean_max'])
    
    # Calculate sensitivity
    sensitivity = np.polyfit(emission_factors, results, 1)[0]
    print(f"Sensitivity: {sensitivity:.2f} µg/m³ per emission factor")
```

## 🔧 Configuration Examples

### High Uncertainty Scenario
```json
{
  "n_members": 50,
  "perturbations": {
    "wind_speed_factor": [0.6, 1.4],
    "wind_direction_offset": [-30, 30],
    "mixing_height_factor": [0.5, 1.5],
    "emission_factor": [0.3, 2.0],
    "turbulence_factor": [0.7, 1.3]
  }
}
```

### Low Uncertainty Scenario
```json
{
  "n_members": 20,
  "perturbations": {
    "wind_speed_factor": [0.95, 1.05],
    "wind_direction_offset": [-5, 5],
    "mixing_height_factor": [0.9, 1.1],
    "emission_factor": [0.9, 1.1]
  }
}
```

## 📈 Interpreting Results

### Good Model Performance
```
Correlation:  > 0.70  ✓
R²:           > 0.50  ✓
NMB:          < ±30%  ✓
Hit Rate:     > 0.70  ✓
```

### Acceptable Uncertainty
```
Coefficient of Variation: < 0.30 (30%)
Arrival Probability:      > 0.70
Relative Std:             < 40% of mean
```

### Need for Improvement
```
Correlation:  < 0.50  ✗
NMB:          > ±50%  ✗
Hit Rate:     < 0.50  ✗
CV:           > 0.50  ✗
```

## 🐛 Troubleshooting

### No observations fetched
- Use `"source": "synthetic"` for testing
- Check geographic bounds for real data
- Verify internet connection for OpenAQ

### High RMSE
- Run simulation longer (more steps)
- Check emission rates are realistic
- Verify wind field is reasonable

### Ensemble takes too long
- Reduce `n_members` (20-30 is usually enough)
- Reduce `n_steps`
- Consider running in background

## 📚 Documentation

- **Full API Docs**: `VALIDATION_ENSEMBLE_API.md`
- **Code Examples**: `examples_validation_ensemble.py`
- **Test Suite**: `test_validation_ensemble.py`

## 🎓 Learning Path

1. **Start**: Run `test_validation_ensemble.py` to see system working
2. **Explore**: Try examples in `examples_validation_ensemble.py`
3. **Understand**: Read `VALIDATION_ENSEMBLE_API.md` for details
4. **Apply**: Use API for your own validation/ensemble needs

## 💡 Best Practices

1. **Always validate** before trusting model results
2. **Use ensembles** for critical decision-making
3. **Start with synthetic data** to test workflow
4. **Document** perturbation choices
5. **Report uncertainty** alongside predictions
6. **Calibrate** based on validation metrics

## 🚀 Production Checklist

- [ ] Validate against real observations
- [ ] Run ensemble with realistic perturbations
- [ ] Document validation statistics
- [ ] Report uncertainty bounds
- [ ] Archive ensemble configurations
- [ ] Compare multiple scenarios
- [ ] Perform sensitivity analysis
- [ ] Generate validation reports

---

**Need Help?** Check the full documentation in `VALIDATION_ENSEMBLE_API.md`
