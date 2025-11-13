"""
Example usage scripts for validation and ensemble features
Copy these examples into browser console or use with requests
"""

# ============================================================================
# BROWSER CONSOLE EXAMPLES (paste into browser developer console)
# ============================================================================

"""
// Example 1: Quick Validation with Synthetic Data
async function quickValidation() {
    // Run simulation first
    await fetch('/api/play', {method: 'POST'});
    for(let i=0; i<10; i++) {
        await fetch('/api/step', {method: 'POST'});
    }
    
    // Validate
    const response = await fetch('/api/validate', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
            source: 'synthetic',
            n_synthetic_stations: 20,
            threshold: 35.0,
            lat_min: 0, lat_max: 200,
            lon_min: 0, lon_max: 200
        })
    });
    
    const result = await response.json();
    console.log('Validation Results:');
    console.log('Correlation:', result.metrics.correlation);
    console.log('RMSE:', result.metrics.rmse, 'µg/m³');
    console.log('Bias:', result.metrics.bias, 'µg/m³');
    console.log('\nFull Report:\n', result.report);
    return result;
}

// Example 2: Run Ensemble Simulation
async function runEnsemble() {
    // Generate ensemble
    const genResponse = await fetch('/api/ensemble/generate', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
            n_members: 25,
            perturbations: {
                wind_speed_factor: [0.8, 1.2],
                wind_direction_offset: [-10, 10],
                mixing_height_factor: [0.7, 1.3],
                emission_factor: [0.6, 1.4],
                turbulence_factor: [0.9, 1.1]
            },
            seed: 42
        })
    });
    
    const genResult = await genResponse.json();
    console.log(`Generated ${genResult.n_members} ensemble members`);
    
    // Run ensemble
    const runResponse = await fetch('/api/ensemble/run', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
            n_steps: 40,
            compute_arrival_stats: true,
            target_location: [150, 150],
            threshold_distance: 10.0
        })
    });
    
    const runResult = await runResponse.json();
    console.log('\nEnsemble Statistics:');
    console.log('Max mean concentration:', runResult.statistics.mean_max, 'µg/m³');
    console.log('Max uncertainty (std):', runResult.statistics.std_max, 'µg/m³');
    console.log('Ensemble spread:', runResult.statistics.spread, 'µg/m³');
    
    if(runResult.arrival_stats) {
        console.log('\nArrival Statistics:');
        console.log('Arrival probability:', (runResult.arrival_stats.arrival_probability * 100).toFixed(1) + '%');
        console.log('Mean arrival time:', runResult.arrival_stats.mean_arrival_time.toFixed(1), 'steps');
    }
    
    console.log('\n' + runResult.report);
    return runResult;
}

// Run both examples
quickValidation().then(() => runEnsemble());
"""

# ============================================================================
# PYTHON EXAMPLES (use with requests library)
# ============================================================================

"""
Example 1: Comprehensive Validation Workflow
"""
def validation_workflow():
    import requests
    import json
    
    BASE = "http://localhost:5000"
    
    # Step 1: Run simulation
    print("Running simulation...")
    requests.post(f"{BASE}/api/play")
    for i in range(15):
        requests.post(f"{BASE}/api/step")
        print(f"  Step {i+1}/15", end='\r')
    print("\n")
    
    # Step 2: Validate with synthetic data
    print("Validating with synthetic observations...")
    val_response = requests.post(f"{BASE}/api/validate", json={
        "source": "synthetic",
        "n_synthetic_stations": 30,
        "threshold": 35.0,
        "lat_min": 0, "lat_max": 200,
        "lon_min": 0, "lon_max": 200
    })
    
    val_result = val_response.json()
    
    print("\n" + "="*60)
    print("VALIDATION RESULTS")
    print("="*60)
    print(f"Stations: {val_result['n_stations']}")
    print(f"Correlation: {val_result['metrics']['correlation']:.3f}")
    print(f"R²: {val_result['metrics']['r2']:.3f}")
    print(f"RMSE: {val_result['metrics']['rmse']:.2f} µg/m³")
    print(f"Bias: {val_result['metrics']['bias']:+.2f} µg/m³")
    print(f"NMB: {val_result['metrics']['nmb']*100:+.1f}%")
    
    if 'threshold' in val_result['metrics']:
        print(f"\nExceedance Metrics (threshold={val_result['metrics']['threshold']} µg/m³):")
        print(f"  Hit Rate: {val_result['metrics']['hit_rate']:.3f}")
        print(f"  False Alarm Ratio: {val_result['metrics']['false_alarm_ratio']:.3f}")
        print(f"  Critical Success Index: {val_result['metrics']['critical_success_index']:.3f}")
    
    print("\n" + val_result['report'])
    
    return val_result

"""
Example 2: Ensemble Uncertainty Quantification
"""
def ensemble_workflow():
    import requests
    import numpy as np
    
    BASE = "http://localhost:5000"
    
    # Step 1: Generate ensemble
    print("Generating ensemble configurations...")
    gen_response = requests.post(f"{BASE}/api/ensemble/generate", json={
        "n_members": 30,
        "perturbations": {
            "wind_speed_factor": [0.7, 1.3],      # ±30% wind
            "wind_direction_offset": [-20, 20],    # ±20° direction
            "mixing_height_factor": [0.6, 1.4],    # ±40% PBL
            "emission_factor": [0.5, 2.0],         # 0.5x to 2x emissions
            "turbulence_factor": [0.8, 1.2]        # ±20% turbulence
        },
        "seed": 123
    })
    
    gen_result = gen_response.json()
    print(f"✓ Generated {gen_result['n_members']} members\n")
    
    # Show sample configurations
    print("Sample Member Configurations:")
    print("-" * 60)
    for cfg in gen_result['configs'][:5]:
        print(f"Member {cfg['member']:2d}: "
              f"Wind×{cfg['wind_speed_pert']:.2f}, "
              f"Dir{cfg['wind_dir_offset']:+.0f}°, "
              f"PBL={cfg['mixing_height']:.0f}m, "
              f"Emis×{cfg['emission_scaling']:.2f}")
    print()
    
    # Step 2: Run ensemble
    print("Running ensemble simulation...")
    run_response = requests.post(f"{BASE}/api/ensemble/run", json={
        "n_steps": 50,
        "compute_arrival_stats": True,
        "target_location": [140, 140],
        "threshold_distance": 12.0
    })
    
    run_result = run_response.json()
    
    print("\n" + "="*60)
    print("ENSEMBLE RESULTS")
    print("="*60)
    
    stats = run_result['statistics']
    print(f"Maximum mean concentration: {stats['mean_max']:.2f} µg/m³")
    print(f"Maximum uncertainty (std):  {stats['std_max']:.2f} µg/m³")
    print(f"Ensemble spread (max-min):  {stats['spread']:.2f} µg/m³")
    print(f"Relative uncertainty:       {(stats['std_max']/stats['mean_max']*100):.1f}%")
    
    if run_result.get('arrival_stats'):
        arr = run_result['arrival_stats']
        print(f"\nArrival at target location:")
        print(f"  Probability: {arr['arrival_probability']*100:.1f}%")
        print(f"  Members arrived: {arr['n_arrived']}/{arr['n_members']}")
        if arr['n_arrived'] > 0:
            print(f"  Mean time: {arr['mean_arrival_time']:.1f} ± {arr['std_arrival_time']:.1f} steps")
            print(f"  Range: {arr['min_arrival_time']:.0f} - {arr['max_arrival_time']:.0f} steps")
    
    print("\n" + run_result['report'])
    
    # Step 3: Retrieve and analyze statistics
    stats_response = requests.get(f"{BASE}/api/ensemble/statistics")
    stats_result = stats_response.json()
    
    # Calculate additional metrics
    mean_field = np.array(stats_result['results']['statistics']['mean'])
    std_field = np.array(stats_result['results']['statistics']['std'])
    cv_field = std_field / (mean_field + 1e-10)
    
    print("\nSpatial Statistics:")
    print(f"  Mean concentration range: {mean_field.min():.2f} - {mean_field.max():.2f} µg/m³")
    print(f"  Uncertainty range: {std_field.min():.2f} - {std_field.max():.2f} µg/m³")
    print(f"  Max coefficient of variation: {cv_field.max():.2f}")
    
    return run_result

"""
Example 3: Sensitivity Analysis
"""
def sensitivity_analysis():
    import requests
    import numpy as np
    import matplotlib.pyplot as plt
    
    BASE = "http://localhost:5000"
    
    # Test different emission factors
    emission_factors = np.linspace(0.5, 2.0, 10)
    max_concentrations = []
    
    print("Running sensitivity analysis on emission factor...")
    
    for ef in emission_factors:
        # Generate single-member ensemble with specific emission factor
        requests.post(f"{BASE}/api/ensemble/generate", json={
            "n_members": 1,
            "perturbations": {"emission_factor": [ef, ef]},
            "seed": 42
        })
        
        # Run simulation
        result = requests.post(f"{BASE}/api/ensemble/run", json={
            "n_steps": 30,
            "compute_arrival_stats": False
        }).json()
        
        max_concentrations.append(result['statistics']['mean_max'])
        print(f"  Emission factor {ef:.2f}: Max conc = {result['statistics']['mean_max']:.2f} µg/m³")
    
    # Plot sensitivity
    plt.figure(figsize=(10, 6))
    plt.plot(emission_factors, max_concentrations, 'o-', linewidth=2, markersize=8)
    plt.xlabel('Emission Scaling Factor', fontsize=12)
    plt.ylabel('Maximum Concentration (µg/m³)', fontsize=12)
    plt.title('Sensitivity to Emission Uncertainty', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('emission_sensitivity.png', dpi=150)
    print("\n✓ Sensitivity plot saved to emission_sensitivity.png")
    
    # Calculate sensitivity coefficient
    slope = np.polyfit(emission_factors, max_concentrations, 1)[0]
    print(f"\nSensitivity: {slope:.2f} µg/m³ per unit emission factor")

"""
Example 4: Compare Different Perturbation Scenarios
"""
def compare_scenarios():
    import requests
    
    BASE = "http://localhost:5000"
    
    scenarios = {
        "Low Uncertainty": {
            "wind_speed_factor": [0.95, 1.05],
            "mixing_height_factor": [0.9, 1.1],
            "emission_factor": [0.9, 1.1]
        },
        "Medium Uncertainty": {
            "wind_speed_factor": [0.85, 1.15],
            "mixing_height_factor": [0.8, 1.2],
            "emission_factor": [0.7, 1.3]
        },
        "High Uncertainty": {
            "wind_speed_factor": [0.7, 1.3],
            "mixing_height_factor": [0.6, 1.4],
            "emission_factor": [0.5, 1.5]
        }
    }
    
    print("Comparing uncertainty scenarios...\n")
    results = {}
    
    for scenario_name, perturbations in scenarios.items():
        print(f"Running {scenario_name}...")
        
        requests.post(f"{BASE}/api/ensemble/generate", json={
            "n_members": 20,
            "perturbations": perturbations
        })
        
        result = requests.post(f"{BASE}/api/ensemble/run", json={
            "n_steps": 40
        }).json()
        
        results[scenario_name] = result['statistics']
        print(f"  Max std: {result['statistics']['std_max']:.2f} µg/m³")
    
    print("\n" + "="*60)
    print("SCENARIO COMPARISON")
    print("="*60)
    for scenario_name, stats in results.items():
        print(f"\n{scenario_name}:")
        print(f"  Mean max: {stats['mean_max']:.2f} µg/m³")
        print(f"  Uncertainty: {stats['std_max']:.2f} µg/m³")
        print(f"  Relative: {(stats['std_max']/stats['mean_max']*100):.1f}%")


# Run all examples
if __name__ == "__main__":
    print("\n" + "="*70)
    print(" VALIDATION & ENSEMBLE EXAMPLES")
    print("="*70 + "\n")
    
    print("1. VALIDATION WORKFLOW")
    print("-" * 70)
    validation_workflow()
    
    print("\n\n2. ENSEMBLE WORKFLOW")
    print("-" * 70)
    ensemble_workflow()
    
    print("\n\n3. SENSITIVITY ANALYSIS")
    print("-" * 70)
    sensitivity_analysis()
    
    print("\n\n4. SCENARIO COMPARISON")
    print("-" * 70)
    compare_scenarios()
    
    print("\n" + "="*70)
    print(" ALL EXAMPLES COMPLETE!")
    print("="*70)
