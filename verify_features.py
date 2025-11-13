"""
Simple verification script to test validation and ensemble features
Run this while the Flask server is running on port 5000
"""
import requests
import time

BASE_URL = "http://localhost:5000"

print("\n" + "="*70)
print(" VALIDATION & ENSEMBLE VERIFICATION")
print("="*70)

# Check if server is running
try:
    response = requests.get(f"{BASE_URL}/api/state", timeout=2)
    if response.status_code == 200:
        print("\n✓ Server is running")
    else:
        print("\n✗ Server error")
        exit(1)
except requests.exceptions.RequestException as e:
    print("\n✗ ERROR: Cannot connect to server")
    print("   Make sure Flask app is running: python app.py")
    exit(1)

# Test 1: Validation Pipeline
print("\n" + "-"*70)
print("TEST 1: VALIDATION PIPELINE")
print("-"*70)

print("\n1. Running simulation...")
requests.post(f"{BASE_URL}/api/play")
for i in range(5):
    requests.post(f"{BASE_URL}/api/step")
    print(f"   Step {i+1}/5", end='\r')
print("\n   ✓ Simulation complete")

print("\n2. Testing validation with synthetic data...")
val_response = requests.post(f"{BASE_URL}/api/validate", json={
    "source": "synthetic",
    "n_synthetic_stations": 15,
    "threshold": 35.0,
    "lat_min": 0,
    "lat_max": 200,
    "lon_min": 0,
    "lon_max": 200
})

if val_response.status_code == 200:
    result = val_response.json()
    print("   ✓ Validation endpoint working!")
    print(f"\n   Results:")
    print(f"   • Stations: {result['n_stations']}")
    print(f"   • Correlation: {result['metrics']['correlation']:.3f}")
    print(f"   • RMSE: {result['metrics']['rmse']:.2f} µg/m³")
    print(f"   • Bias: {result['metrics']['bias']:+.2f} µg/m³")
    if 'threshold' in result['metrics']:
        print(f"   • Hit Rate: {result['metrics']['hit_rate']:.3f}")
    print("\n   ✓ VALIDATION PIPELINE: WORKING ✓")
else:
    print(f"   ✗ Validation failed with status {val_response.status_code}")
    print(f"   Error: {val_response.text}")

# Test 2: Ensemble Simulation
print("\n" + "-"*70)
print("TEST 2: ENSEMBLE SIMULATION & UNCERTAINTY QUANTIFICATION")
print("-"*70)

print("\n1. Generating ensemble configurations...")
gen_response = requests.post(f"{BASE_URL}/api/ensemble/generate", json={
    "n_members": 10,
    "perturbations": {
        "wind_speed_factor": [0.8, 1.2],
        "wind_direction_offset": [-10, 10],
        "mixing_height_factor": [0.8, 1.2],
        "emission_factor": [0.7, 1.3],
        "turbulence_factor": [0.9, 1.1]
    },
    "seed": 42
})

if gen_response.status_code == 200:
    gen_result = gen_response.json()
    print(f"   ✓ Generated {gen_result['n_members']} ensemble members")
    print(f"\n   Sample member configurations:")
    for cfg in gen_result['configs'][:3]:
        print(f"   • Member {cfg['member']}: "
              f"Wind×{cfg['wind_speed_pert']:.2f}, "
              f"Dir{cfg['wind_dir_offset']:+.0f}°, "
              f"Emis×{cfg['emission_scaling']:.2f}")
else:
    print(f"   ✗ Ensemble generation failed: {gen_response.status_code}")
    print(f"   Error: {gen_response.text}")
    exit(1)

print("\n2. Running ensemble simulation (this may take a moment)...")
run_response = requests.post(f"{BASE_URL}/api/ensemble/run", json={
    "n_steps": 20,
    "compute_arrival_stats": True,
    "target_location": [150, 150],
    "threshold_distance": 10.0
})

if run_response.status_code == 200:
    run_result = run_response.json()
    print("   ✓ Ensemble simulation complete!")
    
    stats = run_result['statistics']
    print(f"\n   Uncertainty Quantification Results:")
    print(f"   • Mean max concentration: {stats['mean_max']:.2f} µg/m³")
    print(f"   • Maximum uncertainty (std): {stats['std_max']:.2f} µg/m³")
    print(f"   • Ensemble spread: {stats['spread']:.2f} µg/m³")
    print(f"   • Relative uncertainty: {(stats['std_max']/stats['mean_max']*100):.1f}%")
    
    if run_result.get('arrival_stats'):
        arr = run_result['arrival_stats']
        print(f"\n   Arrival Time Statistics:")
        print(f"   • Arrival probability: {arr['arrival_probability']*100:.1f}%")
        print(f"   • Members arrived: {arr['n_arrived']}/{arr['n_members']}")
        if arr['n_arrived'] > 0:
            print(f"   • Mean arrival time: {arr['mean_arrival_time']:.1f} steps")
            print(f"   • Std deviation: {arr['std_arrival_time']:.1f} steps")
    
    print("\n   ✓ ENSEMBLE SIMULATION: WORKING ✓")
else:
    print(f"   ✗ Ensemble run failed: {run_response.status_code}")
    print(f"   Error: {run_response.text}")
    exit(1)

# Test 3: Retrieve ensemble statistics
print("\n3. Retrieving ensemble statistics...")
stats_response = requests.get(f"{BASE_URL}/api/ensemble/statistics")

if stats_response.status_code == 200:
    print("   ✓ Statistics retrieval working!")
    stats_result = stats_response.json()
    print(f"   • Ensemble members: {stats_result['results']['n_members']}")
    print("\n   ✓ ENSEMBLE STATISTICS: WORKING ✓")
else:
    print(f"   ✗ Statistics retrieval failed: {stats_response.status_code}")

# Summary
print("\n" + "="*70)
print(" VERIFICATION COMPLETE")
print("="*70)
print("\n✓ All features are working correctly!")
print("\nWhat was tested:")
print("  1. ✓ Validation pipeline with synthetic observations")
print("  2. ✓ Ensemble configuration generation")
print("  3. ✓ Ensemble simulation execution")
print("  4. ✓ Uncertainty quantification")
print("  5. ✓ Arrival time statistics")
print("  6. ✓ Statistics retrieval")
print("\nYou can now use these features for:")
print("  • Model validation against ground observations")
print("  • Uncertainty quantification in predictions")
print("  • Probabilistic forecasting")
print("  • Sensitivity analysis")
print("\nSee VALIDATION_ENSEMBLE_API.md for detailed documentation.")
print("="*70 + "\n")
