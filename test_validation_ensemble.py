"""
Test script for validation and ensemble capabilities
"""
import requests
import json

BASE_URL = "http://localhost:5000"

def test_validation_synthetic():
    """Test validation with synthetic observations"""
    print("\n" + "="*60)
    print("TESTING VALIDATION WITH SYNTHETIC DATA")
    print("="*60)
    
    # First, run simulation
    print("\n1. Starting simulation...")
    response = requests.post(f"{BASE_URL}/api/play")
    print(f"   Status: {response.status_code}")
    
    # Run a few steps
    for i in range(5):
        response = requests.post(f"{BASE_URL}/api/step")
        print(f"   Step {i+1}: {response.status_code}")
    
    # Test with synthetic data
    print("\n2. Running validation...")
    validation_request = {
        "source": "synthetic",
        "n_synthetic_stations": 15,
        "threshold": 35.0,  # PM2.5 threshold in µg/m³
        "lat_min": 0,
        "lat_max": 50,  # Updated for 50km domain (was 200)
        "lon_min": 0,
        "lon_max": 50   # Updated for 50km domain (was 200)
    }
    
    response = requests.post(
        f"{BASE_URL}/api/validate",
        json=validation_request
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"\n   ✓ Validation successful!")
        print(f"   Number of stations: {result['n_stations']}")
        print(f"\n   Metrics:")
        metrics = result['metrics']
        print(f"   - Correlation: {metrics['correlation']:.3f}")
        print(f"   - RMSE: {metrics['rmse']:.2f} µg/m³")
        print(f"   - Bias: {metrics['bias']:+.2f} µg/m³")
        print(f"\n{result['report']}")
    else:
        print(f"   ✗ Validation failed: {response.status_code}")
        print(f"   {response.text}")

def test_validation_openaq():
    """Test validation with OpenAQ data"""
    print("\n" + "="*60)
    print("TESTING VALIDATION WITH OpenAQ DATA")
    print("="*60)
    
    validation_request = {
        "source": "openaq",
        "lat_min": 28.4,  # Delhi region
        "lat_max": 28.8,
        "lon_min": 77.0,
        "lon_max": 77.4,
        "parameter": "pm25",
        "threshold": 60.0,
        "limit": 100
    }
    
    response = requests.post(
        f"{BASE_URL}/api/validate",
        json=validation_request
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"\n   ✓ OpenAQ validation successful!")
        print(f"   Number of stations: {result['n_stations']}")
        print(f"\n{result['report']}")
    else:
        print(f"   Note: OpenAQ validation requires active simulation and network access")
        print(f"   Status: {response.status_code}")

def test_ensemble():
    """Test ensemble simulation"""
    print("\n" + "="*60)
    print("TESTING ENSEMBLE SIMULATION")
    print("="*60)
    
    # Generate ensemble
    print("\n1. Generating ensemble configurations...")
    ensemble_config = {
        "n_members": 20,
        "perturbations": {
            "wind_speed_factor": [0.8, 1.2],  # ±20% wind speed
            "wind_direction_offset": [-15, 15],  # ±15° direction
            "mixing_height_factor": [0.7, 1.3],  # ±30% PBL height
            "emission_factor": [0.5, 1.5],  # 0.5x to 1.5x emissions
            "turbulence_factor": [0.8, 1.2]  # ±20% turbulence
        },
        "seed": 42
    }
    
    response = requests.post(
        f"{BASE_URL}/api/ensemble/generate",
        json=ensemble_config
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"   ✓ Generated {result['n_members']} ensemble members")
        print(f"\n   Sample configurations:")
        for i, cfg in enumerate(result['configs'][:3]):
            print(f"   Member {cfg['member']}:")
            print(f"     Wind speed factor: {cfg['wind_speed_pert']:.2f}")
            print(f"     Wind direction offset: {cfg['wind_dir_offset']:+.1f}°")
            print(f"     Mixing height: {cfg['mixing_height']:.1f} m")
            print(f"     Emission scaling: {cfg['emission_scaling']:.2f}")
    else:
        print(f"   ✗ Ensemble generation failed: {response.status_code}")
        return
    
    # Run ensemble
    print("\n2. Running ensemble simulation...")
    run_config = {
        "n_steps": 30,
        "compute_arrival_stats": True,
        "target_location": [37.5, 37.5],  # Center-right of 50km domain (was [150, 150] in 200km)
        "threshold_distance": 10.0
    }
    
    response = requests.post(
        f"{BASE_URL}/api/ensemble/run",
        json=run_config
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"   ✓ Ensemble simulation complete!")
        print(f"\n   Statistics:")
        stats = result['statistics']
        print(f"   - Maximum mean concentration: {stats['mean_max']:.2f} µg/m³")
        print(f"   - Maximum uncertainty (std): {stats['std_max']:.2f} µg/m³")
        print(f"   - Ensemble spread: {stats['spread']:.2f} µg/m³")
        
        if result.get('arrival_stats'):
            arr = result['arrival_stats']
            print(f"\n   Arrival Statistics:")
            print(f"   - Arrival probability: {arr['arrival_probability']*100:.1f}%")
            if arr['n_arrived'] > 0:
                print(f"   - Mean arrival time: {arr['mean_arrival_time']:.1f} steps")
                print(f"   - Std deviation: {arr['std_arrival_time']:.1f} steps")
        
        print(f"\n{result['report']}")
    else:
        print(f"   ✗ Ensemble run failed: {response.status_code}")
        print(f"   {response.text}")
    
    # Get statistics
    print("\n3. Retrieving ensemble statistics...")
    response = requests.get(f"{BASE_URL}/api/ensemble/statistics")
    
    if response.status_code == 200:
        print(f"   ✓ Statistics retrieved successfully")
    else:
        print(f"   ✗ Failed to retrieve statistics")

def test_all():
    """Run all tests"""
    print("\n" + "="*70)
    print(" VALIDATION & ENSEMBLE TESTING SUITE")
    print("="*70)
    
    # Check if server is running
    try:
        response = requests.get(f"{BASE_URL}/api/state", timeout=2)
        if response.status_code != 200:
            print("\n✗ Error: Server not responding properly")
            return
    except requests.exceptions.RequestException:
        print("\n✗ Error: Cannot connect to server. Make sure Flask app is running on port 5000")
        return
    
    print("\n✓ Server is running")
    
    # Run tests
    test_validation_synthetic()
    test_ensemble()
    
    print("\n" + "="*70)
    print(" TESTING COMPLETE")
    print("="*70)

if __name__ == "__main__":
    test_all()
