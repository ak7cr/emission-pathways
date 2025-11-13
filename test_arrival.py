"""
Test arrival statistics with realistic configuration
This test demonstrates how arrival probability works
"""
import requests
import time

BASE_URL = "http://localhost:5000"

print("\n" + "="*70)
print(" ARRIVAL TIME STATISTICS - DEMONSTRATION")
print("="*70)

# Check server
try:
    response = requests.get(f"{BASE_URL}/api/state", timeout=2)
    if response.status_code == 200:
        print("\n✓ Server is running")
        state = response.json()
        hotspots = state.get('hotspots', [[40, 120], [60, 130], [30, 90]])
        print(f"✓ Hotspots (emission sources): {hotspots}")
except:
    print("\n✗ Server not running. Start with: python app.py")
    exit(1)

print("\n" + "-"*70)
print("SCENARIO 1: SHORT-RANGE TARGET (Will Show Arrival)")
print("-"*70)

# Reset simulation
requests.post(f"{BASE_URL}/api/reset")
print("\n1. Resetting simulation...")

# Generate ensemble
print("2. Generating 10 ensemble members...")
ensemble_config = {
    "n_members": 10,
    "perturbations": {
        "wind_speed_factor": [0.9, 1.1],  # ±10% wind speed
        "wind_direction_offset": [-5, 5],  # ±5° direction
        "mixing_height_factor": [0.9, 1.1],
        "emission_factor": [0.9, 1.1],
        "turbulence_factor": [0.9, 1.1]
    },
    "seed": 42
}

response = requests.post(f"{BASE_URL}/api/ensemble/generate", json=ensemble_config)
if response.status_code == 200:
    print("   ✓ Ensemble generated")

# Run with SHORT-RANGE target
print("\n3. Running simulation with SHORT-RANGE target...")
print("   Target: [80, 140] km (close to hotspots)")
print("   Threshold: 15 km radius")
print("   Steps: 100 (50 minutes simulation)")

run_config = {
    "n_steps": 100,
    "compute_arrival_stats": True,
    "target_location": [80, 140],  # Closer target
    "threshold_distance": 15.0      # Larger radius
}

response = requests.post(f"{BASE_URL}/api/ensemble/run", json=run_config)

if response.status_code == 200:
    result = response.json()
    print("\n   ✓ Simulation complete!")
    
    if result.get('arrival_stats'):
        arr = result['arrival_stats']
        print("\n" + "="*70)
        print(" SHORT-RANGE ARRIVAL RESULTS")
        print("="*70)
        print(f"\n   Arrival Probability: {arr['arrival_probability']*100:.1f}%")
        print(f"   Members Arrived: {arr['n_arrived']}/{arr['n_members']}")
        
        if arr['n_arrived'] > 0:
            print(f"\n   Mean Arrival Time: {arr['mean_arrival_time']:.1f} steps")
            print(f"                      ({arr['mean_arrival_time']*0.5:.1f} minutes)")
            print(f"   Std Deviation: {arr['std_arrival_time']:.1f} steps")
            print(f"   Earliest Arrival: {arr['min_arrival_time']} steps")
            print(f"   Latest Arrival: {arr['max_arrival_time']} steps")
        else:
            print("\n   ⚠ No arrivals detected. Try:")
            print("      - Closer target location")
            print("      - Larger threshold distance")
            print("      - More simulation steps")

print("\n" + "-"*70)
print("SCENARIO 2: MEDIUM-RANGE TARGET")
print("-"*70)

# Reset and try medium range
requests.post(f"{BASE_URL}/api/reset")
print("\n1. Resetting simulation...")

response = requests.post(f"{BASE_URL}/api/ensemble/generate", json=ensemble_config)
print("2. Re-generating ensemble...")

print("\n3. Running simulation with MEDIUM-RANGE target...")
print("   Target: [120, 120] km")
print("   Threshold: 20 km radius")
print("   Steps: 200 (100 minutes simulation)")

run_config = {
    "n_steps": 200,
    "compute_arrival_stats": True,
    "target_location": [120, 120],
    "threshold_distance": 20.0
}

response = requests.post(f"{BASE_URL}/api/ensemble/run", json=run_config)

if response.status_code == 200:
    result = response.json()
    print("\n   ✓ Simulation complete!")
    
    if result.get('arrival_stats'):
        arr = result['arrival_stats']
        print("\n" + "="*70)
        print(" MEDIUM-RANGE ARRIVAL RESULTS")
        print("="*70)
        print(f"\n   Arrival Probability: {arr['arrival_probability']*100:.1f}%")
        print(f"   Members Arrived: {arr['n_arrived']}/{arr['n_members']}")
        
        if arr['n_arrived'] > 0:
            print(f"\n   Mean Arrival Time: {arr['mean_arrival_time']:.1f} steps")
            print(f"                      ({arr['mean_arrival_time']*0.5:.1f} minutes)")
            print(f"   Std Deviation: {arr['std_arrival_time']:.1f} steps")
            print(f"   Earliest Arrival: {arr['min_arrival_time']} steps")
            print(f"   Latest Arrival: {arr['max_arrival_time']} steps")

print("\n" + "="*70)
print(" UNDERSTANDING ARRIVAL STATISTICS")
print("="*70)
print("""
Arrival probability depends on:
1. Distance from sources to target
2. Wind speed and direction
3. Simulation duration (number of steps)
4. Threshold distance (detection radius)

For realistic results:
- Hotspots: [40, 120], [60, 130], [30, 90] (west side)
- Wind: ~8-12 m/s eastward (right)
- Time step: 30 seconds
- 100 steps = 50 minutes real time

Examples:
• Target [80, 140] in 50 min  → High arrival (short range)
• Target [120, 120] in 100 min → Medium arrival
• Target [150, 150] in 15 min  → NO arrival (too far!)

The original test used [150, 150] with only 30 steps (15 min),
which is why arrival was 0% - particles need hours to travel 110+ km!
""")
print("="*70 + "\n")
