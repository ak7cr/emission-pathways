"""
Quick test with optimized parameters for faster execution
Uses smaller ensemble and fewer steps for rapid testing
"""
import requests
import time

BASE_URL = "http://localhost:5000"

print("\n" + "="*70)
print(" FAST ENSEMBLE TEST - Optimized Parameters")
print("="*70)

# Check server
try:
    response = requests.get(f"{BASE_URL}/api/state", timeout=2)
    if response.status_code == 200:
        print("\n✓ Server is running")
except:
    print("\n✗ Server not running. Start with: python app.py")
    exit(1)

print("\n" + "-"*70)
print("OPTIMIZED ENSEMBLE TEST (5-10 seconds)")
print("-"*70)

# Reset
requests.post(f"{BASE_URL}/api/reset")
print("\n1. Resetting simulation...")

# Generate SMALLER ensemble (10 members instead of 20)
print("2. Generating 10 ensemble members (half the original)...")
start_gen = time.time()

ensemble_config = {
    "n_members": 10,  # Reduced from 20
    "perturbations": {
        "wind_speed_factor": [0.9, 1.1],      # Smaller range
        "wind_direction_offset": [-5, 5],      # Smaller range
        "mixing_height_factor": [0.9, 1.1],
        "emission_factor": [0.9, 1.1],
        "turbulence_factor": [0.95, 1.05]
    },
    "seed": 42
}

response = requests.post(f"{BASE_URL}/api/ensemble/generate", json=ensemble_config)
gen_time = time.time() - start_gen
if response.status_code == 200:
    print(f"   ✓ Generated in {gen_time:.2f}s")

# Run with FEWER steps (20 instead of 30-50)
print("\n3. Running ensemble with 20 steps (reduced from 30-50)...")
print("   This should take 10-15 seconds...")
start_run = time.time()

run_config = {
    "n_steps": 20,  # Reduced from 30-50
    "compute_arrival_stats": True,
    "target_location": [30, 35],
    "threshold_distance": 10.0
}

response = requests.post(f"{BASE_URL}/api/ensemble/run", json=run_config)
run_time = time.time() - start_run

if response.status_code == 200:
    result = response.json()
    print(f"\n   ✓ Completed in {run_time:.2f}s!")
    
    stats = result['statistics']
    print(f"\n   RESULTS:")
    print(f"   • Mean max concentration: {stats['mean_max']:.2f} µg/m³")
    print(f"   • Uncertainty (std): {stats['std_max']:.2f} µg/m³")
    
    if result.get('arrival_stats'):
        arr = result['arrival_stats']
        print(f"   • Arrival probability: {arr['arrival_probability']*100:.1f}%")
        print(f"   • Members arrived: {arr['n_arrived']}/{arr['n_members']}")
else:
    print(f"   ✗ Failed: {response.status_code}")

print("\n" + "="*70)
print(" PERFORMANCE SUMMARY")
print("="*70)
print(f"""
Configuration:
• Ensemble members: 10 (vs 20 original)
• Steps per member: 20 (vs 30-50 original)
• Total computations: 200 (vs 600-1000 original)
• Speedup: ~3-5× faster

Timing:
• Generation: {gen_time:.2f}s
• Execution: {run_time:.2f}s
• Total: {gen_time + run_time:.2f}s

Trade-offs:
✓ Much faster testing
✓ Still statistically meaningful (10 members is good)
✓ Shows arrival patterns clearly
⚠ Slightly less uncertainty quantification
⚠ May need more members for production

For production use:
• 20-50 members for robust statistics
• 50-100 steps for longer-range transport
• Consider parallel processing for speed
""")
print("="*70 + "\n")
