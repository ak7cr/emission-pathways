"""
Emission Estimation Utilities
Includes FRP to emissions conversion and emission factor database
"""
import numpy as np

# Emission factors (g/kg dry matter burned) for different pollutants
# Source: Based on Andreae & Merlet (2001), updated values
EMISSION_FACTORS = {
    'PM2.5': {
        'savanna': 3.4,
        'tropical_forest': 9.1,
        'extratropical_forest': 13.0,
        'crop_residue': 3.9,
        'peat': 16.0
    },
    'PM10': {
        'savanna': 8.5,
        'tropical_forest': 13.5,
        'extratropical_forest': 17.6,
        'crop_residue': 8.8,
        'peat': 23.0
    },
    'CO': {
        'savanna': 63,
        'tropical_forest': 104,
        'extratropical_forest': 107,
        'crop_residue': 92,
        'peat': 210
    },
    'CO2': {
        'savanna': 1613,
        'tropical_forest': 1580,
        'extratropical_forest': 1569,
        'crop_residue': 1515,
        'peat': 1703
    },
    'NOx': {
        'savanna': 3.9,
        'tropical_forest': 2.55,
        'extratropical_forest': 3.0,
        'crop_residue': 2.5,
        'peat': 3.9
    },
    'SO2': {
        'savanna': 0.48,
        'tropical_forest': 0.40,
        'extratropical_forest': 0.40,
        'crop_residue': 0.40,
        'peat': 0.40
    }
}


def frp_to_fuel_consumption(frp_mw, duration_hours=1.0, combustion_efficiency=0.5):
    """
    Convert Fire Radiative Power (FRP) to fuel consumption
    
    Parameters:
    -----------
    frp_mw : float
        Fire Radiative Power in MW
    duration_hours : float
        Fire duration in hours
    combustion_efficiency : float
        Combustion efficiency (0-1), default 0.5
        
    Returns:
    --------
    fuel_consumed_kg : float
        Fuel consumed in kg
        
    References:
    -----------
    Wooster et al. (2005): FRE = 0.368 * fuel_consumed
    where FRE is Fire Radiative Energy in MJ
    """
    # Convert FRP to FRE (Fire Radiative Energy)
    # FRE (MJ) = FRP (MW) * duration (hours) * 3600 (s/hr)
    fre_mj = frp_mw * duration_hours * 3600
    
    # Wooster coefficient for fuel consumption
    # fuel_consumed (kg) = FRE (MJ) / 0.368
    fuel_consumed_kg = fre_mj / 0.368
    
    # Apply combustion efficiency
    fuel_consumed_kg *= combustion_efficiency
    
    return fuel_consumed_kg


def calculate_emission(fuel_consumed_kg, pollutant='PM2.5', vegetation_type='tropical_forest'):
    """
    Calculate pollutant emission from fuel consumption
    
    Parameters:
    -----------
    fuel_consumed_kg : float
        Fuel consumed in kg (dry matter)
    pollutant : str
        Pollutant type (PM2.5, PM10, CO, CO2, NOx, SO2)
    vegetation_type : str
        Vegetation type (savanna, tropical_forest, etc.)
        
    Returns:
    --------
    emission_g : float
        Total emission in grams
    """
    if pollutant not in EMISSION_FACTORS:
        raise ValueError(f"Unknown pollutant: {pollutant}")
    
    if vegetation_type not in EMISSION_FACTORS[pollutant]:
        vegetation_type = 'tropical_forest'  # default
    
    # Get emission factor (g/kg)
    ef = EMISSION_FACTORS[pollutant][vegetation_type]
    
    # Calculate emission
    emission_g = fuel_consumed_kg * ef
    
    return emission_g


def frp_to_emission(frp_mw, pollutant='PM2.5', vegetation_type='tropical_forest', 
                    duration_hours=1.0, combustion_efficiency=0.5):
    """
    Direct conversion from FRP to pollutant emission
    
    Parameters:
    -----------
    frp_mw : float
        Fire Radiative Power in MW
    pollutant : str
        Pollutant type
    vegetation_type : str
        Vegetation type
    duration_hours : float
        Fire duration in hours
    combustion_efficiency : float
        Combustion efficiency (0-1)
        
    Returns:
    --------
    emission_g : float
        Total emission in grams
    """
    # Step 1: FRP to fuel consumption
    fuel_kg = frp_to_fuel_consumption(frp_mw, duration_hours, combustion_efficiency)
    
    # Step 2: Fuel to emission
    emission_g = calculate_emission(fuel_kg, pollutant, vegetation_type)
    
    return emission_g


def get_emission_factor(pollutant, vegetation_type):
    """
    Get emission factor for a specific pollutant and vegetation type
    
    Returns:
    --------
    ef : float
        Emission factor in g/kg
    """
    if pollutant not in EMISSION_FACTORS:
        return None
    
    return EMISSION_FACTORS[pollutant].get(vegetation_type, None)


def list_vegetation_types():
    """List available vegetation types"""
    # Get from PM2.5 as reference
    return list(EMISSION_FACTORS['PM2.5'].keys())


def list_pollutants():
    """List available pollutants"""
    return list(EMISSION_FACTORS.keys())


# Example usage and testing
if __name__ == '__main__':
    print("=== Emission Estimation Utilities ===\n")
    
    print("Available pollutants:", list_pollutants())
    print("Available vegetation types:", list_vegetation_types())
    print()
    
    # Example 1: FRP to emission
    frp = 100  # MW
    duration = 2  # hours
    
    print(f"Example: FRP = {frp} MW, Duration = {duration} hours")
    print("-" * 50)
    
    for pollutant in ['PM2.5', 'CO', 'CO2']:
        emission = frp_to_emission(frp, pollutant, 'tropical_forest', duration)
        print(f"{pollutant}: {emission:.2f} g")
    
    print()
    
    # Example 2: Emission factors
    print("Emission Factors (g/kg) for tropical forest:")
    print("-" * 50)
    for pollutant in list_pollutants():
        ef = get_emission_factor(pollutant, 'tropical_forest')
        print(f"{pollutant}: {ef} g/kg")
