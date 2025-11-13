# Wind Data Integration Guide

## Overview
The simulator now supports real wind field data from ERA5, GFS, or custom sources.

## Supported Formats

### 1. NumPy Archive (.npz)
Recommended for preprocessed data. Structure:
```python
np.savez('wind_data.npz',
    time=time_array,    # 1D array: time points (hours or seconds)
    lat=lat_array,      # 1D array: latitude values
    lon=lon_array,      # 1D array: longitude values
    u=u_wind,          # 3D array: (time, lat, lon) - eastward wind
    v=v_wind           # 3D array: (time, lat, lon) - northward wind
)
```

### 2. NetCDF (.nc)
Direct ERA5/GFS format. Expected variables:
- `time`: time dimension
- `latitude` or `lat`: latitude coordinates
- `longitude` or `lon`: longitude coordinates
- `u10` or `u`: u-component of wind (m/s)
- `v10` or `v`: v-component of wind (m/s)

## Usage Options

### Option 1: Create Sample Data
Click "Create Sample Wind Data" in the UI to generate synthetic wind data for testing.

### Option 2: Upload Custom Data
1. Prepare your wind data in .npz or .nc format
2. Click "Upload Wind Data" in the UI
3. Select your file
4. The system will automatically load and interpolate the data

### Option 3: Load from File Manually
Place `wind_data.npz` in the project directory. The app will auto-load it when switching to "Real Wind" mode.

## Downloading ERA5 Data

### Using Python (Climate Data Store API)

```python
import cdsapi

c = cdsapi.Client()

c.retrieve(
    'reanalysis-era5-single-levels',
    {
        'product_type': 'reanalysis',
        'variable': ['10m_u_component_of_wind', '10m_v_component_of_wind'],
        'year': '2024',
        'month': '11',
        'day': ['01', '02', '03'],
        'time': ['00:00', '06:00', '12:00', '18:00'],
        'area': [45, -10, 35, 10],  # [N, W, S, E]
        'format': 'netcdf',
    },
    'download.nc')
```

### Converting ERA5 NetCDF to NPZ

```python
import netCDF4 as nc
import numpy as np

# Load ERA5 file
ds = nc.Dataset('download.nc')

# Extract data
time = ds.variables['time'][:]
lat = ds.variables['latitude'][:]
lon = ds.variables['longitude'][:]
u = ds.variables['u10'][:]  # (time, lat, lon)
v = ds.variables['v10'][:]

# Save as npz
np.savez('wind_data.npz', time=time, lat=lat, lon=lon, u=u, v=v)

ds.close()
```

## Domain Mapping

The simulator uses a 200×200 km domain. Ensure your wind data:
- Covers the spatial extent needed
- Has appropriate resolution
- Uses consistent units (m/s for wind speed)

The interpolator will:
- Handle temporal interpolation automatically
- Spatially interpolate to the simulation grid
- Use boundary values for out-of-bounds points

## Performance Tips

1. **Spatial Resolution**: Higher resolution wind data = more accurate but slower interpolation
2. **Temporal Resolution**: 1-6 hour intervals usually sufficient for dispersion modeling
3. **Data Size**: Keep files under 1GB for responsive performance
4. **Preprocessing**: Convert to .npz format for faster loading

## Troubleshooting

**"No data loaded" error:**
- Check file format matches expected structure
- Verify variable names in NetCDF files
- Ensure dimensions are correctly ordered (time, lat, lon)

**Interpolation warnings:**
- Wind data domain may not cover simulation domain
- Check coordinate systems match (degrees vs km)
- Verify time array units

**Slow performance:**
- Reduce spatial resolution of wind data
- Use fewer time steps
- Consider downsampling before upload
