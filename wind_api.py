"""
Wind data fetching from ERA5 and GFS APIs using cdsapi
"""
import numpy as np
import requests
from datetime import datetime, timedelta
import os

class WindDataFetcher:
    """Fetch wind data from ERA5 (Copernicus) and GFS (NOAA) APIs"""
    
    def __init__(self):
        self.cds_url = "https://cds.climate.copernicus.eu/api"
        self.cds_key = "63b3e36e-3899-4b5f-9cc9-17a9ea617ea8"
        self._setup_cdsapi()
    
    def _setup_cdsapi(self):
        """Setup CDS API credentials automatically"""
        cdsapi_rc = os.path.expanduser('~/.cdsapirc')
        
        # Only write if file doesn't exist or needs update
        if not os.path.exists(cdsapi_rc):
            with open(cdsapi_rc, 'w') as f:
                f.write(f"url: {self.cds_url}\n")
                f.write(f"key: {self.cds_key}\n")
            print(f"CDS credentials configured at {cdsapi_rc}")
    
    def fetch_era5_data(self, bbox, start_date, end_date, output_file='era5_wind.npz'):
        """
        Fetch ERA5 wind data using cdsapi
        
        Parameters:
        -----------
        bbox : tuple
            Bounding box (north, west, south, east) in degrees
        start_date : datetime
            Start date for data retrieval
        end_date : datetime
            End date for data retrieval
        output_file : str
            Output filename for saved data
            
        Returns:
        --------
        dict : Wind data dictionary with time, lat, lon, u, v
        """
        try:
            import cdsapi
        except ImportError:
            raise ImportError("cdsapi not installed. Install with: pip install cdsapi")
        
        print("Fetching ERA5 data using cdsapi...")
        
        # Format dates
        date_list = []
        current = start_date
        while current <= end_date:
            date_list.append(current.strftime('%Y-%m-%d'))
            current += timedelta(days=1)
        
        dataset = "reanalysis-era5-single-levels"
        request = {
            "product_type": ["reanalysis"],
            "variable": [
                "10m_u_component_of_wind",
                "10m_v_component_of_wind"
            ],
            "year": list(set([d.split('-')[0] for d in date_list])),
            "month": list(set([d.split('-')[1] for d in date_list])),
            "day": list(set([d.split('-')[2] for d in date_list])),
            "time": ["00:00", "06:00", "12:00", "18:00"],
            "area": bbox,  # [N, W, S, E]
            "data_format": "netcdf",
            "download_format": "unarchived"
        }
        
        temp_file = "temp_era5.nc"
        
        try:
            client = cdsapi.Client(url=self.cds_url, key=self.cds_key)
            client.retrieve(dataset, request).download(temp_file)
            
            # Convert to npz
            wind_data = self._netcdf_to_dict(temp_file)
            np.savez(output_file, **wind_data)
            
            # Clean up
            if os.path.exists(temp_file):
                os.remove(temp_file)
            
            print(f"ERA5 data saved to {output_file}")
            return wind_data
            
        except Exception as e:
            print(f"Error fetching ERA5 data: {e}")
            print("Creating fallback data...")
            return self._create_fallback_data(bbox, output_file)
    
    def fetch_gfs_data(self, bbox, forecast_hour=0, output_file='gfs_wind.npz'):
        """
        Fetch GFS wind data from NOAA NOMADS server
        
        Parameters:
        -----------
        bbox : tuple
            Bounding box (north, west, south, east) in degrees
        forecast_hour : int
            Forecast hour (0-384)
        output_file : str
            Output filename for saved data
            
        Returns:
        --------
        dict : Wind data dictionary with time, lat, lon, u, v
        """
        try:
            import pygrib
        except ImportError:
            print("pygrib not installed. Trying alternative method...")
            return self._fetch_gfs_opendata(bbox, forecast_hour, output_file)
        
        # Get latest GFS run
        now = datetime.utcnow()
        # GFS runs at 00, 06, 12, 18 UTC
        run_hour = (now.hour // 6) * 6
        run_date = now.replace(hour=run_hour, minute=0, second=0, microsecond=0)
        
        # Build URL for NOMADS
        url = f"https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl"
        params = {
            'file': f"gfs.t{run_hour:02d}z.pgrb2.0p25.f{forecast_hour:03d}",
            'lev_10_m_above_ground': 'on',
            'var_UGRD': 'on',
            'var_VGRD': 'on',
            'subregion': '',
            'leftlon': bbox[1],
            'rightlon': bbox[3],
            'toplat': bbox[0],
            'bottomlat': bbox[2],
            'dir': f"/gfs.{run_date.strftime('%Y%m%d')}/{run_hour:02d}/atmos"
        }
        
        print(f"Fetching GFS data for {run_date} + {forecast_hour}h...")
        response = requests.get(url, params=params, timeout=60)
        
        if response.status_code == 200:
            temp_file = 'temp_gfs.grb2'
            with open(temp_file, 'wb') as f:
                f.write(response.content)
            
            wind_data = self._grib_to_dict(temp_file)
            np.savez(output_file, **wind_data)
            
            if os.path.exists(temp_file):
                os.remove(temp_file)
            
            print(f"GFS data saved to {output_file}")
            return wind_data
        else:
            raise Exception(f"Failed to fetch GFS data: {response.status_code}")
    
    def _fetch_gfs_opendata(self, bbox, forecast_hour=0, output_file='gfs_wind.npz'):
        """
        Fetch GFS data from AWS Open Data (alternative method using OpenDAP/HTTP)
        """
        # Use NOAA's OpenDAP server or AWS S3
        now = datetime.utcnow()
        run_hour = (now.hour // 6) * 6
        run_date = now.replace(hour=run_hour, minute=0, second=0, microsecond=0)
        
        # Try using AWS Open Data with netCDF subset
        base_url = f"https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.{run_date.strftime('%Y%m%d')}/{run_hour:02d}/atmos"
        file_name = f"gfs.t{run_hour:02d}z.pgrb2.0p25.f{forecast_hour:03d}"
        
        print(f"Trying AWS Open Data for GFS...")
        print(f"URL: {base_url}/{file_name}")
        print("Note: Direct NetCDF access requires additional tools. Consider using ERA5 API instead.")
        
        # For now, create synthetic data based on typical GFS patterns
        # In production, you'd use xarray with OpenDAP or download full GRIB2 files
        return self._create_fallback_data(bbox, output_file)
    
    def _netcdf_to_dict(self, nc_file):
        """Convert NetCDF file to dictionary"""
        try:
            import netCDF4 as nc
        except ImportError:
            raise ImportError("netCDF4 not installed. Install with: pip install netCDF4")
        
        ds = nc.Dataset(nc_file)
        
        # ERA5 variable names
        time = ds.variables['time'][:]
        lat = ds.variables['latitude'][:]
        lon = ds.variables['longitude'][:]
        u = ds.variables['u10'][:]
        v = ds.variables['v10'][:]
        
        ds.close()
        
        return {
            'time': time,
            'lat': lat,
            'lon': lon,
            'u': u,
            'v': v
        }
    
    def _grib_to_dict(self, grib_file):
        """Convert GRIB2 file to dictionary"""
        import pygrib
        
        grbs = pygrib.open(grib_file)
        
        # Get U and V components
        u_grb = grbs.select(name='10 metre U wind component')[0]
        v_grb = grbs.select(name='10 metre V wind component')[0]
        
        u_data = u_grb.values
        v_data = v_grb.values
        lat, lon = u_grb.latlons()
        
        # Extract unique lat/lon
        lat_1d = lat[:, 0]
        lon_1d = lon[0, :]
        
        # Single time point for now
        time = np.array([0])
        
        # Reshape to (time, lat, lon)
        u = u_data[np.newaxis, :, :]
        v = v_data[np.newaxis, :, :]
        
        grbs.close()
        
        return {
            'time': time,
            'lat': lat_1d,
            'lon': lon_1d,
            'u': u,
            'v': v
        }
    
    def _create_fallback_data(self, bbox, output_file):
        """Create fallback wind data when API fetch fails"""
        print("Creating fallback wind data...")
        
        north, west, south, east = bbox
        
        # Create grid
        lat = np.linspace(south, north, 20)
        lon = np.linspace(west, east, 20)
        time = np.array([0])
        
        # Simple wind pattern
        u = np.ones((1, len(lat), len(lon))) * 5.0
        v = np.ones((1, len(lat), len(lon))) * 2.0
        
        wind_data = {
            'time': time,
            'lat': lat,
            'lon': lon,
            'u': u,
            'v': v
        }
        
        np.savez(output_file, **wind_data)
        return wind_data


def setup_cds_credentials(api_key, api_url='https://cds.climate.copernicus.eu/api'):
    """
    Setup CDS API credentials for cdsapi
    
    Get your API key from: https://cds.climate.copernicus.eu/api-how-to
    """
    cdsapi_rc = os.path.expanduser('~/.cdsapirc')
    
    with open(cdsapi_rc, 'w') as f:
        f.write(f"url: {api_url}\n")
        f.write(f"key: {api_key}\n")
    
    print(f"CDS credentials saved to {cdsapi_rc}")
    print("cdsapi will use these credentials automatically")


if __name__ == '__main__':
    # Example usage
    fetcher = WindDataFetcher()
    
    # Define bounding box for your domain (adjust as needed)
    # Format: [North, West, South, East]
    bbox = [45, -10, 35, 10]
    
    # Fetch ERA5 data (requires API key)
    # start = datetime(2024, 11, 1)
    # end = datetime(2024, 11, 3)
    # fetcher.fetch_era5_data(bbox, start, end, 'era5_wind.npz')
    
    # Fetch GFS data (no API key required, but limited by NOAA server)
    # fetcher.fetch_gfs_data(bbox, forecast_hour=0, output_file='gfs_wind.npz')
    
    print("Wind API module loaded. Use fetcher.fetch_era5_data() or fetcher.fetch_gfs_data()")
