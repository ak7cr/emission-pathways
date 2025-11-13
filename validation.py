"""
Validation pipeline for comparing model output with ground observations
Supports OpenAQ and CPCB (Central Pollution Control Board) data sources
"""
import numpy as np
import requests
from datetime import datetime, timedelta
from scipy.spatial import cKDTree
from typing import Dict, List, Tuple, Optional
import json

class ValidationPipeline:
    """
    Handles fetching ground observations and computing validation metrics
    """
    
    def __init__(self):
        self.openaq_base_url = "https://api.openaq.org/v2"
        self.cpcb_base_url = "https://api.data.gov.in/resource"  # Example endpoint
        self.observations = []
        self.stations = []
    
    def fetch_openaq_data(
        self, 
        lat_min: float, 
        lat_max: float, 
        lon_min: float, 
        lon_max: float,
        parameter: str = "pm25",
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
        limit: int = 1000
    ) -> List[Dict]:
        """
        Fetch OpenAQ observations within bounding box
        
        Parameters:
        -----------
        lat_min, lat_max : float
            Latitude bounds
        lon_min, lon_max : float
            Longitude bounds
        parameter : str
            Pollutant parameter (pm25, pm10, no2, so2, o3, co)
        date_from, date_to : datetime, optional
            Time range for observations
        limit : int
            Maximum number of records to fetch
            
        Returns:
        --------
        observations : list of dict
            List of observation records with location, value, time
        """
        try:
            # Prepare query parameters
            params = {
                'limit': limit,
                'parameter': parameter,
                'coordinates': f'{lat_min},{lon_min},{lat_max},{lon_max}',
                'order_by': 'datetime'
            }
            
            if date_from:
                params['date_from'] = date_from.isoformat()
            if date_to:
                params['date_to'] = date_to.isoformat()
            
            # Fetch measurements
            response = requests.get(
                f"{self.openaq_base_url}/measurements",
                params=params,
                timeout=30
            )
            response.raise_for_status()
            
            data = response.json()
            
            # Parse observations
            observations = []
            for result in data.get('results', []):
                obs = {
                    'station_id': result.get('locationId'),
                    'station_name': result.get('location'),
                    'latitude': result['coordinates']['latitude'],
                    'longitude': result['coordinates']['longitude'],
                    'parameter': result['parameter'],
                    'value': result['value'],
                    'unit': result['unit'],
                    'timestamp': result['date']['utc'],
                    'source': 'OpenAQ'
                }
                observations.append(obs)
            
            self.observations.extend(observations)
            return observations
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching OpenAQ data: {e}")
            return []
    
    def fetch_cpcb_data(
        self,
        state: str,
        city: str,
        parameter: str = "pm2.5",
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
        api_key: Optional[str] = None
    ) -> List[Dict]:
        """
        Fetch CPCB (Central Pollution Control Board) observations
        
        Note: This is a template - actual CPCB API may differ
        
        Parameters:
        -----------
        state : str
            Indian state name
        city : str
            City name
        parameter : str
            Pollutant parameter
        date_from, date_to : datetime, optional
            Time range
        api_key : str, optional
            API key for data.gov.in
            
        Returns:
        --------
        observations : list of dict
            List of observation records
        """
        try:
            # Note: Actual CPCB API endpoint and parameters may differ
            # This is a template implementation
            params = {
                'api-key': api_key or 'demo-key',
                'format': 'json',
                'filters[state]': state,
                'filters[city]': city,
                'filters[parameter]': parameter,
                'limit': 1000
            }
            
            if date_from:
                params['filters[from_date]'] = date_from.strftime('%Y-%m-%d')
            if date_to:
                params['filters[to_date]'] = date_to.strftime('%Y-%m-%d')
            
            # Placeholder - actual endpoint would be different
            # response = requests.get(self.cpcb_base_url, params=params, timeout=30)
            
            # For now, return empty list with note
            print("CPCB data fetching requires valid API credentials")
            return []
            
        except Exception as e:
            print(f"Error fetching CPCB data: {e}")
            return []
    
    def create_synthetic_observations(
        self,
        n_stations: int,
        lat_range: Tuple[float, float],
        lon_range: Tuple[float, float],
        concentration_range: Tuple[float, float] = (10, 100),
        seed: int = 42
    ) -> List[Dict]:
        """
        Create synthetic observations for testing
        
        Parameters:
        -----------
        n_stations : int
            Number of synthetic stations
        lat_range, lon_range : tuple
            Geographic bounds
        concentration_range : tuple
            Range for synthetic concentration values
        seed : int
            Random seed
            
        Returns:
        --------
        observations : list of dict
            Synthetic observation records
        """
        np.random.seed(seed)
        
        observations = []
        for i in range(n_stations):
            obs = {
                'station_id': f'SYNTH_{i:03d}',
                'station_name': f'Synthetic Station {i+1}',
                'latitude': np.random.uniform(*lat_range),
                'longitude': np.random.uniform(*lon_range),
                'parameter': 'pm25',
                'value': np.random.uniform(*concentration_range),
                'unit': 'µg/m³',
                'timestamp': datetime.utcnow().isoformat(),
                'source': 'Synthetic'
            }
            observations.append(obs)
        
        self.observations.extend(observations)
        return observations
    
    def extract_model_values_at_stations(
        self,
        concentration_field: np.ndarray,
        x_grid: np.ndarray,
        y_grid: np.ndarray,
        observations: List[Dict],
        lat_to_y: callable,
        lon_to_x: callable
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract modeled concentration values at observation station locations
        
        Parameters:
        -----------
        concentration_field : ndarray, shape (ny, nx)
            Modeled concentration field
        x_grid, y_grid : ndarray
            Grid coordinates (in km)
        observations : list of dict
            Station observations
        lat_to_y, lon_to_x : callable
            Functions to convert lat/lon to grid coordinates
            
        Returns:
        --------
        modeled_values : ndarray
            Modeled concentrations at station locations
        observed_values : ndarray
            Observed concentrations
        """
        modeled_values = []
        observed_values = []
        
        for obs in observations:
            # Convert lat/lon to grid coordinates
            x_coord = lon_to_x(obs['longitude'])
            y_coord = lat_to_y(obs['latitude'])
            
            # Find nearest grid cell using bilinear interpolation
            if (x_grid[0] <= x_coord <= x_grid[-1] and 
                y_grid[0] <= y_coord <= y_grid[-1]):
                
                # Bilinear interpolation
                i = np.searchsorted(x_grid, x_coord) - 1
                j = np.searchsorted(y_grid, y_coord) - 1
                
                # Ensure indices are within bounds
                i = np.clip(i, 0, len(x_grid) - 2)
                j = np.clip(j, 0, len(y_grid) - 2)
                
                # Interpolation weights
                wx = (x_coord - x_grid[i]) / (x_grid[i+1] - x_grid[i])
                wy = (y_coord - y_grid[j]) / (y_grid[j+1] - y_grid[j])
                
                # Bilinear interpolation
                value = (
                    concentration_field[j, i] * (1 - wx) * (1 - wy) +
                    concentration_field[j, i+1] * wx * (1 - wy) +
                    concentration_field[j+1, i] * (1 - wx) * wy +
                    concentration_field[j+1, i+1] * wx * wy
                )
                
                modeled_values.append(value)
                observed_values.append(obs['value'])
        
        return np.array(modeled_values), np.array(observed_values)
    
    def compute_metrics(
        self,
        modeled: np.ndarray,
        observed: np.ndarray,
        threshold: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Compute validation metrics
        
        Parameters:
        -----------
        modeled : ndarray
            Modeled concentration values
        observed : ndarray
            Observed concentration values
        threshold : float, optional
            Threshold for exceedance hit rate (e.g., 35 µg/m³ for PM2.5)
            
        Returns:
        --------
        metrics : dict
            Dictionary of validation metrics:
            - correlation: Pearson correlation coefficient
            - bias: Mean bias (modeled - observed)
            - rmse: Root mean square error
            - mae: Mean absolute error
            - nmb: Normalized mean bias
            - nme: Normalized mean error
            - r2: Coefficient of determination
            - hit_rate: Hit rate for exceedances (if threshold provided)
            - far: False alarm ratio (if threshold provided)
        """
        # Remove any NaN or inf values
        valid_mask = np.isfinite(modeled) & np.isfinite(observed)
        modeled = modeled[valid_mask]
        observed = observed[valid_mask]
        
        if len(modeled) == 0:
            return {
                'n_pairs': 0,
                'correlation': np.nan,
                'bias': np.nan,
                'rmse': np.nan,
                'mae': np.nan,
                'nmb': np.nan,
                'nme': np.nan,
                'r2': np.nan
            }
        
        # Basic metrics
        bias = np.mean(modeled - observed)
        mae = np.mean(np.abs(modeled - observed))
        rmse = np.sqrt(np.mean((modeled - observed)**2))
        
        # Normalized metrics
        mean_obs = np.mean(observed)
        nmb = bias / mean_obs if mean_obs != 0 else np.nan
        nme = mae / mean_obs if mean_obs != 0 else np.nan
        
        # Correlation and R²
        correlation = np.corrcoef(modeled, observed)[0, 1]
        
        # R² (coefficient of determination)
        ss_res = np.sum((observed - modeled)**2)
        ss_tot = np.sum((observed - mean_obs)**2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else np.nan
        
        metrics = {
            'n_pairs': len(modeled),
            'correlation': float(correlation),
            'bias': float(bias),
            'rmse': float(rmse),
            'mae': float(mae),
            'nmb': float(nmb),
            'nme': float(nme),
            'r2': float(r2)
        }
        
        # Exceedance metrics
        if threshold is not None:
            obs_exceed = observed > threshold
            mod_exceed = modeled > threshold
            
            # Hit rate (probability of detection)
            hits = np.sum(obs_exceed & mod_exceed)
            misses = np.sum(obs_exceed & ~mod_exceed)
            hit_rate = hits / (hits + misses) if (hits + misses) > 0 else np.nan
            
            # False alarm ratio
            false_alarms = np.sum(~obs_exceed & mod_exceed)
            far = false_alarms / (hits + false_alarms) if (hits + false_alarms) > 0 else np.nan
            
            # Critical success index
            csi = hits / (hits + misses + false_alarms) if (hits + misses + false_alarms) > 0 else np.nan
            
            metrics.update({
                'threshold': float(threshold),
                'hit_rate': float(hit_rate),
                'false_alarm_ratio': float(far),
                'critical_success_index': float(csi),
                'n_observed_exceedances': int(np.sum(obs_exceed)),
                'n_modeled_exceedances': int(np.sum(mod_exceed))
            })
        
        return metrics
    
    def generate_validation_report(
        self,
        metrics: Dict[str, float],
        modeled: np.ndarray,
        observed: np.ndarray
    ) -> str:
        """
        Generate a text report of validation statistics
        
        Parameters:
        -----------
        metrics : dict
            Validation metrics from compute_metrics()
        modeled, observed : ndarray
            Modeled and observed values
            
        Returns:
        --------
        report : str
            Formatted validation report
        """
        report = []
        report.append("=" * 60)
        report.append("VALIDATION REPORT")
        report.append("=" * 60)
        report.append(f"\nNumber of station pairs: {metrics['n_pairs']}")
        report.append(f"Observed range: {np.min(observed):.2f} - {np.max(observed):.2f} µg/m³")
        report.append(f"Modeled range: {np.min(modeled):.2f} - {np.max(modeled):.2f} µg/m³")
        report.append("\nSTATISTICAL METRICS:")
        report.append("-" * 60)
        report.append(f"Correlation (r):           {metrics['correlation']:.3f}")
        report.append(f"R² (determination):        {metrics['r2']:.3f}")
        report.append(f"Mean Bias:                 {metrics['bias']:+.2f} µg/m³")
        report.append(f"Root Mean Square Error:    {metrics['rmse']:.2f} µg/m³")
        report.append(f"Mean Absolute Error:       {metrics['mae']:.2f} µg/m³")
        report.append(f"Normalized Mean Bias:      {metrics['nmb']*100:+.1f}%")
        report.append(f"Normalized Mean Error:     {metrics['nme']*100:.1f}%")
        
        if 'threshold' in metrics:
            report.append("\nEXCEEDANCE METRICS:")
            report.append("-" * 60)
            report.append(f"Threshold:                 {metrics['threshold']:.1f} µg/m³")
            report.append(f"Observed exceedances:      {metrics['n_observed_exceedances']}")
            report.append(f"Modeled exceedances:       {metrics['n_modeled_exceedances']}")
            report.append(f"Hit Rate (POD):            {metrics['hit_rate']:.3f}")
            report.append(f"False Alarm Ratio:         {metrics['false_alarm_ratio']:.3f}")
            report.append(f"Critical Success Index:    {metrics['critical_success_index']:.3f}")
        
        report.append("=" * 60)
        
        return "\n".join(report)
