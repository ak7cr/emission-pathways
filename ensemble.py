"""
Ensemble simulation capability with uncertainty quantification
Supports perturbations in wind speed, PBL height, emissions, and turbulence
"""
import numpy as np
from typing import Dict, List, Tuple, Optional
import copy
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing

class EnsembleSimulation:
    """
    Manages ensemble simulations with perturbed parameters
    """
    
    def __init__(self, base_config: Dict):
        """
        Initialize ensemble simulation
        
        Parameters:
        -----------
        base_config : dict
            Base simulation configuration (sim_state)
        """
        self.base_config = copy.deepcopy(base_config)
        self.ensemble_members = []
        self.results = []
    
    def generate_ensemble_configs(
        self,
        n_members: int,
        perturbations: Dict[str, Tuple[float, float]],
        seed: int = 42
    ) -> List[Dict]:
        """
        Generate ensemble member configurations with perturbed parameters
        
        Parameters:
        -----------
        n_members : int
            Number of ensemble members
        perturbations : dict
            Dictionary of parameter perturbations:
            {
                'wind_speed_factor': (min_factor, max_factor),  # e.g., (0.8, 1.2) for ±20%
                'wind_direction_offset': (min_deg, max_deg),    # e.g., (-15, 15) for ±15°
                'mixing_height_factor': (min_factor, max_factor),
                'emission_factor': (min_factor, max_factor),
                'turbulence_factor': (min_factor, max_factor)
            }
        seed : int
            Random seed for reproducibility
            
        Returns:
        --------
        configs : list of dict
            List of perturbed configurations
        """
        np.random.seed(seed)
        configs = []
        
        for i in range(n_members):
            config = copy.deepcopy(self.base_config)
            config['ensemble_member'] = i
            
            # Wind speed perturbation
            if 'wind_speed_factor' in perturbations:
                min_f, max_f = perturbations['wind_speed_factor']
                config['wind_speed_perturbation'] = np.random.uniform(min_f, max_f)
            
            # Wind direction perturbation (degrees)
            if 'wind_direction_offset' in perturbations:
                min_deg, max_deg = perturbations['wind_direction_offset']
                config['wind_direction_offset'] = np.random.uniform(min_deg, max_deg)
            
            # Mixing height (PBL) perturbation
            if 'mixing_height_factor' in perturbations:
                min_f, max_f = perturbations['mixing_height_factor']
                factor = np.random.uniform(min_f, max_f)
                config['mixing_height'] = self.base_config['mixing_height'] * factor
            
            # Emission scaling
            if 'emission_factor' in perturbations:
                min_f, max_f = perturbations['emission_factor']
                config['emission_scaling'] = np.random.uniform(min_f, max_f)
            
            # Turbulence amplitude perturbation
            if 'turbulence_factor' in perturbations:
                min_f, max_f = perturbations['turbulence_factor']
                factor = np.random.uniform(min_f, max_f)
                config['sigma_turb'] = self.base_config['sigma_turb'] * factor
            
            configs.append(config)
        
        self.ensemble_members = configs
        return configs
    
    def apply_wind_perturbations(
        self,
        U: np.ndarray,
        V: np.ndarray,
        speed_factor: float = 1.0,
        direction_offset_deg: float = 0.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply wind perturbations to wind field
        
        Parameters:
        -----------
        U, V : ndarray
            Original wind components (m/s)
        speed_factor : float
            Multiplicative factor for wind speed
        direction_offset_deg : float
            Directional offset in degrees (clockwise)
            
        Returns:
        --------
        U_pert, V_pert : ndarray
            Perturbed wind components
        """
        # Convert to speed and direction
        speed = np.sqrt(U**2 + V**2)
        direction = np.arctan2(V, U)  # radians
        
        # Apply perturbations
        speed_pert = speed * speed_factor
        direction_pert = direction + np.deg2rad(direction_offset_deg)
        
        # Convert back to components
        U_pert = speed_pert * np.cos(direction_pert)
        V_pert = speed_pert * np.sin(direction_pert)
        
        return U_pert, V_pert
    
    def compute_ensemble_statistics(
        self,
        concentration_fields: List[np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """
        Compute ensemble mean and uncertainty metrics
        
        Parameters:
        -----------
        concentration_fields : list of ndarray
            List of concentration fields from ensemble members
            
        Returns:
        --------
        statistics : dict
            {
                'mean': ensemble mean concentration,
                'std': ensemble standard deviation,
                'min': minimum across ensemble,
                'max': maximum across ensemble,
                'median': ensemble median,
                'p10': 10th percentile,
                'p90': 90th percentile
            }
        """
        # Stack all fields
        fields_array = np.stack(concentration_fields, axis=0)  # Shape: (n_members, ny, nx)
        
        statistics = {
            'mean': np.mean(fields_array, axis=0),
            'std': np.std(fields_array, axis=0),
            'min': np.min(fields_array, axis=0),
            'max': np.max(fields_array, axis=0),
            'median': np.median(fields_array, axis=0),
            'p10': np.percentile(fields_array, 10, axis=0),
            'p90': np.percentile(fields_array, 90, axis=0),
            'coefficient_of_variation': np.std(fields_array, axis=0) / (np.mean(fields_array, axis=0) + 1e-10)
        }
        
        return statistics
    
    def compute_arrival_time_statistics(
        self,
        particle_trajectories: List[np.ndarray],
        target_location: Tuple[float, float],
        threshold_distance: float = 5.0
    ) -> Dict[str, float]:
        """
        Compute statistics on arrival times at a target location
        
        Parameters:
        -----------
        particle_trajectories : list of ndarray
            List of particle position arrays from ensemble members
            Each array has shape (n_particles, 2) with (x, y) positions
        target_location : tuple
            (x, y) coordinates of target location
        threshold_distance : float
            Distance threshold (km) to consider arrival
            
        Returns:
        --------
        statistics : dict
            {
                'mean_arrival_time': mean arrival time,
                'std_arrival_time': standard deviation,
                'median_arrival_time': median,
                'arrival_probability': fraction of members that reach target
            }
        """
        arrival_times = []
        
        for traj in particle_trajectories:
            # Compute distances to target
            distances = np.sqrt(
                (traj[:, 0] - target_location[0])**2 + 
                (traj[:, 1] - target_location[1])**2
            )
            
            # Find first time particle reaches target
            arrived = distances < threshold_distance
            if np.any(arrived):
                arrival_time = np.argmax(arrived)  # Index of first True
                arrival_times.append(arrival_time)
        
        if len(arrival_times) == 0:
            return {
                'mean_arrival_time': np.nan,
                'std_arrival_time': np.nan,
                'median_arrival_time': np.nan,
                'arrival_probability': 0.0,
                'n_arrived': 0,
                'n_members': len(particle_trajectories)
            }
        
        arrival_times = np.array(arrival_times)
        
        return {
            'mean_arrival_time': float(np.mean(arrival_times)),
            'std_arrival_time': float(np.std(arrival_times)),
            'median_arrival_time': float(np.median(arrival_times)),
            'min_arrival_time': float(np.min(arrival_times)),
            'max_arrival_time': float(np.max(arrival_times)),
            'arrival_probability': len(arrival_times) / len(particle_trajectories),
            'n_arrived': len(arrival_times),
            'n_members': len(particle_trajectories)
        }
    
    def compute_uncertainty_bands(
        self,
        time_series_list: List[np.ndarray],
        confidence_levels: List[float] = [0.68, 0.95]
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Compute uncertainty bands for time series data
        
        Parameters:
        -----------
        time_series_list : list of ndarray
            List of time series from ensemble members
            Each array has shape (n_timesteps,)
        confidence_levels : list of float
            Confidence levels for uncertainty bands (e.g., [0.68, 0.95] for 1σ and 2σ)
            
        Returns:
        --------
        bands : dict
            Dictionary with keys for each confidence level:
            {
                'mean': mean time series,
                '0.68': {'lower': ..., 'upper': ...},
                '0.95': {'lower': ..., 'upper': ...}
            }
        """
        # Stack time series
        ts_array = np.stack(time_series_list, axis=0)  # Shape: (n_members, n_timesteps)
        
        bands = {
            'mean': np.mean(ts_array, axis=0),
            'median': np.median(ts_array, axis=0)
        }
        
        for cl in confidence_levels:
            alpha = (1 - cl) / 2
            lower_percentile = alpha * 100
            upper_percentile = (1 - alpha) * 100
            
            bands[str(cl)] = {
                'lower': np.percentile(ts_array, lower_percentile, axis=0),
                'upper': np.percentile(ts_array, upper_percentile, axis=0)
            }
        
        return bands
    
    def sensitivity_analysis(
        self,
        results: List[Dict],
        parameter_name: str,
        output_metric: str = 'max_concentration'
    ) -> Dict[str, float]:
        """
        Perform sensitivity analysis for a parameter
        
        Parameters:
        -----------
        results : list of dict
            Results from ensemble members, each containing:
            - config: member configuration
            - outputs: simulation outputs
        parameter_name : str
            Name of perturbed parameter
        output_metric : str
            Output metric to analyze
            
        Returns:
        --------
        sensitivity : dict
            Correlation coefficient and regression slope
        """
        param_values = []
        output_values = []
        
        for result in results:
            if parameter_name in result['config']:
                param_values.append(result['config'][parameter_name])
                output_values.append(result['outputs'][output_metric])
        
        if len(param_values) < 2:
            return {
                'correlation': np.nan,
                'slope': np.nan,
                'n_samples': len(param_values)
            }
        
        param_values = np.array(param_values)
        output_values = np.array(output_values)
        
        # Compute correlation
        correlation = np.corrcoef(param_values, output_values)[0, 1]
        
        # Linear regression slope
        slope = np.cov(param_values, output_values)[0, 1] / np.var(param_values)
        
        return {
            'correlation': float(correlation),
            'slope': float(slope),
            'n_samples': len(param_values),
            'param_range': (float(np.min(param_values)), float(np.max(param_values))),
            'output_range': (float(np.min(output_values)), float(np.max(output_values)))
        }
    
    def generate_ensemble_report(
        self,
        statistics: Dict[str, np.ndarray],
        arrival_stats: Optional[Dict] = None
    ) -> str:
        """
        Generate ensemble uncertainty report
        
        Parameters:
        -----------
        statistics : dict
            Ensemble statistics from compute_ensemble_statistics()
        arrival_stats : dict, optional
            Arrival time statistics
            
        Returns:
        --------
        report : str
            Formatted ensemble report
        """
        report = []
        report.append("=" * 60)
        report.append("ENSEMBLE UNCERTAINTY QUANTIFICATION")
        report.append("=" * 60)
        report.append(f"\nNumber of ensemble members: {len(self.ensemble_members)}")
        
        report.append("\nCONCENTRATION FIELD STATISTICS:")
        report.append("-" * 60)
        report.append(f"Ensemble mean max:         {np.max(statistics['mean']):.2f} µg/m³")
        report.append(f"Ensemble spread (max std): {np.max(statistics['std']):.2f} µg/m³")
        report.append(f"Min across members:        {np.min(statistics['min']):.2f} µg/m³")
        report.append(f"Max across members:        {np.max(statistics['max']):.2f} µg/m³")
        report.append(f"Max coefficient of var:    {np.max(statistics['coefficient_of_variation']):.2f}")
        
        if arrival_stats is not None:
            report.append("\nARRIVAL TIME STATISTICS:")
            report.append("-" * 60)
            report.append(f"Arrival probability:       {arrival_stats['arrival_probability']*100:.1f}%")
            report.append(f"Members arrived:           {arrival_stats['n_arrived']}/{arrival_stats['n_members']}")
            if arrival_stats['n_arrived'] > 0:
                report.append(f"Mean arrival time:         {arrival_stats['mean_arrival_time']:.1f} steps")
                report.append(f"Std deviation:             {arrival_stats['std_arrival_time']:.1f} steps")
                report.append(f"Range:                     {arrival_stats['min_arrival_time']:.0f} - {arrival_stats['max_arrival_time']:.0f} steps")
        
        report.append("=" * 60)
        
        return "\n".join(report)
