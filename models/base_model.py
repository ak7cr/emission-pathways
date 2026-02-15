"""
Base class for all dispersion models
Defines common interface that all models must implement
"""
import numpy as np
from abc import ABC, abstractmethod

class BaseDispersionModel(ABC):
    """Abstract base class for dispersion models"""
    
    def __init__(self, config):
        """
        Initialize model with configuration
        
        Parameters:
        -----------
        config : dict
            Model configuration including domain, physics parameters
        """
        self.config = config
        self.time = 0.0
        self.initialized = False
    
    @abstractmethod
    def initialize(self, hotspots, npph):
        """
        Initialize model state
        
        Parameters:
        -----------
        hotspots : list of [x, y]
            Emission source locations (km)
        npph : int
            Number of particles per hotspot (or grid resolution parameter)
        
        Returns:
        --------
        state : dict
            Initial model state
        """
        pass
    
    @abstractmethod
    def step(self, state, t, dt, wind_U, wind_V):
        """
        Advance model by one time step
        
        Parameters:
        -----------
        state : dict
            Current model state
        t : float
            Current time (seconds)
        dt : float
            Time step (seconds)
        wind_U, wind_V : ndarray
            Wind field components (m/s)
        
        Returns:
        --------
        state : dict
            Updated model state
        """
        pass
    
    @abstractmethod
    def get_concentration(self, state):
        """
        Calculate concentration field from model state
        
        Parameters:
        -----------
        state : dict
            Current model state
        
        Returns:
        --------
        C_normalized : ndarray, shape (ny, nx)
            Normalized concentration (0-1)
        C_physical : ndarray, shape (ny, nx)
            Physical concentration (µg/m³)
        """
        pass
    
    @abstractmethod
    def emit_particles(self, state, hotspots, npph):
        """
        Emit new particles/mass from sources
        
        Parameters:
        -----------
        state : dict
            Current model state
        hotspots : list
            Source locations
        npph : int
            Particles per hotspot
        
        Returns:
        --------
        state : dict
            Updated state with new emissions
        """
        pass
    
    def get_info(self):
        """Get model information and statistics"""
        return {
            'model_type': self.__class__.__name__,
            'time': self.time,
            'initialized': self.initialized
        }
