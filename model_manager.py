"""
Model Manager - Factory and coordinator for dispersion models
"""
from models import AVAILABLE_MODELS

class ModelManager:
    """Manages dispersion model initialization and execution"""
    
    def __init__(self):
        self.current_model = None
        self.model_type = None
    
    def create_model(self, model_type, config):
        """
        Create a dispersion model instance
        
        Parameters:
        -----------
        model_type : str
            Model identifier (lagrangian, eulerian, etc.)
        config : dict
            Model configuration
        
        Returns:
        --------
        model : BaseDispersionModel
            Initialized model instance
        """
        if model_type not in AVAILABLE_MODELS:
            raise ValueError(f"Unknown model type: {model_type}. Available: {list(AVAILABLE_MODELS.keys())}")
        
        model_class = AVAILABLE_MODELS[model_type]['class']
        self.current_model = model_class(config)
        self.model_type = model_type
        
        return self.current_model
    
    def get_model_info(self, model_type=None):
        """Get information about a model"""
        if model_type is None:
            model_type = self.model_type
        
        if model_type in AVAILABLE_MODELS:
            info = AVAILABLE_MODELS[model_type].copy()
            if self.current_model and model_type == self.model_type:
                info['runtime_info'] = self.current_model.get_info()
            return info
        
        return None
    
    def list_available_models(self):
        """List all available models"""
        return {
            model_type: {
                'name': info['name'],
                'description': info['description'],
                'icon': info['icon']
            }
            for model_type, info in AVAILABLE_MODELS.items()
        }

# Global model manager instance
model_manager = ModelManager()
