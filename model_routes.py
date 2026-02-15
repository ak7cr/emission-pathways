"""
API routes for dispersion model management
"""
from flask import jsonify, request
from model_manager import model_manager
import simulation_state

def register_model_routes(app, sim_state):
    """Register model management API routes"""
    
    @app.route('/api/models/list', methods=['GET'])
    def list_models():
        """Get list of available dispersion models"""
        models = model_manager.list_available_models()
        
        # Get detailed info for each model, excluding non-serializable class objects
        models_list = []
        for model_type in models:
            info = model_manager.get_model_info(model_type)
            # Remove class object for JSON serialization
            safe_info = {k: v for k, v in info.items() if k != 'class'}
            safe_info['type'] = model_type
            models_list.append(safe_info)
        
        return jsonify({
            'success': True,
            'models': models_list,
            'current_model': sim_state.get('model_type', 'lagrangian')
        })
    
    @app.route('/api/models/select', methods=['POST'])
    def select_model():
        """Switch to a different dispersion model"""
        data = request.json
        model_type = data.get('model_type','lagrangian')
        
        if model_type not in model_manager.list_available_models():
            return jsonify({
                'success': False,
                'error': f'Unknown model type: {model_type}'
            }), 400
        
        # Update simulation state
        sim_state['model_type'] = model_type
        sim_state['model_state'] = None  # Will be initialized on next step
        sim_state['particles'] = None  # Reset
        sim_state['current_frame'] = 0
        
        # Get model info (exclude non-serializable class object)
        model_info = model_manager.get_model_info(model_type)
        # Remove class object for JSON serialization
        safe_info = {k: v for k, v in model_info.items() if k != 'class'}
        
        return jsonify({
            'success': True,
            'model_type': model_type,
            'model_info': safe_info,
            'message': f'Switched to {safe_info["name"]}'
        })
    
    @app.route('/api/models/info/<model_type>', methods=['GET'])
    def get_model_info(model_type):
        """Get detailed information about a specific model"""
        info = model_manager.get_model_info(model_type)
        
        if info is None:
            return jsonify({
                'success': False,
                'error': f'Unknown model type: {model_type}'
            }), 404
        
        # Remove class object for JSON serialization
        safe_info = {k: v for k, v in info.items() if k != 'class'}
        
        return jsonify({
            'success': True,
            'model_info': safe_info
        })
    
    @app.route('/api/models/current', methods=['GET'])
    def get_current_model():
        """Get information about currently active model"""
        model_type = sim_state.get('model_type', 'lagrangian')
        info = model_manager.get_model_info(model_type)
        
        # Remove class object for JSON serialization
        safe_info = {k: v for k, v in info.items() if k != 'class'} if info else {}
        
        # Add runtime statistics
        if sim_state.get('model_state'):
            if model_type == 'lagrangian':
                n_active = sim_state['model_state'].get('particle_active', []).sum() if sim_state['model_state'].get('particle_active') is not None else 0
                safe_info['runtime_stats'] = {
                    'active_particles': int(n_active),
                    'total_particles': len(sim_state['model_state'].get('particle_active', [])) if sim_state['model_state'].get('particle_active') is not None else 0
                }
            elif model_type == 'puff':
                safe_info['runtime_stats'] = {
                    'active_puffs': len(sim_state['model_state'].get('puffs', []))
                }
            elif model_type in ['eulerian', 'semi_lagrangian']:
                C = sim_state['model_state'].get('concentration')
                if C is not None:
                    import numpy as np
                    safe_info['runtime_stats'] = {
                        'max_concentration': float(C.max()),
                        'mean_concentration': float(C.mean())
                    }
        
        return jsonify({
            'success': True,
            'model_type': model_type,
            'model_info': safe_info
        })
