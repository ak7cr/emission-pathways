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
        return jsonify({
            'success': True,
            'models': models,
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
        
        # Get model info
        model_info = model_manager.get_model_info(model_type)
        
        return jsonify({
            'success': True,
            'model_type': model_type,
            'model_info': model_info,
            'message': f'Switched to {model_info["name"]}'
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
        
        return jsonify({
            'success': True,
            'model_info': info
        })
    
    @app.route('/api/models/current', methods=['GET'])
    def get_current_model():
        """Get information about currently active model"""
        model_type = sim_state.get('model_type', 'lagrangian')
        info = model_manager.get_model_info(model_type)
        
        # Add runtime statistics
        if sim_state.get('model_state'):
            if model_type == 'lagrangian':
                n_active = sim_state['model_state'].get('particle_active', []).sum() if sim_state['model_state'].get('particle_active') is not None else 0
                info['runtime_stats'] = {
                    'active_particles': int(n_active),
                    'total_particles': len(sim_state['model_state'].get('particle_active', [])) if sim_state['model_state'].get('particle_active') is not None else 0
                }
            elif model_type == 'puff':
                info['runtime_stats'] = {
                    'active_puffs': len(sim_state['model_state'].get('puffs', []))
                }
            elif model_type in ['eulerian', 'semi_lagrangian']:
                C = sim_state['model_state'].get('concentration')
                if C is not None:
                    info['runtime_stats'] = {
                        'max_concentration': float(C.max()),
                        'mean_concentration': float(C.mean())
                    }
        
        return jsonify({
            'success': True,
            'model_type': model_type,
            'model_info': info
        })
