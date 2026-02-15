"""
Dispersion modeling approaches
Modular implementation of different atmospheric dispersion models
"""

from .lagrangian_model import LagrangianModel
from .eulerian_model import EulerianModel
from .gaussian_plume_model import GaussianPlumeModel
from .puff_model import PuffModel
from .semi_lagrangian_model import SemiLagrangianModel
from .hybrid_model import HybridModel

__all__ = [
    'LagrangianModel',
    'EulerianModel', 
    'GaussianPlumeModel',
    'PuffModel',
    'SemiLagrangianModel',
    'HybridModel'
]

AVAILABLE_MODELS = {
    'lagrangian': {
        'name': 'Lagrangian (Particle Tracking)',
        'class': LagrangianModel,
        'description': 'Track individual particles through space - best for point sources',
        'icon': '🎯'
    },
    'eulerian': {
        'name': 'Eulerian (Grid-Based PDE)',
        'class': EulerianModel,
        'description': 'Solve advection-diffusion on fixed grid - best for large domains',
        'icon': '📐'
    },
    'gaussian_plume': {
        'name': 'Gaussian Plume (Analytical)',
        'class': GaussianPlumeModel,
        'description': 'Analytical steady-state solution - fastest, regulatory standard',
        'icon': '📊'
    },
    'puff': {
        'name': 'Puff Model (Discrete Puffs)',
        'class': PuffModel,
        'description': 'Track expanding Gaussian puffs - good for intermittent sources',
        'icon': '💨'
    },
    'semi_lagrangian': {
        'name': 'Semi-Lagrangian (Hybrid)',
        'class': SemiLagrangianModel,
        'description': 'Backward trajectory on grid - reduces numerical diffusion',
        'icon': '↩️'
    },
    'hybrid': {
        'name': 'Hybrid (Lagrangian + Eulerian)',
        'class': HybridModel,
        'description': 'Particles near source, grid far away - best accuracy',
        'icon': '🔀'
    }
}
