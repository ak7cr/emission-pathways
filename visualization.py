"""
Visualization and plotting functions
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
import io
import base64

def generate_frame(particles, particle_active, t, concentration_func, wind_type, wind_data_cache,
                  show_wind_vectors, hotspots, x, y, nx, ny, xmin, xmax, ymin, ymax):
    """
    Generate a single frame and return as base64 encoded PNG
    
    Parameters:
    -----------
    particles : ndarray
        Particle array
    particle_active : ndarray
        Boolean array of active particles
    t : float
        Current time in seconds
    concentration_func : function
        Function to calculate concentration fields
    wind_type : str
        'synthetic' or 'real'
    wind_data_cache : dict
        Wind data cache
    show_wind_vectors : bool
        Show wind vectors on plot
    hotspots : list
        List of hotspot positions
    x, y : ndarray
        Grid coordinates
    nx, ny : int
        Grid dimensions
    xmin, xmax, ymin, ymax : float
        Domain boundaries
    
    Returns:
    --------
    img_base64 : str
        Base64 encoded PNG image
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Get both normalized and physical concentrations
    H_normalized, H_physical = concentration_func()
    
    # Get max physical concentration for display
    max_conc = H_physical.max()
    
    # Generic title (works for all models)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_title(f"Atmospheric Dispersion - t = {t:.1f} s | Max: {max_conc:.2f} µg/m³", fontsize=14)
    ax.set_xlabel("X (km)")
    ax.set_ylabel("Y (km)")
    
    # Use normalized concentration for visualization
    im = ax.imshow(H_normalized, origin='lower', extent=(xmin, xmax, ymin, ymax), 
                   alpha=0.7, vmin=0, vmax=1, cmap=cm.inferno)
    
    # Plot wind vectors if enabled
    if show_wind_vectors:
        # Get wind field at current time
        if wind_type == 'real':
            from wind_field import real_wind_field
            U, V = real_wind_field(t, x, y, nx, ny, wind_data_cache)
        else:
            from wind_field import synthetic_wind_field
            U, V = synthetic_wind_field(t, x, y, nx, ny)
        
        # Create wind vector field (subsample for clarity)
        skip = 8  # Show every 8th arrow
        X_grid, Y_grid = np.meshgrid(x[::skip], y[::skip])
        U_sub = U[::skip, ::skip]
        V_sub = V[::skip, ::skip]
        
        # Plot wind vectors
        quiver = ax.quiver(X_grid, Y_grid, U_sub, V_sub,
                           color='white', alpha=0.6, scale=50, width=0.003,
                           headwidth=4, headlength=5, linewidths=0.5)
        
        # Add quiver key (scale reference)
        ax.quiverkey(quiver, 0.9, 0.95, 5, '5 m/s', labelpos='E',
                     coordinates='axes', color='white', labelcolor='white',
                     fontproperties={'size': 10})
    
    # Sample particles for display (only active particles, if using particle-based model)
    if particles is not None and particle_active is not None:
        particles_active = particles[particle_active]
        
        if len(particles_active) > 0:
            n_display = min(1500, len(particles_active))
            sample_idx = np.random.choice(len(particles_active), size=n_display, replace=False)
            sample = particles_active[sample_idx]
            ax.scatter(sample[:, 0], sample[:, 1], s=1, alpha=0.5, c='white')
    
    # Plot hotspots
    for hx, hy in hotspots:
        ax.plot(hx, hy, 'wo', markersize=10, markeredgecolor='k', markeredgewidth=2)
    
    plt.colorbar(im, ax=ax, label='Normalized Concentration')
    
    # Save to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    
    # Encode as base64
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    return img_base64
