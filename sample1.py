import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import imageio
import os

np.random.seed(0)

xmin, xmax, ymin, ymax = 0.0, 200.0, 0.0, 200.0
nx, ny = 120, 120
x = np.linspace(xmin, xmax, nx)
y = np.linspace(ymin, ymax, ny)

def synthetic_wind_field(t):
    uu = 3.0 + 1.0 * np.sin(2*np.pi*(y/200.0 + 0.06*t))
    vv = 0.3 * np.cos(2*np.pi*(x/200.0 - 0.03*t))
    U = np.tile(uu, (nx,1)).T
    V = np.tile(vv, (ny,1))
    return U, V

hotspots = np.array([[40.0, 120.0],[60.0, 130.0],[30.0, 90.0]])

npph = 2500
n_particles = npph * len(hotspots)

particles = np.zeros((n_particles,2))
for i,h in enumerate(hotspots):
    a = i*npph
    b = (i+1)*npph
    particles[a:b,0] = h[0] + np.random.normal(scale=2.0, size=npph)
    particles[a:b,1] = h[1] + np.random.normal(scale=2.0, size=npph)

dt = 0.2
nt = 240
sigma_turb = 0.9

def advect(p, t, dt):
    U, V = synthetic_wind_field(t)
    u = np.interp(p[:,0], x, U.mean(axis=0))
    v = np.interp(p[:,1], y, V.mean(axis=1))
    p[:,0] += u*dt + np.sqrt(2*sigma_turb*dt)*np.random.randn(len(p))
    p[:,1] += v*dt + np.sqrt(2*sigma_turb*dt)*np.random.randn(len(p))
    p[:,0] = np.clip(p[:,0], xmin, xmax)
    p[:,1] = np.clip(p[:,1], ymin, ymax)
    return p

def concentration_field(p):
    H, xe, ye = np.histogram2d(p[:,0], p[:,1], bins=[nx,ny], range=[[xmin,xmax],[ymin,ymax]])
    H = H.T
    H = H / (H.max() + 1e-9)
    return H

outdir = "frames_demo"
os.makedirs(outdir, exist_ok=True)
frames = []
fig = plt.figure(figsize=(6,6))
for ti in range(nt):
    t = ti*dt
    particles = advect(particles, t, dt)
    H = concentration_field(particles)
    plt.clf()
    ax = plt.gca()
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_title(f"t = {t:.1f} s")
    im = ax.imshow(H, origin='lower', extent=(xmin,xmax,ymin,ymax), alpha=0.7, vmin=0, vmax=1, cmap=cm.inferno)
    sample = particles[np.random.choice(len(particles), size=1500, replace=False)]
    ax.scatter(sample[:,0], sample[:,1], s=1, alpha=0.5)
    for hx, hy in hotspots:
        ax.plot(hx, hy, 'wo', markersize=6, markeredgecolor='k')
    fname = os.path.join(outdir, f"frame_{ti:04d}.png")
    fig.savefig(fname, dpi=100, bbox_inches='tight')
    frames.append(fname)

gif_path = "lagrangian_demo.gif"
with imageio.get_writer(gif_path, mode='I', duration=0.08) as writer:
    for fname in frames:
        image = imageio.imread(fname)
        writer.append_data(image)

print("Saved animation to", gif_path)
