
import numpy as np
from skimage import measure
import matplotlib.pyplot as plt
import matplotlib.animation as animation

R, r, dr, w = 1, 0.15, 0.08, 8
dx, dy, dz  = 0.05, 0.05, 0.05
r2 = np.sqrt(2)/2
iters = 6

Rx = np.array([[1,0,0],[0,0,-1],[0,1,0]])
Ry = np.array([[0,0,1],[0,1,0],[-1,0,0]])
RL = Ry @ Rx @ Rx @ Rx

def croissant(x, y, z):
    return (np.sqrt(y**2 + z**2) - R)**2 + x**2 - (r - dr * z)**2 + np.exp( - w * (x**2 + y**2 +(z - R)**2) )
    
F = croissant(*np.mgrid[-1.5:1.5:dx, -1.5:1.5:dy, -1.5:1.5:dz])
verts, faces, normals, values = measure.marching_cubes(F, 0, spacing = [dx, dy, dz])
verts -= 1.5

for i in range(iters):
    vertsR = (0.55 * Rx @ verts.T + r2 * np.array([[0],[+0.21],[1]])).T    
    vertsL = (0.55 * RL @ verts.T + r2 * np.array([[0],[-0.21],[1]])).T
    faces  = np.vstack([faces, faces + verts.shape[0], faces + 2 * verts.shape[0]])
    verts  = np.vstack([verts, vertsR, vertsL])
    
##### Do the plotting
fig  = plt.figure(figsize = (8,8))
ax   = fig.add_subplot(projection='3d')

ax.axis('off')
ax.set_xlim(-0.45, 0.45)
ax.set_ylim(-0.45, 0.45)
ax.set_zlim(0.4, 1.05)
ax.view_init(elev = 20, azim = 160)
ax.set_facecolor("black")
fig.set_facecolor("black")

ax.plot_trisurf(*verts.T, triangles = faces, cmap = 'Wistia', edgecolor = 'black', linewidth = 0.05, antialiased = True, vmin = 0.4, vmax = 0.95)
plt.savefig('MIB0054_HornedSphere.png', dpi = 1000, bbox_inches='tight', pad_inches = 0)
