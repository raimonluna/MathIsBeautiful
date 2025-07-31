import numpy as np
from skimage import measure
import matplotlib.pyplot as plt
import matplotlib.animation as animation

R          = 7
dr, dt, dp = 0.05, 0.02, 0.02
frames     = 500

def sph2cart(r, t, p):
    x = R * r * np.sin(np.pi * t) * np.cos(2 * np.pi * p)
    y = R * r * np.sin(np.pi * t) * np.sin(2 * np.pi * p)
    z = R * r * np.cos(np.pi * t)
    return x, y, z

def gyroid(r, t, p):
    x, y, z = sph2cart(r, t, p)
    return np.sin(x) * np.cos(y) + np.sin(y) * np.cos(z) + np.sin(z) * np.cos(x)
    
F = gyroid(*np.mgrid[0:1:dr, 0:1:dt, 0:(1 + dp):dp])
verts, faces, normals, values = measure.marching_cubes(F, 0, spacing = [dr, dt, dp])
new_verts = np.vstack(sph2cart(*verts.T)).T

##### Do the plotting
fig  = plt.figure(figsize = (6,6))
fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
ax   = fig.add_subplot(projection='3d')
plt.close()

ax.plot_trisurf(new_verts[:, 0], new_verts[:, 1], faces, new_verts[:, 2], cmap='ocean', alpha = 0.6)

ax.axis('off')
ax.set_xlim(-0.7 * R, 0.7 * R)
ax.set_ylim(-0.7 * R, 0.7 * R)
ax.set_zlim(-0.5 * R, 0.5 * R)
ax.set_facecolor("black")
fig.set_facecolor("black")
    
animation_fig = animation.FuncAnimation(fig, lambda i: ax.view_init(elev = 20, azim = - 360 * i / frames), frames = frames, interval = 50)
animation_fig.save("MIB0048_Gyroid.mp4", dpi = 200)
