import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.animation as animation

frames = 800
np.random.seed(42)

colors     = list(mpl.colormaps['summer'](np.linspace(0, 1, 4)))
voxels     = np.zeros((6, 5, 4), dtype = bool)
facecolors = np.empty(voxels.shape, dtype = object)

voxels[:-1, 0, 0]  = True
voxels[0, :, 0]    = True
voxels[-2, 0, :-1] = True
voxels[-1, 0, -1]  = True

facecolors[0, 1:, 0]   = colors[::-1]
facecolors[:-2, 0, 0]  = colors
facecolors[-2, 0, :-1] = colors[:-1]
facecolors[-1, 0, -1]  = colors[-1]

x, y, z   = np.mgrid[0:7:1, 0:6:1, 0:5:1].astype(float)
xs, ys    = - 15 + 30 * np.random.rand(2, 2000)
x[-1,:,:] = 5.0001

fig  = plt.figure(figsize = (6, 6))
fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
ax  = fig.add_subplot(projection='3d')
ax.axis('off')
plt.close()

ax.scatter(xs, ys, color = 'white', s = np.random.rand(2000))
ax.voxels(x, y, z, voxels, facecolors = facecolors, edgecolors = 'k', alpha = 0.9)

ax.set_xlim(-1.5, 5)
ax.set_ylim(-1.5, 5)
ax.set_zlim(-1, 3.4)
ax.set_proj_type('ortho') 
ax.set_facecolor('black')
fig.set_facecolor('black')

def update(i):
    p  = (i%200 - 50) * (i%200 > 50) / (200 - 51)
    sp = 6*p**5 - 15*p**4 + 10*p**3
    ax.view_init(38.1, -45 - 360 * sp)

animation_fig = animation.FuncAnimation(fig, update, frames = frames, interval = 50)
animation_fig.save("MIB0049_PenroseTriangle.mp4", dpi = 200)
