import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

frames = 200
side   = 30
bgnd   = 'wheat'
cmap   = plt.get_cmap('RdPu')

a, b = np.mgrid[-4:5, -4:5]
u, v = np.mgrid[0:2*np.pi:side*1j, 0:4*np.pi:side*3j]
x    = np.select([v < 2 * np.pi, v < 3 * np.pi], [(2.5 - 1.5 * np.cos(v)) * np.cos(u), -2 + (2 + np.cos(u)) * np.cos(v)], - 2 + 2 * np.cos(v) - np.cos(u))
y    = np.select([v < 2 * np.pi], [(2.5 - 1.5 * np.cos(v)) * np.sin(u)], np.sin(u))
z    = np.select([v < np.pi, v < 2 * np.pi, v < 3 * np.pi], [-2.5 * np.sin(v), 3 * v - 3 * np.pi, (2 + np.cos(u)) * np.sin(v) + 3 * np.pi ], -3 * v + 12 * np.pi)

fig  = plt.figure(figsize = (6, 6))
ax   = fig.add_subplot(projection='3d', computed_zorder=False)
fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
#plt.close()

facecolors = cmap( 0.5 + np.sin( 2*v )**2 / 2  )[:-1, :-1, :]

bd = ax.plot_surface(2 * a, 2 * b, 0 * a, facecolors = np.choose((a + b + 1) % 2, ['w', 'k']), lw = 0.01)
sf = ax.plot_surface(x, y, z + 2.5, facecolors = facecolors, edgecolor='k', lw = 0.1, alpha=0.75, rstride = 1, cstride = 1, shade = False, zorder = +100)
ax.scatter([0], [0], [0], s = 10, color = 'r')
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
ax.set_zlim(3.0, 12.0)
ax.set_facecolor(bgnd)
fig.set_facecolor(bgnd)
ax.axis('off')

def update(i):
    facecolors = cmap( 0.5 + np.sin( 2 * v + 4 * np.pi * i / frames )**2 / 2 )
    sf.set_facecolors(facecolors[:-1, :-1, :].reshape(-1, 4))
    ax.view_init(elev = 20, azim = 225 - 360 * i / frames)
    
animation_fig = animation.FuncAnimation(fig, update, frames = frames, interval = 50)
animation_fig.save("MIB0065_KleinBottle.mp4", dpi = 200)
