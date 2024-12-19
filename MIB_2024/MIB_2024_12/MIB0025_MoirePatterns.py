import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib as mpl

shape  = (2000, 2000)
xrange = (-200, 200)
yrange = (-200, 200)
frames = 100

x = np.linspace(*xrange, shape[0])[:, None]
y = np.linspace(*yrange, shape[1])[None, :]

cmap1 = mpl.colors.LinearSegmentedColormap.from_list("", [(0.0, 'black'), (1, 'black')])
cmap2 = mpl.colors.LinearSegmentedColormap.from_list("", [(0.0, 'blue'), (1, 'blue')])

def grid(theta, factor = 0):
    r = np.sqrt(x**2 + y**2)
    u = (x * np.cos(theta) - y * np.sin(theta)) * (1 + factor * r)
    v = (x * np.sin(theta) + y * np.cos(theta)) * (1 + factor * r)
    f = np.cos(u) + np.cos(u/2 + np.sqrt(3)*v/2) + np.cos(u/2 - np.sqrt(3)*v/2)
    g = np.ones_like(f)
    g[f > 2] = np.nan
    return g

fig, ax = plt.subplots(1,1, figsize = (8, 8))
fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
plt.close()

ax.axis('off')
ax.imshow(grid(0, 0).T, cmap = cmap2, extent = xrange + yrange)
gr = ax.imshow(grid(0, 0.0001).T, cmap = cmap1, extent = xrange + yrange)

angles = np.linspace(0, 0.05, frames) * np.pi
angles = np.hstack([angles, angles[::-1]])

animation_fig = animation.FuncAnimation(fig, lambda i: gr.set_data(grid(angles[i], 0.0001 ).T), 
                                        frames = len(angles), interval = 50)
animation_fig.save("MIB0025_MoirePatterns.mp4", dpi = 200)
