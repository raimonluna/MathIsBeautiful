import numpy as np
from scipy.ndimage import convolve
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.animation as animation

shape     = 150
figsize   = 8
frames    = 400

alpha = 1.0
beta  = 0.5
gamma = 0.01

HEX_KERNEL = np.ones((3,3)) - np.eye(3)[::-1]

r, q = np.indices((shape, shape))
x    = q - r / 2.0 
y    = r * (np.sqrt(3) / 2.0)
x   -= x[shape//2, shape//2]
y   -= y[shape//2, shape//2]

s = beta * np.ones((shape, shape))
s[shape//2, shape//2] = 1

fig, ax = plt.subplots(figsize = (figsize, figsize))
fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
fig.set_facecolor("black")
ax.set_xlim(-shape/3.15, shape/3.15)
ax.set_ylim(-shape/3.15, shape/3.15)
plt.close()
ax.axis('off')

blues_colors = ['#f7fbff', '#deebf7', '#c6dbef', '#9ecae1', '#6baed6', 
                '#4292c6', '#2171b5', '#08519c', '#000000'][::-1]
cmap = mpl.colors.LinearSegmentedColormap.from_list('custom_Blues', blues_colors, N=256)

flake = ax.scatter(x, y, c = s.ravel(), cmap = cmap, marker = 'h', s = 1.2e4 * (figsize / shape)**2 )

def update(i):
    global s
    if i < frames:
        frozen    = s >= 1
        neighbors = convolve(frozen, HEX_KERNEL, mode = 'constant', cval = 0)
        receptive = frozen | (neighbors > 0)
         
        # Diffusing part of vapor: non-receptive cells.
        u = np.where(receptive, 0, s)
        avg_u = convolve(u, HEX_KERNEL, mode='constant', cval=beta) / 6
        u += alpha * (avg_u - u) / 2
        
        # Non-diffusing part of water: receptive cells.
        v = np.where(receptive, s, 0)
        v[receptive] += gamma
        
        s = u + v
        
        flake.set_array((s * frozen).ravel())
        flake.autoscale()
    
animation_fig = animation.FuncAnimation(fig, update, frames = frames + 100, interval = 100)
animation_fig.save("MIB0070_ReiterSnowflake.mp4", dpi = 200)
