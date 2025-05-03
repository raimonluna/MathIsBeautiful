import numpy as np
import cupy  as cp
import matplotlib.pyplot as plt
import matplotlib.animation as animation

shape        = (1000, 1000)
max_radius   = 1000000
total_iters  = 100
xrange       = (-1.6, 1.6)
yrange       = (-1.6, 1.6)
frames       = 400

angles = np.linspace(0, 2* cp.pi, frames + 1)[:-1]
rez    = cp.linspace(*xrange, shape[0])[:, None]
imz    = cp.linspace(*yrange, shape[1])[None, :]

fig, ax = plt.subplots(frameon = False, figsize = (8, 8))
fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
ax.axis('off')
plt.close()

julia = total_iters * cp.ones(shape)
ish   = ax.imshow(cp.asnumpy(julia).T, cmap = 'jet', norm = 'log', origin = 'lower', extent = xrange + yrange)
            
def update(frame):
    z   = rez + 1j * imz
    c   = 0.7885 * cp.exp( angles[frame] * 1j)
    
    julia = total_iters * cp.ones(shape)
    
    for i in range(total_iters):
        z = z**2 + c
        diverged = (cp.abs(z) > max_radius) & (julia == total_iters)
        z[diverged] = 0
        julia[diverged] = i
        
    ish.set_data(cp.asnumpy(julia).T)
    ish.set_clim(vmin = 4, vmax = 20 + 80 * cp.sin(0.5 * angles[frame])**2)

animation_fig = animation.FuncAnimation(fig, update, frames = frames, interval = 50)
animation_fig.save("MIB0035_JuliaSet.mp4", dpi = 200)

