import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.animation as animation
np.random.seed(42)

frames = 600

def perlin_noise(cells = (5,5), npix = 32):
    xy = np.mgrid[0:cells[0]:1/npix, 0:cells[1]:1/npix].transpose(1, 2, 0) % 1
    sm = 6*xy**5 - 15*xy**4 + 10*xy**3 # Smoothstep function
    
    angles = 2 * np.pi * np.random.rand(*cells)
    grads  = np.dstack([np.cos(angles), np.sin(angles)])
    cshape = np.ones((npix, npix, 1))
    
    perlin = 0
    for di, dj in [(di, dj) for di in (0, 1) for dj in (0, 1)]:
        dot_product = np.kron(np.roll(grads, (-di, -dj), axis = (0, 1)), cshape) * (xy - [di, dj])
        smoothing   = sm + (1 - 2 * sm) * [1 - di, 1 - dj]
        perlin     += np.sqrt(2) * np.prod(smoothing, axis = 2) * np.sum(dot_product, axis = 2)

    return perlin

def fractal_noise(cells = (5,5), npix = 32, octaves = 5):
    return np.sum([perlin_noise((2**n * cells[0], 2**n * cells[1]), npix // 2**n) / 2**n for n in range(octaves)], axis = 0)

times = np.linspace(0, 4 * np.pi, frames)

fig, ax = plt.subplots(1,1, figsize = (6, 6))
fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
ax.axis('off')
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
plt.close()

cmap  = mpl.colors.LinearSegmentedColormap.from_list("", [(0.0, 'black'), (0.5, 'black'), (1, 'darkblue')])
board = fractal_noise(cells = (5, 5), npix = 160, octaves = 5)
ax.imshow(board.T, cmap = cmap, origin = 'lower', vmin = -2, vmax = 1, extent = (-1,+1,-1,+1))

vguides   = [ ax.plot([2, 2], [-1,1], ls = ':', lw = 0.5, color = 'blue')[0] for i in range(8)]
hguides   = [ ax.plot([-1,1], [2, 2], ls = ':', lw = 0.5, color = 'blue')[0] for j in range(8)]
lissajous = [[ax.plot([2, 2], [2, 2], ls = '-', lw = 1, color = ('blue', 'greenyellow')[min(i*j, 1)])[0] for i in range(8)] for j in range(8)]
plot_dots = [[ax.scatter([2, 2], [2, 2], s = 10, color = ('blue', 'greenyellow')[min(i*j, 1)], alpha = 1.0) for i in range(8)] for j in range(8)]
aura_dots = [[ax.scatter([2, 2], [2, 2], s = 75, color = ('blue', 'greenyellow')[min(i*j, 1)], alpha = 0.3) for i in range(8)] for j in range(8)]
plot_dots[0][0].set_visible(False)
aura_dots[0][0].set_visible(False)

def update(frame):
    t, f = times[:frame], times[frame]
    for i in range(8):
        if i > 0:
            vguides[i].set_xdata(2 * [- 0.875 + 0.25 * i + 0.1 * np.cos(i * f)])
            hguides[i].set_ydata(2 * [+ 0.875 - 0.25 * i + 0.1 * np.sin(i * f)])
        for j in range(8):
            if i*j == 0:
                k = max(i,j)
                lissajous[i][j].set_data(     - 0.875 + 0.25 * i + 0.1 * np.cos(k * t), + 0.875 - 0.25 * j + 0.1 * np.sin(k * t))
                plot_dots[i][j].set_offsets([[- 0.875 + 0.25 * i + 0.1 * np.cos(k * f), + 0.875 - 0.25 * j + 0.1 * np.sin(k * f)]])
                aura_dots[i][j].set_offsets([[- 0.875 + 0.25 * i + 0.1 * np.cos(k * f), + 0.875 - 0.25 * j + 0.1 * np.sin(k * f)]])
            else:
                lissajous[i][j].set_data(     - 0.875 + 0.25 * i + 0.1 * np.cos(i * t), + 0.875 - 0.25 * j + 0.1 * np.sin(j * t))
                plot_dots[i][j].set_offsets([[- 0.875 + 0.25 * i + 0.1 * np.cos(i * f), + 0.875 - 0.25 * j + 0.1 * np.sin(j * f)]])
                aura_dots[i][j].set_offsets([[- 0.875 + 0.25 * i + 0.1 * np.cos(i * f), + 0.875 - 0.25 * j + 0.1 * np.sin(j * f)]])
    
animation_fig = animation.FuncAnimation(fig, update, frames = frames, interval = 50)
animation_fig.save("MIB0069_LissajousCurves.mp4", dpi = 200)
