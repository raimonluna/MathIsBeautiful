import numpy as np
import matplotlib.pyplot as plt

shape        = (2000, 2000)
max_radius   = 1000
total_iters  = 100
epsilon      = 1e-10
xrange       = (-1.0, 1.0)
yrange       = (-1.05, 0.95)

rec = np.linspace(*xrange, shape[0])[:, None]
imc = np.linspace(*yrange, shape[1])[None, :]
c   = rec + 1j * imc
z   = 0 * c
mandelbrot = total_iters * np.ones(shape)

for i in range(total_iters):
    z = np.cos(z) - 1j / c
    diverged = (np.abs(z) > max_radius) & (mandelbrot == total_iters)
    z[diverged] = c[diverged] = 0
    mandelbrot[diverged] = i

plt.axis('off')
plt.imshow(mandelbrot.T, cmap = 'hot_r', norm = 'log', origin = 'lower', extent = xrange + yrange)
plt.savefig('MIB0026_BalrogFractal.png', dpi = 2000, bbox_inches='tight', pad_inches = 0)
