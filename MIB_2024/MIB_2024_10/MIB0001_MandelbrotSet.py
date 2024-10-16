import numpy as np
import matplotlib.pyplot as plt

shape        = (2000, 2000)
max_radius   = 10
total_iters  = 100
xrange       = (-2.1, 0.6)
yrange       = (-1.35, 1.35)

rec, imc = np.meshgrid(np.linspace(*xrange, shape[0]), np.linspace(*yrange, shape[1]), indexing='ij')
c   = rec + 1j * imc
z   = 0 * c
mandelbrot = total_iters * np.ones(shape)

for i in range(total_iters):
    z = z**2 + c
    diverged = (np.abs(z) > max_radius) & (mandelbrot == total_iters)
    z[diverged] = c[diverged] = 0
    mandelbrot[diverged] = i

plt.axis('off')
plt.imshow(mandelbrot.T, cmap = 'inferno', norm = 'log', extent = xrange + yrange)
plt.savefig('MIB0001_MandelbrotSet.png', dpi = 1000, bbox_inches='tight', pad_inches = 0)
