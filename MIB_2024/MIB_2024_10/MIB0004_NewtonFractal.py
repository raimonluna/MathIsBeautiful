import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

shape        = (2000, 2000)
total_iters  = 100
xrange       = (-2, 2)
yrange       = (-2, 2)

rec, imc = np.meshgrid(np.linspace(*xrange, shape[0]), np.linspace(*yrange, shape[1]), indexing='ij')
z   = rec + 1j * imc

iters = - np.ones(shape)
canvas   = np.ones(shape + (3,))

for i in range(total_iters):
    change = (z**3 - 1) / (3 * z**2)
    iters[(np.abs(change) < 1e-12) & (iters == -1)] = i
    z = z - change

iters  = np.log(iters)
iters -= np.min(iters)
iters /= np.max(iters)

canvas[..., 0] = (np.angle(-z) + np.pi) / (2 * np.pi)
canvas[..., 2] = 1 - iters
canvas = mpl.colors.hsv_to_rgb(canvas)

plt.axis('off')
plt.imshow(canvas.transpose(1,0,2), origin = 'lower', extent = xrange + yrange)
plt.savefig('MIB0004_NewtonFractal.png', dpi = 1000, bbox_inches='tight', pad_inches = 0)
