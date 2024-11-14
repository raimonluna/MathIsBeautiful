import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

shape        = (2000, 2000)
max_radius   = 1000000
total_iters  = 100
xrange       = (-1.8, -1.7)
yrange       = (-0.015, 0.085)

rec = np.linspace(*xrange, shape[0])[:, None]
imc = np.linspace(*yrange, shape[1])[None, :]
c   = rec + 1j * imc
z   = 0 * c
burningship = total_iters * np.ones(shape)

for i in range(total_iters):
    z = (np.abs(z.real) - 1j * np.abs(z.imag))**2 + c
    diverged = (np.abs(z) > max_radius) & (burningship == total_iters)
    z[diverged] = c[diverged] = 0
    burningship[diverged] = i

cmap = mpl.colors.LinearSegmentedColormap.from_list("", [(0.0, 'navy'), (0.3, 'darkgreen'),  
                                                         (1-1e-9, 'lime'),(1, 'black')])
                                                         
plt.axis('off')
plt.imshow(burningship.T, cmap = cmap, norm = 'log', origin = 'lower', extent = xrange + yrange)
plt.savefig('MIB0015_BurningShip.png', dpi = 1000, bbox_inches='tight', pad_inches = 0)
