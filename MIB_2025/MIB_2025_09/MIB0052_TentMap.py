import numpy as np
import matplotlib.pyplot as plt

shape       = (2000, 2000)
total_iters = 10000
wait_iters  = 800
xrange      = (1.0, 2.0)
yrange      = (0.0, 1.0)

mu = np.linspace(*xrange, shape[0])
xn = 0.5 * np.ones(shape[0]) 
bifurcation = np.ones(shape)

for it in range(total_iters):
    xn = mu * np.minimum(xn, 1 - xn)
    if it > wait_iters:
        idx2 = np.int32(shape[1] * (xn - yrange[0]) / (yrange[1] - yrange[0]))
        bifurcation[range(shape[0]), idx2] += 1

plt.axis('off')
plt.imshow(bifurcation[::-1,:], cmap = 'Blues_r', norm = 'log', origin = 'lower', extent = xrange + yrange, vmin = 1, vmax = 50)
plt.savefig('MIB0052_TentMap.png', dpi = 1000, bbox_inches = 'tight', pad_inches = 0)
