import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

shape       = (2000, 2000)
total_iters = 2000
wait_iters  = 800
xrange      = (2.9, 4.0)
yrange      = (-0.05, 1.05)

cmap  = mpl.colors.LinearSegmentedColormap.from_list("", [(0.0, 'wheat'), (0.1, 'orange'), (0.15, 'saddlebrown'), (0.3, 'black'), (1, 'black')])

r, x = np.meshgrid(np.linspace(*xrange, shape[0]), np.linspace(*yrange, shape[1]), indexing='ij')
xn = 0.5 * np.ones(shape[0]) 
bifurcation = np.ones(shape)

for it in range(total_iters):
    xn = r[:, 0] * xn * (1 - xn)
    if it > wait_iters:
        idx2 = np.int32(shape[1] * (xn - yrange[0]) / (yrange[1] - yrange[0]))
        bifurcation[range(shape[0]), idx2] += 1

plt.axis('off')
plt.imshow(bifurcation.T, cmap = cmap, norm = 'log', origin = 'lower', extent = xrange + yrange)
plt.savefig('MIB0006_LogisticMap.png', dpi = 1000, bbox_inches='tight', pad_inches = 0)
