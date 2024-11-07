import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

shape  = (800, 800)
cells  = (5, 5)
xrange = (-0.8, 0.8)
yrange = (-0.8, 0.8)
np.random.seed(42)

##### Perlin Noise

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

##### Dragon Curve

points = np.array([[0, 1],[0, 0]])
rotate = np.array([[0,-1],[1, 0]])

for i in range(18):
    newpoints = points.copy()
    pivot     = newpoints[:, [-1]]
    newpoints = rotate @ (newpoints - pivot) + pivot
    points    = np.hstack([points, newpoints[:, -2::-1]]) / np.sqrt(2)
    points   -= np.mean(points, axis = 1)[:, None]

board = 7e-2 * fractal_noise(cells = cells, npix = shape[0] // cells[0], octaves = 5)
idx1  = np.int32( shape[0] * (points[0, :] - xrange[0]) / (xrange[1] - xrange[0]) )
idx2  = np.int32( shape[1] * (points[1, :] - yrange[0]) / (yrange[1] - yrange[0]) )
board[idx1, idx2] = 1.5 - np.arange(points.shape[1]) / points.shape[1] 

##### Plotting

cmap = mpl.colors.LinearSegmentedColormap.from_list("", [(0.0, 'black'), (0.1, 'gray'), 
                                                         (0.2, 'saddlebrown'), (0.3, 'crimson'), 
                                                         (0.8, 'gold'), (1, 'forestgreen')])

plt.axis('off')
plt.imshow(board.T, cmap = cmap, origin = 'lower', extent = xrange + yrange)
plt.savefig('MIB0014_DragonCurve.png', dpi = 1000, bbox_inches='tight', pad_inches = 0)
