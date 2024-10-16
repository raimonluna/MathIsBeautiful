import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

shape       = (1000, 1000)
total_iters = 5000000
xrange      = (-5.25, 5.25)
yrange      = (0, 10.5)

board  = 0.001 * np.ones(shape)
coords = np.zeros((2, 1))
x, y   = np.meshgrid(np.linspace(*xrange, shape[0]), np.linspace(*yrange, shape[1]), indexing='ij')
cmap   = mpl.colors.LinearSegmentedColormap.from_list("", [(0.0, 'lightyellow'), (0.2, 'lime'), (0.85, 'black'), (1, 'black')])

M = np.asarray([[0,	0,	0,	0.16,	0,	0,	0.01],
[0.85,	0.04,	-0.04,	0.85,	0,	1.60,	0.85],
[0.20,	-0.26,	0.23,	0.22,	0,	1.60,	0.07],
[-0.15,	0.28,	0.26,	0.24,	0,	0.44,	0.07]])

for i in range(total_iters):
    r = np.random.rand()
    c = 0.0
    for j in range(4):
        c += M[j, 6]
        if r < c:
            coords  = M[j, :4].reshape(2, 2) @ coords + M[j, 4:6].reshape(2, 1)
            idx1 = np.int32( shape[0] * (coords[0, 0] - xrange[0]) / (xrange[1] - xrange[0]) )
            idx2 = np.int32( shape[1] * (coords[1, 0] - yrange[0]) / (yrange[1] - yrange[0]) )
            board[idx1, idx2] += 1
            break
        
plt.axis('off')
plt.imshow(board.T, cmap = cmap, norm = 'log', origin = 'lower', extent = xrange + yrange)
plt.savefig('MIB0002_BarnsleyFern.png', dpi = 1000, bbox_inches='tight', pad_inches = 0)
