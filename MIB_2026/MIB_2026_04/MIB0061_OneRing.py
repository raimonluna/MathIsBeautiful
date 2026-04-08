import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from tqdm.auto import tqdm

shape       = (1000, 1000)
batch       = 100000
total_iters = 1000
xrange      = (-11, 11)
yrange      = (-11, 11)
a, b, c, d  = 1.40, 1.56, 1.40, -6.56

board  = 0.1 * np.ones(shape)
cmap   = mpl.colors.LinearSegmentedColormap.from_list("", [(0.0, 'black'), (0.6, 'gold'), 
                                                           (0.8, 'darkorange'), (1, 'saddlebrown')])
x, y = np.random.rand(2, batch)
for i in tqdm(range(total_iters)):
    x, y   = d * np.sin(a * x) - np.sin(b * y), c * np.cos(a * x) + np.cos(b * y)
    idx1   = np.int32( shape[0] * (x + 2*y - xrange[0]) / (xrange[1] - xrange[0]) )
    idx2   = np.int32( shape[1] * (2*y - x - yrange[0]) / (yrange[1] - yrange[0]) )
    board[idx1, idx2] += 1
        
plt.axis('off')
plt.imshow(board.T, cmap = cmap, norm = 'log', origin = 'lower', extent = xrange + yrange)
plt.savefig('MIB0061_OneRing.png', dpi = 1000, bbox_inches='tight', pad_inches = 0)
