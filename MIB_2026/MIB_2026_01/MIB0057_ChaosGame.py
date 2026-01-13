import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import matplotlib as mpl

shape       = (2000, 2000)
vertices    = 5
total_iters = int(5e6)
xrange      = (-0.9, 0.9)
yrange      = (-0.9, 0.9)

board  = np.zeros(shape)
coords = vertex = 0

for i in tqdm(range(total_iters)):
    vertex = (vertex + np.random.randint(1, vertices)) % vertices
    coords = (coords + 1j * np.exp(2j * np.pi * vertex / vertices) ) / 2
    idx1   = np.int32( shape[0] * (coords.real - xrange[0]) / (xrange[1] - xrange[0]) )
    idx2   = np.int32( shape[1] * (coords.imag - yrange[0]) / (yrange[1] - yrange[0]) )
    board[idx1, idx2] += 1

plt.axis('off')
plt.imshow(board.T, cmap = 'Blues_r', origin = 'lower', interpolation = 'bilinear', extent = xrange + yrange, vmin = 0, vmax = 10)
plt.savefig('MIB0057_ChaosGame.png', dpi = 1000, bbox_inches = 'tight', pad_inches = 0)
