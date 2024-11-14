import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

side = 81 # Make it odd!

###### Prepare the checkerboard ######
cmap_board  = mpl.colors.LinearSegmentedColormap.from_list("", [(0.0, 'mediumblue'), (1, 'darkblue')])
cmap_knight = mpl.colors.LinearSegmentedColormap.from_list("", [(0.0, 'white'), (0.05, 'saddlebrown'), (0.15, 'saddlebrown'),
                                                                (0.25, 'darkgreen'), (0.6, 'green'), (1, 'orange')])
x = np.arange(side)[:, None]
board = (x + x.T) % 2

###### Prepare the number spiral ######
spiral = np.zeros((side, side), dtype = np.int64)
i, j   = side//2, side//2
di, dj = 1, 0
spiral[i, j] = side**2 + 1

for k in range(2, side**2 + 1):
    i, j = i + di, j + dj
    spiral[i, j] = k
    if spiral[i - dj, j + di] == 0:
        di, dj = - dj, di

###### Make the knight jumps ######
i, j   = side//2, side//2
n = 0

while True:
    n += 1
    knight_jumps = [(di, dj) for di in range(-2,3) for dj in range(-2,3) if (abs(di) != abs(dj)) & (di*dj != 0)]
    min_indices  = np.argmin([spiral[i + di, j + dj] for di, dj in knight_jumps])
    di, dj       = knight_jumps[min_indices]
    if spiral[i + di, j + dj] == side**2 + 1:
        break
    plt.plot([i + 0.1*di, i + 0.9*di], [j+0.1*dj, j + 0.9*dj], color = 'k', linewidth = 2.25, solid_capstyle='round')
    plt.plot([i, i + di], [j, j + dj], color = cmap_knight(256 * n // 2015), linewidth = 1.5, solid_capstyle='round')
    spiral[i + di, j + dj] = side**2 + 1
    i, j = i + di, j + dj

plt.xlim(10.5, 69.5)
plt.ylim(10.5, 69.5)
plt.axis('off')
plt.plot([i], [j], 'x', color = 'red')
plt.imshow(board, cmap = cmap_board, origin = 'lower', extent = (-0.5, side - 0.5, -0.5, side - 0.5))
plt.savefig('MIB0007_TrappedKnight.png', dpi = 1000, bbox_inches='tight', pad_inches = 0)
