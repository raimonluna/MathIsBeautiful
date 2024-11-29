import numpy as np
import cupy as cp
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.animation as animation

side        = 500
walkers     = 50000
total_iters = 20000
save_every  = 50

cp.random.seed(42)
frames = (total_iters - 1)// save_every + 1
stored = cp.zeros((side, side, frames))

board = cp.zeros((side, side)) 
board[side//2, side//2] = 1 # Place the seed

i, j = cp.random.randint(0, side, size = (2, walkers))
mask = cp.ones(walkers, dtype = cp.int32)

###### Make the random walks

for it in tqdm(range(total_iters)):

    # Propose movement of agents who are still free
    di, dj = cp.random.randint(-1, 2, size = (2, walkers))
    newi = (i + di * mask) % side
    newj = (j + dj * mask) % side

    # Update the mask for the agents who will be trapped
    trapped = board[newi, newj] > 0
    mask[trapped] = 0
    board[i[trapped], j[trapped]] = 1

    # Do the movement for the agents who are still free
    i = (i + di * mask) % side
    j = (j + dj * mask) % side

    if it % save_every == 0:
        stored[..., it//save_every] = board
        #stored[i[~trapped], j[~trapped]] = -1

stored = cp.asnumpy(stored)

print('Simulation finished!')

###### Make the animation

cmap = mpl.colors.LinearSegmentedColormap.from_list("", [(0.0, 'darkblue'), (1, 'aqua')])

fig, ax = plt.subplots(frameon=False, figsize= (8,8))
fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
ish = ax.imshow(stored[..., 0], cmap = cmap)

def update(i):
    ii = min(i, frames - 1)
    ish.set_data(stored[..., ii])

ax.axis('off')
animation_fig = animation.FuncAnimation(fig, update, frames = frames + 100, interval = 50)
animation_fig.save("MIB0012_BrownianTree.mp4", dpi = 200)

print('Animation finished!')
