import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

size    = 200
start   = 1000
frames  = 500
np.random.seed(42)

saved     = np.zeros((size, size, 3, frames), dtype = np.int32)
neighbors = [(di, dj) for di in range(-1, 2) for dj in range(-1, 2) if di**2 + dj**2 > 0]
initial   = np.random.randint(0, 3, size = (size, size))
board     = np.dstack([np.int32(initial == i) for i in range(3)])

for k in range(start + frames):
    surr  = np.sum([np.roll(board, (di, dj), axis = (0, 1)) for di,dj in neighbors], axis = 0)
    board = np.dstack([  (board[...,i] * (surr[..., (i+1)%3] <  3 )) 
                       + (board[...,(i-1)%3] * (surr[..., i] >= 3 ))  for i in range(3)])
    if k >= start:
        saved[..., k - start] = board

fig, ax = plt.subplots(frameon=False, figsize= (8,8))
fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
ish = ax.imshow(255*saved[...,0])
for k in range(size):
    ax.axhline(y = k - 0.5, color = 'k', linewidth = 0.5)
    ax.axvline(x = k - 0.5, color = 'k', linewidth = 0.5)

ax.axis('off')
animation_fig = animation.FuncAnimation(fig, lambda i: ish.set_data(255*saved[...,i]), frames = frames, interval = 50)
animation_fig.save("MIB0016_RockPaperScissors.mp4", dpi = 200)
