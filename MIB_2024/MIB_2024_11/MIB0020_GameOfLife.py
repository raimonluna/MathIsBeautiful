import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

size   = 115
frames = 30 * 21

gun = [
'000000000000000000000000100000000000',
'000000000000000000000010100000000000',
'000000000000110000001100000000000011',
'000000000001000100001100000000000011',
'110000000010000010001100000000000000',
'110000000010001011000010100000000000',
'000000000010000010000000100000000000',
'000000000001000100000000000000000000',
'000000000000110000000000000000000000']

neighbors = [(di, dj) for di in range(-1, 2) for dj in range(-1, 2) if di**2 + dj**2 > 0]
board     = np.zeros((size, size))

for i in range(len(gun)):
    for j in range(len(gun[0])):
        board[i + 20, j + 10]  = np.int32(gun[i][j])
        board[i + 20, 105 - j] = np.int32(gun[i][j])
        board[90 - i, j + 10]  = np.int32(gun[i][j])
        board[90 - i, 105 - j] = np.int32(gun[i][j])

fig, ax = plt.subplots(frameon=False, figsize= (5, 5))
fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
ish = ax.imshow(board, cmap = 'gnuplot', vmin = 0, vmax = 10)

def update(i):
    global board, board_mem
    surr  = np.sum([np.roll(board, (di, dj), axis = (0, 1)) for (di,dj) in neighbors], axis = 0)
    board = board * (surr > 1) * (surr < 4) + (1 - board) * (surr == 3)
    ish.set_data(surr * (1 - board) + 10 * board)

for i in range(90):
    update(i)

ax.axis('off')
animation_fig = animation.FuncAnimation(fig, update, frames = frames, interval = 50)
animation_fig.save("MIB0020_GameOfLife.mp4", dpi = 200)
