import numpy as np
import matplotlib.pyplot as plt
import time

##### 3x3 Grid Properties

checks = [
[],
[(1,2,0,0)],
[(2,1,1,3)],
[(3,0,2,2), (3,1,0,3)],
[(4,0,3,2)],
[(5,3,4,1),(5,0,0,2)],
[(6,3,5,1)],
[(7,2,6,0),(7,3,0,1)],
[(8,2,7,0),(8,3,1,1)]
]

order = np.array([6, 7, 8, 5, 0, 1, 4, 3, 2])

CALLS = 0

##### Functions

def scramble(cards):
    rand_config = np.vstack([np.random.permutation(9), np.random.randint(0,4,9)])
    rand_board  = cards[rand_config[0]]
    rand_board  = np.asarray([np.roll(rand_board[i], rand_config[1, i]) for i in range(9)])
    return rand_board

def plot_config(config):
    
    board    = cards[config[0]]
    board    = np.asarray([np.roll(board[i], config[1, i]) for i in range(9)])
    board    = board[order]
    
    plane    = np.array([[1, 1, 0.5], [-0.5, 0.5, -0.5]])
    rot      = np.array([[0, -1],[1, 0]])
    colors   = ('red', 'green', 'white', 'orange')
    
    fig, axs = plt.subplots(3, 3, figsize = (8,8))
    for i, ax in enumerate(axs.ravel()):
        ax.set_facecolor("darkblue")
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        ax.set_xlim(-1,1)
        ax.set_ylim(-1,1)
        if board[i,0] == 0:
            continue
        plot_plane = plane
        for j in range(4):
            sym = np.roll(np.array([[1], [np.sign(board[i, j])]]), j%2, 0)
            ax.fill(*(sym * plot_plane), color = colors[np.abs(board[i, j]) - 1])
            plot_plane = rot @ plot_plane
    plt.tight_layout()
    plt.savefig('MIB0021_ScrambleSquares.png', dpi = fig.dpi,
                bbox_inches='tight', pad_inches = 0)

def fits(k, config, new_card, new_rot):
    for a,b,c,d in checks[k]:
        new_edge = cards[new_card, (b - new_rot) % 4]
        old_edge = cards[config[0, c], (d - config[1, c]) % 4]
        if new_edge + old_edge != 0:
            return False
    return True

def solve(k, config, available):
    global CALLS
    CALLS += 1
    
    if k == 9:
        return True
        
    for new_card in np.arange(9)[available]:
        for new_rot in range(1 + 3*(k > 0)):
            if fits(k, config, new_card, new_rot):
                
                config[0, k] = new_card
                config[1, k] = new_rot
                available[new_card] = False
                
                if solve(k + 1, config, available):
                    return True
    
                config[0, k] = 0
                config[1, k] = 0
                available[new_card] = True
    return False

##### Solve the Game!

cards = np.array([
[ 3,  2,  4, -1],
[-4, -1,  2,  3],
[-4, -4, -1, -3],
[ 3, -4, -1, -2],
[ 4, -4,  1,  2],
[-3, -2, -3,  1],
[ 3, -2,  4, -1],
[ 2,  3, -4, -1],
[ 1, -2,  2, -3]])

np.random.seed(2)

cards      = scramble(cards)
config     = np.zeros((2,9), dtype = np.int32)
available  = np.ones(9, dtype = bool)

tic        = time.time()
if solve(0, config, available):
    toc = 1000 * (time.time() - tic)
    print('Solution:\n', config, '\n')
    print(f'Elapsed time: {toc:.2f} milliseconds.')
    print(f'Explored nodes: {CALLS}.\n')
    plot_config(config)
else:
    print('No solution found!')

