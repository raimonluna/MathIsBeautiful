import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.animation as animation
from tqdm.auto import tqdm
from scipy.ndimage import gaussian_filter
from skimage import exposure

####################### PARAMETERS ########################

shape       = (2000, 2000)
total_iters = 10000000
xrange      = (-5.25, 5.25)
yrange      = (0, 10.5)
frames      = 400

leaves = [3, 2]
moves  = [[2] + (l-1)*[1] for l in leaves]

coefs = np.asarray([
[0,	0,	0,	0.16,	0,	0,	0.01],
[0.85,	0.04,	-0.04,	0.85,	0,	1.60,	0.85],
[0.20,	-0.26,	0.23,	0.22,	0,	1.60,	0.07],
[-0.15,	0.28,	0.26,	0.24,	0,	0.44,	0.07]])

####################### FUNCTIONS #########################

def get_transform(index):
    return coefs[index, :4].reshape(2, 2), coefs[index, 4:6].reshape(2, 1)

def inverse_transform(M, b):
    Minv = np.linalg.inv(M)
    return Minv, -Minv @ b

def partial_transform(M, b, t):
    I = np.eye(2)
    return M + t * (I - M), (1 - t) * b

def base_fern():
    coords  = np.zeros((2, 1))
    dotlist = []
    print('Generating base fern...')
    for i in tqdm(range(total_iters)):
        r = np.random.rand()
        c = 0.0
        for j in range(4):
            c += coefs[j, 6]
            if r < c:
                M, b = get_transform(j)
                coords  = M @ coords + b
                dotlist.append(coords[:,0])
                break
    return np.array(dotlist).T

def transform_chain(chain):
    M = np.eye(2)
    b = np.zeros((2,1))
    for t in chain:
        Mn, bn = get_transform(t)
        M, b   = Mn @ M, Mn @ b + bn 
    return M, b

def transform_at_stage(n):
    chain = []
    for i in range(n+1):
        chain += (2*moves)[i]
    return transform_chain(chain)

def print_fern(dotlist, board):
    idx1 = np.int32( shape[0] * (dotlist[0] - xrange[0]) / (xrange[1] - xrange[0]) )
    idx2 = np.int32( shape[1] * (dotlist[1] - yrange[0]) / (yrange[1] - yrange[0]) )
    
    big_enough = (np.max(idx1) - np.min(idx1) > 10) | (np.max(idx2) - np.min(idx2) > 10)
    in_range   = (idx1 > 0) & (idx1 < shape[0]) & (idx2 > 0) & (idx2 < shape[1])
    to_plot    = in_range & big_enough
    
    if to_plot.any():
        board[idx1[in_range], idx2[in_range]] = 1
        return True
    return False

def backwards_cascade(Mt, bt, fern):
    
    board  = np.zeros(shape)
    
    for i in range(len(leaves) + 2):
        Mi, bi  = inverse_transform(*transform_at_stage(i))
        dotlist = Mi @ fern + bi
        dotlist = Mt @ dotlist + bt
        
        size_x    = np.max(dotlist[0,:]) - np.min(dotlist[0,:])
        subsample = np.int32(1000 / size_x**2) + 1
        #print(subsample)
        
        print_fern(dotlist[:, ::subsample], board)

    board = gaussian_filter(board, shape[0] / 2000)
    board = exposure.equalize_adapthist(board, clip_limit = 0.1)

    return board
        
################### GENERATE FERN #########################

fern   = base_fern()
M0, b0 = transform_at_stage(len(leaves) - 1)

##################### MAKE VIDEO ##########################

print('Making movie...')

fig, ax = plt.subplots(figsize = (8, 8))
fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
plt.close()
ax.axis('off')

cmap   = mpl.colors.LinearSegmentedColormap.from_list("", [(0.0, 'black'), (0.2, 'darkblue'), (0.95, 'cyan'), (1, 'cyan')])

Mt, bt  = partial_transform(M0, b0, 0)
board   = backwards_cascade(Mt, bt, fern.copy())

pt1  = ax.imshow(board.T, cmap = cmap, interpolation = 'bilinear', origin = 'lower', extent = xrange + yrange)

def update(i):
    global board
    print(f'{i+1}/{frames}', end="\r", flush=True)
    
    Mt, bt  = partial_transform(M0, b0, (i / (frames-1))**2)
    board   = backwards_cascade(Mt, bt, fern.copy())

    pt1.set_data(board.T)

    return fig

animation_fig = animation.FuncAnimation(fig, update, frames = frames, interval = 50)
animation_fig.save("MIB0060_BarnsleyZoom.mp4", dpi = fig.dpi)

###########################################################


