import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from tqdm.auto import tqdm

shape       = (1000, 1000)
xrange      = (-145, 155)
yrange      = (-50, 250)

board       = 1e-6 * np.ones(shape)
factor      = 1.7
angle       = 0.25

cmap = mpl.colors.LinearSegmentedColormap.from_list("", [(0.0, 'darkblue'), (0.3, 'yellow') , (1, 'crimson')])
phase_up = np.exp(+ angle * 1j)
phase_dn = np.exp(- factor * angle * 1j)

def collatz(n):
    point   = 0 + 0j
    segment = np.exp(- np.pi * 1j / 4)
    steps   = [] 
    
    while n > 1:
        if n % 2 == 0:
            n /= 2
            steps.append(True)
        else:
            n = 3*n + 1
            steps.append(False)
            
    for step in steps[::-1]:
        if step:
            segment *= phase_up
        else:
            segment *= phase_dn

        for s in range(10):
            point += segment / 10
            idx1 = np.int32( shape[0] * (point.real - xrange[0]) / (xrange[1] - xrange[0]) )
            idx2 = np.int32( shape[1] * (point.imag - yrange[0]) / (yrange[1] - yrange[0]) )
            board[idx1, idx2] += 1
        
for n in tqdm(range(1, 50001)):
    collatz(n)

plt.axis('off')
plt.imshow(board.T, cmap = cmap, norm = 'log', interpolation = 'bilinear', origin = 'lower', extent = xrange + yrange)
plt.savefig('MIB0011_CollatzConjecture.png', dpi = 1000, bbox_inches='tight', pad_inches = 0)
