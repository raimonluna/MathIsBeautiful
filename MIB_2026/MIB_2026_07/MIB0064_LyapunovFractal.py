import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from tqdm.auto import tqdm

shape        = (1000, 1000)
arange       = (3.1, 4.0)
brange       = (2.5, 3.4)
total_iters  = 100
string       = 'BBBBBBAAAAAA'

a   = np.linspace(*arange, shape[0])[:, None]
b   = np.linspace(*brange, shape[1])[None, :]
rs  = (a, b)
eps = 1e-15
x   = np.ones(shape) / 2
lam = np.zeros(shape)

for it in tqdm(range(total_iters)):
    r    = rs[ord(string[it % len(string)]) - 65]
    if it > 0:
        lam += np.log(np.abs(r * (1 - 2 * x)) + eps) / total_iters
    x    = r * x * (1 - x)
    
plt.axis('off')
plt.imshow(-lam, cmap = 'inferno',   norm = 'log', origin = 'lower', extent = brange + arange)
plt.imshow(+lam, cmap = 'inferno_r', norm = 'log', origin = 'lower', extent = brange + arange)
plt.savefig('MIB0064_LyapunovFractal.png', dpi = 1000, bbox_inches='tight', pad_inches = 0)
