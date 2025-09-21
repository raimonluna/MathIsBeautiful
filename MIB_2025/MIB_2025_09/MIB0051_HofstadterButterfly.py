import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

xrange  = (-4, 4)
yrange  = (0, 1)
shape   = (2000, 2000)
max_q   = 100

cmap = mpl.colors.LinearSegmentedColormap.from_list("", [(0.0, 'wheat'), (0.1, 'wheat'), (0.5, 'green') , (1, 'blue')])

def gcd(a, b): 
    if b == 0: return a
    return gcd(b, a % b)
    
def H(p, q, kx, ky):
    M  = np.diag(2 * np.cos(ky - 2 * np.pi * np.arange(q) * p / q) + 0j)
    ar = np.arange(q - 1)
    M[ar, ar + 1] = M[ar + 1, ar] = 1
    M[0, -1] += np.exp( - q * 1j * kx)
    M[-1, 0] += np.exp( + q * 1j * kx)
    return M

rationals = [(p, q) for q in range(1, max_q + 1) for p in range(1, q) if gcd(p, q) == 1 ]
butterfly = np.ones(shape)

for p, q in rationals:

    x1 = np.linalg.eigvalsh(H(p, q, kx = 0, ky = 0))
    x2 = np.linalg.eigvalsh(H(p, q, kx = np.pi / q, ky = np.pi / q))

    idx1 = np.int32( shape[0] * (x1 - xrange[0]) / (xrange[1] - xrange[0]) )
    idx2 = np.int32( shape[0] * (x2 - xrange[0]) / (xrange[1] - xrange[0]) )
    idx1, idx2 = np.minimum(idx1, idx2), np.maximum(idx1, idx2)
    
    idx3 = np.int32( shape[1] * (p/q - yrange[0]) / (yrange[1] - yrange[0]) )
    
    for i1, i2 in zip(idx1, idx2):
        butterfly[i1 - 1:i2 + 1, idx3] += 1

plt.axis('off')
plt.imshow(butterfly.T, cmap = cmap, norm = 'log', origin = 'lower', vmin = 1, vmax = 5)
plt.savefig('MIB0051_HofstadterButterfly.png', dpi = 1000, bbox_inches='tight', pad_inches = 0)
