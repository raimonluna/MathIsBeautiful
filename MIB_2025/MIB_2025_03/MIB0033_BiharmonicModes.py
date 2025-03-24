import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eig
from scipy.interpolate import RectBivariateSpline

def ChebyshevD(N):
    row = np.arange(N + 1)[:, None]
    col = np.arange(N + 1)[None, :]
    x  = np.cos(np.pi * row / N)
    c  = ((1 + (col % N == 0)) * (-1) ** col).T
    X  = np.ones(N + 1) * x
    dX = X - X.T
    D  = c @ (1 / c.T) / (dX + np.eye(N + 1))
    D1 = D - np.diag(np.sum(D, axis=1))
    return x, D1

# Prepare grid and differentiation matrices
N = 17
x, D1 = ChebyshevD(N)
xf = x.flatten()
xl = np.linspace(-1, 1, 100)[: None]
I  = np.eye(N-1)
D2 = D1 @ D1 
D3 = D1 @ D2
D4 = D1 @ D3

# Build the differential biharmonic operator
S  = np.diag(np.hstack([0, 1 / (1 - xf[1:-1]**2), 0]))
L4 = np.diag(1 - xf**2) @ D4 - 8 * np.diag(xf) @ D3 - 12 * D2
L2, L4 = D2[1:-1,1:-1], (L4 @ S)[1:-1,1:-1]
L  = np.kron(I, L4) + np.kron(L4, I) + 2 * np.kron(L2, I) @ np.kron(I, L2)

# Solve the eigenvalue problem and sort indices
lam2, VV = eig(L)
indices  = np.argsort(lam2.real)
lam      = np.sqrt(lam2.real[indices])
VV       = VV.real[:, indices]
V        = np.zeros((N+1, N+1, (N-1)**2))
V[1:-1, 1:-1, :] = VV.reshape(N-1, N-1, (N-1)**2)
V       /= np.linalg.norm(V, axis = (0,1))

# Do the interpolation and plotting
fig, axes = plt.subplots(5, 5, figsize= (20, 20), sharex=True, sharey=True)
fig.subplots_adjust(0,0,1,1,0,0)
plt.rc('font', size = 18)

for i, ax in enumerate(axes.ravel()):
    ax.axis('off')
    interp = RectBivariateSpline(-xf, -xf, V[..., i])
    ax.imshow(np.abs(interp(xl, xl.T)).T, norm = 'log', cmap = 'binary', vmin = 1e-5, vmax = 1,
              interpolation='bilinear', origin = 'lower', extent = (-1,1,-1,1))
    #ax.set_title(f'{lam[i]/lam[0]:.4f}')

plt.savefig('MIB0033_BiharmonicModes.png', dpi = fig.dpi, bbox_inches='tight', pad_inches = 0)
