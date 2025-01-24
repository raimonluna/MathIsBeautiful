import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def hopf_fibration(phi, theta, alpha):
    X0 = np.cos((alpha + phi)/2) * np.sin(theta/2)
    X1 = np.sin((alpha + phi)/2) * np.sin(theta/2)
    X2 = np.cos((alpha - phi)/2) * np.cos(theta/2)
    X3 = np.sin((alpha - phi)/2) * np.cos(theta/2)
    return np.array([X0, X1, X2]) / (1 - X3)

def get_family(theta, N = 1000):
    t_vals     = np.arange(0, 1, 0.02)
    alpha_vals = np.linspace(0, 4*np.pi, N)
    all_fibers = np.hstack([hopf_fibration(t * (4/3) * np.pi, theta, alpha_vals) for t in t_vals])
    return all_fibers

fibers1 = get_family(np.pi/2 + 0.5, N = 500 )
fibers2 = get_family(np.pi/2,       N = 1000)
fibers3 = get_family(np.pi/2 - 0.5, N = 2500)

all_fibers = np.hstack([fibers1, fibers2, fibers3])
all_colors = np.linspace(0, 1, all_fibers.shape[1])

##### Plotting part

fig  = plt.figure()
fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
ax   = fig.add_subplot(projection='3d')

ax.view_init(elev = 20, azim = 0)
ax.axis('off')

ax.set_xlim(-2.7, 2.7)
ax.set_ylim(-2.7, 2.7)
ax.set_zlim(-1.1, 1.2)
ax.set_facecolor("black")

ax.scatter(*all_fibers, c = all_colors, s= 0.3, cmap = 'hsv', alpha = 0.5)
plt.savefig('MIB0028_HopfFibration.png', dpi = 1000, bbox_inches='tight', pad_inches = 0)
