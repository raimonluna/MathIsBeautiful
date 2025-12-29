import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

def gamma(z):
    return (z - 1) / (z + 1)

def plot_patch(ax, prevsize, size, step, color = 'k', lw = 0.5):
    for y in np.arange(0, size + step, step):
        x = np.linspace(prevsize * (y < prevsize), size, 100)
        Gh, Gv = gamma(x + 1j * y), gamma(y + 1j * x)
        ax.plot(Gh.real, Gh.imag, Gh.real, -Gh.imag, color = color, lw = lw)
        ax.plot(Gv.real, Gv.imag, Gv.real, -Gv.imag, color = color, lw = lw)

fig, ax = plt.subplots(1,1, figsize = (10,10))
cmap = mpl.colormaps['tab20']
fig.set_facecolor('k')
ax.axis('off')

#Circle
theta = np.linspace(0, 2 * np.pi, 100)
ax.fill(np.cos(theta), np.sin(theta), 'wheat', zorder = -1)

#First patch
prevsize, size, step, box = 0, 0.2, 0.01, 5

#Iterate patches
for i in range(10):
    plot_patch(ax, prevsize, size, step, color = cmap(i/10 + 0.05), lw = 1.5)
    plot_patch(ax, prevsize, size, box * step, color = cmap(i/10), lw = 1.5)
    prevsize = size
    size *= [5/2, 2, 2][i%3]
    step *= [2, 5/2, 2, 2, 5][i * (i < 5) - ((i + 1) % 2) * (i > 4)]
    box   = [5, 2, 2, 5][min(i,3)]

plt.tight_layout()
plt.savefig('MIB0056_SmithChart.png', dpi = 200, bbox_inches='tight', pad_inches = 0)
