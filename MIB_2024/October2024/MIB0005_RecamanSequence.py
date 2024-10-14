import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

total_terms = 100
angle       = np.pi/4

ax     = plt.figure().add_subplot()
cmap   = mpl.colormaps['hsv']
rotate = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])

recaman = [0]
for n in range(1, total_terms):
    an   = recaman[-1]
    sign = 2 * ((an - n in recaman) or (an < n)) - 1
    next = an + sign * n
    recaman.append(next)

    r = (next - an) / 2
    theta = np.linspace(0, np.pi, 300)
    xy = np.vstack((an + r + r * np.cos(theta), sign * (-1)**n * r * np.sin(theta)))
    xy = rotate @ xy
    ax.plot(*xy, color = cmap(256 * n // total_terms), linewidth = 0.5)

ax.set_facecolor("black")
ax.xaxis.set_visible(False)
ax.yaxis.set_visible(False)
plt.gca().set_aspect('equal')
plt.savefig('MIB0005_RecamanSequence.png', dpi = 1000, bbox_inches='tight', pad_inches = 0)
