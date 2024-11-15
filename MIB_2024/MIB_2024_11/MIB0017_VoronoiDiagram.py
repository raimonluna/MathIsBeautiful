import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

size      = 1000
point_num = 50
np.random.seed(42)

x, y = np.arange(size)[:, None, None], np.arange(size)[None, :, None]
x0, y0 = np.random.uniform(0, size, size = (2, 1, 1, point_num))

voronoi = np.sort(np.sqrt((x - x0)**2 + (y - y0)**2), axis = 2)
voronoi = np.abs(voronoi[..., 0] - voronoi[..., 1]) / np.maximum(voronoi[..., 0], voronoi[..., 1])

cmap = mpl.colors.LinearSegmentedColormap.from_list("", [(0.0, 'black'), (0.2, 'black'),
                                                         (0.5, 'white'), (0.7, 'white'), 
                                                         (0.8, 'brown'), (0.9, 'brown'), 
                                                         (0.95,'black'), (1, 'black')])
plt.axis('off')
plt.imshow(voronoi.T + 1, cmap = cmap, norm = 'log', origin = 'lower', extent = (0, size, 0, size))
plt.savefig('MIB0017_VoronoiDiagram.png', dpi = 1000, bbox_inches='tight', pad_inches = 0)
