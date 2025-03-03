import numpy as np
from scipy.ndimage import gaussian_filter, sobel
import matplotlib.pyplot as plt
import matplotlib as mpl

shape        = (1000, 1000)
xrange       = (-1.1, 1.1)
yrange       = (-1.1, 1.1)
np.random.seed(42)

x = np.linspace(*xrange, shape[0])[:, None]
y = np.linspace(*yrange, shape[1])[None, :]

turn = np.exp(1j * np.pi / 3)
vecs = 0.7 * turn ** np.array([2,4,6])
for _ in range(7):
    vecs = np.array(list(map(lambda p: [p/3, turn.conj() * p/3, turn * p/3, p/3], vecs))).ravel()
boundary = 2*np.cumsum(vecs)
boundary -= np.mean(boundary)

idx1  = np.int32( shape[0] * (boundary.real - xrange[0]) / (xrange[1] - xrange[0]) )
idx2  = np.int32( shape[1] * (boundary.imag - yrange[0]) / (yrange[1] - yrange[0]) )

cookie = np.float64(x**2 + y**2 < 1) 
cookie[idx1, idx2] -= 10
cookie = gaussian_filter(cookie, 20)
cookie = sobel(cookie, axis = 0) + sobel(cookie, axis = 1)
cookie += 0.3 * np.random.rand(*shape) * ( x**2 + y**2 < 1.05**2 )

cookie = (cookie - np.min(cookie)) / (np.max(cookie) - np.min(cookie))

cmap = mpl.colors.LinearSegmentedColormap.from_list("", [(0.0, 'saddlebrown'), (0.6, 'orange'), (1, 'yellow')])

plt.axis('off')
plt.imshow(cookie.T, cmap = cmap, origin = 'lower', extent = xrange + yrange)
plt.plot(boundary.real, boundary.imag, color = 'saddlebrown', linewidth = 0.5)
plt.savefig('MIB0029_KochSnowflake.png', dpi = 1000, bbox_inches='tight', pad_inches = 0)
